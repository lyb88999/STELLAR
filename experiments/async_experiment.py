
from experiments.grouping_experiment import SimilarityGroupingExperiment
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import copy

class AsyncGroupingExperiment(SimilarityGroupingExperiment):
    """
    异步/延迟更新实验
    模拟卫星网络中的异步更新和陈旧梯度问题
    """
    def __init__(self, config_path, delay_prob=0.3, max_delay=3, staleness_alpha=0.5):
        super().__init__(config_path)
        self.delay_prob = delay_prob  # 发生延迟的概率
        self.max_delay = max_delay    # 最大延迟轮数
        self.staleness_alpha = staleness_alpha # 陈旧更新的衰减系数
        
        # 延迟缓冲区: {delivery_round: [list of results]}
        self.delayed_buffer = {}
        
        self.logger.info(f"初始化异步实验: DelayProb={delay_prob}, MaxDelay={max_delay}, StalenessAlpha={staleness_alpha}")

    def _station_aggregation(self, station_id, station):
        """
        重写: 获取聚合结果但不立即发送给全局聚合器，而是返回给调用者进行异步调度
        """
        try:
            # 在调用 get_aggregated_update 之前，先计算总样本数，因为 get_aggregated_update 会清除 pending_updates
            updates = station.pending_updates.get(self.current_round, {})
            if not updates:
                return False, None, 0
            
            current_total_samples = sum(u.num_clients for u in updates.values())
            
            # 获取聚合结果 (这会消耗掉 pending_updates)
            aggregated_update = station.get_aggregated_update(self.current_round)
            if aggregated_update:
                self.logger.info(f"地面站 {station_id} 完成聚合 (将被异步调度)")
                # 返回 (成功标志, 模型更新, 样本数量)
                return True, aggregated_update, current_total_samples
            
            return False, None, 0
        except Exception as e:
            self.logger.error(f"地面站 {station_id} 聚合异常: {e}")
            return False, None, 0             

    def train(self):
        """重写训练循环以引入异步更新"""
        # 初始化记录列表
        accuracies = []
        losses = []
        # 新增的分类指标列表
        precision_macros = []
        recall_macros = []
        f1_macros = []
        precision_weighteds = []
        recall_weighteds = []
        f1_weighteds = []
        fresh_counts = []
        stale_counts = []
        energy_stats = {
            'training_energy': [],
            'communication_energy': [],
            'total_energy': []
        }
        satellite_stats = {
            'training_satellites': [],
            'receiving_satellites': [],
            'total_active': []
        }

        current_time = datetime.now().timestamp()
        self.current_round = 0
        best_accuracy = 0
        rounds_without_improvement = 0
        best_f1 = 0
        
        # 禁用早停
        self.disable_early_stopping = True

        # 初始化所有轨道的分组和代表节点
        # 尝试初始化 0 到 N，自动检测存在的轨道
        for orbit in range(0, self.config['fl']['num_orbits']+1):
            self._init_orbit_structures(orbit)

        for round_num in range(self.config['fl']['num_rounds']):
            self.current_round = round_num
            self.logger.info(f"\n=== 开始第 {round_num + 1} 轮训练 (Async) === 时间：{datetime.fromtimestamp(current_time)}")
            
            # 1. 执行轨道训练 (计算开销等)
            orbit_successes = 0
            round_training_energy = 0
            round_comm_energy = 0
            round_training_sats = set()
            round_receiving_sats = set()
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_orbit = {}
                # 仅提交已成功初始化的轨道
                valid_orbits = list(self.orbit_groups.keys())
                self.logger.info(f"本轮处理轨道: {valid_orbits}")
                
                for orbit_id in valid_orbits:
                    future = executor.submit(
                        self._handle_orbit_training,
                        None,
                        orbit_id,
                        current_time
                    )
                    future_to_orbit[future] = orbit_id

                for future in as_completed(future_to_orbit):
                    orbit_id = future_to_orbit[future]
                    try:
                        result = future.result()
                        if isinstance(result, tuple) and len(result) == 2:
                            success, orbit_stats = result
                            if success:
                                orbit_successes += 1
                            if orbit_stats:
                                round_training_energy += orbit_stats['training_energy']
                                round_comm_energy += orbit_stats['communication_energy']
                                round_training_sats.update(orbit_stats['training_satellites'])
                                round_receiving_sats.update(orbit_stats['receiving_satellites'])
                    except Exception as e:
                        self.logger.error(f"处理轨道 {orbit_id} 时出错: {str(e)}")
            
            # 记录本轮统计信息
            energy_stats['training_energy'].append(round_training_energy)
            energy_stats['communication_energy'].append(round_comm_energy)
            energy_stats['total_energy'].append(round_training_energy + round_comm_energy)
            satellite_stats['training_satellites'].append(len(round_training_sats))
            satellite_stats['receiving_satellites'].append(len(round_receiving_sats))
            satellite_stats['total_active'].append(len(round_training_sats | round_receiving_sats))

            # 2. 地面站聚合 & 异步缓冲
            # 先执行标准的地面站聚合，得到 station_results
            station_results = [] # This line is effectively replaced by current_round_updates
            if orbit_successes > 0:
                self.logger.info(f"\n=== 地面站聚合阶段 (Async Buffer) === ({orbit_successes} 个轨道成功)")
                
                # 获取本轮"原本"应该到达的更新
                current_round_updates = []
                with ThreadPoolExecutor(max_workers=3) as executor:
                    # 使用我们重写的 _station_aggregation
                    future_to_station = {
                        executor.submit(self._station_aggregation, station_id, station): station_id
                        for station_id, station in self.ground_stations.items()
                    }
                    for future in as_completed(future_to_station):
                        try:
                            # 返回值: (success, model, samples)
                            success, model, samples = future.result()
                            if success and model:
                                station_id = future_to_station[future]
                                # 存入
                                current_round_updates.append((station_id, model, samples))
                        except Exception as e:
                            self.logger.error(f"地面站聚合出错: {str(e)}")
                
                # --- 异步逻辑核心 ---
                # 1. 将本轮更新分流：立即提交 vs 延迟提交
                ready_updates = []
                
                for station_id, model, samples in current_round_updates:
                    if random.random() < self.delay_prob:
                        # 延迟
                        delay = random.randint(1, self.max_delay)
                        delivery_round = round_num + delay
                        if delivery_round not in self.delayed_buffer:
                            self.delayed_buffer[delivery_round] = []
                        # 深度拷贝模型以防引用问题
                        self.delayed_buffer[delivery_round].append((station_id, copy.deepcopy(model), samples, round_num))
                        self.logger.info(f"ASYNC: 地面站 {station_id} 更新延迟 {delay} 轮 -> Round {delivery_round+1}")
                    else:
                        # 立即
                        ready_updates.append((station_id, model, samples, round_num))
                
                # 2. 检查缓冲区，提取本轮到期的旧更新
                if round_num in self.delayed_buffer:
                    stale_list = self.delayed_buffer[round_num]
                    self.logger.info(f"ASYNC: 提取到 {len(stale_list)} 个滞后更新")
                    for item in stale_list:
                        ready_updates.append(item)
                    del self.delayed_buffer[round_num]
                
                # 3. 将 ready_updates 注入回 self.ground_stations 以供 _perform_global_aggregation 使用
                # 这是一个 Hack。因为 _perform_global_aggregation 是遍历 self.ground_stations 并读取其 .model
                # 为了让它读到我们混合了新旧模型的列表，我们需要：
                # 临时修改 self.ground_stations 的状态？
                # 或者重写 _perform_global_aggregation。
                # 鉴于 _perform_global_aggregation 代码较短，我们在下面直接重写这部分的聚合逻辑。
                
                if len(ready_updates) >= 1:
                    self.logger.info(f"\n=== 全局聚合阶段 ({len(ready_updates)} 个可用更新) ===")
                    
                    # 统计本轮的新鲜/陈旧更新数
                    n_fresh = sum(1 for _, _, _, origin in ready_updates if origin == round_num)
                    n_stale = sum(1 for _, _, _, origin in ready_updates if origin < round_num)
                    fresh_counts.append(n_fresh)
                    stale_counts.append(n_stale)
                    
                    # 手动执行加权平均 (代替 _perform_global_aggregation)
                    total_weight = 0
                    global_weights = None
                    
                    for station_id, model, samples, origin_round in ready_updates:
                        # 计算权重 (Staleness Decay)
                        delay = round_num - origin_round
                        decay = 1.0 / (1.0 + self.staleness_alpha * delay)
                        weight = samples * decay
                        
                        if global_weights is None:
                            global_weights = {k: v * weight for k, v in model.items()}
                        else:
                            for k, v in model.items():
                                global_weights[k] += v * weight
                        
                        total_weight += weight
                    
                    # 平均
                    if global_weights and total_weight > 0:
                        for k in global_weights:
                            global_weights[k] = global_weights[k] / total_weight
                        
                        # 使用 strict=False 以忽略可能缺失的 BatchNorm 统计量 (running_mean/var)
                        self.model.load_state_dict(global_weights, strict=False)
                        self.logger.info("全局模型更新完成 (Async Aggregation)")
                        
                        success = True

                        metrics = self.evaluate() 
                        if isinstance(metrics, (float, int)):
                            metrics = {'accuracy': metrics, 'f1_macro': 0, 'precision_macro': 0, 'recall_macro': 0, 'f1_weighted': 0, 'precision_weighted': 0, 'recall_weighted': 0, 'test_loss': 0}

                        accuracies.append(metrics['accuracy'])
                        f1_weighteds.append(metrics['f1_weighted'])
                        # 记录真实的测试损失，而非由指标函数返回的
                        # evaluate() 返回的 metrics['test_loss'] 是准确的
                        losses.append(metrics.get('test_loss', 0))

                        self.logger.info(f"第 {round_num + 1} 轮 (Async) 指标: Accuracy={metrics['accuracy']:.2f}%, F1={metrics['f1_weighted']:.4f}")
                    else:
                        success = False
                        self.logger.warning("聚合权重为0")
                        accuracies.append(0); f1_weighteds.append(0); losses.append(0)
                else:
                    success = False
                    self.logger.warning("ASYNC: 本轮没有可用更新 (全部延迟或失败)")
                    accuracies.append(0); f1_weighteds.append(0); losses.append(0)
                    fresh_counts.append(0); stale_counts.append(0)
            else:
                self.logger.warning("本轮无轨道成功")
                accuracies.append(0); f1_weighteds.append(0); losses.append(0)
                fresh_counts.append(0); stale_counts.append(0)

        return {
            "accuracies": accuracies,
            "f1_weighteds": f1_weighteds,
            "losses": losses,
            "fresh_counts": fresh_counts,
            "stale_counts": stale_counts
        }

