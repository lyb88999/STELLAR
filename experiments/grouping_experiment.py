from experiments.baseline_experiment import BaselineExperiment
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

class SimilarityGroupingExperiment(BaselineExperiment):
    def __init__(self, config_path: str = "configs/similarity_grouping_config.yaml"):
        """
        初始化基于数据相似度分组的联邦学习实验
        Args:
            config_path: 配置文件路径
        """
        super().__init__(config_path)
        
        # 获取分组配置
        self.max_distance = self.config['group'].get('max_distance', 2)
        self.max_group_size = self.config['group'].get('max_group_size', 5)
        self.similarity_threshold = self.config['group'].get('similarity_threshold', 0.5)
        self.max_group_size_threshold = self.config['group'].get('max_group_size_threshold', 4)
        self.similarity_refresh_rounds = self.config['group'].get('similarity_refresh_rounds', 5)
        
        # 获取权重配置
        weights_config = self.config['group'].get('weights', {})
        self.weights = {
            'alpha': weights_config.get('alpha', 0.4),
            'beta': weights_config.get('beta', 0.3),
            'gamma': weights_config.get('gamma', 0.3)
        }
        self.logger.info(f"使用相似度权重: alpha={self.weights['alpha']}, beta={self.weights['beta']}, gamma={self.weights['gamma']}")
        
        # 获取鲁棒性测试配置
        robustness_config = self.config.get('robustness', {})
        self.parameter_noise_level = robustness_config.get('parameter_noise_level', 0.0)
        self.noise_start_round = robustness_config.get('noise_start_round', 5)
        if self.parameter_noise_level > 0:
            self.logger.info(f"启用鲁棒性测试: 参数噪声等级={self.parameter_noise_level}, 起始轮次={self.noise_start_round}")
        
        # 初始化分组信息
        self.orbit_groups = {}  # {orbit_id: {satellite_id: group_id}}
        self.orbit_visited = {}  # {orbit_id: {satellite_id: True/False}}
        self.orbit_similarity_thresholds = {}  # {orbit_id: {satellite_id: threshold}}
        self.orbit_representatives = {}  # {orbit_id: {group_id: representative_id}}
        self.orbit_coordinators = {}  # {orbit_id: coordinator_id}
        
        # 存储轨道模型缓存，用于相似度计算
        self.satellite_model_cache = {}
        
        # 禁用早停 - 添加以下几行代码
        self.disable_early_stopping = True  # 添加一个标志，表示禁用早停
        
        self.logger.info("初始化基于数据相似度分组的联邦学习实验")
    
    def _init_orbit_structures(self, orbit_id: int):
        """
        初始化单个轨道的数据结构
        
        Args:
            orbit_id: 轨道ID (从1开始)
        """
        n_sats = self.config['fl']['satellites_per_orbit']
        
        # 获取该轨道实际存在的卫星
        existing_sats = []
        for i in range(1, n_sats+1):
            sat_id = f"satellite_{orbit_id}-{i}"
            if sat_id in self.clients:
                existing_sats.append(sat_id)
        
        if not existing_sats:
            self.logger.warning(f"轨道 {orbit_id} 没有找到任何卫星")
            return

        # 初始化分组信息，每个卫星初始独自为一组
        if orbit_id not in self.orbit_groups:
            self.orbit_groups[orbit_id] = {sat_id: f"group_{orbit_id}-{i+1}" 
                                        for i, sat_id in enumerate(existing_sats)}
        
        # 初始化访问状态，所有卫星初始未访问
        if orbit_id not in self.orbit_visited:
            self.orbit_visited[orbit_id] = {sat_id: False for sat_id in existing_sats}
        
        # 初始化相似度阈值，所有卫星初始阈值相同
        if orbit_id not in self.orbit_similarity_thresholds:
            self.orbit_similarity_thresholds[orbit_id] = {sat_id: self.similarity_threshold 
                                                        for sat_id in existing_sats}
        
        # 初始化代表节点信息，暂无代表节点
        if orbit_id not in self.orbit_representatives:
            self.orbit_representatives[orbit_id] = {}
        
        # 选择协调者节点（选择轨道中的第一个卫星）
        if orbit_id not in self.orbit_coordinators:
            self.orbit_coordinators[orbit_id] = existing_sats[0]
        
        self.logger.info(f"轨道 {orbit_id} 数据结构初始化完成，协调者: {self.orbit_coordinators[orbit_id]}，卫星数: {len(existing_sats)}")
    
    # def compute_similarity(self, model1: Dict[str, torch.Tensor], 
    #                       model2: Dict[str, torch.Tensor]) -> float:
    #     """
    #     计算两个模型的相似度（余弦相似度）
        
    #     Args:
    #         model1: 第一个模型的参数字典
    #         model2: 第二个模型的参数字典
            
    #     Returns:
    #         float: 模型相似度，范围在[-1, 1]
    #     """
    #     # 展平模型参数
    #     vec1 = self._flatten_model(model1)
    #     vec2 = self._flatten_model(model2)
        
    #     # 计算余弦相似度
    #     dot_product = torch.sum(vec1 * vec2)
    #     norm1 = torch.norm(vec1)
    #     norm2 = torch.norm(vec2)
        
    #     if norm1 == 0 or norm2 == 0:
    #         return 0.0
            
    #     similarity = dot_product / (norm1 * norm2)
    #     return similarity.item()

    def compute_similarity(self, model1: Dict[str, torch.Tensor], 
                     model2: Dict[str, torch.Tensor]) -> float:
        """
        改进的相似度计算 - 对参数进行更好的过滤和调整
        
        Args:
            model1: 第一个模型的参数字典
            model2: 第二个模型的参数字典
            
        Returns:
            float: 相似度值，范围在[0, 1]
        """
        try:
            # 过滤参数，只保留权重参数
            important_keys = [k for k in model1.keys() 
                            if "weight" in k or "bias" in k]
            
            vectors = []
            for name in important_keys:
                if name in model1 and name in model2:
                    param1 = model1[name].flatten()
                    param2 = model2[name].flatten()
                    
                    # 确保参数长度相同
                    if param1.shape == param2.shape:
                        # 规范化向量
                        norm1 = torch.norm(param1)
                        norm2 = torch.norm(param2)
                        
                        if norm1 > 0 and norm2 > 0:
                            param1 = param1 / norm1
                            param2 = param2 / norm2
                            
                            # 计算余弦相似度
                            cos_sim = torch.sum(param1 * param2).item()
                            vectors.append((cos_sim + 1) / 2)  # 将[-1,1]映射到[0,1]
            
            if not vectors:
                return 0.0
                
            # 返回所有参数相似度的平均值
            similarity = sum(vectors) / len(vectors)
            
            # 指数调整，增加区分度
            adjusted_similarity = similarity ** 0.5  # 平方根调整，增大低相似度
            
            return adjusted_similarity
            
        except Exception as e:
            self.logger.error(f"计算模型相似度时出错: {str(e)}")
            return 0.0
    
    def _flatten_model(self, model: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        将模型参数展平为一维向量
        
        Args:
            model: 模型参数字典
            
        Returns:
            torch.Tensor: 展平后的一维向量
        """
        # 过滤batchnorm参数
        filtered_params = []
        for name, param in model.items():
            if 'running_mean' not in name and 'running_var' not in name and 'num_batches_tracked' not in name:
                filtered_params.append(param.flatten())
                
        return torch.cat(filtered_params) if filtered_params else torch.tensor([])
    
    def _get_satellite_neighbors(self, sat_id: str, distance: int = 1) -> List[str]:
        """
        获取给定卫星的邻居节点
        
        Args:
            sat_id: 卫星ID
            distance: 距离
            
        Returns:
            List[str]: 邻居卫星ID列表
        """
        try:
            orbit_id, sat_num = self._parse_satellite_id(sat_id)
            neighbors = []
            satellites_per_orbit = self.config['fl']['satellites_per_orbit']
            
            # 向左遍历
            for i in range(1, distance + 1):
                left_num = sat_num - i
                if left_num <= 0:  # 环形处理
                    left_num = satellites_per_orbit + left_num
                
                neighbor_id = f"satellite_{orbit_id}-{left_num}"
                if neighbor_id in self.clients:
                    neighbors.append(neighbor_id)
                
            # 向右遍历
            for i in range(1, distance + 1):
                right_num = (sat_num + i - 1) % satellites_per_orbit + 1
                neighbor_id = f"satellite_{orbit_id}-{right_num}"
                if neighbor_id in self.clients:
                    neighbors.append(neighbor_id)
                
            return neighbors
        except Exception as e:
            self.logger.error(f"获取卫星 {sat_id} 的邻居出错: {str(e)}")
            return []
    
    def _get_group_members(self, orbit_id: int, group_id: str) -> List[str]:
        """
        获取组内成员
        
        Args:
            orbit_id: 轨道ID
            group_id: 组ID
            
        Returns:
            List[str]: 组内成员ID列表
        """
        return [sat_id for sat_id, g_id in self.orbit_groups[orbit_id].items() 
                if g_id == group_id]
    
    # def perform_grouping(self, orbit_id: int):
    #     """
    #     执行轨道内卫星分组
        
    #     Args:
    #         orbit_id: 轨道ID
    #     """
    #     self.logger.info(f"\n=== 开始轨道 {orbit_id} 的分组过程 ===")
        
    #     # 确保轨道已初始化
    #     self._init_orbit_structures(orbit_id)
        
    #     # 重置访问状态
    #     for sat_id in self.orbit_visited[orbit_id]:
    #         self.orbit_visited[orbit_id][sat_id] = False
        
    #     # 获取协调者节点
    #     coordinator = self.orbit_coordinators[orbit_id]
    #     self.logger.info(f"协调者: {coordinator}")
        
    #     # 更新模型缓存
    #     self._update_model_cache(orbit_id)
        
    #     # 遍历未访问的卫星节点作为源节点
    #     for sat_id in sorted(self.orbit_visited[orbit_id].keys()):
    #         try:
    #             if self.orbit_visited[orbit_id][sat_id]:
    #                 continue
                    
    #             # 标记当前源节点为已访问
    #             self.orbit_visited[orbit_id][sat_id] = True
                
    #             # 获取源节点当前所在组
    #             source_group = self.orbit_groups[orbit_id][sat_id]
    #             self.logger.info(f"处理源节点: {sat_id}, 当前组: {source_group}")
                
    #             # 初始化组内成员计数
    #             group_members = [sat_id]
                
    #             # 访问邻居节点
    #             neighbors = self._get_satellite_neighbors(sat_id, self.max_distance)
                
    #             # 维护一个已经检查过的节点集合
    #             checked_neighbors = set()
                
    #             # 遍历邻居节点
    #             for neighbor in neighbors:
    #                 # 如果源节点的邻居未被划入当前组，则停止考虑更远节点
    #                 if (len(checked_neighbors) > 0 and 
    #                     all(self.orbit_groups[orbit_id].get(n, "") != source_group 
    #                         for n in checked_neighbors)):
    #                     self.logger.info(f"邻居节点未被划入当前组，停止考虑更远节点")
    #                     break
                        
    #                 checked_neighbors.add(neighbor)
                    
    #                 # 如果邻居不在当前轨道，跳过
    #                 if neighbor not in self.orbit_visited[orbit_id]:
    #                     continue
                        
    #                 # 计算源节点与邻居节点的相似度
    #                 if sat_id in self.satellite_model_cache and neighbor in self.satellite_model_cache:
    #                     similarity = self.compute_similarity(
    #                         self.satellite_model_cache[sat_id], 
    #                         self.satellite_model_cache[neighbor]
    #                     )
    #                 else:
    #                     self.logger.debug(f"缺少模型缓存: {sat_id} 或 {neighbor}")
    #                     similarity = 0.0
                    
    #                 # 获取当前相似度阈值
    #                 current_threshold = self.orbit_similarity_thresholds[orbit_id][sat_id]
    #         except Exception as e:
    #             self.logger.error(f"处理源节点 {sat_id} 时出错: {str(e)}")
    #             continue
                
    #             if not self.orbit_visited[orbit_id][neighbor]:
    #                 # 邻居节点未被访问
    #                 if similarity >= current_threshold:
    #                     # 将邻居节点划入当前组
    #                     self.orbit_groups[orbit_id][neighbor] = source_group
    #                     self.orbit_visited[orbit_id][neighbor] = True
    #                     group_members.append(neighbor)
    #                     self.logger.info(f"将未访问节点 {neighbor} 划入组 {source_group}, 相似度: {similarity:.4f}")
    #             else:
    #                 # 邻居节点已被其他组访问
    #                 neighbor_group = self.orbit_groups[orbit_id][neighbor]
    #                 if (neighbor_group != source_group and 
    #                     similarity >= current_threshold):
    #                     # 计算邻居与其当前组代表的相似度
    #                     neighbor_group_rep = self._get_group_representative(orbit_id, neighbor_group)
    #                     if (neighbor_group_rep and 
    #                         neighbor_group_rep in self.satellite_model_cache and 
    #                         neighbor in self.satellite_model_cache):
                            
    #                         neighbor_group_similarity = self.compute_similarity(
    #                             self.satellite_model_cache[neighbor], 
    #                             self.satellite_model_cache[neighbor_group_rep]
    #                         )
                            
    #                         # 如果与当前源节点的相似度更高，则重新分配
    #                         if similarity > neighbor_group_similarity:
    #                             self.orbit_groups[orbit_id][neighbor] = source_group
    #                             group_members.append(neighbor)
    #                             self.logger.info(f"将已访问节点 {neighbor} 从组 {neighbor_group} 重新划入组 {source_group}")
                
    #             # 检查是否达到最大组大小阈值，若达到则调整相似度阈值
    #             if len(group_members) >= self.max_group_size_threshold:
    #                 # 计算组内最小相似度
    #                 min_similarity = 1.0
    #                 for member1 in group_members:
    #                     for member2 in group_members:
    #                         if member1 != member2:
    #                             if (member1 in self.satellite_model_cache and 
    #                                 member2 in self.satellite_model_cache):
                                    
    #                                 sim = self.compute_similarity(
    #                                     self.satellite_model_cache[member1],
    #                                     self.satellite_model_cache[member2]
    #                                 )
    #                                 min_similarity = min(min_similarity, sim)
                                
    #                 # 更新相似度阈值，确保在0.6到1之间
    #                 new_threshold = max(0.6, min(1.0, min_similarity))
    #                 self.orbit_similarity_thresholds[orbit_id][sat_id] = new_threshold
    #                 self.logger.info(f"组 {source_group} 达到阈值大小，调整相似度阈值为 {new_threshold:.4f}")
            
    #         # 记录源节点为代表节点
    #         self.orbit_representatives[orbit_id][source_group] = sat_id
            
    #         # 检查是否达到最大组大小，若达到则停止添加
    #         if len(group_members) >= self.max_group_size:
    #             self.logger.info(f"组 {source_group} 达到最大大小 {self.max_group_size}")
        
    #     # 总结分组结果
    #     groups = {}
    #     for sat_id, group_id in self.orbit_groups[orbit_id].items():
    #         if group_id not in groups:
    #             groups[group_id] = []
    #         groups[group_id].append(sat_id)
            
    #     self.logger.info(f"轨道 {orbit_id} 分组结果:")
    #     for group_id, members in groups.items():
    #         self.logger.info(f"  组 {group_id}: {members}")
            
    #     return groups

    def _perform_position_grouping(self, orbit_id: int, satellites: List[str]):
        """
        执行基于位置的卫星分组
        
        Args:
            orbit_id: 轨道ID
            satellites: 轨道内的卫星列表
        """
        self.logger.info(f"轨道 {orbit_id} 执行基于位置分组")
        
        # 每个组的大小 (从配置读取，默认为3)
        group_size = self.config.get('group', {}).get('initial_group_size', 3)
        
        # 分组计数器
        group_counter = 0
        
        # 创建分组
        for i in range(0, len(satellites), group_size):
            # 获取当前组的成员
            group_members = satellites[i:min(i+group_size, len(satellites))]
            
            # 如果是最后一组且成员数少于2，则合并到前一组
            # 注意：如果原本设定的group_size就是1，则不允许合并
            if group_size > 1 and len(group_members) < 2 and i > 0:
                # 获取前一组的ID
                prev_group_id = f"group_{orbit_id}-{group_counter-1}"
                
                # 将这些成员添加到前一组
                for member in group_members:
                    self.orbit_groups[orbit_id][member] = prev_group_id
                    self.orbit_visited[orbit_id][member] = True
                    
                self.logger.info(f"将最后 {len(group_members)} 个卫星合并到组 {prev_group_id}")
            else:
                # 创建新组
                current_group_id = f"group_{orbit_id}-{group_counter}"
                
                # 将成员添加到新组
                for member in group_members:
                    self.orbit_groups[orbit_id][member] = current_group_id
                    self.orbit_visited[orbit_id][member] = True
                    
                # 设置代表节点为组内第一个卫星
                self.orbit_representatives[orbit_id][current_group_id] = group_members[0]
                
                self.logger.debug(f"创建组 {current_group_id} 包含 {len(group_members)} 个卫星: {group_members}")
                
                # 递增组计数器
                group_counter += 1
        
        # 总结分组结果
        groups = {}
        for sat_id, group_id in self.orbit_groups[orbit_id].items():
            if group_id not in groups:
                groups[group_id] = []
            groups[group_id].append(sat_id)
            
        self.logger.debug(f"轨道 {orbit_id} 分组结果:")
        for group_id, members in groups.items():
            self.logger.debug(f"  组 {group_id}: {members}")
            
        return groups

    def perform_grouping(self, orbit_id: int):
        """
        执行轨道内卫星分组 - 严格按照STELLAR论文实现的物理邻域贪婪分组算法
        
        Algorithm Steps:
        1. Initialization: Start with unassigned satellites.
        2. Greedy Strategy: Source node absorbs PHYSICAL NEIGHBORS based on similarity.
        3. Dynamic Thresholding: Adjust threshold if group grows too large.
        4. Regrouping (Stealing): Allow reassignment if similarity is better.
        
        Args:
            orbit_id: 轨道ID
        """
        self.logger.info(f"\n=== 开始轨道 {orbit_id} 的分组过程 (Physical Neighbor Greedy) ===")
        
        # --- 0. 鲁棒性测试：注入参数噪声 (保持原逻辑) ---
        original_params = {}
        if self.current_round >= self.noise_start_round and self.parameter_noise_level > 0:
            self.logger.info(f"=== [鲁棒性测试] 轨道 {orbit_id} 注入参数传输噪声 (Level={self.parameter_noise_level}) ===")
            n_sats = self.config['fl']['satellites_per_orbit']
            orbit_satellites = [f"satellite_{orbit_id}-{i}" for i in range(1, n_sats+1)
                              if f"satellite_{orbit_id}-{i}" in self.clients]
            for sat_id in orbit_satellites:
                if sat_id in self.clients:
                    client = self.clients[sat_id]
                    # 保存原始参数
                    original_params[sat_id] = {k: v.clone() for k, v in client.model.state_dict().items()}
                    with torch.no_grad():
                        for param in client.model.parameters():
                            noise = torch.randn_like(param) * self.parameter_noise_level
                            param.add_(noise)
            # 清除缓存强制重算
            if orbit_id in self.satellite_model_cache:
                 for sat_id in orbit_satellites:
                     if sat_id in self.satellite_model_cache:
                         del self.satellite_model_cache[sat_id]

        try:
            # --- 1. Initialization ---
            self._init_orbit_structures(orbit_id)
            
            # Reset visited status (used here to track if 'processed as source node')
            # Note: In this algorithm, 'visited' means 'has been considered as a Source Node'
            for sat_id in self.orbit_visited[orbit_id]:
                self.orbit_visited[orbit_id][sat_id] = False
            
            coordinator = self.orbit_coordinators[orbit_id]
            self.logger.info(f"协调者: {coordinator}")
            self._update_model_cache(orbit_id)
            
            # Reset groupings
            self.orbit_groups[orbit_id] = {}
            self.orbit_representatives[orbit_id] = {}
            satellite_to_group = {} # Map sat_id -> group_id
            
            # Sort satellites by position in orbit (critical for sequential processing)
            satellites = sorted(list(self.orbit_visited[orbit_id].keys()), 
                            key=lambda x: int(x.split('-')[1]))
            
            # Special Case: First round or cold start -> Position Grouping
            if self.current_round == 0:
                self.logger.info(f"轨道 {orbit_id} 使用基于位置的分组策略 (Cold Start)")
                return self._perform_position_grouping(orbit_id, satellites)
            
            # Parameters from config or defaults
            max_distance = self.config.get('group', {}).get('max_distance', 2) # Search radius (hops)
            initial_threshold = 0.6 # Starting similarity threshold
            
            group_counter = 0
            
            # --- 2. Greedy Grouping Strategy ---
            # Iterate through each satellite as a potential 'Source Node'
            for source_sat in satellites:
                # Skip if already ASSIGNED to a group (passive) - Wait, paper says: 
                # "Coordinator iterates... considers unassigned satellite i as source"
                if source_sat in satellite_to_group:
                    continue
                
                # Start a new group with Source Node
                group_id = f"group_{orbit_id}-{group_counter}"
                current_threshold = initial_threshold
                
                # Assign Source Node to Group
                satellite_to_group[source_sat] = group_id
                self.orbit_groups[orbit_id][source_sat] = group_id
                # Source is initially the representative
                self.orbit_representatives[orbit_id][group_id] = source_sat
                
                group_members = [source_sat]
                self.logger.debug(f"新组 {group_id} 以 {source_sat} 为中心初始化")
                
                # Search PHYSICAL NEIGHBORS
                # Note: We scan neighbors up to max_distance hops
                neighbors = self._get_satellite_neighbors(source_sat, max_distance)
                
                # Sort neighbors by proximity (closest first) - optional but good for locality
                # (Simple list from _get_satellite_neighbors is usually ordered by hop)
                
                added_count = 0
                
                for neighbor in neighbors:
                    # Skip if neighbor is the source itself (safety check)
                    if neighbor == source_sat: 
                        continue
                        
                    # Calculate Similarity
                    if source_sat in self.satellite_model_cache and neighbor in self.satellite_model_cache:
                        try:
                            sim = self.compute_enhanced_similarity(source_sat, neighbor)
                        except Exception as e:
                            sim = 0.0
                    else:
                        sim = 0.0
                    
                    # --- Case A: Neighbor is Unassigned ---
                    if neighbor not in satellite_to_group:
                        if sim >= current_threshold:
                            # Absorb into group
                            satellite_to_group[neighbor] = group_id
                            self.orbit_groups[orbit_id][neighbor] = group_id
                            group_members.append(neighbor)
                            added_count += 1
                            self.logger.debug(f"  -> 吸纳未分组邻居 {neighbor} (Sim={sim:.3f})")
                    
                    # --- Case B: Neighbor is Assigned (Regrouping/Stealing) ---
                    else:
                        current_group = satellite_to_group[neighbor]
                        # Don't steal from self
                        if current_group == group_id:
                            continue
                            
                        # Get current representative similarity
                        current_rep = self.orbit_representatives[orbit_id].get(current_group)
                        if current_rep and current_rep in self.satellite_model_cache and neighbor in self.satellite_model_cache:
                             curr_sim = self.compute_enhanced_similarity(current_rep, neighbor)
                        else:
                             curr_sim = 0.0
                        
                        # Steal if significantly better
                        if sim > curr_sim + 0.05: # Add margin to prevent oscillation
                            # Steal!
                            # Remove from old group (logical removal only, cleaning up old group list usually not needed for simplistic logic unless rep changes)
                            # Note: If we steal the REP of another group, that group might break. 
                            # Simplification: Don't steal representatives to ensure stability
                            if neighbor == current_rep:
                                continue
                                
                            satellite_to_group[neighbor] = group_id
                            self.orbit_groups[orbit_id][neighbor] = group_id
                            group_members.append(neighbor)
                            added_count += 1
                            self.logger.info(f"  -> 从组 {current_group} 抢夺邻居 {neighbor} (NewSim={sim:.3f} > OldSim={curr_sim:.3f})")

                    # --- 3. Dynamic Thresholding ---
                    if len(group_members) >= self.max_group_size_threshold:
                        # Calculate min similarity in group to tighten threshold
                        min_sim = 1.0
                        for m in group_members:
                            if m == source_sat: continue
                            if source_sat in self.satellite_model_cache and m in self.satellite_model_cache:
                                s = self.compute_enhanced_similarity(source_sat, m)
                                min_sim = min(min_sim, s)
                        
                        current_threshold = max(0.6, min(1.0, min_sim))
                        # Stop growing if full
                        if len(group_members) >= self.max_group_size:
                            break
                
                # Check for "Directional Stop" triggers (continuous failure) - implicit in limited neighbor scan
                
                group_counter += 1

            # --- 4. Final Cleanup & Validation ---
            # Ensure every satellite has a group (Singletons)
            for sat_id in satellites:
                if sat_id not in satellite_to_group:
                     group_id = f"group_{orbit_id}-{group_counter}"
                     satellite_to_group[sat_id] = group_id
                     self.orbit_groups[orbit_id][sat_id] = group_id
                     self.orbit_representatives[orbit_id][group_id] = sat_id
                     group_counter += 1
            
            # Reconstruct groups dict for return
            final_groups = {}
            for sat_id, g_id in self.orbit_groups[orbit_id].items():
                if g_id not in final_groups:
                    final_groups[g_id] = []
                final_groups[g_id].append(sat_id)
            
            # Update representatives for any groups that might have lost members or formed late
            # (Already handled basic assignment, checking for empty groups)
            final_groups = {k:v for k,v in final_groups.items() if len(v) > 0}
            
            self.logger.info(f"轨道 {orbit_id} 分组结果 (Groups: {len(final_groups)}):")
            for gid, mem in final_groups.items():
                self.logger.info(f"  {gid}: {mem} (Rep: {self.orbit_representatives[orbit_id].get(gid)})")
                
            return final_groups

        except Exception as e:
            self.logger.error(f"分组过程出错: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return self._perform_position_grouping(orbit_id, satellites)
            
        finally:
            # 鲁棒性测试：恢复原始参数
            if original_params:
                self.logger.info(f"=== [鲁棒性测试] 轨道 {orbit_id} 恢复原始参数用于聚合 ===")
                for sat_id, params in original_params.items():
                    if sat_id in self.clients:
                        self.clients[sat_id].model.load_state_dict(params)

    def compute_enhanced_similarity(self, sat1_id: str, sat2_id: str) -> float:
        """
        增强的相似度计算，综合考虑多个维度的相似性
        
        Args:
            sat1_id: 第一个卫星ID
            sat2_id: 第二个卫星ID
            
        Returns:
            float: 综合相似度值(0-1)
        """
        try:
            if sat1_id not in self.clients or sat2_id not in self.clients:
                return 0.0
                
            client1 = self.clients[sat1_id]
            client2 = self.clients[sat2_id]
            
            # 1. 模型参数相似度 (权重: 0.4)
            if sat1_id in self.satellite_model_cache and sat2_id in self.satellite_model_cache:
                model1 = self.satellite_model_cache[sat1_id]
                model2 = self.satellite_model_cache[sat2_id]
                param_similarity = self._compute_parameter_similarity(model1, model2)
            else:
                param_similarity = 0.0
                
            # 2. 训练损失曲线相似度 (权重: 0.3)
            loss_similarity = self._compute_loss_curve_similarity(client1, client2)
            
            # 3. 预测行为相似度 (权重: 0.3)
            prediction_similarity = self._compute_prediction_similarity(client1, client2)
            
            # 综合计算最终相似度
            final_similarity = (self.weights['alpha'] * param_similarity + 
                            self.weights['beta'] * loss_similarity + 
                            self.weights['gamma'] * prediction_similarity)
            
            self.logger.debug(f"卫星 {sat1_id}-{sat2_id} 相似度: 参数={param_similarity:.4f}, " +
                            f"损失曲线={loss_similarity:.4f}, 预测={prediction_similarity:.4f}, " +
                            f"最终={final_similarity:.4f}")
            
            return final_similarity
            
        except Exception as e:
            self.logger.error(f"计算增强相似度时出错: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return 0.0

    def _compute_parameter_similarity(self, model1, model2):
        """计算模型参数相似度，考虑层的重要性"""
        try:
            # 只选择重要的权重参数
            important_keys = [k for k in model1.keys() 
                            if ("weight" in k or "bias" in k) and 
                            not any(x in k for x in ["running_mean", "running_var", "num_batches"])]
            
            # 分层计算相似度
            layer_similarities = []
            layer_weights = []
            
            for i, name in enumerate(important_keys):
                if name in model1 and name in model2:
                    param1 = model1[name].flatten()
                    param2 = model2[name].flatten()
                    
                    if param1.shape == param2.shape and param1.shape[0] > 0:
                        # 规范化向量
                        norm1 = torch.norm(param1)
                        norm2 = torch.norm(param2)
                        
                        if norm1 > 0 and norm2 > 0:
                            param1 = param1 / norm1
                            param2 = param2 / norm2
                            
                            # 计算余弦相似度
                            cos_sim = torch.sum(param1 * param2).item()
                            sim = (cos_sim + 1) / 2  # 映射到[0,1]范围
                            
                            # 后面的层通常更重要，给予更高权重
                            weight = 1.0 + i * 0.1
                            
                            layer_similarities.append(sim)
                            layer_weights.append(weight)
            
            if not layer_similarities:
                return 0.5  # 默认中等相似度
                
            # 加权平均
            total_weight = sum(layer_weights)
            weighted_sim = sum(s * w for s, w in zip(layer_similarities, layer_weights)) / total_weight
            
            return weighted_sim ** 0.8  # 增强中等相似度的区分度
        except Exception as e:
            self.logger.error(f"计算参数相似度出错: {str(e)}")
            return 0.5  # 出错时返回中等相似度

    def _compute_loss_curve_similarity(self, client1, client2):
        """比较两个卫星的训练Loss相似度 (简化版本 - 鲁棒)"""
        try:
            # 获取训练损失历史
            if not hasattr(client1, 'train_stats') or not hasattr(client2, 'train_stats'):
                return 0.5
            
            if not client1.train_stats or not client2.train_stats:
                return 0.5
            
            # 提取最后一次训练的损失曲线
            loss1 = client1.train_stats[-1]['summary'].get('train_loss', [])
            loss2 = client2.train_stats[-1]['summary'].get('train_loss', [])
            
            # 数据不足时返回默认值
            if len(loss1) < 1 or len(loss2) < 1:
                return 0.5
            
            # 简化版本: 比较平均Loss水平
            # Reverted from vector-based Cosine to scalar comparison.
            # Rationale: SGD noise makes epoch-to-epoch trends unreliable.
            mean_loss1 = np.mean(loss1)
            mean_loss2 = np.mean(loss2)
            
            # Scale factor 50 to make differences meaningful in [0,1] range
            diff = abs(mean_loss1 - mean_loss2)
            sim = 1.0 / (1.0 + diff * 50.0)
            
            return max(0.0, min(1.0, sim))
        except Exception as e:
            self.logger.error(f"计算损失曲线相似度出错: {str(e)}")
            return 0.5

    def _compute_prediction_similarity(self, client1, client2):
        """通过比较卫星在同样输入上的预测结果来评估相似度"""
        try:
            # 如果没有数据集，返回默认值
            if not hasattr(client1, 'dataset') or not hasattr(client2, 'dataset'):
                return 0.5
            
            if client1.dataset is None or client2.dataset is None:
                return 0.5
            
            if len(client1.dataset) == 0 or len(client2.dataset) == 0:
                return 0.5
            
            # 从两个数据集中采样共同的数据点
            import numpy as np
            import torch
            from torch.utils.data import DataLoader
            
            # 创建一个小的评估集（从客户端1的数据集取前N个样本）
            sample_size = min(10, len(client1.dataset))
            indices = np.arange(sample_size)
            eval_data = [client1.dataset[i][0] for i in indices]  # 只取特征，不取标签
            
            # 将评估数据转换为batch
            eval_tensor = torch.stack(eval_data)
            
            # 获取两个模型的预测
            device = next(client1.model.parameters()).device
            eval_tensor = eval_tensor.to(device)
            
            with torch.no_grad():
                client1.model.eval()
                client2.model.eval()
                
                pred1 = client1.model(eval_tensor).softmax(dim=1)
                pred2 = client2.model(eval_tensor).softmax(dim=1)
            
            # Match Paper: Hellinger Distance
            # Eq: Sim_pred = 1 - (1/K) * sum( Hellinger(y_i, y_j) )
            # Hellinger(P, Q) = (1/sqrt(2)) * sqrt( sum( (sqrt(p_c) - sqrt(q_c))^2 ) )
            
            # pred1, pred2 shape: [Batch, Classes]
            # Ensure probabilities (softmax) - already done above
            
            # 1. Sqrt of probs
            sqrt_p = torch.sqrt(pred1)
            sqrt_q = torch.sqrt(pred2)
            
            # 2. Sum of squared differences per sample
            # sum((sqrt_p - sqrt_q)^2) -> shape: [Batch]
            sum_sq_diff = torch.sum((sqrt_p - sqrt_q) ** 2, dim=1)
            
            # 3. Hellinger per sample
            # hellinger = (1/sqrt(2)) * sqrt(sum_sq_diff)
            hellinger_dists = (1.0 / np.sqrt(2.0)) * torch.sqrt(sum_sq_diff)
            
            # 4. Average Hellinger over batch (1/K sum)
            avg_hellinger = torch.mean(hellinger_dists).item()
            
            # 5. Similarity = 1 - Distance
            similarity = 1.0 - avg_hellinger
            
            return max(0.0, min(1.0, similarity))
        except Exception as e:
            self.logger.error(f"计算预测相似度出错: {str(e)}")
            return 0.5
    def _perform_spectral_grouping(self, orbit_id: int):
        """
        执行轨道内卫星分组 - 使用改进的相似度和谱聚类 (备用方法)

        Args:
            orbit_id: 轨道ID
        """
        self.logger.info(f"\n=== 开始轨道 {orbit_id} 的分组过程 ===")
        
        # 初始化轨道结构
        self._init_orbit_structures(orbit_id)
        
        # 获取卫星列表并排序
        satellites = sorted(list(self.orbit_visited[orbit_id].keys()), 
                        key=lambda x: int(x.split('-')[1]))
        
        # 第一轮或模型缓存为空时使用位置分组
        if self.current_round == 0:
            self.logger.info(f"轨道 {orbit_id}: 第一轮使用基于位置的分组策略")
            return self._perform_position_grouping(orbit_id, satellites)
        
        # 如果不是刷新周期，保持当前分组
        if self.current_round % self.similarity_refresh_rounds != 0 and self.current_round > 0:
            self.logger.info(f"轨道 {orbit_id}: 保持当前分组 (当前轮次 {self.current_round})")
            groups = {}
            for sat_id, group_id in self.orbit_groups[orbit_id].items():
                if group_id not in groups:
                    groups[group_id] = []
                groups[group_id].append(sat_id)
            return groups
        
        # 更新模型缓存并检查状态
        self._update_model_cache(orbit_id)
        
        # 检查模型缓存状态
        if not all(sat in self.satellite_model_cache for sat in satellites):
            missing = [sat for sat in satellites if sat not in self.satellite_model_cache]
            self.logger.warning(f"轨道 {orbit_id}: 模型缓存不完整，缺少 {len(missing)} 个卫星: {missing[:5]}...")
            # Debug: print why they are missing
            self.logger.info(f"轨道 {orbit_id}: 使用基于位置的分组策略 (因为缓存不完整)")
            return self._perform_position_grouping(orbit_id, satellites)
        
        # 记录开始使用基于相似度的分组策略
        self.logger.info(f"轨道 {orbit_id}: 使用基于相似度的分组策略")
        
        # 计算相似度矩阵
        n = len(satellites)
        similarity_matrix = np.zeros((n, n))
        
        # 填充相似度矩阵
        for i in range(n):
            similarity_matrix[i, i] = 1.0  # 自己和自己的相似度为1
            for j in range(i+1, n):
                sim = self.compute_enhanced_similarity(satellites[i], satellites[j])
                similarity_matrix[i, j] = similarity_matrix[j, i] = sim
        
        # 使用谱聚类进行分组
        try:
            from sklearn.cluster import SpectralClustering
            
            # 计算合适的分组数
            n_clusters = max(2, min(n // 3, n // self.max_group_size + 1))
            
            # 使用谱聚类
            clustering = SpectralClustering(
                n_clusters=n_clusters,
                affinity='precomputed',  # 使用提供的相似度矩阵
                assign_labels='kmeans',
                random_state=42
            ).fit(similarity_matrix)
            
            labels = clustering.labels_
            
            # 创建分组
            groups = {}
            for i, label in enumerate(labels):
                group_id = f"group_{orbit_id}-{label}"
                if group_id not in groups:
                    groups[group_id] = []
                groups[group_id].append(satellites[i])
                
                # 更新卫星状态
                self.orbit_groups[orbit_id][satellites[i]] = group_id
                self.orbit_visited[orbit_id][satellites[i]] = True
            
            # 选择代表节点
            for group_id, members in groups.items():
                if members:
                    # 选择组内相似度中心性最高的卫星作为代表
                    best_member = None
                    max_centrality = -1
                    
                    for sat in members:
                        idx = satellites.index(sat)
                        # 计算与组内其他成员的平均相似度
                        centrality = 0
                        for other in members:
                            if sat != other:
                                other_idx = satellites.index(other)
                                centrality += similarity_matrix[idx, other_idx]
                        
                        if len(members) > 1:
                            centrality /= (len(members) - 1)
                            
                        if centrality > max_centrality:
                            max_centrality = centrality
                            best_member = sat
                    
                    self.orbit_representatives[orbit_id][group_id] = best_member or members[0]
            
            # 输出分组结果
            self.logger.info(f"轨道 {orbit_id} 分组结果:")
            for group_id, members in groups.items():
                rep = self.orbit_representatives[orbit_id].get(group_id, "未设置")
                self.logger.info(f"  组 {group_id}: {len(members)} 成员, 代表: {rep}")
            
            return groups
            
        except Exception as e:
            self.logger.error(f"谱聚类分组失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            self.logger.info(f"轨道 {orbit_id}: 回退到基于位置的分组策略")
            return self._perform_position_grouping(orbit_id, satellites)
    
    def _update_model_cache(self, orbit_id: int):
        """
        更新卫星模型缓存，用于相似度计算
        
        Args:
            orbit_id: 轨道ID
        """
        self.logger.info(f"更新轨道 {orbit_id} 的模型缓存")
        
        for sat_id, client in self.clients.items():
            orbit_num, _ = self._parse_satellite_id(sat_id)
            
            if orbit_num == orbit_id:
                # 保存模型参数，用于相似度计算
                self.satellite_model_cache[sat_id] = {
                    name: param.data.clone()
                    for name, param in client.model.state_dict().items()
                }
        
        self.logger.info(f"缓存了 {len(self.satellite_model_cache)} 个卫星模型")
    
    def _get_group_representative(self, orbit_id: int, group_id: str) -> Optional[str]:
        """
        获取组的代表节点
        
        Args:
            orbit_id: 轨道ID
            group_id: 组ID
            
        Returns:
            str: 代表节点ID，如果不存在则返回None
        """
        return self.orbit_representatives[orbit_id].get(group_id)
    
    def assign_representatives(self, orbit_id: int, groups: Dict[str, List[str]]):
        """
        分配初始代表节点
        
        Args:
            orbit_id: 轨道ID
            groups: 分组信息
        """
        self.logger.info(f"为轨道 {orbit_id} 分配初始代表节点")
        
        for group_id, members in groups.items():
            if not members:
                continue
                
            # 获取源节点（组内第一个节点）作为初始代表节点
            representative = members[0]
            self.orbit_representatives[orbit_id][group_id] = representative
            self.logger.info(f"组 {group_id} 的初始代表节点: {representative}")
        
        return self.orbit_representatives[orbit_id]
    
    def rotate_representatives(self, orbit_id: int, groups: Dict[str, List[str]]):
        """
        轮换代表节点
        
        Args:
            orbit_id: 轨道ID
            groups: 分组信息
        """
        self.logger.info(f"为轨道 {orbit_id} 轮换代表节点")
        
        new_representatives = {}
        
        for group_id, members in groups.items():
            if len(members) <= 1:
                # 如果组内只有一个节点，保持不变
                if group_id in self.orbit_representatives[orbit_id]:
                    new_representatives[group_id] = self.orbit_representatives[orbit_id][group_id]
                continue
                
            current_rep = self.orbit_representatives[orbit_id].get(group_id)
            if not current_rep or current_rep not in members:
                # 如果当前代表节点无效，选择组内第一个节点
                new_representatives[group_id] = members[0]
                continue
                
            # 获取当前代表节点的索引
            try:
                current_idx = members.index(current_rep)
            except ValueError:
                current_idx = -1
                
            # 按照顺时针方向选择下一个节点
            next_idx = (current_idx + 1) % len(members)
            next_rep = members[next_idx]
            
            # 检查下一个节点是否属于其他组(这种情况不应该发生，因为members已经是同一组)
            # 如果发生，则选择组内逆时针方向第一个节点
            if self.orbit_groups[orbit_id].get(next_rep, "") != group_id:
                prev_idx = (current_idx - 1) % len(members)
                next_rep = members[prev_idx]
                
            new_representatives[group_id] = next_rep
            self.logger.info(f"组 {group_id} 的代表节点从 {current_rep} 轮换为 {next_rep}")
        
        # 更新代表节点信息
        self.orbit_representatives[orbit_id] = new_representatives
        
        return new_representatives
    
    def _handle_orbit_training(self, station_id: str, orbit_id: int, current_time: float):
        """
        处理单个轨道的训练过程，重写父类方法
        
        Args:
            station_id: 地面站ID
            orbit_id: 轨道ID
            current_time: 当前时间戳
            
        Returns:
            bool: 训练是否成功完成
            dict: 轨道统计信息
        """
        try:
            orbit_num = orbit_id + 1
            # station = self.ground_stations[station_id] # station_id is None now
            self.logger.info(f"\n=== 处理轨道 {orbit_num} ===")
            
            # 记录本轮轨道的统计信息
            orbit_stats = {
                'training_energy': 0,  # 训练能耗
                'communication_energy': 0,  # 通信能耗
                'training_satellites': set(),  # 训练的卫星
                'receiving_satellites': set()  # 接收参数的卫星
            }
            
            # 此轮使用上一轮计算的分组
            # 获取已有分组或使用默认位置分组
            if self.current_round == 0 or orbit_num not in self.orbit_groups:
                # 第一轮使用位置分组
                orbit_satellites = self._get_orbit_satellites(orbit_id)
                groups = self._perform_position_grouping(orbit_num, orbit_satellites)
                self.assign_representatives(orbit_num, groups)
            else:
                # 否则使用已有分组
                groups = {}
                for sat_id, group_id in self.orbit_groups[orbit_num].items():
                    if group_id not in groups:
                        groups[group_id] = []
                    groups[group_id].append(sat_id)
                
                # 如果不是第一轮，且上一轮有代表节点，则轮换代表节点
                if self.current_round > 0 and self.orbit_representatives.get(orbit_num):
                    self.rotate_representatives(orbit_num, groups)
            # 如果配置了最大卫星数量限制
            if 'max_satellites_per_orbit' in self.config.get('limit', {}):
                max_sats = self.config['limit']['max_satellites_per_orbit']
                groups = self.limit_satellite_groups(orbit_num, groups, max_sats)
            
            # 2. 等待并选择可见卫星作为协调者
            coordinator = None
            orbit_satellites = self._get_orbit_satellites(orbit_id)
            max_wait_time = current_time + self.config['fl']['round_interval'] * 0.5
            
            while not coordinator and current_time < max_wait_time:
                for sat_id in orbit_satellites:
                    # 检查卫星是否对任意地面站可见
                    is_visible = False
                    for st_id in self.ground_stations:
                        if self.network_model._check_visibility(st_id, sat_id, current_time):
                            is_visible = True
                            break
                    
                    if is_visible:
                        coordinator = sat_id
                        break
                if not coordinator:
                    self.logger.info(f"轨道 {orbit_num} 当前无可见卫星，等待60秒...")
                    current_time += 60
                    self.topology_manager.update_topology(current_time)

            if not coordinator:
                self.logger.warning(f"轨道 {orbit_num} 在指定时间内未找到可见卫星")
                return False, orbit_stats
            
            self.logger.info(f"轨道 {orbit_num} 选择 {coordinator} 作为协调者")
            self.orbit_coordinators[orbit_num] = coordinator

            # 3. 分发初始参数给协调者
            model_state = self.model.state_dict()
            self.logger.info(f"\n=== 轨道 {orbit_num} 内参数分发 ===")
            pre_comm_energy = self.energy_model.get_battery_level(coordinator)
            self.clients[coordinator].apply_model_update(model_state)
            post_comm_energy = self.energy_model.get_battery_level(coordinator)
            orbit_stats['communication_energy'] += (pre_comm_energy - post_comm_energy)
            orbit_stats['receiving_satellites'].add(coordinator)
            
            # 4. 选择代表节点进行训练
            self.logger.info(f"\n=== 轨道 {orbit_num} 代表节点训练 ===")
            
            # 获取当前分组和代表节点
            orbit_groups = {}
            for sat_id, group_id in self.orbit_groups[orbit_num].items():
                if group_id not in orbit_groups:
                    orbit_groups[group_id] = []
                orbit_groups[group_id].append(sat_id)
            
            representatives = self.orbit_representatives[orbit_num]
            
            # 只有代表节点进行训练
            trained_satellites = set()
            training_stats = {}
            
            for group_id, rep_id in representatives.items():
                try:
                    # 检查代表节点是否存在
                    if rep_id not in self.clients:
                        self.logger.warning(f"代表节点 {rep_id} 不存在于客户端列表中")
                        continue
                    
                    # 检查协调者是否存在
                    if coordinator not in self.clients:
                        self.logger.warning(f"协调者节点 {coordinator} 不存在于客户端列表中")
                        continue
                    
                    # 从协调者获取初始模型
                    self.clients[rep_id].apply_model_update(self.clients[coordinator].model.state_dict())
                    orbit_stats['receiving_satellites'].add(rep_id)
                    
                    # 记录训练前能耗
                    pre_train_energy = self.energy_model.get_battery_level(rep_id)
                    
                    # 代表节点执行训练
                    stats = self.clients[rep_id].train(self.current_round)
                    
                    if stats['summary']['train_loss']:
                        post_train_energy = self.energy_model.get_battery_level(rep_id)
                        orbit_stats['training_energy'] += (pre_train_energy - post_train_energy)
                        orbit_stats['training_satellites'].add(rep_id)
                        trained_satellites.add(rep_id)
                        training_stats[rep_id] = stats
                        
                        self.logger.info(f"代表节点 {rep_id} 完成训练: "
                                        f"Loss={stats['summary']['train_loss'][-1]:.4f}, "
                                        f"Acc={stats['summary']['train_accuracy'][-1]:.2f}%, "
                                        f"能耗={stats['summary']['energy_consumption']:.4f}Wh")
                        
                        # 将代表节点的模型参数分发给组内其他成员
                        group_members = orbit_groups.get(group_id, [])
                        for member in group_members:
                            if member != rep_id and member in self.clients:
                                try:
                                    # 分发模型参数
                                    pre_dist_energy = self.energy_model.get_battery_level(rep_id)
                                    self.clients[member].apply_model_update(self.clients[rep_id].model.state_dict())
                                    post_dist_energy = self.energy_model.get_battery_level(rep_id)
                                    
                                    orbit_stats['communication_energy'] += (pre_dist_energy - post_dist_energy)
                                    orbit_stats['receiving_satellites'].add(member)
                                    
                                    self.logger.info(f"组内成员 {member} 接收代表节点 {rep_id} 的模型参数")
                                except Exception as e:
                                    self.logger.error(f"分发模型参数给 {member} 时出错: {str(e)}")
                except Exception as e:
                    self.logger.error(f"处理代表节点 {rep_id} 时出错: {str(e)}")
            
            # 关键修改: 训练后更新模型缓存
            self._update_model_cache(orbit_num)
            
            # 关键修改: 训练后执行下一轮的分组，如果满足刷新条件
            if self.current_round % self.similarity_refresh_rounds == 0:
                self.logger.info(f"轨道 {orbit_num}: 基于训练后的模型参数计算下一轮分组")
                next_groups = self.perform_grouping(orbit_num)
                # 保存新的分组结果，但不立即使用
                self.assign_representatives(orbit_num, next_groups)
                
                # 调试: 打印几个卫星的相似度矩阵
                debug_sats = orbit_satellites[:min(3, len(orbit_satellites))]
                for i, sat1 in enumerate(debug_sats):
                    for sat2 in debug_sats[i+1:]:
                        try:
                            if sat1 in self.satellite_model_cache and sat2 in self.satellite_model_cache:
                                sim = self.compute_similarity(self.satellite_model_cache[sat1], 
                                                            self.satellite_model_cache[sat2])
                                self.logger.info(f"训练后相似度: {sat1}-{sat2} = {sim:.4f}")
                        except Exception as e:
                            self.logger.error(f"计算相似度出错: {str(e)}")
            
            # 5. 轨道内聚合
            min_updates_required = self.config['aggregation']['min_updates']
            self.logger.info(f"需要至少 {min_updates_required} 个代表节点更新，当前有 {len(trained_satellites)} 个")

            if len(trained_satellites) >= min_updates_required:
                self.logger.info(f"\n=== 轨道 {orbit_num} 聚合 ===")
                aggregator = self.intra_orbit_aggregators.get(orbit_id)
                if not aggregator:
                    from fl_core.aggregation.intra_orbit import IntraOrbitAggregator, AggregationConfig
                    aggregator = IntraOrbitAggregator(AggregationConfig(**self.config['aggregation']))
                    self.intra_orbit_aggregators[orbit_id] = aggregator

                # 收集代表节点的更新并聚合
                updates_collected = 0
                for rep_id in trained_satellites:
                    try:
                        model_diff, stats = self.clients[rep_id].get_model_update()
                        if model_diff:
                            self.logger.info(f"收集代表节点 {rep_id} 的模型更新")
                            aggregator.receive_update(rep_id, self.current_round, model_diff, current_time)
                            updates_collected += 1
                        else:
                            self.logger.warning(f"代表节点 {rep_id} 的模型更新为空")
                    except Exception as e:
                        self.logger.error(f"收集代表节点 {rep_id} 更新时出错: {str(e)}")

                self.logger.info(f"成功收集了 {updates_collected} 个代表节点的更新")

                orbit_update = aggregator.get_aggregated_update(self.current_round)
                if orbit_update:
                    self.logger.info(f"轨道 {orbit_num} 完成聚合")
                    
                    # 更新轨道内所有卫星的模型
                    update_success = 0
                    for sat_id in orbit_satellites:
                        try:
                            self.clients[sat_id].apply_model_update(orbit_update)
                            update_success += 1
                            orbit_stats['receiving_satellites'].add(sat_id)
                            self.logger.info(f"更新卫星 {sat_id} 的模型参数")
                        except Exception as e:
                            self.logger.error(f"更新卫星 {sat_id} 模型时出错: {str(e)}")

                    self.logger.info(f"成功更新了 {update_success} 个卫星的模型")

                    # 6. 尝试通过任意可见卫星发送轨道聚合结果到地面站
                    # 只要卫星已经更新了模型，它就可以作为中继
                    potential_relays = [sat_id for sat_id in orbit_satellites 
                                      if sat_id in orbit_stats['receiving_satellites']]
                    
                    # 确保协调者也在列表中（如果它还没被标记为接收者）
                    if coordinator not in potential_relays:
                        potential_relays.append(coordinator)
                        
                    self.logger.info(f"尝试通过 {len(potential_relays)} 个潜在中继卫星发送更新")
                    
                    upload_success = False
                    relay_satellite = None
                    
                    # 1. 首先检查当前时刻是否有可见卫星
                    import random
                    for sat_id in potential_relays:
                        # 找出所有可见的地面站
                        visible_stations = []
                        for station_id in self.ground_stations:
                            if self.network_model._check_visibility(station_id, sat_id, current_time):
                                visible_stations.append(station_id)
                        
                        if visible_stations:
                            # 随机选择一个可见的地面站，以实现负载均衡
                            station_id = random.choice(visible_stations)
                            relay_satellite = sat_id
                            self.logger.info(f"找到当前可见的中继卫星: {sat_id} (可见地面站: {visible_stations} -> 选择 {station_id})")
                            found_visible_station = True
                            break
                            
                    # 2. 如果当前不可见，搜索未来窗口
                    if not relay_satellite:
                        visibility_start = current_time
                        max_search_time = 600  # 增加搜索窗口到10分钟
                        
                        for check_time in range(int(visibility_start), int(visibility_start + max_search_time), 30):
                            for sat_id in potential_relays:
                                found_visible_station = False
                                for station_id in self.ground_stations:
                                    if self.network_model._check_visibility(station_id, sat_id, check_time):
                                        relay_satellite = sat_id
                                        current_time = check_time
                                        self.logger.info(f"找到未来可见的中继卫星: {sat_id} (在 {check_time}s, 可见地面站: {station_id})")
                                        self.topology_manager.update_topology(current_time)
                                        found_visible_station = True
                                        break
                                if found_visible_station:
                                    break
                            if relay_satellite:
                                break

                    # 7. 发送轨道聚合结果到地面站
                    # 尝试找到任意可见的地面站
                    upload_success = False
                    
                    if relay_satellite:
                        try:
                            # 获取模型更新（所有更新过的卫星应该有相同的模型）
                            model_diff, _ = self.clients[relay_satellite].get_model_update()
                            
                            if model_diff:
                                # 遍历所有地面站，寻找可见的
                                for target_station_id, target_station in self.ground_stations.items():
                                    # 检查中继卫星是否可见该地面站
                                    # 注意：这里我们假设如果在搜索窗口内找到了中继，它在current_time是对某个地面站可见的
                                    # 但我们需要确认是对哪个地面站可见。
                                    # 之前的逻辑是针对特定station_id搜索的。
                                    # 现在我们需要针对所有station搜索。
                                    
                                    # 重新检查可见性
                                    if self.network_model._check_visibility(target_station_id, relay_satellite, current_time):
                                        success = target_station.receive_orbit_update(
                                            str(orbit_id),
                                            self.current_round,
                                            model_diff,
                                            len(trained_satellites)
                                        )
                                        if success:
                                            self.logger.info(f"轨道 {orbit_num} 的模型通过 {relay_satellite} 成功发送给地面站 {target_station_id}")
                                            upload_success = True
                                            break
                                        else:
                                            self.logger.warning(f"地面站 {target_station_id} 拒绝接收轨道 {orbit_num} 的更新")
                                
                                if not upload_success:
                                    self.logger.warning(f"轨道 {orbit_num} 有中继卫星 {relay_satellite} 但无法连接任何地面站")
                                else:
                                    return True, orbit_stats
                        except Exception as e:
                            self.logger.error(f"通过 {relay_satellite} 发送更新时出错: {str(e)}")
                    else:
                        self.logger.warning(f"轨道 {orbit_num} 在搜索窗口内无卫星可见任何地面站")
                else:
                    self.logger.error(f"轨道 {orbit_num} 聚合失败: 无法获取有效的聚合结果")
            else:
                self.logger.warning(f"轨道 {orbit_num} 训练的代表节点数量不足: {len(trained_satellites)} < {min_updates_required}")
                self.logger.warning(f"代表节点列表: {representatives}")

            return False, orbit_stats

        except Exception as e:
            self.logger.error(f"处理轨道 {orbit_num} 时出错: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False, orbit_stats

    """
    为相似度分组实验添加卫星数量限制功能
    在grouping_experiment.py中添加此功能
    """

    def limit_satellite_groups(self, orbit_id: int, groups: dict, max_satellites: int = 8) -> dict:
        """
        限制每个轨道使用的卫星数量以满足总体限制
        
        Args:
            orbit_id: 轨道ID
            groups: 原始分组字典 {group_id: [satellite_ids]}
            max_satellites: 限制的最大卫星数量
            
        Returns:
            限制后的分组字典
        """
        self.logger.info(f"轨道 {orbit_id}: 限制卫星数量，目标数量: {max_satellites}")
        
        # 获取当前轨道数量
        current_satellites = sum(len(members) for members in groups.values())
        self.logger.info(f"轨道 {orbit_id}: 当前卫星数量: {current_satellites}")
        
        if current_satellites <= max_satellites:
            self.logger.info(f"轨道 {orbit_id}: 无需限制，当前卫星数 {current_satellites} <= 限制 {max_satellites}")
            return groups
            
        # 计算需要减少的卫星数量
        reduction_needed = current_satellites - max_satellites
        self.logger.info(f"轨道 {orbit_id}: 需要减少 {reduction_needed} 颗卫星")
        
        # 策略：优先保留代表节点，然后根据相似度保留最相似的卫星
        limited_groups = {}
        
        # 1. 首先确保每个组至少保留代表节点
        for group_id, members in groups.items():
            representative = self.orbit_representatives[orbit_id].get(group_id)
            if representative and representative in members:
                limited_groups[group_id] = [representative]
            elif members:
                # 如果没有代表节点或代表节点不在成员中，选择第一个成员作为代表
                limited_groups[group_id] = [members[0]]
        
        # 计算已经使用的卫星数量
        used_satellites = sum(len(members) for members in limited_groups.values())
        
        # 2. 按相似度排序，分配剩余的卫星配额
        remaining_slots = max_satellites - used_satellites
        
        if remaining_slots > 0:
            # 收集所有已分配代表节点的组和剩余待分配的卫星
            representatives = {}
            remaining_satellites = []
            
            for group_id, members in groups.items():
                rep = limited_groups[group_id][0]
                representatives[group_id] = rep
                
                # 将除代表节点外的成员添加到待分配列表
                for sat in members:
                    if sat != rep:
                        remaining_satellites.append((group_id, sat))
            
            # 计算每个待分配卫星与其组代表的相似度
            satellite_similarities = []
            for group_id, sat in remaining_satellites:
                rep = representatives[group_id]
                
                # 如果可能，计算与代表节点的相似度
                if sat in self.satellite_model_cache and rep in self.satellite_model_cache:
                    similarity = self.compute_similarity(
                        self.satellite_model_cache[sat],
                        self.satellite_model_cache[rep]
                    )
                else:
                    # 如果无法计算相似度，使用默认值
                    similarity = 0.5
                    
                satellite_similarities.append((group_id, sat, similarity))
            
            # 按相似度降序排序
            satellite_similarities.sort(key=lambda x: x[2], reverse=True)
            
            # 分配剩余的卫星配额
            for i in range(min(remaining_slots, len(satellite_similarities))):
                group_id, sat, _ = satellite_similarities[i]
                limited_groups[group_id].append(sat)
                
        # 计算最终卫星数
        final_satellite_count = sum(len(members) for members in limited_groups.values())
        self.logger.info(f"轨道 {orbit_id}: 限制后卫星数量: {final_satellite_count}")
        
        # 记录每个组的卫星数量
        for group_id, members in limited_groups.items():
            self.logger.info(f"轨道 {orbit_id}: 组 {group_id} 成员数: {len(members)}")
        
        return limited_groups

    
    def train(self):
        """
        执行训练过程，保留父类的大部分逻辑
        """
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

        # 新增：追踪最佳F1值
        best_f1 = 0
        
        # 如果禁用了早停，则修改相关参数 
        if hasattr(self, 'disable_early_stopping') and self.disable_early_stopping: 
            max_rounds_without_improvement = float('inf') # 设置为无穷大 
            min_rounds = self.config['fl']['num_rounds'] # 最小轮数设为总轮数 
            accuracy_threshold = 100.0 # 设置一个不可能达到的准确率阈值 
        else: 
            max_rounds_without_improvement = 3 # 默认值 
            min_rounds = 5 # 默认值 
            accuracy_threshold = 95.0 # 默认值


        # 初始化所有轨道的分组和代表节点
        # 注意：轨道ID实际是从1开始的
        for orbit in range(1, self.config['fl']['num_orbits'] + 1):
            self._init_orbit_structures(orbit)

        for round_num in range(self.config['fl']['num_rounds']):
            self.current_round = round_num
            self.logger.info(f"\n=== 开始第 {round_num + 1} 轮训练 === 时间：{datetime.fromtimestamp(current_time)}")
            
            # 使用线程池并行处理每个地面站的轨道
            orbit_successes = 0
            # 收集所有轨道的统计信息
            round_training_energy = 0
            round_comm_energy = 0
            round_training_sats = set()
            round_receiving_sats = set()
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # 创建所有任务
                future_to_orbit = {}
                # 直接遍历所有轨道
                for orbit_id in range(self.config['fl']['num_orbits']):
                    future = executor.submit(
                        self._handle_orbit_training,
                        None, # station_id 不再预先指定
                        orbit_id,
                        current_time
                    )
                    future_to_orbit[future] = orbit_id

                # 等待所有任务完成
                for future in as_completed(future_to_orbit):
                    orbit_id = future_to_orbit[future]
                    try:
                        result = future.result()  # result 是一个元组 (success, orbit_stats)
                        if isinstance(result, tuple) and len(result) == 2:
                            success, orbit_stats = result
                            if success:
                                orbit_successes += 1
                            if orbit_stats:  # 如果有统计信息就收集
                                round_training_energy += orbit_stats['training_energy']
                                round_comm_energy += orbit_stats['communication_energy']
                                round_training_sats.update(orbit_stats['training_satellites'])
                                round_receiving_sats.update(orbit_stats['receiving_satellites'])
                        else:
                            self.logger.warning(f"轨道 {orbit_id} 返回的结果格式不正确")
                    except Exception as e:
                        self.logger.error(f"处理轨道 {orbit_id} 时出错: {str(e)}")
            
            # 记录本轮统计信息
            energy_stats['training_energy'].append(round_training_energy)
            energy_stats['communication_energy'].append(round_comm_energy)
            energy_stats['total_energy'].append(round_training_energy + round_comm_energy)
            
            satellite_stats['training_satellites'].append(len(round_training_sats))
            satellite_stats['receiving_satellites'].append(len(round_receiving_sats))
            satellite_stats['total_active'].append(len(round_training_sats | round_receiving_sats))

            # 地面站聚合
            if orbit_successes > 0:
                self.logger.info(f"\n=== 地面站聚合阶段 === ({orbit_successes} 个轨道成功)")
                station_results = []
                with ThreadPoolExecutor(max_workers=3) as executor:
                    future_to_station = {
                        executor.submit(self._station_aggregation, station_id, station): station_id
                        for station_id, station in self.ground_stations.items()
                    }
                    
                    for future in as_completed(future_to_station):
                        station_id = future_to_station[future]
                        try:
                            result = future.result()
                            if result:
                                station_results.append(station_id)
                        except Exception as e:
                            self.logger.error(f"地面站 {station_id} 聚合出错: {str(e)}")

                # 全局聚合
                # 只要有至少一个地面站完成聚合，就可以进行全局聚合
                if len(station_results) >= 1:
                    self.logger.info("\n=== 全局聚合阶段 ===")
                    success = self._perform_global_aggregation(round_num)
                    
                    if success:
                        # 评估准确率
                        # accuracy = self.evaluate()
                        # accuracies.append(accuracy)
                        metrics = self.evaluate() 
                        
                        # 兼容性处理：如果evaluate返回的是float（旧版本），则转换为字典
                        if isinstance(metrics, (float, int)):
                            metrics = {
                                'accuracy': metrics,
                                'precision_macro': 0,
                                'recall_macro': 0,
                                'f1_macro': 0,
                                'precision_weighted': 0,
                                'recall_weighted': 0,
                                'f1_weighted': 0
                            } 

                        # 收集所有指标
                        accuracies.append(metrics['accuracy'])
                        precision_macros.append(metrics['precision_macro'])
                        recall_macros.append(metrics['recall_macro'])
                        f1_macros.append(metrics['f1_macro'])
                        precision_weighteds.append(metrics['precision_weighted'])
                        recall_weighteds.append(metrics['recall_weighted'])
                        f1_weighteds.append(metrics['f1_weighted'])
                        
                        # 计算当前轮次的总损失
                        round_loss = 0
                        count = 0
                        for client in self.clients.values():
                            if client.train_stats and client.client_id in round_training_sats:
                                round_loss += client.train_stats[-1]['summary']['train_loss'][-1]
                                count += 1
                                
                        losses.append(round_loss / max(1, count))  # 平均损失

                        # self.logger.info(f"第 {round_num + 1} 轮全局准确率: {accuracy:.4f}")
                        self.logger.info(f"第 {round_num + 1} 轮指标: "
                           f"准确率={metrics['accuracy']:.2f}%, "
                           f"F1={metrics['f1_macro']:.2f}%, "
                           f"精确率={metrics['precision_macro']:.2f}%, "
                           f"召回率={metrics['recall_macro']:.2f}%")
                        

                        current_accuracy = metrics['accuracy']
                        current_f1 = metrics['f1_macro']
                        # 更新最佳准确率和检查提升情况
                        improvement_found = False
                        if current_f1 > best_f1:
                            best_f1 = current_f1
                            improvement_found = True
                            self.logger.info(f"找到更好的模型！新的最佳F1值: {current_f1:.2f}%")
                        
                        if current_accuracy > best_accuracy:
                            best_accuracy = current_accuracy
                            if not improvement_found:  # 只有在F1值没有改善时才记录准确率改善
                                improvement_found = True
                                self.logger.info(f"找到更好的模型！新的最佳准确率: {current_accuracy:.2f}%")
                        
                        if improvement_found:
                            rounds_without_improvement = 0
                        else:
                            rounds_without_improvement += 1
                            self.logger.info(f"性能未提升，已经 {rounds_without_improvement} 轮没有改进")

                        # # 检查是否满足停止条件
                        # if round_num + 1 >= min_rounds:  # 已达到最小轮数
                        #     if accuracy >= accuracy_threshold:
                        #         self.logger.info(f"达到目标准确率 {accuracy:.4f}，停止训练")
                        #         break
                        #     elif rounds_without_improvement >= max_rounds_without_improvement:
                        #         self.logger.info(f"连续 {max_rounds_without_improvement} 轮准确率未提升，停止训练")
                        #         break
                    else:
                        self.logger.warning("全局聚合失败")
                else:
                    self.logger.warning(f"只有 {len(station_results)}/{len(self.ground_stations)} 个地面站完成聚合，跳过全局聚合")
                    # 即使失败也要填充数据，保持列表长度一致
                    accuracies.append(0)
                    losses.append(0)
                    precision_macros.append(0)
                    recall_macros.append(0)
                    f1_macros.append(0)
                    precision_weighteds.append(0)
                    recall_weighteds.append(0)
                    f1_weighteds.append(0)
            else:
                self.logger.warning("所有轨道训练失败，跳过聚合阶段")

            current_time += self.config['fl']['round_interval']
            
        self.logger.info(f"\n=== 训练结束 ===")
        self.logger.info(f"总轮次: {round_num + 1}")
        self.logger.info(f"最佳准确率: {best_accuracy:.4f}")
        self.logger.info(f"最佳F1值: {best_f1:.4f}")  # 新增    
        
        # 收集所有统计信息
        stats = {
            'accuracies': accuracies,
            'losses': losses,
             # 新增的分类指标
            'precision_macros': precision_macros,
            'recall_macros': recall_macros,
            'f1_macros': f1_macros,
            'precision_weighteds': precision_weighteds,
            'recall_weighteds': recall_weighteds,
            'f1_weighteds': f1_weighteds,
            'energy_stats': energy_stats,
            'satellite_stats': satellite_stats
        }

        # 生成可视化
        self.visualizer.plot_training_metrics(
            accuracies=stats['accuracies'],
            losses=stats['losses'],
            energy_stats=stats['energy_stats'],
            satellite_stats=stats['satellite_stats'],
            save_path=self.log_dir / 'training_metrics.png'  # 保存在实验日志目录
        )

        return stats
        
    def run(self):
        """运行分组实验"""
        self.logger.info("开始基于数据相似度的分组实验")
        
        # 准备数据
        self.prepare_data()
        
        # 设置客户端
        self.setup_clients()
        
        # 执行训练并获取统计信息
        stats = self.train()
        
        self.logger.info("实验完成")
        
        # 返回统计信息供后续比较
        return stats