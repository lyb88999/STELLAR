import logging
from typing import List, Tuple, Dict
import numpy as np
from datetime import datetime, timedelta, timezone
from skyfield.api import load, EarthSatellite, utc, wgs84
from astropy import units as u
from astropy.coordinates import CartesianRepresentation

class SatelliteNetwork:
    def __init__(self, tle_file: str, max_isl_distance: float = 4000.0):
        """
        初始化卫星网络模型
        Args:
            tle_file: TLE数据文件路径
            max_isl_distance: 星间链路最大通信距离 (km)
        """
        # 添加logger
        self.logger = logging.getLogger(__name__)

        self.max_isl_distance = max_isl_distance

        self.ts = load.timescale()
        self.satellites = self._load_satellites(tle_file)
        self.positions_cache = {}  # 位置缓存

        # 添加地面站位置 (经度,纬度,高度)
        # 使用高纬度位置以更好地覆盖极轨道卫星
        self.ground_stations = {
            "station_0": (70.0, 30.0, 0.1),      # 北纬70度，东经30度 (Cluster 1)
            "station_1": (70.0, 60.0, 0.1),      # 北纬70度，东经60度 (Cluster 1)
            "station_2": (70.0, 90.0, 0.1),      # 北纬70度，东经90度 (Cluster 1)
            "station_3": (70.0, -30.0, 0.1),     # 北纬70度，西经30度 (Cluster 2)
            "station_4": (70.0, -60.0, 0.1),     # 北纬70度，西经60度 (Cluster 2)
            "station_5": (70.0, -90.0, 0.1),     # 北纬70度，西经90度 (Cluster 2)
        }
        
    def _load_satellites(self, tle_file: str) -> Dict[str, EarthSatellite]:
        """
        从TLE文件加载卫星数据，并将其映射到轨道结构
        新增：过滤异常卫星（inc/MM/ecc不符，可能是备份/失效），确保12纯净轨道
        """
        raw_satellites = []
        try:
            with open(tle_file, 'r') as f:
                lines = f.readlines()
                
            if len(lines) == 0:
                raise ValueError("TLE file is empty")
                
            self.logger.info(f"读取到 {len(lines)} 行TLE数据")
            
            # 1. 加载&过滤原始卫星数据
            valid_count = 0
            for i in range(0, len(lines), 3):
                if i + 2 >= len(lines):
                    break
                try:
                    name = lines[i].strip()
                    line1 = lines[i + 1].strip()
                    line2 = lines[i + 2].strip()
                    
                    # 提取TLE参数过滤异常（备份/失效卫星）
                    inc_deg = float(line2[8:16])
                    mean_motion = float(line2[52:63])  # rev/day
                    ecc_str = line2[26:33].strip()
                    ecc = float('0.' + ecc_str)
                    
                    if not (87.8 < inc_deg < 88.0 and 
                            13.10 < mean_motion < 13.25 and 
                            ecc < 0.001):
                        self.logger.debug(f"过滤异常卫星 {name}: inc={inc_deg:.3f}, MM={mean_motion:.3f}, ecc={ecc:.6f}")
                        continue  # 丢弃~5颗无效
                    
                    valid_count += 1
                    satellite = EarthSatellite(line1, line2, name, self.ts)
                    
                    # 提取轨道参数
                    raan = satellite.model.nodeo  # rad
                    ma = satellite.model.mo       # rad
                    
                    raw_satellites.append({
                        'sat': satellite,
                        'raan': raan,
                        'ma': ma,
                        'original_name': name
                    })
                    
                except Exception as e:
                    self.logger.error(f"加载卫星 {name} 时出错: {str(e)}")
                    continue
            
            self.logger.info(f"过滤后有效卫星: {valid_count} 颗（总{len(raw_satellites)}）")

            if not raw_satellites:
                raise ValueError("No valid satellites loaded from TLE file")

            # 2. 根据RAAN分组
            raw_satellites.sort(key=lambda x: x['raan'])
            
            orbits = []
            current_orbit = [raw_satellites[0]]
            
            threshold = 0.25  # ~14°，优化
            
            for i in range(1, len(raw_satellites)):
                diff = raw_satellites[i]['raan'] - raw_satellites[i-1]['raan']
                if diff > threshold:
                    orbits.append(current_orbit)
                    current_orbit = []
                current_orbit.append(raw_satellites[i])
            
            if current_orbit:
                orbits.append(current_orbit)
            
            # 环绕合并
            if len(orbits) > 1:
                first_avg = sum(s['raan'] for s in orbits[0]) / len(orbits[0])
                last_avg = sum(s['raan'] for s in orbits[-1]) / len(orbits[-1])
                wrap_diff = (first_avg + 2 * np.pi - last_avg) % (2 * np.pi)
                wrap_threshold = threshold  # 或0.20 rad固定
                if wrap_diff < wrap_threshold:
                    ...
                    self.logger.info(f"Merged true wrap-around (diff={wrap_diff:.3f} rad)")
                else:
                    self.logger.info(f"Wrap gap {wrap_diff:.3f} rad >阈值，保持分离（正常相邻面）")
            
            # 小簇清理（<5颗合并到最近大簇，避免孤立）
            cleaned_orbits = []
            for orb in orbits:
                if len(orb) >= 5:
                    cleaned_orbits.append(orb)
                else:
                    self.logger.info(f"丢弃小簇 {len(orb)}颗（异常）")
            orbits = cleaned_orbits
            
            self.logger.info(f"最终轨道平面: {len(orbits)} 个")
            for i, orb in enumerate(orbits):
                avg_raan_deg = np.degrees(sum(s['raan'] for s in orb) / len(orb))
                self.logger.info(f"  轨道 {i+1}: {len(orb)} 颗 (RAAN~{avg_raan_deg:.1f}°)")
            
            # 3. 重命名&映射
            satellites = {}
            for orbit_idx, orbit_sats in enumerate(orbits):
                orbit_sats.sort(key=lambda x: x['ma'])
                for sat_idx, item in enumerate(orbit_sats):
                    new_name = f"satellite_{orbit_idx + 1}-{sat_idx + 1}"
                    satellites[new_name] = item['sat']
                    self.logger.debug(f"映射: {item['original_name']} -> {new_name}")
            
            self.logger.info(f"总映射卫星: {len(satellites)} 颗")
            return satellites
            
        except Exception as e:
            self.logger.error(f"加载TLE文件出错: {str(e)}")
            raise

    
    def compute_position(self, sat_name: str, time: float) -> np.ndarray:
        """计算卫星位置"""
        try:
            # 优先使用TLE计算
            if sat_name in self.satellites:
                dt = datetime.fromtimestamp(time, timezone.utc)
                t = self.ts.from_datetime(dt)
                geocentric = self.satellites[sat_name].at(t)
                return geocentric.position.km
                
            # 如果找不到卫星，尝试解析ID并回退到解析模型（仅作为后备）
            # 统一使用 satellite_X-X 格式
            if sat_name.startswith('Iridium'):
                parts = sat_name.split()[1].split('-')
                sat_name = f"satellite_{parts[0]}-{parts[1]}"
            
            if sat_name in self.satellites:
                dt = datetime.fromtimestamp(time, timezone.utc)
                t = self.ts.from_datetime(dt)
                geocentric = self.satellites[sat_name].at(t)
                return geocentric.position.km

            # 如果还是找不到，记录错误并返回零
            import traceback
            self.logger.warning(f"找不到卫星 {sat_name} 的TLE数据，无法计算位置. 调用栈:\n{''.join(traceback.format_stack()[-5:])}")
            
            # Debug: check what keys ARE in self.satellites
            if sat_name.startswith("satellite_6-"):
                keys_6 = [k for k in self.satellites.keys() if k.startswith("satellite_6-")]
                self.logger.warning(f"Orbit 6 satellites in network_model: {keys_6}")
                
            return np.array([0.0, 0.0, 0.0])
            
        except Exception as e:
            self.logger.error(f"计算卫星 {sat_name} 位置时出错: {str(e)}")
            return np.array([0.0, 0.0, 0.0])

    
    # def check_visibility(self, src: str, dst: str, time: float) -> bool:
    #     """
    #     检查源节点和目标节点之间是否可见
    #     Args:
    #         src: 源节点ID(可以是卫星或地面站)
    #         dst: 目标节点ID(可以是卫星或地面站)
    #         time: 时间戳
    #     Returns:
    #         bool: 是否可见
    #     """
    #     try:
    #         # 转换卫星ID格式
    #         def convert_sat_id(sat_id: str) -> str:
    #             if sat_id.startswith('satellite_'):
    #                 num = sat_id.split('_')[1]
    #                 return f"Iridium {num}"
    #             return sat_id

    #         # 如果包含地面站
    #         if src.startswith('station_') or dst.startswith('station_'):
    #             station_id = src if src.startswith('station_') else dst
    #             sat_id = dst if dst.startswith('satellite_') else src
    #             return self.check_ground_station_visibility(station_id, convert_sat_id(sat_id), time)
                
    #         # 卫星间可见性检查
    #         src_id = convert_sat_id(src)
    #         dst_id = convert_sat_id(dst)
            
    #         try:
    #             sat1_num = int(src_id.split()[-1])
    #             sat2_num = int(dst_id.split()[-1])
                
    #             # 计算轨道平面差异
    #             plane_diff = abs(sat1_num - sat2_num)
    #             if plane_diff > 3:  # 如果相差超过半个星座，取补值
    #                 plane_diff = 6 - plane_diff
                    
    #             # 检查是否为相邻轨道平面或同一轨道平面
    #             if plane_diff > 1:
    #                 return False  # 非相邻轨道平面不可见
                    
    #             pos1 = self.compute_position(src, time)
    #             pos2 = self.compute_position(dst, time)
                
    #             # 计算距离
    #             distance = float(np.linalg.norm(pos2 - pos1))  # 转换为Python float
                
    #             # 根据轨道平面关系设置阈值
    #             if plane_diff == 1:  # 相邻轨道平面
    #                 is_visible = distance <= 7500.0  # 稍大于标称距离7155km
    #             else:  # 同一轨道平面
    #                 is_visible = distance <= 4000.0
                    
    #             return bool(is_visible)  # 显式转换为Python布尔值
                
    #         except Exception as e:
    #             self.logger.error(f"可见性检查时出错 ({src}-{dst}): {str(e)}")
    #             return False
                
    #     except Exception as e:
    #         self.logger.error(f"可见性检查主方法出错: {str(e)}")
    #         return False

    def _is_earth_blocked(self, pos1: np.ndarray, pos2: np.ndarray) -> bool:
        """
        检查两点连线是否被地球遮挡（地球遮挡检测）
        Args:
            pos1, pos2: ECEF 坐标 (km)
        Returns:
            True 表示连线穿过地球，链路不可用
        """
        R_EARTH = 6371.0  # km
        d = pos2 - pos1
        denom = np.dot(d, d)
        if denom == 0:
            return False
        # 线段上离地心最近点的参数 t
        t = np.clip(-np.dot(pos1, d) / denom, 0.0, 1.0)
        closest = pos1 + t * d
        return bool(np.linalg.norm(closest) < R_EARTH)

    def _check_visibility(self, src: str, dst: str, time: float) -> bool:
        """
        检查源节点和目标节点之间是否可见
        """
        try:
            # 如果包含地面站
            if src.startswith('station_') or dst.startswith('station_'):
                station_id = src if src.startswith('station_') else dst
                sat_id = dst if dst.startswith('satellite_') else src
                return self.check_ground_station_visibility(station_id, sat_id, time)

            # 卫星间可见性检查
            pos1 = self.compute_position(src, time)
            pos2 = self.compute_position(dst, time)

            # 位置无效（0,0,0）则不可见
            if np.all(pos1 == 0) or np.all(pos2 == 0):
                return False

            dist = np.linalg.norm(pos1 - pos2)

            # 超过最大 ISL 距离（从配置读取）
            if dist > self.max_isl_distance:
                return False

            # 地球遮挡检查
            if self._is_earth_blocked(pos1, pos2):
                return False

            return True

        except Exception as e:
            self.logger.error(f"可见性检查错误 ({src}-{dst}): {str(e)}")
            return False
        
    def _parse_satellite_id(self, sat_id: str) -> Tuple[int, int]:
        """解析卫星ID，返回(轨道号, 卫星号)"""
        try:
            # 格式: satellite_{orbit}-{sat}
            parts = sat_id.split('_')[1].split('-')
            return int(parts[0]), int(parts[1])
        except:
            return 0, 0
        
    def compute_doppler_shift(self, sat1: str, sat2: str, 
                            time: float, frequency: float) -> float:
        """
        计算多普勒频移
        Args:
            sat1: 发送卫星名称
            sat2: 接收卫星名称
            time: 时间戳
            frequency: 载波频率(Hz)
        Returns:
            频移量(Hz)
        """
        dt = 0.1  # 时间差分间隔(s)
        
        # 计算t和t+dt时刻的位置
        pos1_t = self.compute_position(sat1, time)
        pos2_t = self.compute_position(sat2, time)
        pos1_dt = self.compute_position(sat1, time + dt)
        pos2_dt = self.compute_position(sat2, time + dt)
        
        # 计算相对速度
        vel1 = (pos1_dt - pos1_t) / dt
        vel2 = (pos2_dt - pos2_t) / dt
        rel_vel = vel2 - vel1
        
        # 计算视线方向
        los = pos2_t - pos1_t
        los_unit = los / np.linalg.norm(los)
        
        # 计算径向速度
        radial_vel = np.dot(rel_vel, los_unit)
        
        # 计算多普勒频移
        c = 299792.458  # 光速(km/s)
        doppler_shift = frequency * radial_vel / c
        
        return doppler_shift

    def get_orbit_plane(self, sat_name: str) -> np.ndarray:
        """
        计算卫星轨道平面的法向量
        Args:
            sat_name: 卫星名称
        Returns:
            轨道平面法向量
        """
        # 计算三个时间点的位置
        t0 = self.ts.now()
        positions = []
        for dt in [0, 10, 20]:  # 取20分钟内的三个点
            t = self.ts.from_datetime(t0.utc_datetime() + timedelta(minutes=dt))
            geocentric = self.satellites[sat_name].at(t)
            positions.append(geocentric.position.km)
            
        # 使用叉积计算轨道平面法向量
        v1 = positions[1] - positions[0]
        v2 = positions[2] - positions[0]
        normal = np.cross(v1, v2)
        return normal / np.linalg.norm(normal)
    
    def check_ground_station_visibility(self, station_id: str, sat_id: str, time: float, min_elevation: float = 10.0) -> bool:
        """
        检查地面站和卫星之间是否可见
        Args:
            station_id: 地面站ID
            sat_id: 卫星ID
            time: 时间戳
            min_elevation: 最小仰角(度)，默认为10度以模拟真实通信环境(避开地形遮挡和大气衰减)
        Returns:
            bool: 是否可见
        """
        try:
            if station_id not in self.ground_stations:
                self.logger.error(f"地面站 {station_id} 不存在")
                return False
            
            # 获取地面站位置 (lat, lon, alt)
            station_coords = self.ground_stations[station_id]
            
            # 获取卫星位置
            sat_pos = self.compute_position(sat_id, time)
            station_pos = self._geodetic_to_ecef(*station_coords)
            
            # 计算距离
            range_vector = sat_pos - station_pos
            distance = np.linalg.norm(range_vector)
            
            # 最大可见距离 (km)
            # 1200km轨道高度的几何地平线视距约为4100km
            # 考虑到10度仰角，有效通信距离通常小于4000km
            max_range = 4000.0 
            
            if distance > max_range:
                return False
                
            # 计算仰角
            # 简单的仰角计算：基于地心向量和站心向量的夹角
            # 更精确的计算需要坐标转换，但这里作为近似
            up_vector = station_pos / np.linalg.norm(station_pos)
            
            # 计算视线向量与天顶方向的夹角余弦
            cos_zenith = np.dot(range_vector/distance, up_vector)
            
            # 仰角 = 90 - 天顶角
            elevation = np.degrees(np.arcsin(cos_zenith))
            
            # self.logger.debug(f"地面站 {station_id} -> 卫星 {sat_id}: 距离={distance:.2f}km, 仰角={elevation:.2f}度")
            
            is_visible = elevation >= min_elevation
            
            # 添加调试日志：如果不可见且是问题站点，记录详细信息
            if not is_visible and station_id in ['station_2', 'station_3'] and elevation > -10.0:
                 # 只记录接近可见的情况 (-10度到0度) 以避免日志爆炸
                 self.logger.debug(f"可见性检查失败: {station_id} -> {sat_id}, 仰角={elevation:.2f}, 距离={distance:.2f}, 阈值={min_elevation}")
            
            return is_visible
            
        except Exception as e:
            self.logger.error(f"地面站可见性检查出错: {str(e)}")
            return False

    def _geodetic_to_ecef(self, lat: float, lon: float, alt: float) -> np.ndarray:
        """
        将大地坐标（经纬度）转换为ECEF坐标
        Args:
            lat: 纬度(度)
            lon: 经度(度)
            alt: 高度(km)
        Returns:
            np.ndarray: ECEF坐标(x, y, z)
        """
        # WGS84椭球体参数
        a = 6378.137  # 长半轴(km)
        e2 = 0.006694379990141  # 第一偏心率平方
        
        # 转换为弧度
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        
        # 计算卯酉圈曲率半径
        N = a / np.sqrt(1 - e2 * np.sin(lat_rad)**2)
        
        # 计算ECEF坐标
        x = (N + alt) * np.cos(lat_rad) * np.cos(lon_rad)
        y = (N + alt) * np.cos(lat_rad) * np.sin(lon_rad)
        z = (N * (1 - e2) + alt) * np.sin(lat_rad)
        
        return np.array([x, y, z])