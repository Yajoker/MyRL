"""高层事件触发器 - 决定何时更新子目标和导航模式"""

import numpy as np
import time

class MultiSourceTrigger:
    """
    多源事件触发器 - 集成多种触发条件
    """
    def __init__(
        self,
        risk_threshold=0.7,
        proximity_threshold=0.5,
        heading_threshold=0.3,
        subgoal_check_dist=2.0,
        min_trigger_interval=1.0,
        env_complexity_factor=0.3
    ):
        """
        初始化多源事件触发器
        
        参数:
            risk_threshold: 风险触发阈值
            proximity_threshold: 障碍物接近阈值 (米)
            heading_threshold: 航向变化阈值 (弧度)
            subgoal_check_dist: 子目标可达性检查距离
            min_trigger_interval: 最小触发间隔 (秒)
            env_complexity_factor: 环境复杂度影响因子
        """
        self.base_risk_threshold = risk_threshold
        self.risk_threshold = risk_threshold
        self.proximity_threshold = proximity_threshold
        self.heading_threshold = heading_threshold
        self.subgoal_check_dist = subgoal_check_dist
        self.min_trigger_interval = min_trigger_interval
        self.env_complexity_factor = env_complexity_factor
        
        # 触发相关状态
        self.last_trigger_time = 0
        self.trigger_count = 0
        self.trigger_reasons = []
        
        # 调试信息
        self.debug_info = {
            'risk_level': 0.0,
            'min_distance': float('inf'),
            'heading_change': 0.0,
            'is_subgoal_reachable': True,
            'env_complexity': 0.0
        }
    
    def check_trigger(self, state, goal, subgoal, obstacles=None, scan_data=None, local_path=None):
        """
        检查是否触发高层决策更新
        
        参数:
            state: 当前机器人状态 [x, y, theta, v, omega]
            goal: 最终目标 [x, y]
            subgoal: 当前子目标 [x, y]
            obstacles: 障碍物列表 [[x1, y1], [x2, y2], ...]
            scan_data: 激光雷达数据
            local_path: 局部路径点列表
            
        返回:
            triggered: 是否触发
            reason: 触发原因
        """
        current_time = time.time()
        
        # 冷却时间检查
        if current_time - self.last_trigger_time < self.min_trigger_interval:
            return False, None
        
        # 计算环境复杂度和调整阈值
        env_complexity = self._estimate_environment_complexity(obstacles, scan_data)
        self.debug_info['env_complexity'] = env_complexity
        
        # 根据环境复杂度调整风险阈值
        self.risk_threshold = self.base_risk_threshold * (1 - self.env_complexity_factor * env_complexity)
        
        # 1. 风险评估触发
        risk_level = self._calculate_risk(state, obstacles, scan_data)
        self.debug_info['risk_level'] = risk_level
        
        if risk_level > self.risk_threshold:
            self._register_trigger(current_time, 'risk')
            return True, 'risk'
        
        # 2. 障碍物接近触发
        min_distance = self._calculate_min_distance(state, obstacles, scan_data)
        self.debug_info['min_distance'] = min_distance
        
        if min_distance < self.proximity_threshold:
            self._register_trigger(current_time, 'proximity')
            return True, 'proximity'
        
        # 3. 航向变化触发
        heading_change = self._calculate_heading_change(state, goal, local_path)
        self.debug_info['heading_change'] = heading_change
        
        if abs(heading_change) > self.heading_threshold:
            self._register_trigger(current_time, 'heading')
            return True, 'heading'
        
        # 4. 子目标可达性触发
        is_subgoal_reachable = self._is_subgoal_reachable(state, subgoal, obstacles, scan_data)
        self.debug_info['is_subgoal_reachable'] = is_subgoal_reachable
        
        if not is_subgoal_reachable:
            self._register_trigger(current_time, 'unreachable')
            return True, 'unreachable'
        
        return False, None
    
    def _calculate_risk(self, state, obstacles=None, scan_data=None):
        """
        计算当前状态的风险级别
        
        参数:
            state: 机器人状态
            obstacles: 障碍物列表
            scan_data: 激光雷达数据
            
        返回:
            risk_level: [0, 1]之间的风险评分
        """
        risk = 0.0
        
        # 使用激光雷达数据计算风险
        if scan_data is not None and len(scan_data) > 0:
            min_scan = min(scan_data)
            scan_risk = 1.0 / (1.0 + min_scan)  # 转换为[0,1]区间
            risk = max(risk, scan_risk)
        
        # 使用障碍物列表计算风险
        if obstacles is not None and len(obstacles) > 0:
            robot_pos = np.array(state[:2])
            robot_vel = np.array(state[3:5]) if len(state) > 4 else np.array([0, 0])
            
            for obstacle in obstacles:
                # 计算到障碍物的距离
                obs_pos = np.array(obstacle[:2])
                distance = np.linalg.norm(robot_pos - obs_pos)
                
                # 如果障碍物有速度信息，计算相对速度
                if len(obstacle) > 2 and len(obstacle) >= 4:
                    obs_vel = np.array(obstacle[2:4])
                    rel_vel = np.linalg.norm(robot_vel - obs_vel)
                    
                    # 障碍物接近速度越快，风险越高
                    vel_factor = min(1.0, rel_vel)
                else:
                    vel_factor = 0.5  # 默认值
                
                # 距离越近，风险越高
                obstacle_risk = (1.0 / (1.0 + distance)) * (0.5 + 0.5 * vel_factor)
                risk = max(risk, obstacle_risk)
        
        return risk
    
    def _calculate_min_distance(self, state, obstacles=None, scan_data=None):
        """
        计算到最近障碍物的距离
        
        参数:
            state: 机器人状态
            obstacles: 障碍物列表
            scan_data: 激光雷达数据
            
        返回:
            min_distance: 最小距离(米)
        """
        min_distance = float('inf')
        
        # 使用激光雷达数据
        if scan_data is not None and len(scan_data) > 0:
            min_distance = min(min_distance, min(scan_data))
        
        # 使用障碍物列表
        if obstacles is not None and len(obstacles) > 0:
            robot_pos = np.array(state[:2])
            
            for obstacle in obstacles:
                obs_pos = np.array(obstacle[:2])
                distance = np.linalg.norm(robot_pos - obs_pos)
                min_distance = min(min_distance, distance)
        
        return min_distance
    
    def _calculate_heading_change(self, state, goal, local_path=None):
        """
        计算当前航向与目标方向的偏差
        
        参数:
            state: 机器人状态 [x, y, theta, ...]
            goal: 目标位置 [x, y]
            local_path: 局部路径
            
        返回:
            heading_change: 航向变化(弧度)
        """
        # 提取机器人位置和朝向
        robot_pos = np.array(state[:2])
        robot_theta = state[2] if len(state) > 2 else 0
        
        if local_path is not None and len(local_path) > 1:
            # 如果有局部路径，使用局部路径方向
            next_point = np.array(local_path[1])
            direction = next_point - robot_pos
        else:
            # 否则使用目标方向
            goal_pos = np.array(goal)
            direction = goal_pos - robot_pos
        
        # 计算目标方向的角度
        target_theta = np.arctan2(direction[1], direction[0])
        
        # 计算航向差，并归一化到[-pi, pi]
        heading_diff = target_theta - robot_theta
        heading_diff = (heading_diff + np.pi) % (2 * np.pi) - np.pi
        
        return heading_diff
    
    def _is_subgoal_reachable(self, state, subgoal, obstacles=None, scan_data=None):
        """
        检查当前子目标是否可达
        
        参数:
            state: 机器人状态
            subgoal: 子目标位置
            obstacles: 障碍物列表
            scan_data: 激光雷达数据
            
        返回:
            is_reachable: 是否可达
        """
        # 提取机器人位置
        robot_pos = np.array(state[:2])
        subgoal_pos = np.array(subgoal)
        
        # 计算方向和距离
        direction = subgoal_pos - robot_pos
        distance = np.linalg.norm(direction)
        
        # 如果距离过远，仅检查机器人前方一定距离
        check_distance = min(distance, self.subgoal_check_dist)
        direction = direction / distance if distance > 0 else np.array([1, 0])
        
        # 使用障碍物列表检查路径是否被阻挡
        if obstacles is not None and len(obstacles) > 0:
            # 检查从机器人到子目标的直线路径是否有障碍物
            for obstacle in obstacles:
                obs_pos = np.array(obstacle[:2])
                
                # 计算障碍物到路径的距离
                v = obs_pos - robot_pos
                proj = np.dot(v, direction)
                proj = max(0, min(check_distance, proj))  # 限制在检查距离内
                
                # 计算投影点
                proj_point = robot_pos + proj * direction
                
                # 计算障碍物到投影点的距离
                obs_distance = np.linalg.norm(obs_pos - proj_point)
                
                # 如果障碍物太接近路径，认为路径被阻挡
                if obs_distance < self.proximity_threshold:
                    return False
        
        # 使用激光雷达数据检查前方是否有障碍物
        if scan_data is not None and len(scan_data) > 0:
            # 计算子目标相对于机器人的角度
            subgoal_angle = np.arctan2(direction[1], direction[0])
            
            # 归一化为激光雷达数据的索引
            num_beams = len(scan_data)
            angle_increment = 2 * np.pi / num_beams
            beam_index = int((subgoal_angle % (2 * np.pi)) / angle_increment)
            
            # 检查子目标方向上的几个激光束
            angle_range = int(np.pi / 6 / angle_increment)  # 30度范围
            for i in range(beam_index - angle_range, beam_index + angle_range + 1):
                idx = i % num_beams
                
                # 如果该方向的障碍物距离小于检查距离，认为路径被阻挡
                if scan_data[idx] < check_distance:
                    return False
        
        return True
    
    def _estimate_environment_complexity(self, obstacles=None, scan_data=None):
        """
        估计环境复杂度
        
        参数:
            obstacles: 障碍物列表
            scan_data: 激光雷达数据
            
        返回:
            complexity: [0, 1]之间的复杂度评分
        """
        complexity = 0.0
        
        # 基于障碍物数量的复杂度
        if obstacles is not None:
            num_obstacles = len(obstacles)
            # 假设环境中最多有20个障碍物时复杂度达到最大
            obstacle_complexity = min(1.0, num_obstacles / 20.0)
            complexity += obstacle_complexity * 0.5  # 权重为0.5
        
        # 基于激光雷达数据的复杂度
        if scan_data is not None and len(scan_data) > 0:
            # 计算激光雷达读数的变异系数(CV)，表示环境不均匀程度
            mean_scan = np.mean(scan_data)
            if mean_scan > 0:
                std_scan = np.std(scan_data)
                cv = std_scan / mean_scan
                scan_complexity = min(1.0, cv / 0.5)  # 归一化，CV=0.5时复杂度为1
            else:
                scan_complexity = 1.0
                
            # 另外考虑最小距离
            min_scan = min(scan_data)
            dist_complexity = 1.0 / (1.0 + min_scan)  # 最近障碍物越近，环境越复杂
            
            # 组合两种复杂度
            scan_complexity = 0.7 * scan_complexity + 0.3 * dist_complexity
            complexity += scan_complexity * 0.5  # 权重为0.5
        
        return complexity
    
    def _register_trigger(self, time_stamp, reason):
        """
        记录触发事件
        
        参数:
            time_stamp: 触发时间戳
            reason: 触发原因
        """
        self.last_trigger_time = time_stamp
        self.trigger_count += 1
        self.trigger_reasons.append((time_stamp, reason))
        
        # 仅保留最近100条记录
        if len(self.trigger_reasons) > 100:
            self.trigger_reasons.pop(0)
    
    def reset(self):
        """重置触发器状态"""
        self.last_trigger_time = 0
        self.trigger_count = 0
        self.trigger_reasons = []