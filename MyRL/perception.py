"""感知模块 - 处理观测数据并生成特征表示"""

"""感知模块 - 处理观测数据并生成特征表示"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

class LidarProcessor:
    """
    激光雷达数据处理器 - 将原始激光数据转换为栅格表示或特征向量
    """
    def __init__(self, grid_size=64, resolution=0.1, max_range=10.0, history_length=8):
        """
        初始化激光雷达数据处理器
        
        参数:
            grid_size: 栅格地图大小
            resolution: 栅格分辨率 (米/像素)
            max_range: 最大有效距离 (米)
            history_length: 历史缓冲区长度
        """
        self.grid_size = grid_size
        self.resolution = resolution
        self.max_range = max_range
        self.history_length = history_length
        
        # 历史缓冲区
        self.scan_history = deque(maxlen=history_length)
        self.robot_state_history = deque(maxlen=history_length)
        
        # 最新栅格图和障碍物
        self.latest_grid_map = None
        self.latest_obstacles = None
        self.latest_scan = None
    
    def update(self, scan_data, robot_state=None):
        """
        更新内部状态并处理扫描数据
        
        参数:
            scan_data: 激光雷达扫描数据 [距离1, 距离2, ...]
            robot_state: 机器人状态 [x, y, theta, v, omega]
            
        返回:
            obstacles: 提取的障碍物列表
        """
        # 保存最新扫描数据
        self.latest_scan = scan_data
        
        # 处理扫描数据生成栅格图
        self.latest_grid_map = self.process_scan(scan_data, robot_state)
        
        # 提取障碍物信息
        self.latest_obstacles = self._extract_obstacles_from_scan(scan_data)
        
        return self.latest_obstacles
        
    def process_scan(self, scan_data, robot_state=None):
        """
        处理单帧激光雷达数据
        
        参数:
            scan_data: 激光雷达扫描数据 [距离1, 距离2, ...]
            robot_state: 机器人状态 [x, y, theta, v, omega]
            
        返回:
            grid_map: 栅格地图表示
        """
        # 更新历史缓冲区
        if robot_state is not None:
            self.robot_state_history.append(np.array(robot_state))
        
        # 将激光数据转换为栅格图
        grid_map = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        
        # 计算激光束的角度增量
        num_beams = len(scan_data)
        angle_increment = 2 * np.pi / num_beams
        
        # 栅格地图中心点（机器人位置）
        center_x = self.grid_size // 2
        center_y = self.grid_size // 2
        
        # 处理每一束激光数据
        for i, distance in enumerate(scan_data):
            # 如果距离超过最大范围，则忽略
            if distance > self.max_range:
                continue
            
            # 计算激光束角度（以机器人朝向为0度）
            angle = i * angle_increment
            
            # 计算障碍物的相对坐标（局部坐标系）
            obstacle_x = distance * np.cos(angle)
            obstacle_y = distance * np.sin(angle)
            
            # 将障碍物坐标转换为栅格坐标
            grid_x = int(center_x + obstacle_x / self.resolution)
            grid_y = int(center_y + obstacle_y / self.resolution)
            
            # 确保坐标在栅格地图范围内
            if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
                grid_map[grid_x, grid_y] = 1.0
        
        # 将当前栅格图加入历史缓冲区
        self.scan_history.append(grid_map)
        
        return grid_map
    
    def get_occupancy_grid(self):
        """
        获取最新的占据栅格地图
        
        返回:
            grid_map: 栅格地图，值为0-1之间表示占据概率
        """
        if self.latest_grid_map is None:
            # 如果还没有处理过扫描数据，返回空栅格图
            return np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        
        return self.latest_grid_map
    
    def _extract_obstacles_from_scan(self, scan_data):
        """
        从激光雷达扫描数据中提取障碍物
        
        参数:
            scan_data: 激光雷达扫描数据 [距离1, 距离2, ...]
            
        返回:
            obstacles: 障碍物列表 [(x1, y1), (x2, y2), ...]
        """
        obstacles = []
        
        if scan_data is None or len(scan_data) == 0:
            return obstacles
        
        # 计算激光束的角度增量
        num_beams = len(scan_data)
        angle_increment = 2 * np.pi / num_beams
        
        # 提取障碍物点
        for i, distance in enumerate(scan_data):
            # 如果距离超过最大范围或无效，则忽略
            if distance > self.max_range or distance <= 0:
                continue
            
            # 计算激光束角度（以机器人朝向为0度）
            angle = i * angle_increment
            
            # 计算障碍物的相对坐标（局部坐标系）
            obstacle_x = distance * np.cos(angle)
            obstacle_y = distance * np.sin(angle)
            
            # 添加到障碍物列表
            obstacles.append((obstacle_x, obstacle_y))
        
        return obstacles
    
    def get_feature_vector(self, scan_data, robot_state):
        """
        将激光雷达数据和机器人状态转换为特征向量
        
        参数:
            scan_data: 激光雷达扫描数据
            robot_state: 机器人状态 [x, y, theta, v, omega]
            
        返回:
            feature_vector: 特征向量
        """
        # 首先处理激光数据
        grid_map = self.process_scan(scan_data, robot_state)
        
        # 提取一些简单的特征
        # 1. 最小距离及其角度
        min_dist_idx = np.argmin(scan_data)
        min_dist = scan_data[min_dist_idx]
        min_angle = min_dist_idx * (2 * np.pi / len(scan_data))
        
        # 2. 四个象限内的最小距离
        quad_size = len(scan_data) // 4
        front_min = np.min(scan_data[:quad_size])
        right_min = np.min(scan_data[quad_size:2*quad_size])
        back_min = np.min(scan_data[2*quad_size:3*quad_size])
        left_min = np.min(scan_data[3*quad_size:])
        
        # 3. 距离梯度（表示环境变化）
        gradients = np.diff(scan_data, append=scan_data[0])
        max_gradient = np.max(np.abs(gradients))
        
        # 4. 机器人当前速度和角速度
        v = robot_state[3] if len(robot_state) > 3 else 0.0
        omega = robot_state[4] if len(robot_state) > 4 else 0.0
        
        # 组合特征
        features = [
            min_dist,
            min_angle,
            front_min,
            right_min,
            back_min,
            left_min,
            max_gradient,
            v,
            omega
        ]
        
        # 归一化特征
        features = np.array(features, dtype=np.float32)
        
        return features
    
    def get_observation_vector(self, scan_data, robot_state, goal_state):
        """
        生成完整的观测向量，用于强化学习输入
        
        参数:
            scan_data: 激光雷达扫描数据
            robot_state: 机器人状态 [x, y, theta, v, omega]
            goal_state: 目标状态 [x, y]
            
        返回:
            observation: 观测向量
        """
        # 提取特征向量
        features = self.get_feature_vector(scan_data, robot_state)
        
        # 计算机器人与目标的相对位置
        robot_x, robot_y = robot_state[:2]
        goal_x, goal_y = goal_state
        
        # 相对距离和方向
        dx = goal_x - robot_x
        dy = goal_y - robot_y
        distance = np.sqrt(dx**2 + dy**2)
        
        # 计算目标相对于机器人朝向的角度
        robot_theta = robot_state[2]
        goal_angle = np.arctan2(dy, dx) - robot_theta
        goal_angle = (goal_angle + np.pi) % (2 * np.pi) - np.pi  # 归一化到[-pi, pi]
        
        # 将目标信息转换为cos和sin形式，避免角度的不连续性
        goal_cos = np.cos(goal_angle)
        goal_sin = np.sin(goal_angle)
        
        # 将所有信息组合成一个观测向量
        observation = np.concatenate([
            scan_data,           # 原始激光数据
            features,            # 提取的特征
            [distance],          # 到目标的距离
            [goal_cos],          # 目标方向的余弦
            [goal_sin]           # 目标方向的正弦
        ])
        
        return observation


class AttentionEncoder(nn.Module):
    """
    注意力增强编码器 - 提取观测中的关键信息
    """
    def __init__(self, input_dim, hidden_dim=128, output_dim=64, num_heads=4):
        super(AttentionEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        
        # 输入投影层
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 自注意力层
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=num_heads
        )
        
        # 输出投影层
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入特征 [batch_size, input_dim]
            
        返回:
            output: 编码后的特征 [batch_size, output_dim]
            attention_weights: 注意力权重
        """
        batch_size = x.size(0)
        
        # 投影到隐藏维度
        projected = self.input_projection(x)
        
        # 重塑为序列形式，用于自注意力机制
        # [batch_size, hidden_dim] -> [1, batch_size, hidden_dim]
        # 我们将batch视为序列长度为1的多个样本
        projected = projected.unsqueeze(0)
        
        # 应用自注意力
        attn_output, attention_weights = self.self_attention(
            projected, projected, projected
        )
        
        # 重塑回原始形状
        # [1, batch_size, hidden_dim] -> [batch_size, hidden_dim]
        attn_output = attn_output.squeeze(0)
        
        # 输出投影
        output = self.output_projection(attn_output)
        
        return output, attention_weights


class HistoryEncoder:
    """
    历史编码器 - 管理观测和动作的历史，提供信念状态
    """
    def __init__(self, observation_dim, action_dim, history_length=8):
        """
        初始化历史编码器
        
        参数:
            observation_dim: 观测向量维度
            action_dim: 动作向量维度
            history_length: 历史长度
        """
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.history_length = history_length
        
        # 初始化历史缓冲区
        self.observation_history = deque(maxlen=history_length)
        self.action_history = deque(maxlen=history_length)
        
        # 用零填充初始历史
        for _ in range(history_length):
            self.observation_history.append(np.zeros(observation_dim))
            self.action_history.append(np.zeros(action_dim))
    
    def update(self, observation, action=None):
        """
        更新历史缓冲区
        
        参数:
            observation: 当前观测
            action: 当前动作 (如果有)
        """
        self.observation_history.append(np.array(observation))
        
        if action is not None:
            self.action_history.append(np.array(action))
    
    def get_belief_state(self):
        """
        获取当前信念状态（历史观测和动作的组合）
        
        返回:
            belief_state: 信念状态向量
        """
        # 将历史观测和动作平展并连接
        observations = np.array(list(self.observation_history))
        actions = np.array(list(self.action_history))
        
        # 组合成信念状态
        belief_state = np.concatenate([
            observations.flatten(),
            actions.flatten()
        ])
        
        return belief_state
