"""
高层规划器模块
基于神经网络和事件触发机制的子目标生成器
处理环境信息和全局目标，生成安全的中间子目标
"""

import numpy as np
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


class SubgoalNetwork(nn.Module):
    """
    子目标生成神经网络
    处理激光雷达数据和目标信息，生成导航子目标（距离和角度）
    为机器人提供中间路径点，实现更安全高效的导航
    """

    def __init__(self, belief_dim=90, hidden_dim=256):
        """
        初始化子目标生成网络

        Args:
            belief_dim: 信念状态输入的维度
            hidden_dim: 隐藏层的维度
        """
        super(SubgoalNetwork, self).__init__()

        # CNN层用于处理激光雷达数据
        self.cnn1 = nn.Conv1d(1, 8, kernel_size=5, stride=2)  # 第一层CNN
        self.cnn2 = nn.Conv1d(8, 16, kernel_size=3, stride=2)  # 第二层CNN
        self.cnn3 = nn.Conv1d(16, 8, kernel_size=3, stride=1)  # 第三层CNN

        # 全局目标信息处理层
        self.goal_embed = nn.Linear(3, 32)  # 处理距离、余弦、正弦三个目标信息

        # 全连接层
        cnn_output_dim = self._get_cnn_output_dim(belief_dim)  # 计算CNN输出维度
        self.fc1 = nn.Linear(cnn_output_dim + 32, hidden_dim)  # 第一层全连接
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)  # 第二层全连接

        # 输出层：子目标距离和角度
        self.distance_head = nn.Linear(hidden_dim // 2, 1)  # 距离输出头
        self.angle_head = nn.Linear(hidden_dim // 2, 1)  # 角度输出头

    def _get_cnn_output_dim(self, belief_dim):
        """计算CNN层展平后的输出维度"""
        # 创建虚拟输入来计算输出维度
        x = torch.zeros(1, 1, belief_dim)
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        return x.numel()  # 返回元素总数

    def forward(self, belief_state, goal_info):
        """
        子目标网络的前向传播

        Args:
            belief_state: 包含激光雷达数据的张量
            goal_info: 包含[距离, cos, sin]全局目标信息的张量

        Returns:
            包含(子目标距离, 子目标角度)的元组
        """
        # 处理激光雷达数据
        laser = belief_state.unsqueeze(1)  # 增加通道维度
        x = F.relu(self.cnn1(laser))  # 第一层CNN + ReLU激活
        x = F.relu(self.cnn2(x))  # 第二层CNN + ReLU激活
        x = F.relu(self.cnn3(x))  # 第三层CNN + ReLU激活
        x = x.flatten(start_dim=1)  # 展平特征图

        # 处理目标信息
        g = F.relu(self.goal_embed(goal_info))  # 目标嵌入 + ReLU激活

        # 合并特征
        combined = torch.cat((x, g), dim=1)

        # 全连接层处理
        x = F.relu(self.fc1(combined))  # 第一层全连接 + ReLU
        x = F.relu(self.fc2(x))  # 第二层全连接 + ReLU

        # 生成子目标参数
        distance = 3 * torch.sigmoid(self.distance_head(x))  # 距离范围[0, 3]米
        angle = torch.tanh(self.angle_head(x)) * np.pi  # 角度范围[-π, π]

        return distance, angle


class EventTrigger:
    """
    多源事件触发机制实现
    包含多种触发条件，基于环境因素、机器人状态和时间约束决定何时生成新子目标
    """

    def __init__(self,
                 base_threshold=0.7,
                 complexity_factor=0.3,
                 safe_distance=0.5,
                 heading_threshold=0.3,
                 min_interval=1.0):
        """
        初始化事件触发机制

        Args:
            base_threshold: 基础风险阈值 τ0
            complexity_factor: 环境复杂性影响系数 α
            safe_distance: 安全距离阈值（米）
            heading_threshold: 航向变化阈值（弧度）
            min_interval: 触发之间的最小时间间隔（秒）
        """
        self.base_threshold = base_threshold
        self.complexity_factor = complexity_factor
        self.safe_distance = safe_distance
        self.heading_threshold = heading_threshold
        self.min_interval = min_interval

        # 状态变量
        self.last_trigger_time = 0.0  # 上次触发时间
        self.last_heading = 0.0  # 上次航向
        self.last_subgoal = None  # 上次子目标
        self.current_threshold = base_threshold  # 当前阈值

    def risk_assessment_trigger(self, risk_level, env_complexity):
        """
        基于环境风险评估的触发

        Args:
            risk_level: 当前评估的风险等级 [0, 1]
            env_complexity: 计算的环境复杂性 [0, 1]

        Returns:
            布尔值，指示是否满足触发条件
        """
        # 基于环境复杂性动态调整阈值
        self.current_threshold = self.base_threshold * (1 - self.complexity_factor * env_complexity)
        return risk_level > self.current_threshold  # 风险超过阈值则触发

    def obstacle_proximity_trigger(self, min_obstacle_dist):
        """
        基于障碍物接近度的触发

        Args:
            min_obstacle_dist: 到最近障碍物的距离（米）

        Returns:
            布尔值，指示是否满足触发条件
        """
        return min_obstacle_dist < self.safe_distance * 1.5  # 距离小于安全距离的1.5倍则触发

    def heading_change_trigger(self, current_heading):
        """
        基于航向显著变化的触发

        Args:
            current_heading: 当前机器人航向（弧度）

        Returns:
            布尔值，指示是否满足触发条件
        """
        heading_change = abs(self.last_heading - current_heading)  # 计算航向变化

        # 处理±π环绕
        if heading_change > np.pi:
            heading_change = 2 * np.pi - heading_change

        self.last_heading = current_heading  # 更新上次航向
        return heading_change > self.heading_threshold  # 变化超过阈值则触发

    def subgoal_reachability_trigger(self, current_pos, subgoal_pos, min_obstacle_dist):
        """
        基于子目标可达性评估的触发

        Args:
            current_pos: 当前机器人位置 [x, y]
            subgoal_pos: 当前子目标位置 [x, y]
            min_obstacle_dist: 到最近障碍物的距离

        Returns:
            布尔值，指示是否满足触发条件
        """
        if self.last_subgoal is None:
            return False  # 没有子目标时不触发

        # 计算到子目标的距离
        subgoal_dist = np.linalg.norm(np.array(current_pos) - np.array(subgoal_pos))

        # 如果子目标很近，检查是否已到达
        if subgoal_dist < 0.3:
            return True  # 已到达子目标，需要新子目标

        # 如果到子目标的路径被阻塞
        if min_obstacle_dist < self.safe_distance and subgoal_dist > self.safe_distance:
            # 计算到障碍物和子目标的向量点积
            if np.dot(current_pos, subgoal_pos) / (np.linalg.norm(current_pos) * np.linalg.norm(subgoal_pos)) > 0.7:
                return True  # 路径阻塞，需要新子目标

        return False  # 不需要新子目标

    def time_based_trigger(self):
        """
        确保触发之间的最小时间间隔

        Returns:
            布尔值，指示自上次触发以来是否经过了足够时间
        """
        current_time = time.time()
        if current_time - self.last_trigger_time > self.min_interval:
            return True  # 时间间隔满足
        return False  # 时间间隔不满足

    def reset_time(self):
        """重置上次触发时间为当前时间"""
        self.last_trigger_time = time.time()


class HighLevelPlanner:
    """
    高层规划器类
    基于事件触发机制生成导航子目标
    使用神经网络计算子目标，并管理事件触发机制决定何时计算新子目标
    """

    def __init__(self,
                 belief_dim=90,
                 device=None,
                 save_directory=Path("ethsrl/models/high_level"),
                 model_name="high_level_planner",
                 load_model=False,
                 load_directory=None):
        """
        初始化高层规划器

        Args:
            belief_dim: 信念状态的维度
            device: 计算设备（CPU/GPU）
            save_directory: 模型检查点保存目录
            model_name: 模型文件名
            load_model: 是否加载预训练模型
            load_directory: 模型加载目录（如果为None则使用save_directory）
        """
        self.belief_dim = belief_dim
        # 设置计算设备，默认为GPU（如果可用）否则CPU
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 初始化子目标生成网络
        self.subgoal_network = SubgoalNetwork(belief_dim=belief_dim).to(self.device)

        # 初始化事件触发器
        self.event_trigger = EventTrigger()

        # 训练设置
        self.optimizer = torch.optim.Adam(self.subgoal_network.parameters(), lr=1e-4)  # 优化器
        self.writer = SummaryWriter(comment=model_name)  # TensorBoard记录器
        self.iter_count = 0  # 迭代计数器
        self.model_name = model_name  # 模型名称
        self.save_directory = save_directory  # 保存目录

        # 当前状态跟踪
        self.current_subgoal = None  # 当前子目标
        self.last_goal_distance = float('inf')  # 上次目标距离
        self.last_goal_direction = 0.0  # 上次目标方向

        # 如果请求则加载预训练模型
        if load_model:
            load_dir = load_directory if load_directory else save_directory
            self.load_model(filename=model_name, directory=load_dir)

    def process_laser_scan(self, laser_scan):
        """
        处理原始激光雷达数据为信念状态表示

        Args:
            laser_scan: 原始激光雷达读数

        Returns:
            处理后的激光雷达张量
        """
        laser_scan = np.array(laser_scan)

        # 处理无穷大值
        inf_mask = np.isinf(laser_scan)
        laser_scan[inf_mask] = 7.0  # 用最大范围值替换

        # 归一化到[0, 1]范围
        laser_scan = laser_scan / 7.0

        return torch.FloatTensor(laser_scan).to(self.device)

    def process_goal_info(self, distance, cos_angle, sin_angle):
        """
        处理目标信息为张量

        Args:
            distance: 到全局目标的距离
            cos_angle: 到全局目标角度的余弦值
            sin_angle: 到全局目标角度的正弦值

        Returns:
            处理后的目标信息张量
        """
        # 归一化距离
        norm_distance = min(distance / 10.0, 1.0)  # 最大10米，归一化到[0,1]

        # 组合成张量
        goal_info = torch.FloatTensor([norm_distance, cos_angle, sin_angle]).to(self.device)

        return goal_info

    def check_triggers(self, laser_scan, robot_pose, goal_info, min_obstacle_dist=None):
        """
        检查是否有任何事件触发器被激活

        Args:
            laser_scan: 当前激光雷达读数
            robot_pose: 当前机器人位姿 [x, y, theta]
            goal_info: 全局目标信息 [distance, cos, sin]
            min_obstacle_dist: 到最近障碍物的距离（如果为None，则从laser_scan计算）

        Returns:
            布尔值，指示是否应生成新子目标
        """
        # 如果最小时间未过，不触发
        if not self.event_trigger.time_based_trigger():
            return False

        # 如果未提供则计算最小障碍物距离
        if min_obstacle_dist is None:
            valid_scans = laser_scan[~np.isinf(laser_scan)]  # 过滤有效扫描值
            min_obstacle_dist = np.min(valid_scans) if valid_scans.size > 0 else float('inf')

        # 计算环境复杂性
        env_complexity = self.compute_environment_complexity(laser_scan)

        # 计算风险等级（简化版）
        risk_level = 1.0 - min(min_obstacle_dist / 3.0, 1.0)  # 距离越近风险越高

        # 检查各个触发器
        risk_trigger = self.event_trigger.risk_assessment_trigger(risk_level, env_complexity)  # 风险评估触发
        obstacle_trigger = self.event_trigger.obstacle_proximity_trigger(min_obstacle_dist)  # 障碍物接近触发
        heading_trigger = self.event_trigger.heading_change_trigger(robot_pose[2])  # 航向变化触发

        # 从当前子目标创建子目标位置（如果存在）
        subgoal_pos = None
        if self.current_subgoal is not None:
            distance, angle = self.current_subgoal
            # 计算子目标的世界坐标
            subgoal_x = robot_pose[0] + distance * np.cos(robot_pose[2] + angle)
            subgoal_y = robot_pose[1] + distance * np.sin(robot_pose[2] + angle)
            subgoal_pos = [subgoal_x, subgoal_y]

        # 可达性触发检查
        reachability_trigger = self.event_trigger.subgoal_reachability_trigger(
            [robot_pose[0], robot_pose[1]],  # 当前位置
            subgoal_pos if subgoal_pos else [robot_pose[0], robot_pose[1]],  # 子目标位置
            min_obstacle_dist  # 最小障碍物距离
        )

        # 组合所有触发器（任一触发即生成新子目标）
        trigger_new_subgoal = risk_trigger or obstacle_trigger or heading_trigger or reachability_trigger

        # 如果触发，重置时间计数器
        if trigger_new_subgoal:
            self.event_trigger.reset_time()

        return trigger_new_subgoal

    def generate_subgoal(self, laser_scan, goal_distance, goal_cos, goal_sin):
        """
        基于当前状态生成新子目标

        Args:
            laser_scan: 处理后的激光雷达数据
            goal_distance: 到全局目标的距离
            goal_cos: 到全局目标角度的余弦值
            goal_sin: 到全局目标角度的正弦值

        Returns:
            包含(子目标距离, 子目标角度)的元组
        """
        # 处理输入
        laser_tensor = self.process_laser_scan(laser_scan)  # 激光数据张量化
        goal_tensor = self.process_goal_info(goal_distance, goal_cos, goal_sin)  # 目标信息张量化

        # 使用网络生成子目标（不计算梯度）
        with torch.no_grad():
            distance, angle = self.subgoal_network(
                laser_tensor.unsqueeze(0),  # 增加批次维度
                goal_tensor.unsqueeze(0)  # 增加批次维度
            )

        # 转换为numpy数组
        subgoal_distance = distance.cpu().numpy().item()
        subgoal_angle = angle.cpu().numpy().item()

        # 存储供将来参考
        self.current_subgoal = (subgoal_distance, subgoal_angle)
        self.last_goal_distance = goal_distance
        self.last_goal_direction = np.arctan2(goal_sin, goal_cos)  # 计算目标方向角度

        return subgoal_distance, subgoal_angle

    def compute_environment_complexity(self, laser_scan):
        """
        基于激光雷达数据计算当前环境的复杂性

        Args:
            laser_scan: 激光雷达读数

        Returns:
            环境复杂性得分，范围[0, 1]
        """
        # 用最大范围值替换无穷大值
        scan = np.array(laser_scan)
        scan[np.isinf(scan)] = 7.0

        # 基于以下因素的简单复杂性度量：
        # 1. 扫描读数的方差（方差越大越复杂）
        # 2. 平均距离（障碍物越近越复杂）
        variance = np.var(scan) / 10.0  # 归一化方差
        avg_distance = np.mean(scan) / 7.0  # 归一化均值

        # 计算复杂性得分（平均距离的倒数，用方差加权）
        complexity = (1.0 - avg_distance) * (0.5 + 0.5 * min(variance, 1.0))

        return min(complexity, 1.0)  # 确保在[0, 1]范围内

    def filter_unsafe_subgoals(self, laser_scan, candidate_subgoals):
        """
        基于激光雷达数据过滤不安全的子目标选项

        Args:
            laser_scan: 当前激光雷达数据
            candidate_subgoals: (距离, 角度)元组的列表

        Returns:
            安全的(距离, 角度)元组列表
        """
        safe_subgoals = []

        # 将扫描转换为笛卡尔坐标以便处理
        angles = np.linspace(-np.pi, np.pi, len(laser_scan))  # 生成角度数组

        for subgoal in candidate_subgoals:
            distance, angle = subgoal

            # 检查到子目标的路径是否清晰
            index = int((angle + np.pi) / (2 * np.pi) * len(laser_scan))  # 计算对应索引
            index = max(0, min(index, len(laser_scan) - 1))  # 确保有效索引

            # 检查子目标是否在安全距离内
            if distance < laser_scan[index] - self.event_trigger.safe_distance:
                safe_subgoals.append(subgoal)  # 路径清晰，子目标安全

        # 如果所有子目标都不安全，选择最安全的一个
        if not safe_subgoals and candidate_subgoals:
            safest_distance = 0  # 最远距离
            safest_subgoal = None

            for subgoal in candidate_subgoals:
                distance, angle = subgoal
                index = int((angle + np.pi) / (2 * np.pi) * len(laser_scan))
                index = max(0, min(index, len(laser_scan) - 1))

                if laser_scan[index] > safest_distance:  # 找到最远障碍物的方向
                    safest_distance = laser_scan[index]
                    safest_subgoal = subgoal

            if safest_subgoal:
                safe_subgoals.append(safest_subgoal)  # 添加最安全的子目标

        return safe_subgoals

    def update_planner(self, states, actions, rewards, next_states, dones, batch_size=64):
        """
        使用收集的经验更新规划器的神经网络

        Args:
            states: 环境状态批次
            actions: 采取的动作批次
            rewards: 获得的奖励批次
            next_states: 结果状态批次
            dones: 完成标志批次
            batch_size: 训练批次大小

        Returns:
            训练指标字典
        """
        # 转换为张量
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device).unsqueeze(1)

        # 将状态分割为激光和目标组件
        laser_scans = states[:, :-3]  # 激光数据部分
        goal_info = states[:, -3:]  # 目标信息部分

        # 生成子目标
        subgoal_distances, subgoal_angles = self.subgoal_network(laser_scans, goal_info)
        subgoals = torch.cat([subgoal_distances, subgoal_angles], dim=1)  # 合并距离和角度

        # 计算损失（简化示例使用MSE）
        # 在实际实现中，可能使用更复杂的损失函数
        loss = F.mse_loss(subgoals, actions)

        # 优化
        self.optimizer.zero_grad()  # 清零梯度
        loss.backward()  # 反向传播
        self.optimizer.step()  # 更新参数

        # 更新训练计数器
        self.iter_count += 1

        # 记录指标
        self.writer.add_scalar('planner/loss', loss.item(), self.iter_count)

        return {
            'loss': loss.item(),
            'avg_distance': subgoal_distances.mean().item(),  # 平均子目标距离
            'avg_angle': subgoal_angles.mean().item()  # 平均子目标角度
        }

    def save_model(self, filename, directory):
        """
        保存模型参数到文件

        Args:
            filename: 保存文件的基础名称
            directory: 保存目录
        """
        # 如果目录不存在则创建
        Path(directory).mkdir(parents=True, exist_ok=True)

        # 保存模型
        torch.save(self.subgoal_network.state_dict(), f"{directory}/{filename}.pth")
        print(f"模型已保存到 {directory}/{filename}.pth")

    def load_model(self, filename, directory):
        """
        从文件加载模型参数

        Args:
            filename: 要加载的文件的基础名称
            directory: 加载目录
        """
        try:
            self.subgoal_network.load_state_dict(torch.load(f"{directory}/{filename}.pth"))
            print(f"模型已从 {directory}/{filename}.pth 加载")
        except FileNotFoundError as e:
            print(f"加载模型时出错: {e}")
