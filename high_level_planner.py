"""
高层规划器模块
基于神经网络和事件触发机制的子目标生成器
处理环境信息和全局目标，生成安全的中间子目标
"""

import math
import numpy as np
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from ethsrl.global_planner import WaypointWindow
from ethsrl.config import MotionConfig, TriggerConfig


def _logit(value: torch.Tensor) -> torch.Tensor:
    """Compute element-wise logit for values in (0, 1)."""

    return torch.log(value) - torch.log1p(-value)


def _artanh(value: torch.Tensor) -> torch.Tensor:
    """Compute element-wise inverse tanh for values in (-1, 1)."""

    return 0.5 * (torch.log1p(value) - torch.log1p(-value))

class SubgoalNetwork(nn.Module):
    """
    子目标生成神经网络
    处理激光雷达数据和目标信息，生成导航子目标（距离和角度）
    为机器人提供中间路径点，实现更安全高效的导航
    """

    def __init__(self, belief_dim=90, goal_info_dim=3, hidden_dim=256):
        """
        初始化子目标生成网络

        Args:
            belief_dim: 信念状态输入的维度（激光雷达数据点数）
            goal_info_dim: 目标信息（含航点特征）的维度
            hidden_dim: 隐藏层的维度
        """
        super(SubgoalNetwork, self).__init__()

        # CNN层用于处理激光雷达数据（一维卷积，处理序列数据）
        self.cnn1 = nn.Conv1d(1, 8, kernel_size=5, stride=2)  # 第一层CNN：输入1通道，输出8通道
        self.cnn2 = nn.Conv1d(8, 16, kernel_size=3, stride=2)  # 第二层CNN：输入8通道，输出16通道
        self.cnn3 = nn.Conv1d(16, 8, kernel_size=3, stride=1)  # 第三层CNN：输入16通道，输出8通道

        # 全局目标信息处理层
        self.goal_embed = nn.Linear(goal_info_dim, 64)  # 处理距离、余弦、正弦及航点特征

        # 历史动作嵌入层
        self.action_embed = nn.Linear(2, 16)  # 处理历史线速度和角速度，输出16维

        # 全连接层 - 更新输入维度
        cnn_output_dim = self._get_cnn_output_dim(belief_dim)  # 计算CNN输出维度
        # 第一层全连接：输入=CNN输出+目标嵌入+动作嵌入，输出=隐藏层维度
        self.fc1 = nn.Linear(cnn_output_dim + 64 + 16, hidden_dim)
        # 第二层全连接：输入=隐藏层维度，输出=隐藏层维度的一半
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        # 输出层：距离调整系数与角度偏移量
        self.distance_head = nn.Linear(hidden_dim // 2, 1)  # 距离缩放输出头
        self.angle_head = nn.Linear(hidden_dim // 2, 1)  # 角度偏移输出头

    def _get_cnn_output_dim(self, belief_dim):
        """计算CNN层展平后的输出维度"""
        # 创建虚拟输入来计算输出维度
        x = torch.zeros(1, 1, belief_dim)  # 批次大小1，通道数1，输入维度belief_dim
        x = self.cnn1(x)  # 通过第一层CNN
        x = self.cnn2(x)  # 通过第二层CNN
        x = self.cnn3(x)  # 通过第三层CNN
        return x.numel()  # 返回元素总数（展平后的维度）

    def forward(self, belief_state, goal_info, prev_action):
        """
        子目标网络的前向传播

        Args:
            belief_state: 包含激光雷达数据的张量，形状[batch_size, belief_dim]
            goal_info: 包含全局目标与航点特征的张量，形状[batch_size, goal_info_dim]
            prev_action: 包含[线速度, 角速度]历史动作的张量，形状[batch_size, 2]

        Returns:
            包含(距离调整系数, 角度偏移量)的元组
        """
        # 处理激光雷达数据
        laser = belief_state.unsqueeze(1)  # 增加通道维度：[batch_size, 1, belief_dim]
        x = F.relu(self.cnn1(laser))  # 第一层CNN + ReLU激活
        x = F.relu(self.cnn2(x))  # 第二层CNN + ReLU激活
        x = F.relu(self.cnn3(x))  # 第三层CNN + ReLU激活
        x = x.flatten(start_dim=1)  # 展平特征图：[batch_size, cnn_output_dim]

        # 处理目标信息
        g = F.relu(self.goal_embed(goal_info))  # 目标嵌入 + ReLU激活：[batch_size, 32]

        # 处理历史动作
        a = F.relu(self.action_embed(prev_action))  # 动作嵌入 + ReLU激活：[batch_size, 16]

        # 合并特征 - 更新为包含动作
        combined = torch.cat((x, g, a), dim=1)  # 拼接所有特征：[batch_size, cnn_output_dim+64+16]

        # 全连接层处理
        x = F.relu(self.fc1(combined))  # 第一层全连接 + ReLU：[batch_size, hidden_dim]
        x = F.relu(self.fc2(x))  # 第二层全连接 + ReLU：[batch_size, hidden_dim//2]

        # 生成未约束的距离与角度表示
        distance_logits = self.distance_head(x)
        angle_logits = self.angle_head(x)

        # 在推理阶段再压回合法区间，并在距离上保留 ε 以避免饱和
        eps = 1e-3
        distance_scale = torch.sigmoid(distance_logits) * (1 - 2 * eps) + eps
        angle_offset = torch.tanh(angle_logits) * (np.pi / 4)

        # squeeze掉最后一维，保持批量维度，便于后续堆叠或索引
        return (
            distance_scale.squeeze(-1),
            angle_offset.squeeze(-1),
            distance_logits.squeeze(-1),
            angle_logits.squeeze(-1),
        )


class EventTrigger:
    """事件触发器，聚焦安全距离与进度两类触发条件。"""

    def __init__(
        self,
        *,
        trigger_config: TriggerConfig,
        motion_config: MotionConfig,
    ) -> None:
        """
        初始化事件触发器

        Args:
            trigger_config: 高层触发相关阈值配置
            motion_config: 运动学与时间步配置
        """
        # 触发参数设置
        self.config = trigger_config
        self.motion = motion_config
        self.safety_trigger_distance = trigger_config.safety_trigger_distance
        self.subgoal_reach_threshold = trigger_config.subgoal_reach_threshold
        self.stagnation_steps = max(1, int(trigger_config.stagnation_steps))  # 确保至少1步
        self.progress_epsilon_floor = float(max(trigger_config.progress_epsilon, 0.0))
        self.progress_epsilon_ratio = float(max(trigger_config.progress_epsilon_ratio, 0.0))
        self.current_progress_epsilon = self.progress_epsilon_floor
        self.stagnation_turn_threshold = float(max(trigger_config.stagnation_turn_threshold, 0.0))
        self.window_inside_hold = max(0, int(trigger_config.window_inside_hold))
        self.step_duration = motion_config.dt
        self.min_interval = float(max(trigger_config.min_interval, 0.0))

        requested_steps = getattr(trigger_config, "min_step_interval", None)
        if requested_steps is not None and requested_steps > 0:
            self.min_step_interval = max(1, int(requested_steps))
        elif self.step_duration > 0:
            ratio = self.min_interval / self.step_duration if self.min_interval > 0 else 0
            self.min_step_interval = max(1, int(math.ceil(ratio)))
        else:
            self.min_step_interval = 1

        # 同步时间阈值到步数设定，保证后续日志一致
        if self.step_duration > 0:
            self.min_interval = self.min_step_interval * self.step_duration

        # 状态变量初始化
        self.last_trigger_step = -self.min_step_interval  # 上次触发时间步，初始化为负值确保第一次可触发
        self.last_subgoal: Optional[np.ndarray] = None  # 上次子目标位置
        self.best_goal_distance: Optional[float] = None  # 最佳目标距离（用于进度跟踪）
        self.last_progress_step = 0  # 上次有进展的时间步
        self.cumulative_turn = 0.0
        self.last_heading_error: Optional[float] = None

    def safe_distance_trigger(self, min_obstacle_dist: float) -> bool:
        """若最近障碍物距离低于安全阈值则触发。"""

        if np.isnan(min_obstacle_dist):  # 检查距离是否为NaN
            return False
        return min_obstacle_dist <= self.safety_trigger_distance  # 距离小于等于安全阈值则触发

    def reset_progress(self, goal_distance: float, current_step: int) -> None:
        """在生成新子目标时重置进度基准。"""

        if not np.isfinite(goal_distance):  # 检查目标距离是否有限（非无穷大/NaN）
            self.best_goal_distance = None  # 重置最佳距离
            self.last_progress_step = current_step  # 更新进度时间步
            return

        # 设置新的最佳距离基准
        self.best_goal_distance = goal_distance
        self.last_progress_step = current_step
        self.reset_turn_metrics()

    def update_progress(self, goal_distance: float, current_step: int) -> None:
        """根据当前全局目标距离更新最优进度。"""

        if not np.isfinite(goal_distance):  # 检查距离是否有效
            return

        if self.best_goal_distance is None:  # 如果还没有最佳距离记录
            self.best_goal_distance = goal_distance  # 设置初始最佳距离
            self.last_progress_step = current_step  # 记录当前时间步
            return

        # 如果当前距离比最佳距离小（更接近目标），且超过进度容差
        if goal_distance + self.current_progress_epsilon < self.best_goal_distance:
            self.best_goal_distance = goal_distance  # 更新最佳距离
            self.last_progress_step = current_step  # 更新进度时间步
            self.reset_turn_metrics()

    def set_progress_context(self, window_radius: Optional[float]) -> None:
        """根据窗口几何动态调整进度判据。"""

        epsilon = self.progress_epsilon_floor
        if window_radius is not None and window_radius > 0:
            epsilon = max(epsilon, window_radius * self.progress_epsilon_ratio)
        self.current_progress_epsilon = epsilon

    @staticmethod
    def _wrap(angle: float) -> float:
        wrapped = (float(angle) + math.pi) % (2.0 * math.pi) - math.pi
        if wrapped <= -math.pi:
            wrapped += 2.0 * math.pi
        return wrapped

    def update_heading_metrics(self, heading_error: Optional[float]) -> None:
        """累计当前窗口阶段内的转向幅度。"""

        if heading_error is None:
            self.last_heading_error = None
            return

        angle = float(heading_error)
        if self.last_heading_error is None:
            self.last_heading_error = angle
            return

        delta = abs(self._wrap(angle - self.last_heading_error))
        self.cumulative_turn += delta
        self.last_heading_error = angle

    def reset_turn_metrics(self) -> None:
        """在进度重置或换子目标时清空累计转向。"""

        self.cumulative_turn = 0.0
        self.last_heading_error = None

    def is_stagnated(self, current_step: int) -> bool:
        """判断是否出现长时间进展停滞。"""

        if self.best_goal_distance is None:  # 如果没有进度基准
            return False
        # 检查从上次进展到现在是否超过停滞步数阈值
        return (current_step - self.last_progress_step) >= self.stagnation_steps

    def _ray_distance_to_angle(self, laser_scan: Optional[np.ndarray], angle: float) -> float:
        """将角度转换为激光雷达扫描中的距离值"""
        
        if laser_scan is None or len(laser_scan) == 0 or np.isnan(angle):  # 检查输入有效性
            return float('inf')  # 无效输入返回无穷大

        # 将角度从[-π, π]归一化到[0, 1]
        normalized = (angle + np.pi) / (2 * np.pi)
        normalized = float(np.clip(normalized, 0.0, 1.0))  # 确保在0-1范围内
        # 计算对应的激光雷达索引
        index = int(round(normalized * (len(laser_scan) - 1)))
        distance = laser_scan[index]  # 获取该方向的激光距离
        if np.isnan(distance):  # 检查距离是否有效
            return float('inf')
        return float(distance)  # 返回有效距离

    def intelligent_progress_trigger(
        self,
        *,
        dist_to_subgoal: Optional[float],
        current_step: int,
        subgoal_angle: Optional[float],
        laser_scan: Optional[np.ndarray],
        min_obstacle_dist: float,
    ) -> bool:
        """组合子目标完成、进度停滞和子目标受阻的智能触发。"""

        if dist_to_subgoal is None:  # 如果没有当前子目标
            return False

        # 条件1: 子目标到达触发 - 距离子目标足够近
        if dist_to_subgoal <= self.subgoal_reach_threshold:
            return True

        # 条件2: 进度停滞触发 - 长时间没有向全局目标进展且累计转向超阈值
        self.update_heading_metrics(subgoal_angle)
        turn_excess = (
            self.stagnation_turn_threshold <= 0.0
            or self.cumulative_turn >= self.stagnation_turn_threshold
        )
        stagnation = self.is_stagnated(current_step) and turn_excess

        # 条件3: 子目标受阻触发 - 子目标方向有障碍物
        blocked = False
        if laser_scan is not None and subgoal_angle is not None:  # 检查输入有效性
            ray_distance = self._ray_distance_to_angle(laser_scan, subgoal_angle)  # 获取子目标方向的距离
            blocked = (
                np.isfinite(ray_distance)  # 距离有效
                and ray_distance <= self.safety_trigger_distance  # 子目标方向有近距离障碍物
                and min_obstacle_dist <= self.safety_trigger_distance  # 整体环境有近距离障碍物
            )

        # 任一进度相关条件满足即触发
        return stagnation or blocked

    def time_based_trigger(self, current_step: int) -> bool:
        """确保触发之间满足最小步数间隔。"""

        # 检查自上次触发以来是否经过足够步数
        return current_step - self.last_trigger_step >= self.min_step_interval

    def reset_time(self, current_step: int) -> None:
        """记录触发发生的时间步。"""

        self.last_trigger_step = current_step  # 更新上次触发时间步

    def reset_state(self) -> None:
        """重置触发器在回合之间的内部状态。"""

        # 重置所有状态变量
        self.last_trigger_step = -self.min_step_interval
        self.last_subgoal = None
        self.best_goal_distance = None
        self.last_progress_step = 0
        self.reset_turn_metrics()
        self.current_progress_epsilon = self.progress_epsilon_floor


class HighLevelPlanner:
    """
    高层规划器类
    基于事件触发机制生成导航子目标
    使用神经网络计算子目标，并管理事件触发机制决定何时计算新子目标
    """

    def __init__(
        self,
        belief_dim=90,
        device=None,
        save_directory=Path("ethsrl/models/high_level"),
        model_name="high_level_planner",
        load_model=False,
        load_directory=None,
        waypoint_lookahead: int = 3,
        *,
        trigger_config: Optional[TriggerConfig] = None,
        motion_config: Optional[MotionConfig] = None,
        rwr_temperature: float = 2.0,
        rwr_min_temperature: float = 0.4,
        rwr_temperature_decay: float = 0.999,
    ):
        """
        初始化高层规划器

        Args:
            belief_dim: 信念状态的维度（激光雷达数据点数）
            device: 计算设备（CPU/GPU）
            save_directory: 模型检查点保存目录
            model_name: 模型文件名
            load_model: 是否加载预训练模型
            load_directory: 模型加载目录（如果为None则使用save_directory）
            trigger_config: 高层触发阈值配置
            motion_config: 运动学与时间步配置
        """
        self.belief_dim = belief_dim
        # 设置计算设备，默认为GPU（如果可用）否则CPU
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 航点窗口相关参数
        self.motion_config = motion_config or MotionConfig()
        self.trigger_config = trigger_config or TriggerConfig()
        self.waypoint_lookahead = max(1, int(waypoint_lookahead))
        self.active_window_feature_dim = 6  # [dist, cos, sin, radius, inside, margin]
        self.per_window_feature_dim = 4  # [dist, cos, sin, radius]
        self.goal_feature_dim = 3 + self.active_window_feature_dim + self.per_window_feature_dim * self.waypoint_lookahead

        # 初始化子目标生成网络
        self.subgoal_network = SubgoalNetwork(
            belief_dim=belief_dim,
            goal_info_dim=self.goal_feature_dim,
        ).to(self.device)

        # 初始化事件触发器
        self.event_trigger = EventTrigger(
            trigger_config=self.trigger_config,
            motion_config=self.motion_config,
        )
        self.step_duration = self.motion_config.dt

        # 训练设置
        self.optimizer = torch.optim.Adam(self.subgoal_network.parameters(), lr=3e-4)  # Adam优化器
        self.writer = SummaryWriter(comment=model_name)  # TensorBoard记录器，用于可视化训练过程
        self.iter_count = 0  # 迭代计数器，记录训练步数
        self.model_name = model_name  # 模型名称
        self.save_directory = save_directory  # 保存目录

        # RWR温度调度参数，用于软最大化权重
        self.rwr_temperature = float(max(rwr_temperature, 1e-6))
        self.rwr_min_temperature = float(max(rwr_min_temperature, 1e-6))
        self.rwr_temperature_decay = float(max(min(rwr_temperature_decay, 1.0), 0.0))

        # 当前状态跟踪
        self.current_subgoal = None  # 当前子目标（相对坐标：距离，角度）
        self.last_goal_distance = float('inf')  # 上次目标距离，初始化为无穷大
        self.last_goal_direction = 0.0  # 上次目标方向角度
        self.prev_action = [0.0, 0.0]  # 上一动作（归一化坐标）
        self.current_subgoal_world: Optional[np.ndarray] = None  # 当前子目标的世界坐标[x, y]
        self.subgoal_smoothing_alpha = float(np.clip(self.trigger_config.subgoal_smoothing_alpha, 0.0, 0.999))
        self._smoothed_distance: Optional[float] = None
        self._smoothed_angle: Optional[float] = None

        # 如果请求则加载预训练模型
        if load_model:
            load_dir = load_directory if load_directory else save_directory
            self.load_model(filename=model_name, directory=load_dir)

    def get_relative_subgoal(self, robot_pose: Optional[Sequence[float]]) -> Tuple[Optional[float], Optional[float]]:
        """计算当前子目标相对于机器人姿态的距离和角度。"""

        if robot_pose is None or self.current_subgoal_world is None:  # 检查输入有效性
            return None, None

        robot_xy = np.asarray(robot_pose[:2], dtype=np.float32)  # 提取机器人位置[x, y]
        subgoal_world = np.asarray(self.current_subgoal_world, dtype=np.float32)  # 子目标世界坐标
        delta = subgoal_world - robot_xy  # 计算相对位移向量
        distance = float(np.linalg.norm(delta))  # 计算欧几里得距离

        if distance <= 1e-6:  # 如果距离非常小（接近到达）
            return 0.0, 0.0  # 返回零距离和零角度

        heading = float(robot_pose[2])  # 机器人朝向角度
        # 计算子目标相对于机器人朝向的角度
        angle = math.atan2(float(delta[1]), float(delta[0])) - heading
        # 规范化角度到[-π, π]范围
        angle = math.atan2(math.sin(angle), math.cos(angle))

        return distance, angle  # 返回相对距离和角度

    def process_laser_scan(self, laser_scan):
        """
        处理原始激光雷达数据为信念状态表示

        Args:
            laser_scan: 原始激光雷达读数

        Returns:
            处理后的激光雷达张量
        """
        laser_scan = np.array(laser_scan)  # 转换为numpy数组

        # 处理无穷大值（表示没有障碍物检测）
        inf_mask = np.isinf(laser_scan)
        laser_scan[inf_mask] = 7.0  # 用最大范围值替换（假设激光雷达最大范围7米）

        # 归一化到[0, 1]范围
        laser_scan = laser_scan / 7.0

        return torch.FloatTensor(laser_scan).to(self.device)  # 转换为PyTorch张量并移动到设备

    def process_goal_info(self, distance, cos_angle, sin_angle, waypoint_features=None):
        """将目标与航点特征组合成网络输入张量。"""

        norm_distance = min(float(distance) / 10.0, 1.0)
        base_features: List[float] = [norm_distance, float(cos_angle), float(sin_angle)]

        tail_len = max(0, self.goal_feature_dim - 3)
        if tail_len > 0:
            if waypoint_features is None:
                waypoint_features = [0.0] * tail_len
            else:
                waypoint_features = list(waypoint_features)[:tail_len]
                if len(waypoint_features) < tail_len:
                    waypoint_features.extend([0.0] * (tail_len - len(waypoint_features)))
            base_features.extend(float(value) for value in waypoint_features)

        return torch.FloatTensor(base_features).to(self.device)

    def process_action_info(self, prev_action):
        """
        处理历史动作为张量

        Args:
            prev_action: 上一步的归一化动作 [-1, 1] 区间

        Returns:
            处理后的动作信息张量
        """
        prev = np.asarray(prev_action, dtype=np.float32).flatten()
        if prev.size < 2:
            raise ValueError("prev_action must contain at least two elements (a_lin, a_ang).")

        lin_vel = float(np.clip(prev[0], -1.0, 1.0))
        ang_vel = float(np.clip(prev[1], -1.0, 1.0))

        action_info = torch.FloatTensor([lin_vel, ang_vel]).to(self.device)

        return action_info

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    def _smooth_subgoal(self, distance: float, angle: float) -> Tuple[float, float]:
        """Apply EMA smoothing to consecutive subgoals to damp oscillations."""

        alpha = self.subgoal_smoothing_alpha
        if not (0.0 < alpha < 1.0):
            self._smoothed_distance = distance
            self._smoothed_angle = angle
            return distance, angle

        if self._smoothed_distance is None or self._smoothed_angle is None:
            self._smoothed_distance = distance
            self._smoothed_angle = angle
            return distance, angle

        smoothed_distance = alpha * self._smoothed_distance + (1.0 - alpha) * distance

        prev_vec = np.array(
            [math.cos(self._smoothed_angle), math.sin(self._smoothed_angle)],
            dtype=np.float32,
        )
        new_vec = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)
        blended = alpha * prev_vec + (1.0 - alpha) * new_vec
        if float(np.linalg.norm(blended)) <= 1e-6:
            smoothed_angle = angle
        else:
            smoothed_angle = math.atan2(float(blended[1]), float(blended[0]))

        smoothed_angle = math.atan2(math.sin(smoothed_angle), math.cos(smoothed_angle))

        self._smoothed_distance = smoothed_distance
        self._smoothed_angle = smoothed_angle
        return smoothed_distance, smoothed_angle

    def reset_smoothing(self) -> None:
        """Clear the smoothing state when episodes reset or plans restart."""

        self._smoothed_distance = None
        self._smoothed_angle = None

    def _world_to_relative(self, robot_pose, waypoint) -> Tuple[float, float]:
        if robot_pose is None:
            return 0.0, 0.0

        waypoint_vec = np.asarray(waypoint, dtype=np.float32)
        dx = float(waypoint_vec[0] - robot_pose[0])
        dy = float(waypoint_vec[1] - robot_pose[1])
        distance = math.hypot(dx, dy)
        angle = self._wrap_angle(math.atan2(dy, dx) - robot_pose[2])
        return distance, angle

    def _relative_to_world(self, robot_pose, distance: float, angle: float) -> np.ndarray:
        if robot_pose is None:
            return np.zeros(2, dtype=np.float32)
        world_x = robot_pose[0] + distance * math.cos(robot_pose[2] + angle)
        world_y = robot_pose[1] + distance * math.sin(robot_pose[2] + angle)
        return np.array([world_x, world_y], dtype=np.float32)

    def build_waypoint_features(self, waypoints, robot_pose) -> List[float]:
        active_features = [0.0] * self.active_window_feature_dim
        sequence_features: List[float] = []

        if robot_pose is None:
            sequence_features.extend([0.0] * self.per_window_feature_dim * self.waypoint_lookahead)
            return active_features + sequence_features

        for idx in range(self.waypoint_lookahead):
            window: Optional[WaypointWindow] = None
            if waypoints is not None and idx < len(waypoints):
                entry = waypoints[idx]
                if isinstance(entry, tuple) and len(entry) == 2:
                    window = entry[1]
                elif isinstance(entry, WaypointWindow):
                    window = entry
                else:
                    waypoint_vec = np.asarray(entry, dtype=np.float32)
                    window = WaypointWindow(center=waypoint_vec, radius=0.0)

            if window is not None:
                centre = np.asarray(window.center, dtype=np.float32)
                distance, angle = self._world_to_relative(robot_pose, centre)
                norm_distance = min(distance / 10.0, 1.0)
                cos_rel = math.cos(angle)
                sin_rel = math.sin(angle)
                norm_radius = min(float(window.radius) / 5.0, 1.0)
                sequence_features.extend([norm_distance, cos_rel, sin_rel, norm_radius])
                margin = (distance - float(window.radius)) / max(float(window.radius), 1e-3)
                margin = float(np.clip(margin, -1.0, 1.0))
                inside_flag = 1.0 if distance <= float(window.radius) else 0.0
                if idx == 0:
                    active_features = [norm_distance, cos_rel, sin_rel, norm_radius, inside_flag, margin]
            else:
                sequence_features.extend([0.0, 0.0, 0.0, 0.0])

        total_expected = self.per_window_feature_dim * self.waypoint_lookahead
        if len(sequence_features) < total_expected:
            sequence_features.extend([0.0] * (total_expected - len(sequence_features)))

        return active_features + sequence_features

    def build_state_vector(self, laser_scan, distance, cos_angle, sin_angle, prev_action, waypoints=None, robot_pose=None):
        """构造高层规划器训练所需的状态向量"""

        with torch.no_grad():
            laser_tensor = self.process_laser_scan(laser_scan)
            waypoint_features = self.build_waypoint_features(waypoints, robot_pose)
            goal_tensor = self.process_goal_info(distance, cos_angle, sin_angle, waypoint_features)
            action_tensor = self.process_action_info(prev_action)
            state_tensor = torch.cat((laser_tensor, goal_tensor, action_tensor))

        return state_tensor.cpu().numpy()

    def check_triggers(
        self,
        laser_scan,
        robot_pose,
        goal_info,
        prev_action=None,
        min_obstacle_dist=None,
        current_step: int = 0,
        window_metrics: Optional[dict] = None,
        *,
        throttle_ready: bool = True,
    ):
        """
        检查是否有任何事件触发器被激活

        Args:
            laser_scan: 当前激光雷达读数
            robot_pose: 当前机器人位姿 [x, y, theta]
            goal_info: 全局目标信息 [distance, cos, sin]
            prev_action: 上一步的动作（归一化）[a_lin, a_ang]
            min_obstacle_dist: 到最近障碍物的距离（如果为None，则从laser_scan计算）
            current_step: 当前时间步
            throttle_ready: 是否已满足最小触发间隔（由集成层统筹）

        Returns:
            布尔值，指示是否应生成新子目标
        """
        # 提取目标距离信息
        goal_distance = float(goal_info[0]) if goal_info else float('inf')
        laser_scan = np.asarray(laser_scan, dtype=np.float32)  # 确保激光数据为numpy数组

        window_radius: Optional[float] = None
        inside_window = False
        steps_inside = 0
        limit_exceeded = False
        if window_metrics:
            radius_val = window_metrics.get("radius")
            if radius_val is not None:
                window_radius = float(radius_val)
            inside_window = bool(window_metrics.get("inside", False))
            steps_inside = int(window_metrics.get("steps_inside", 0))
            limit_exceeded = bool(window_metrics.get("limit_exceeded", False))

        self.event_trigger.set_progress_context(window_radius)

        # 如果未提供则计算最小障碍物距离
        if min_obstacle_dist is None:
            valid_scans = laser_scan[np.isfinite(laser_scan)]  # 过滤有效扫描值（非NaN/无穷大）
            min_obstacle_dist = np.min(valid_scans) if valid_scans.size > 0 else float('inf')

        # 更新进度信息（用于停滞检测）
        self.event_trigger.update_progress(goal_distance, current_step)

        # 计算当前子目标的空间信息
        dist_to_subgoal, subgoal_angle = self.get_relative_subgoal(robot_pose)
        if dist_to_subgoal is not None:  # 如果有有效子目标
            # 更新事件触发器中的子目标位置
            self.event_trigger.last_subgoal = np.asarray(self.current_subgoal_world, dtype=np.float32).tolist()
        else:
            self.event_trigger.last_subgoal = None

        # 核心触发条件检查
        safe_trigger = self.event_trigger.safe_distance_trigger(min_obstacle_dist)  # 安全距离触发
        progress_trigger = self.event_trigger.intelligent_progress_trigger(  # 智能进度触发
            dist_to_subgoal=dist_to_subgoal,
            current_step=current_step,
            subgoal_angle=subgoal_angle,
            laser_scan=laser_scan,
            min_obstacle_dist=min_obstacle_dist,
        )

        # 最终触发决策：满足节流条件并触发安全/进度事件，窗口强制触发单独处理
        trigger_new_subgoal = (safe_trigger or progress_trigger) and throttle_ready
        if limit_exceeded:
            trigger_new_subgoal = True

        # 如果触发，重置时间计数器
        if window_metrics:
            if window_metrics.get("limit_exceeded", False):
                trigger_new_subgoal = True
            if window_metrics.get("entered", False):
                # 新窗口被进入，刷新进度基准以避免重复触发
                self.event_trigger.reset_progress(goal_distance, current_step)

        hold_threshold = self.event_trigger.window_inside_hold
        if (
            trigger_new_subgoal
            and not limit_exceeded
            and inside_window
            and steps_inside < hold_threshold
        ):
            trigger_new_subgoal = False

        if trigger_new_subgoal:
            self.event_trigger.reset_time(current_step)

        return trigger_new_subgoal

    def generate_subgoal(
        self,
        laser_scan,
        goal_distance,
        goal_cos,
        goal_sin,
        prev_action=None,
        robot_pose=None,
        current_step: Optional[int] = None,
        waypoints=None,
        window_metrics: Optional[dict] = None,
    ):
        """
        基于当前状态生成新子目标

        Args:
            laser_scan: 处理后的激光雷达数据
            goal_distance: 到全局目标的距离
            goal_cos: 到全局目标角度的余弦值
            goal_sin: 到全局目标角度的正弦值
            prev_action: 上一步的动作（归一化）[a_lin, a_ang]
            robot_pose: 机器人位姿 [x, y, theta]（用于计算世界坐标）
            current_step: 当前全局时间步（用于进度重置）
            waypoints: 当前全局规划提供的候选航点
            window_metrics: 当前目标窗口的状态统计信息

        Returns:
            包含(子目标距离, 子目标角度)的元组
        """
        # 处理输入数据
        laser_tensor = self.process_laser_scan(laser_scan)  # 激光数据张量化
        waypoint_features = self.build_waypoint_features(waypoints, robot_pose)
        goal_tensor = self.process_goal_info(goal_distance, goal_cos, goal_sin, waypoint_features)

        # 如果未提供动作，则使用存储的上一步动作
        if prev_action is None:
            prev_action = self.prev_action

        # 处理动作信息
        action_tensor = self.process_action_info(prev_action)

        # 使用网络生成子目标（不计算梯度）
        with torch.no_grad():  # 推理模式，不计算梯度
            (
                distance_scale_tensor,
                angle_offset_tensor,
                _,
                _,
            ) = self.subgoal_network(
                laser_tensor.unsqueeze(0),  # 增加批次维度：[1, belief_dim] -> [1, 1, belief_dim]
                goal_tensor.unsqueeze(0),  # 增加批次维度：[3] -> [1, 3]
                action_tensor.unsqueeze(0),  # 增加批次维度：[2] -> [1, 2]
            )

        distance_scale = float(distance_scale_tensor.cpu().numpy().item())
        angle_offset = float(angle_offset_tensor.cpu().numpy().item())

        # 规范化输出范围
        distance_scale = float(np.clip(distance_scale, 0.0, 1.0))
        angle_offset = float(self._wrap_angle(angle_offset))

        candidate_info = []
        active_window_index: Optional[int] = None
        active_window_radius: Optional[float] = None
        if robot_pose is not None and waypoints:
            for entry in waypoints:
                if isinstance(entry, tuple) and len(entry) == 2:
                    idx = int(entry[0]) if entry[0] is not None else None
                    window_obj = entry[1]
                elif isinstance(entry, WaypointWindow):
                    idx = None
                    window_obj = entry
                else:
                    idx = None
                    window_vec = np.asarray(entry, dtype=np.float32)
                    window_obj = WaypointWindow(center=window_vec, radius=0.0)

                if not isinstance(window_obj, WaypointWindow):
                    continue

                centre = np.asarray(window_obj.center, dtype=np.float32)
                rel_dist, rel_angle = self._world_to_relative(robot_pose, centre)
                candidate_info.append(
                    {
                        "index": idx,
                        "window": window_obj,
                        "position": centre,
                        "distance": rel_dist,
                        "angle": rel_angle,
                        "radius": float(window_obj.radius),
                    }
                )

            if candidate_info:
                active_window_index = candidate_info[0]["index"]
                active_window_radius = candidate_info[0]["radius"]

        selected_index = None

        anchor_info = candidate_info[0] if candidate_info else None

        if anchor_info is not None and anchor_info["index"] is not None:
            selected_index = int(anchor_info["index"])

        if active_window_index is None and window_metrics:
            active_window_index = window_metrics.get("index")
            active_window_radius = window_metrics.get("radius")

        metadata = {
            "selected_waypoint": selected_index,
            "active_window_index": active_window_index,
            "active_window_radius": active_window_radius,
            "window_metrics": dict(window_metrics) if window_metrics else {},
            "candidate_info": candidate_info,
            "raw_distance_scale": distance_scale,
            "raw_angle_offset": angle_offset,
        }

        if metadata["selected_waypoint"] is None and active_window_index is not None:
            metadata["selected_waypoint"] = active_window_index

        return distance_scale, angle_offset, metadata

    def commit_subgoal(
        self,
        *,
        distance: float,
        angle: float,
        world_target: Optional[np.ndarray],
        goal_distance: float,
        goal_direction: float,
        prev_action: Optional[Sequence[float]],
        current_step: int,
        robot_pose: Optional[Sequence[float]] = None,
    ) -> None:
        """记录最新子目标并刷新触发器的内部状态。"""

        wrapped_angle = float(self._wrap_angle(angle))
        smoothed_distance, smoothed_angle = self._smooth_subgoal(float(distance), wrapped_angle)
        self.current_subgoal = (smoothed_distance, smoothed_angle)
        self.last_goal_distance = float(goal_distance)
        self.last_goal_direction = float(self._wrap_angle(goal_direction))

        if prev_action is not None:
            prev_arr = np.asarray(prev_action, dtype=np.float32).flatten()
            if prev_arr.size >= 2:
                self.prev_action = [
                    float(np.clip(prev_arr[0], -1.0, 1.0)),
                    float(np.clip(prev_arr[1], -1.0, 1.0)),
                ]

        if robot_pose is not None:
            target_vec = self._relative_to_world(robot_pose, smoothed_distance, smoothed_angle)
            self.current_subgoal_world = target_vec
            self.event_trigger.last_subgoal = target_vec.tolist()
        elif world_target is not None:
            target_vec = np.asarray(world_target, dtype=np.float32)
            self.current_subgoal_world = target_vec
            self.event_trigger.last_subgoal = target_vec.tolist()
        else:
            self.current_subgoal_world = None
            self.event_trigger.last_subgoal = None

        progress_step = int(current_step)
        self.event_trigger.reset_progress(goal_distance, progress_step)

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
        scan[np.isinf(scan)] = 7.0  # 替换无穷大为最大范围值

        # 基于以下因素的简单复杂性度量：
        # 1. 扫描读数的方差（方差越大越复杂）
        # 2. 平均距离（障碍物越近越复杂）
        variance = np.var(scan) / 10.0  # 归一化方差（假设最大方差10）
        avg_distance = np.mean(scan) / 7.0  # 归一化均值（最大范围7米）

        # 计算复杂性得分（平均距离的倒数，用方差加权）
        # 障碍物越近、分布越不均匀，复杂性越高
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
        safe_subgoals = []  # 存储安全子目标

        # 将扫描转换为笛卡尔坐标以便处理（生成对应的角度数组）
        angles = np.linspace(-np.pi, np.pi, len(laser_scan))  # 生成角度数组，覆盖360度

        for subgoal in candidate_subgoals:  # 遍历所有候选子目标
            distance, angle = subgoal  # 解包子目标距离和角度

            # 检查到子目标的路径是否清晰
            # 将角度转换为激光雷达索引
            index = int((angle + np.pi) / (2 * np.pi) * len(laser_scan))  # 计算对应索引
            index = max(0, min(index, len(laser_scan) - 1))  # 确保有效索引范围

            # 检查子目标是否在安全距离内
            # 条件：子目标距离 < 激光读数 - 安全距离
            if distance < laser_scan[index] - self.event_trigger.safety_trigger_distance:
                safe_subgoals.append(subgoal)  # 路径清晰，子目标安全

        # 如果所有子目标都不安全，选择最安全的一个（fallback策略）
        if not safe_subgoals and candidate_subgoals:
            safest_distance = 0  # 最远距离（初始化为0）
            safest_subgoal = None  # 最安全子目标

            for subgoal in candidate_subgoals:  # 重新遍历所有候选
                distance, angle = subgoal
                index = int((angle + np.pi) / (2 * np.pi) * len(laser_scan))
                index = max(0, min(index, len(laser_scan) - 1))

                if laser_scan[index] > safest_distance:  # 找到最远障碍物的方向
                    safest_distance = laser_scan[index]  # 更新最远距离
                    safest_subgoal = subgoal  # 更新最安全子目标

            if safest_subgoal:
                safe_subgoals.append(safest_subgoal)  # 添加最安全的子目标

        return safe_subgoals  # 返回安全子目标列表

    def update_planner(self, states, actions, rewards, subgoal_dones, next_states, batch_size=64):
        """
        使用收集的经验更新规划器的神经网络

        Args:
            states: 环境状态批次
            actions: 采取的动作批次（真实的子目标）
            rewards: 获得的奖励批次
            next_states: 结果状态批次
            subgoal_dones: 子目标终止标志批次
            batch_size: 训练批次大小

        Returns:
            训练指标字典
        """
        # 转换为PyTorch张量并移动到设备
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1)  # 增加维度便于广播
        subgoal_dones = torch.FloatTensor(subgoal_dones).to(self.device).unsqueeze(1)
        next_states = torch.FloatTensor(next_states).to(self.device)

        goal_feature_dim = self.goal_feature_dim
        action_feature_dim = 2
        split_index = states.shape[1] - (goal_feature_dim + action_feature_dim)
        laser_scans = states[:, :split_index]
        goal_info = states[:, split_index : split_index + goal_feature_dim]
        prev_action = states[:, -action_feature_dim:]

        # 生成子目标（网络预测）
        (
            distance_scales,
            angle_offsets,
            distance_logits,
            angle_logits,
        ) = self.subgoal_network(laser_scans, goal_info, prev_action)

        # 合并调整量为完整动作 [distance_scale, angle_offset]
        subgoals = torch.stack((distance_scales, angle_offsets), dim=1)

        # 将监督信号反变换到未约束空间
        eps = 1e-3
        target_distance = actions[:, 0:1]
        target_angle = actions[:, 1:2]

        clamped_distance = target_distance.clamp(eps, 1.0 - eps)
        distance_targets = _logit(clamped_distance)

        angle_scale = np.pi / 4
        normalized_angle = (target_angle / angle_scale).clamp(-1.0 + eps, 1.0 - eps)
        angle_targets = _artanh(normalized_angle)

        distance_logits = distance_logits.unsqueeze(1)
        angle_logits = angle_logits.unsqueeze(1)

        # 基于软最大化的奖励加权，温度参数控制集中度
        with torch.no_grad():
            temperature = max(self.rwr_temperature, 1e-6)
            scaled = (rewards - rewards.max()) / temperature
            weights = torch.softmax(scaled.squeeze(1), dim=0).unsqueeze(1)

        # 计算每个样本的Huber损失
        distance_loss = F.smooth_l1_loss(distance_logits, distance_targets, reduction='none')
        angle_loss = F.smooth_l1_loss(angle_logits, angle_targets, reduction='none')
        per_sample_loss = distance_loss + angle_loss
        # 加权损失：突出高回报样本的重要性
        loss = (per_sample_loss * weights).sum() / weights.sum()

        # 优化步骤
        self.optimizer.zero_grad()  # 清零梯度
        loss.backward()  # 反向传播计算梯度
        self.optimizer.step()  # 更新网络参数

        # 更新训练计数器
        self.iter_count += 1

        # 按调度退火温度，避免学习后期过于平滑
        if self.rwr_temperature > self.rwr_min_temperature:
            updated_temp = self.rwr_temperature * self.rwr_temperature_decay
            self.rwr_temperature = max(updated_temp, self.rwr_min_temperature)

        # 记录指标到TensorBoard
        self.writer.add_scalar('planner/loss', loss.item(), self.iter_count)  # 损失值
        self.writer.add_scalar('planner/reward_weight_mean', weights.mean().item(), self.iter_count)  # 平均权重
        self.writer.add_scalar('planner/rwr_temperature', self.rwr_temperature, self.iter_count)

        # 返回训练指标
        weight_entropy = float(-(weights * (weights.clamp_min(1e-8).log())).sum().item())

        return {
            'loss': loss.item(),  # 损失值
            'avg_distance_scale': distance_scales.mean().item(),  # 平均距离缩放量
            'avg_angle_offset': angle_offsets.mean().item(),  # 平均角度偏移量
            'weight_mean': weights.mean().item(),  # 平均权重
            'weight_entropy': weight_entropy,
            'temperature': self.rwr_temperature,
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

        # 保存模型状态字典
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
            # 加载模型状态字典
            self.subgoal_network.load_state_dict(torch.load(f"{directory}/{filename}.pth"))
            print(f"模型已从 {directory}/{filename}.pth 加载")
        except FileNotFoundError as e:  # 处理文件不存在的情况
            print(f"加载模型时出错: {e}")
