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

from config import HighLevelCostConfig, PlannerConfig, SafetyCriticConfig, TriggerConfig


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

        cnn_output_dim = self._get_cnn_output_dim(belief_dim)  # 计算CNN输出维度
        combined_dim = cnn_output_dim + 64  # 融合激光与目标特征后的维度

        # 序列建模层：在触发之间捕捉子目标演化
        self.rnn = nn.GRU(
            input_size=combined_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        # 全连接层
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        # 输出层：距离调整系数与角度偏移量
        self.distance_head = nn.Linear(hidden_dim // 2, 1)  # 距离调整输出头
        self.angle_head = nn.Linear(hidden_dim // 2, 1)  # 角度偏移输出头

    def _get_cnn_output_dim(self, belief_dim):
        """计算CNN层展平后的输出维度"""
        # 创建虚拟输入来计算输出维度
        x = torch.zeros(1, 1, belief_dim)  # 批次大小1，通道数1，输入维度belief_dim
        x = self.cnn1(x)  # 通过第一层CNN
        x = self.cnn2(x)  # 通过第二层CNN
        x = self.cnn3(x)  # 通过第三层CNN
        return x.numel()  # 返回元素总数（展平后的维度）

    def forward(self, belief_state, goal_info, hidden_state: Optional[torch.Tensor] = None):
        """
        子目标网络的前向传播

        Args:
            belief_state: 包含激光雷达数据的张量，形状[batch_size, belief_dim]
            goal_info: 包含全局目标与航点特征的张量，形状[batch_size, goal_info_dim]
            hidden_state: 可选的RNN隐状态，形状[num_layers, batch_size, hidden_dim]

        Returns:
            包含(距离调整系数, 角度偏移量, 下一个隐状态)的元组
        """
        # 处理激光雷达数据
        laser = belief_state.unsqueeze(1)  # 增加通道维度：[batch_size, 1, belief_dim]
        x = F.relu(self.cnn1(laser))  # 第一层CNN + ReLU激活
        x = F.relu(self.cnn2(x))  # 第二层CNN + ReLU激活
        x = F.relu(self.cnn3(x))  # 第三层CNN + ReLU激活
        x = x.flatten(start_dim=1)  # 展平特征图：[batch_size, cnn_output_dim]

        # 处理目标信息
        g = F.relu(self.goal_embed(goal_info))  # 目标嵌入 + ReLU激活：[batch_size, 32]

        # 合并特征 - 更新为仅包含激光与目标信息
        combined = torch.cat((x, g), dim=1)  # 拼接所有特征：[batch_size, combined_dim]

        # 通过GRU引入时序记忆
        x_seq = combined.unsqueeze(1)  # [batch_size, 1, combined_dim]
        rnn_out, next_hidden = self.rnn(x_seq, hidden_state)
        h_t = rnn_out[:, -1, :]  # 取序列末端输出作为表示

        # 全连接层处理
        x = F.relu(self.fc1(h_t))  # 第一层全连接 + ReLU
        x = F.relu(self.fc2(x))  # 第二层全连接 + ReLU：[batch_size, hidden_dim//2]

        # 生成子目标调整量
        distance_adjust = torch.tanh(self.distance_head(x))  # 距离调整系数∈[-1, 1]
        angle_offset = torch.tanh(self.angle_head(x)) * (np.pi / 4)  # 角度偏移量∈[-π/4, π/4]

        # squeeze掉最后一维，保持批量维度，便于后续堆叠或索引
        return distance_adjust.squeeze(-1), angle_offset.squeeze(-1), next_hidden


class SafetyCritic(nn.Module):
    """Safety critic that predicts short-horizon obstacle risk for candidate goals."""

    def __init__(self, belief_dim=90, goal_info_dim=3, hidden_dim=192):
        super().__init__()

        # CNN层用于处理激光雷达数据
        self.cnn1 = nn.Conv1d(1, 8, kernel_size=5, stride=2)  # 第一层CNN：输入1通道，输出8通道
        self.cnn2 = nn.Conv1d(8, 16, kernel_size=3, stride=2)  # 第二层CNN：输入8通道，输出16通道
        self.cnn3 = nn.Conv1d(16, 8, kernel_size=3, stride=1)  # 第三层CNN：输入16通道，输出8通道

        # 目标信息嵌入层
        self.goal_embed = nn.Linear(goal_info_dim, 64)  # 处理目标信息，输出64维
        # 子目标几何信息嵌入层
        self.subgoal_embed = nn.Linear(3, 16)  # 处理子目标几何信息，输出16维

        # 计算CNN输出维度
        cnn_output_dim = self._get_cnn_output_dim(belief_dim)
        # 第一层全连接：输入=CNN输出+目标嵌入+子目标嵌入，输出=隐藏层维度
        self.fc1 = nn.Linear(cnn_output_dim + 64 + 16, hidden_dim)
        # 第二层全连接：输入=隐藏层维度，输出=隐藏层维度的一半
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        # 输出层：风险预测值
        self.output_head = nn.Linear(hidden_dim // 2, 1)  # 输出风险值

    def _get_cnn_output_dim(self, belief_dim: int) -> int:
        """计算CNN层展平后的输出维度"""
        # 创建虚拟输入来计算输出维度
        x = torch.zeros(1, 1, belief_dim)  # 批次大小1，通道数1，输入维度belief_dim
        x = self.cnn1(x)  # 通过第一层CNN
        x = self.cnn2(x)  # 通过第二层CNN
        x = self.cnn3(x)  # 通过第三层CNN
        return x.numel()  # 返回元素总数（展平后的维度）

    def forward(self, belief_state, goal_info, subgoal_geom):
        """
        Safety critic的前向传播

        Args:
            belief_state: 信念状态（激光雷达数据）
            goal_info: 目标信息
            subgoal_geom: 子目标几何信息

        Returns:
            风险预测值
        """
        # 处理激光雷达数据
        laser = belief_state.unsqueeze(1)  # 增加通道维度
        x = F.relu(self.cnn1(laser))  # 第一层CNN + ReLU激活
        x = F.relu(self.cnn2(x))  # 第二层CNN + ReLU激活
        x = F.relu(self.cnn3(x))  # 第三层CNN + ReLU激活
        x = x.flatten(start_dim=1)  # 展平特征图

        # 处理目标信息
        g = F.relu(self.goal_embed(goal_info))  # 目标嵌入 + ReLU激活
        # 处理子目标几何信息
        geom = F.relu(self.subgoal_embed(subgoal_geom))  # 子目标几何嵌入 + ReLU激活

        # 合并所有特征
        combined = torch.cat((x, g, geom), dim=1)  # 拼接所有特征
        # 全连接层处理
        x = F.relu(self.fc1(combined))  # 第一层全连接 + ReLU
        x = F.relu(self.fc2(x))  # 第二层全连接 + ReLU
        # 输出风险值（使用softplus确保非负）
        risk = F.softplus(self.output_head(x))
        return risk.squeeze(-1)  # 去掉最后一维


class CostCritic(nn.Module):
    """Long-horizon cost critic predicting cumulative safety cost for a subgoal."""

    def __init__(self, belief_dim=90, goal_info_dim=3, geom_dim=3, hidden_dim=192):
        super().__init__()

        self.cnn1 = nn.Conv1d(1, 8, kernel_size=5, stride=2)
        self.cnn2 = nn.Conv1d(8, 16, kernel_size=3, stride=2)
        self.cnn3 = nn.Conv1d(16, 8, kernel_size=3, stride=1)

        self.goal_embed = nn.Linear(goal_info_dim, 64)
        self.subgoal_embed = nn.Linear(geom_dim, 16)

        cnn_output_dim = self._get_cnn_output_dim(belief_dim)
        self.fc1 = nn.Linear(cnn_output_dim + 64 + 16, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.output_head = nn.Linear(hidden_dim // 2, 1)

    def _get_cnn_output_dim(self, belief_dim: int) -> int:
        x = torch.zeros(1, 1, belief_dim)
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        return x.numel()

    def forward(self, belief_state, goal_info, subgoal_geom):
        laser = belief_state.unsqueeze(1)
        x = F.relu(self.cnn1(laser))
        x = F.relu(self.cnn2(x))
        x = F.relu(self.cnn3(x))
        x = x.flatten(start_dim=1)

        g = F.relu(self.goal_embed(goal_info))
        geom = F.relu(self.subgoal_embed(subgoal_geom))

        combined = torch.cat((x, g, geom), dim=1)
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))
        cost = F.softplus(self.output_head(x))
        return cost.squeeze(-1)


class EventTrigger:
    """事件触发器类，基于多种条件决定何时生成新子目标"""

    def __init__(
        self,
        *,
        config: "TriggerConfig",
        step_duration: float,
        min_interval: Optional[float] = None,
        subgoal_reach_threshold: Optional[float] = None,
        progress_epsilon: Optional[float] = None,
    ) -> None:
        """初始化事件触发器并与集中配置对齐。

        修复点：
        - 若显式提供了 min_interval（秒），则优先采用它，并用 step_duration 反推 min_step_interval（步）。
        - 若未提供显式时间但 config.min_interval > 0，则同样按时间→步数映射。
        - 仅当两者都未提供有效时间时，才回退到 config.min_step_interval（步）。
        - 不再无条件用 config.min_step_interval 作为下限去覆盖基于时间的设置。
        """

        self._config = config  # 触发器配置
        self.safety_trigger_distance = config.safety_trigger_distance  # 安全触发距离
        self.subgoal_reach_threshold = (
            subgoal_reach_threshold
            if subgoal_reach_threshold is not None
            else config.subgoal_reach_threshold
        )  # 子目标到达阈值
        self.stagnation_steps = max(1, int(config.stagnation_steps))  # 停滞步数阈值
        self.progress_epsilon = (
            progress_epsilon if progress_epsilon is not None else config.progress_epsilon
        )  # 进度容差
        self.step_duration = float(step_duration)  # 步长持续时间

        # ---------- 1) 先确定"时间下限"的来源（显式 > 配置时间 > 配置步数） ----------
        if min_interval is not None and min_interval > 0:
            # 显式时间优先
            self.min_interval = float(min_interval)
            _time_source = "explicit_time"
        elif getattr(config, "min_interval", 0) and config.min_interval > 0:
            # 配置里的时间次之
            self.min_interval = float(config.min_interval)
            _time_source = "config_time"
        else:
            # 都没有时间，就用"步数 × dt"作为时间的派生显示值（仅用于日志/可读）
            steps_cfg = max(1, int(getattr(config, "min_step_interval", 1)))
            self.min_interval = float(steps_cfg * self.step_duration) if self.step_duration > 0 else 0.0
            _time_source = "config_steps"

        # ---------- 2) 由时间反推"步数下限"；仅在纯步数配置时直接用配置步数 ----------
        if _time_source in ("explicit_time", "config_time"):
            if self.step_duration > 0:
                steps_from_time = int(math.ceil(self.min_interval / self.step_duration))
                self.min_step_interval = max(1, steps_from_time)
            else:
                # 极端兜底：没有有效 dt 时，退回到配置步数或 1
                self.min_step_interval = max(1, int(getattr(config, "min_step_interval", 1)))
        else:
            # 使用配置步数作为唯一来源
            self.min_step_interval = max(1, int(getattr(config, "min_step_interval", 1)))
            # 同步一份与之对应的时间，便于日志可读（不影响逻辑）
            self.min_interval = float(self.min_step_interval * self.step_duration) if self.step_duration > 0 else 0.0

        # ---------- 3) 状态变量初始化 ----------
        # 置为"负的步数阈值"，保证初始化后允许立即触发一次
        self.last_trigger_step = -self.min_step_interval  # 上次触发步数
        self.last_subgoal: Optional[np.ndarray] = None  # 上次子目标
        self.best_goal_distance: Optional[float] = None  # 最佳目标距离
        self.last_progress_step = 0  # 上次进展步数

    def safe_distance_trigger(self, min_obstacle_dist: float) -> bool:
        """若最近障碍物距离低于安全阈值则触发。"""

        if np.isnan(min_obstacle_dist):  # 检查距离是否为NaN
            return False
        return min_obstacle_dist <= self.safety_trigger_distance  # 距离小于等于安全阈值则触发

    def subgoal_reached(self, dist_to_subgoal: Optional[float]) -> bool:
        """是否已经进入子目标半径内。"""

        return dist_to_subgoal is not None and dist_to_subgoal <= self.subgoal_reach_threshold

    def global_progress_stagnant(self, goal_distance: float, current_step: int) -> bool:
        """检测全局进展停滞：长时间未取得显著距离改进。"""

        if not np.isfinite(goal_distance):
            return False

        epsilon = max(self.progress_epsilon, self._config.progress_epsilon_ratio * goal_distance)
        if self.best_goal_distance is None:
            self.best_goal_distance = goal_distance
            self.last_progress_step = current_step
            return False

        if goal_distance + epsilon < self.best_goal_distance:
            self.best_goal_distance = goal_distance
            self.last_progress_step = current_step
            return False

        return (current_step - self.last_progress_step) >= self.stagnation_steps

    def reset_progress(self, goal_distance: float, current_step: int) -> None:
        """在生成新子目标时重置进度基准。"""

        if not np.isfinite(goal_distance):  # 检查目标距离是否有限（非无穷大/NaN）
            self.best_goal_distance = None  # 重置最佳距离
            self.last_progress_step = current_step  # 更新进度时间步
            return

        # 设置新的最佳距离基准
        self.best_goal_distance = goal_distance
        self.last_progress_step = current_step

    def update_progress(self, goal_distance: float, current_step: int) -> None:
        """根据当前全局目标距离更新最优进度。"""

        if not np.isfinite(goal_distance):  # 检查距离是否有效
            return

        if self.best_goal_distance is None:  # 如果还没有最佳距离记录
            self.best_goal_distance = goal_distance  # 设置初始最佳距离
            self.last_progress_step = current_step  # 记录当前时间步
            return

        # 如果当前距离比最佳距离小（更接近目标），且超过进度容差
        if goal_distance + self.progress_epsilon < self.best_goal_distance:
            self.best_goal_distance = goal_distance  # 更新最佳距离
            self.last_progress_step = current_step  # 更新进度时间步

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

        # 条件2: 进度停滞触发 - 长时间没有向全局目标进展
        stagnation = self.is_stagnated(current_step)

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
        step_duration=0.3,
        min_interval: Optional[float] = None,
        subgoal_reach_threshold: Optional[float] = None,
        waypoint_lookahead: Optional[int] = None,
        *,
        trigger_config: Optional[TriggerConfig] = None,
        planner_config: Optional[PlannerConfig] = None,
        safety_config: Optional[SafetyCriticConfig] = None,
        high_level_cost_config: Optional[HighLevelCostConfig] = None,
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
            step_duration: 步长持续时间（秒）
            min_interval: 最小触发间隔（秒），为None时从TriggerConfig获取
            trigger_config: 事件触发器配置
            planner_config: 航点规划配置
        """
        self.belief_dim = belief_dim  # 信念状态维度
        trigger_cfg = trigger_config or TriggerConfig()  # 触发器配置
        planner_cfg = planner_config or PlannerConfig()  # 规划器配置
        self.safety_config = safety_config or SafetyCriticConfig()  # 安全评估配置
        self.cost_config = high_level_cost_config or HighLevelCostConfig()
        self.planner_config = planner_cfg

        # 设置参数，优先使用传入参数，否则使用配置默认值
        if subgoal_reach_threshold is None:
            subgoal_reach_threshold = trigger_cfg.subgoal_reach_threshold
        if waypoint_lookahead is None:
            waypoint_lookahead = planner_cfg.waypoint_lookahead
        if min_interval is None:
            min_interval = (
                trigger_cfg.min_interval
                if trigger_cfg.min_interval > 0
                else trigger_cfg.min_step_interval * step_duration
            )
        # 设置计算设备，默认为GPU（如果可用）否则CPU
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 航点窗口相关参数
        self.waypoint_lookahead = max(1, int(waypoint_lookahead))  # 前瞻航点数
        self.active_window_feature_dim = 6  # [dist, cos, sin, radius, inside, margin]
        self.per_window_feature_dim = 4  # [dist, cos, sin, radius]
        self.goal_feature_dim = 3 + self.active_window_feature_dim + self.per_window_feature_dim * self.waypoint_lookahead

        # 初始化子目标生成网络
        self.subgoal_network = SubgoalNetwork(
            belief_dim=belief_dim,
            goal_info_dim=self.goal_feature_dim,
        ).to(self.device)

        # 战术层风险评估网络（Safety-Critic）
        self.safety_critic = SafetyCritic(
            belief_dim=belief_dim,
            goal_info_dim=self.goal_feature_dim,
        ).to(self.device)

        # 初始化事件触发器
        self.event_trigger = EventTrigger(
            config=trigger_cfg,
            step_duration=step_duration,
            min_interval=min_interval,
            subgoal_reach_threshold=subgoal_reach_threshold,
        )
        self.step_duration = step_duration  # 步长持续时间

        self.subgoal_reach_threshold = subgoal_reach_threshold  # 子目标到达阈值

        # 训练设置
        self.optimizer = torch.optim.Adam(self.subgoal_network.parameters(), lr=3e-4)  # Adam优化器
        self.safety_optimizer = torch.optim.Adam(self.safety_critic.parameters(), lr=3e-4)  # 安全评估优化器
        self.safety_loss_fn = nn.MSELoss()  # 安全评估损失函数
        self.cost_critic = CostCritic(
            belief_dim=belief_dim,
            goal_info_dim=self.goal_feature_dim,
        ).to(self.device)
        self.cost_optimizer = torch.optim.Adam(self.cost_critic.parameters(), lr=self.cost_config.lr)
        self.writer = SummaryWriter(comment=model_name)  # TensorBoard记录器，用于可视化训练过程
        self.iter_count = 0  # 迭代计数器，记录训练步数
        self.safety_update_count = 0  # 安全评估更新计数
        self.safety_sample_count = 0  # 安全评估样本计数
        self._safety_buffer: List[Tuple[np.ndarray, np.ndarray, float]] = []  # 安全评估样本缓冲区
        self._cost_buffer: List[Tuple[np.ndarray, np.ndarray, float]] = []
        self.cost_sample_count = 0
        self.cost_update_count = 0
        self.model_name = model_name  # 模型名称
        self.save_directory = save_directory  # 保存目录

        # RWR温度调度参数，用于软最大化权重
        self.rwr_temperature = float(max(rwr_temperature, 1e-6))  # RWR温度
        self.rwr_min_temperature = float(max(rwr_min_temperature, 1e-6))  # 最小RWR温度
        self.rwr_temperature_decay = float(max(min(rwr_temperature_decay, 1.0), 0.0))  # 温度衰减率

        # 当前状态跟踪
        self.current_subgoal = None  # 当前子目标（相对坐标：距离，角度）
        self.last_goal_distance = float('inf')  # 上次目标距离，初始化为无穷大
        self.last_goal_direction = 0.0  # 上次目标方向角度
        self.current_subgoal_world: Optional[np.ndarray] = None  # 当前子目标的世界坐标[x, y]
        self.subgoal_hidden: Optional[torch.Tensor] = None  # GRU隐状态

        # 目标–间隙引导的候选生成参数
        self.ogds_num_candidates = planner_cfg.ogds_num_candidates
        self.ogds_min_distance = planner_cfg.ogds_min_distance
        self.ogds_max_distance = planner_cfg.ogds_max_distance
        self.ogds_front_angle = planner_cfg.ogds_front_angle
        self.ogds_gap_min_width = planner_cfg.ogds_gap_min_width

        # 如果请求则加载预训练模型
        if load_model:
            load_dir = load_directory if load_directory else save_directory
            self.load_model(filename=model_name, directory=load_dir)

    def reset_subgoal_hidden(self) -> None:
        """在回合之间重置子目标网络的隐状态。"""

        self.subgoal_hidden = None

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

        norm_distance = min(float(distance) / 30.0, 1.0)  # 归一化距离
        base_features: List[float] = [norm_distance, float(cos_angle), float(sin_angle)]  # 基础特征

        tail_len = max(0, self.goal_feature_dim - 3)  # 计算尾部特征长度
        if tail_len > 0:
            if waypoint_features is None:
                waypoint_features = [0.0] * tail_len  # 默认填充零
            else:
                waypoint_features = list(waypoint_features)[:tail_len]  # 截取前tail_len个
                if len(waypoint_features) < tail_len:
                    waypoint_features.extend([0.0] * (tail_len - len(waypoint_features)))  # 填充剩余部分
            base_features.extend(float(value) for value in waypoint_features)  # 添加航点特征

        return torch.FloatTensor(base_features).to(self.device)  # 转换为张量并移动到设备

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        """将角度包装到[-π, π]范围内"""
        return math.atan2(math.sin(angle), math.cos(angle))

    def _world_to_relative(self, robot_pose, waypoint) -> Tuple[float, float]:
        """将世界坐标转换为相对坐标（距离和角度）"""
        if robot_pose is None:
            return 0.0, 0.0

        waypoint_vec = np.asarray(waypoint, dtype=np.float32)  # 航点坐标
        dx = float(waypoint_vec[0] - robot_pose[0])  # x方向差值
        dy = float(waypoint_vec[1] - robot_pose[1])  # y方向差值
        distance = math.hypot(dx, dy)  # 计算距离
        angle = self._wrap_angle(math.atan2(dy, dx) - robot_pose[2])  # 计算相对角度
        return distance, angle

    def _relative_to_world(self, robot_pose, distance: float, angle: float) -> np.ndarray:
        """将相对坐标（距离和角度）转换为世界坐标"""
        if robot_pose is None:
            return np.zeros(2, dtype=np.float32)
        world_x = robot_pose[0] + distance * math.cos(robot_pose[2] + angle)  # 计算世界x坐标
        world_y = robot_pose[1] + distance * math.sin(robot_pose[2] + angle)  # 计算世界y坐标
        return np.array([world_x, world_y], dtype=np.float32)  # 返回世界坐标

    def build_waypoint_features(self, waypoints, robot_pose) -> List[float]:
        """当前版本不使用全局航点，返回零特征占位。"""
        active_features = [0.0] * self.active_window_feature_dim
        sequence_features = [0.0] * (self.per_window_feature_dim * self.waypoint_lookahead)
        return active_features + sequence_features

    def _extract_gaps_from_lidar(
        self,
        laser_scan: np.ndarray,
        front_angle: Optional[float] = None,
    ) -> List[Tuple[float, float, float]]:
        """从激光雷达数据中提取前方可通行的间隙。

        Args:
            laser_scan: 原始激光雷达距离数组。
            front_angle: 可选的前方视场角（弧度）。

        Returns:
            间隙列表，每个元素为 (theta_start, theta_end, gap_distance)。
        """

        scan = np.asarray(laser_scan, dtype=np.float32)
        if scan.size == 0:
            return []

        # 处理无效值：NaN 视为被阻塞，Inf 视为远处空旷
        scan = np.copy(scan)
        scan[np.isnan(scan)] = 0.0
        scan[np.isinf(scan)] = 10.0

        num_rays = scan.shape[0]
        angles = np.linspace(-math.pi, math.pi, num=num_rays, endpoint=False)

        front_span = float(front_angle) if front_angle is not None else float(self.ogds_front_angle)
        half_span = max(front_span * 0.5, 0.0)
        front_mask = (angles >= -half_span) & (angles <= half_span)
        if not np.any(front_mask):
            return []

        sub_scan = scan[front_mask]
        sub_angles = angles[front_mask]

        distance_threshold = max(self.event_trigger.safety_trigger_distance, 1e-3)
        free_mask = sub_scan > distance_threshold

        gaps: List[Tuple[float, float, float]] = []
        start_idx: Optional[int] = None
        for idx, is_free in enumerate(free_mask):
            if is_free and start_idx is None:
                start_idx = idx
            elif not is_free and start_idx is not None:
                end_idx = idx - 1
                theta_start = float(sub_angles[start_idx])
                theta_end = float(sub_angles[end_idx])
                if (theta_end - theta_start) >= self.ogds_gap_min_width:
                    gap_distance = float(np.nanmedian(sub_scan[start_idx : end_idx + 1]))
                    gaps.append((theta_start, theta_end, gap_distance))
                start_idx = None

        # 末尾区间收尾
        if start_idx is not None:
            theta_start = float(sub_angles[start_idx])
            theta_end = float(sub_angles[-1])
            if (theta_end - theta_start) >= self.ogds_gap_min_width:
                gap_distance = float(np.nanmedian(sub_scan[start_idx:]))
                gaps.append((theta_start, theta_end, gap_distance))

        return gaps

    def _generate_ogds_candidates(
        self,
        *,
        laser_scan: np.ndarray,
        goal_distance: float,
        goal_cos: float,
        goal_sin: float,
        robot_pose: np.ndarray,
    ) -> Tuple[List[dict], Optional[int], Optional[float]]:
        """生成基于目标–间隙引导的候选子目标集合。"""

        theta_goal = math.atan2(goal_sin, goal_cos)
        gaps = self._extract_gaps_from_lidar(laser_scan)

        candidate_info: List[dict] = []
        active_window_radius: Optional[float] = None

        base_radius = max(float(self.planner_config.anchor_radius), 0.3)

        if not gaps:
            base_distance = float(np.clip(goal_distance, self.ogds_min_distance, self.ogds_max_distance))
            world_xy = self._relative_to_world(robot_pose, base_distance, theta_goal)
            candidate_info.append(
                {
                    "index": None,
                    "window": None,
                    "position": world_xy,
                    "distance": base_distance,
                    "angle": theta_goal,
                    "radius": base_radius,
                }
            )
            return candidate_info, None, base_radius

        weights: List[float] = []
        sigma = max(self.ogds_front_angle, 1e-3)
        for theta_start, theta_end, gap_distance in gaps:
            theta_center = 0.5 * (theta_start + theta_end)
            delta_goal = theta_center - theta_goal
            width = max(theta_end - theta_start, 1e-6)
            distance_term = max(min(gap_distance, self.ogds_max_distance), self.ogds_min_distance)
            weight = width * math.exp(-(delta_goal ** 2) / (sigma ** 2)) * distance_term
            weights.append(weight)

        total_weight = float(sum(weights))
        if total_weight <= 0.0:
            weights = [1.0 for _ in gaps]
            total_weight = float(len(gaps))

        num_candidates = max(1, int(self.ogds_num_candidates))
        counts = [1 for _ in gaps]
        remaining = num_candidates - len(gaps)
        if remaining > 0:
            probabilities = np.asarray(weights, dtype=np.float32) / total_weight
            expected = probabilities * float(remaining)
            floor_counts = np.floor(expected).astype(int)
            counts = [c + int(f) for c, f in zip(counts, floor_counts)]
            allocated = int(floor_counts.sum())
            leftover = remaining - allocated
            if leftover > 0:
                order = list(np.argsort(-probabilities))
                for idx in order[:leftover]:
                    counts[int(idx)] += 1

        for (theta_start, theta_end, gap_distance), gap_count in zip(gaps, counts):
            if gap_count <= 0:
                continue
            step = (theta_end - theta_start) / float(gap_count + 1)
            for i in range(gap_count):
                theta = theta_start + step * float(i + 1)
                distance = min(gap_distance * 0.7, self.ogds_max_distance)
                distance = float(np.clip(distance, self.ogds_min_distance, self.ogds_max_distance))
                distance = min(distance, float(goal_distance)) if np.isfinite(goal_distance) else distance
                world_xy = self._relative_to_world(robot_pose, distance, theta)
                candidate_info.append(
                    {
                        "index": None,
                        "window": None,
                        "position": world_xy,
                        "distance": distance,
                        "angle": theta,
                        "radius": base_radius,
                    }
                )

        if candidate_info:
            active_window_radius = float(np.mean([entry.get("radius", base_radius) for entry in candidate_info]))

        return candidate_info, None, active_window_radius

    def _extract_gaps_from_lidar(
        self,
        laser_scan: np.ndarray,
        front_angle: Optional[float] = None,
    ) -> List[Tuple[float, float, float]]:
        """从激光雷达数据中提取前方可通行的间隙。

        Args:
            laser_scan: 原始激光雷达距离数组。
            front_angle: 可选的前方视场角（弧度）。

        Returns:
            间隙列表，每个元素为 (theta_start, theta_end, gap_distance)。
        """

        scan = np.asarray(laser_scan, dtype=np.float32)
        if scan.size == 0:
            return []

        # 处理无效值：NaN 视为被阻塞，Inf 视为远处空旷
        scan = np.copy(scan)
        scan[np.isnan(scan)] = 0.0
        scan[np.isinf(scan)] = 10.0

        num_rays = scan.shape[0]
        angles = np.linspace(-math.pi, math.pi, num=num_rays, endpoint=False)

        front_span = float(front_angle) if front_angle is not None else float(self.ogds_front_angle)
        half_span = max(front_span * 0.5, 0.0)
        front_mask = (angles >= -half_span) & (angles <= half_span)
        if not np.any(front_mask):
            return []

        sub_scan = scan[front_mask]
        sub_angles = angles[front_mask]

        distance_threshold = max(self.event_trigger.safety_trigger_distance, 1e-3)
        free_mask = sub_scan > distance_threshold

        gaps: List[Tuple[float, float, float]] = []
        start_idx: Optional[int] = None
        for idx, is_free in enumerate(free_mask):
            if is_free and start_idx is None:
                start_idx = idx
            elif not is_free and start_idx is not None:
                end_idx = idx - 1
                theta_start = float(sub_angles[start_idx])
                theta_end = float(sub_angles[end_idx])
                if (theta_end - theta_start) >= self.ogds_gap_min_width:
                    gap_distance = float(np.nanmedian(sub_scan[start_idx : end_idx + 1]))
                    gaps.append((theta_start, theta_end, gap_distance))
                start_idx = None

        # 末尾区间收尾
        if start_idx is not None:
            theta_start = float(sub_angles[start_idx])
            theta_end = float(sub_angles[-1])
            if (theta_end - theta_start) >= self.ogds_gap_min_width:
                gap_distance = float(np.nanmedian(sub_scan[start_idx:]))
                gaps.append((theta_start, theta_end, gap_distance))

        return gaps

    def _generate_ogds_candidates(
        self,
        *,
        laser_scan: np.ndarray,
        goal_distance: float,
        goal_cos: float,
        goal_sin: float,
        robot_pose: np.ndarray,
    ) -> Tuple[List[dict], Optional[int], Optional[float]]:
        """生成基于目标–间隙引导的候选子目标集合。"""

        theta_goal = math.atan2(goal_sin, goal_cos)
        gaps = self._extract_gaps_from_lidar(laser_scan)

        candidate_info: List[dict] = []
        active_window_radius: Optional[float] = None

        base_radius = max(float(self.planner_config.anchor_radius), 0.3)

        if not gaps:
            base_distance = float(np.clip(goal_distance, self.ogds_min_distance, self.ogds_max_distance))
            world_xy = self._relative_to_world(robot_pose, base_distance, theta_goal)
            candidate_info.append(
                {
                    "index": None,
                    "window": None,
                    "position": world_xy,
                    "distance": base_distance,
                    "angle": theta_goal,
                    "radius": base_radius,
                }
            )
            return candidate_info, None, base_radius

        weights: List[float] = []
        sigma = max(self.ogds_front_angle, 1e-3)
        for theta_start, theta_end, gap_distance in gaps:
            theta_center = 0.5 * (theta_start + theta_end)
            delta_goal = theta_center - theta_goal
            width = max(theta_end - theta_start, 1e-6)
            distance_term = max(min(gap_distance, self.ogds_max_distance), self.ogds_min_distance)
            weight = width * math.exp(-(delta_goal ** 2) / (sigma ** 2)) * distance_term
            weights.append(weight)

        total_weight = float(sum(weights))
        if total_weight <= 0.0:
            weights = [1.0 for _ in gaps]
            total_weight = float(len(gaps))

        num_candidates = max(1, int(self.ogds_num_candidates))
        counts = [1 for _ in gaps]
        remaining = num_candidates - len(gaps)
        if remaining > 0:
            probabilities = np.asarray(weights, dtype=np.float32) / total_weight
            expected = probabilities * float(remaining)
            floor_counts = np.floor(expected).astype(int)
            counts = [c + int(f) for c, f in zip(counts, floor_counts)]
            allocated = int(floor_counts.sum())
            leftover = remaining - allocated
            if leftover > 0:
                order = list(np.argsort(-probabilities))
                for idx in order[:leftover]:
                    counts[int(idx)] += 1

        for (theta_start, theta_end, gap_distance), gap_count in zip(gaps, counts):
            if gap_count <= 0:
                continue
            step = (theta_end - theta_start) / float(gap_count + 1)
            for i in range(gap_count):
                theta = theta_start + step * float(i + 1)
                distance = min(gap_distance * 0.7, self.ogds_max_distance)
                distance = float(np.clip(distance, self.ogds_min_distance, self.ogds_max_distance))
                distance = min(distance, float(goal_distance)) if np.isfinite(goal_distance) else distance
                world_xy = self._relative_to_world(robot_pose, distance, theta)
                candidate_info.append(
                    {
                        "index": None,
                        "window": None,
                        "position": world_xy,
                        "distance": distance,
                        "angle": theta,
                        "radius": base_radius,
                    }
                )

        if candidate_info:
            active_window_radius = float(np.mean([entry.get("radius", base_radius) for entry in candidate_info]))

        return candidate_info, None, active_window_radius

    def build_state_vector(self, laser_scan, distance, cos_angle, sin_angle, waypoints=None, robot_pose=None):
        """构造高层规划器训练所需的状态向量"""

        with torch.no_grad():
            laser_tensor = self.process_laser_scan(laser_scan)  # 处理激光数据
            waypoint_features = self.build_waypoint_features(waypoints, robot_pose)  # 构建航点特征
            goal_tensor = self.process_goal_info(distance, cos_angle, sin_angle, waypoint_features)  # 处理目标信息
            state_tensor = torch.cat((laser_tensor, goal_tensor))  # 拼接所有特征

        return state_tensor.cpu().numpy()  # 转换为numpy数组返回

    def store_safety_sample(self, state_vector: np.ndarray, subgoal_geom: np.ndarray, target: float) -> None:
        """缓存一次子目标的风险监督样本。"""

        clipped_target = float(np.clip(target, self.safety_config.target_clip_min, self.safety_config.target_clip_max))  # 裁剪目标值
        state_arr = np.asarray(state_vector, dtype=np.float32)  # 状态向量
        geom_arr = np.asarray(subgoal_geom, dtype=np.float32)  # 子目标几何信息

        self._safety_buffer.append((state_arr, geom_arr, clipped_target))  # 添加到缓冲区
        self.safety_sample_count += 1  # 增加样本计数

        overflow = len(self._safety_buffer) - self.safety_config.max_buffer_size  # 计算溢出量
        if overflow > 0:
            del self._safety_buffer[:overflow]  # 删除最旧的样本

    def maybe_update_safety_critic(self, batch_size: Optional[int] = None) -> Optional[dict]:
        """在样本足够时执行一次Safety-Critic的更新。"""

        batch = batch_size or self.safety_config.update_batch_size  # 批次大小
        required = max(self.safety_config.min_buffer_size, batch)  # 所需最小样本数
        if len(self._safety_buffer) < required:
            return None  # 样本不足，不更新

        samples = self._safety_buffer[:batch]  # 取前batch个样本
        del self._safety_buffer[:batch]  # 从缓冲区删除已取样本

        states = np.stack([entry[0] for entry in samples])  # 堆叠状态
        geoms = np.stack([entry[1] for entry in samples])  # 堆叠几何信息
        targets = np.array([entry[2] for entry in samples], dtype=np.float32)  # 堆叠目标值

        return self.update_safety_critic(states, geoms, targets)  # 更新安全评估器

    def update_safety_critic(self, states: np.ndarray, subgoal_geoms: np.ndarray, targets: np.ndarray) -> dict:
        """优化Safety-Critic，使其拟合未来最小障碍距离。"""

        self.safety_critic.train()  # 设置为训练模式

        tensor_states = torch.as_tensor(states, dtype=torch.float32, device=self.device)  # 状态张量
        tensor_geoms = torch.as_tensor(subgoal_geoms, dtype=torch.float32, device=self.device)  # 几何信息张量
        tensor_targets = torch.as_tensor(targets, dtype=torch.float32, device=self.device)  # 目标值张量

        goal_feature_dim = self.goal_feature_dim  # 目标特征维度
        laser_dim = tensor_states.shape[1] - goal_feature_dim  # 激光数据维度
        laser_scans = tensor_states[:, :laser_dim]  # 激光数据
        goal_info = tensor_states[:, laser_dim:]  # 目标信息

        preds = self.safety_critic(laser_scans, goal_info, tensor_geoms)  # 预测风险
        loss = self.safety_loss_fn(preds, tensor_targets)  # 计算损失

        self.safety_optimizer.zero_grad()  # 清零梯度
        loss.backward()  # 反向传播
        self.safety_optimizer.step()  # 更新参数

        self.safety_update_count += 1  # 增加更新计数

        with torch.no_grad():
            avg_pred = float(preds.mean().item())  # 平均预测值
            avg_target = float(tensor_targets.mean().item())  # 平均目标值

        # 记录到TensorBoard
        self.writer.add_scalar('planner/safety_loss', loss.item(), self.safety_update_count)
        self.writer.add_scalar('planner/safety_target_mean', avg_target, self.safety_update_count)
        self.writer.add_scalar('planner/safety_pred_mean', avg_pred, self.safety_update_count)
        self.writer.add_scalar('planner/safety_buffer_size', len(self._safety_buffer), self.safety_update_count)

        return {
            'safety_loss': float(loss.item()),  # 损失值
            'safety_pred_mean': avg_pred,  # 平均预测值
            'safety_target_mean': avg_target,  # 平均目标值
        }

    def store_cost_sample(self, state_vector: np.ndarray, subgoal_geom: np.ndarray, cost: float) -> None:
        """缓存一次长期成本监督样本。"""

        state_arr = np.asarray(state_vector, dtype=np.float32)
        geom_arr = np.asarray(subgoal_geom, dtype=np.float32)
        clipped_cost = float(np.clip(cost, 0.0, np.finfo(np.float32).max))
        self._cost_buffer.append((state_arr, geom_arr, clipped_cost))
        self.cost_sample_count += 1

        overflow = len(self._cost_buffer) - self.cost_config.max_buffer_size
        if overflow > 0:
            del self._cost_buffer[:overflow]

    def maybe_update_cost_critic(self, batch_size: Optional[int] = None) -> Optional[dict]:
        """在样本足够时执行一次长期成本 Critic 的更新。"""

        batch = batch_size or self.cost_config.update_batch_size
        required = max(self.cost_config.min_buffer_size, batch)
        if len(self._cost_buffer) < required:
            return None

        samples = self._cost_buffer[:batch]
        del self._cost_buffer[:batch]

        states = np.stack([entry[0] for entry in samples])
        geoms = np.stack([entry[1] for entry in samples])
        costs = np.array([entry[2] for entry in samples], dtype=np.float32)
        return self.update_cost_critic(states, geoms, costs)

    def update_cost_critic(self, states: np.ndarray, subgoal_geoms: np.ndarray, targets: np.ndarray) -> dict:
        """优化长期成本 Critic，使其拟合子目标周期内的累计安全成本。"""

        self.cost_critic.train()

        tensor_states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        tensor_geoms = torch.as_tensor(subgoal_geoms, dtype=torch.float32, device=self.device)
        tensor_targets = torch.as_tensor(targets, dtype=torch.float32, device=self.device)

        goal_feature_dim = self.goal_feature_dim
        laser_dim = tensor_states.shape[1] - goal_feature_dim
        laser_scans = tensor_states[:, :laser_dim]
        goal_info = tensor_states[:, laser_dim:]

        preds = self.cost_critic(laser_scans, goal_info, tensor_geoms)

        with torch.no_grad():
            safety_preds = self.safety_critic(laser_scans, goal_info, tensor_geoms)

        loss_main = F.mse_loss(preds, tensor_targets)
        loss_aux = self.cost_config.aux_align_weight * F.mse_loss(preds, safety_preds)
        loss = loss_main + loss_aux

        self.cost_optimizer.zero_grad()
        loss.backward()
        self.cost_optimizer.step()

        self.cost_update_count += 1
        return {
            "cost_loss": float(loss.item()),
            "cost_loss_main": float(loss_main.item()),
            "cost_loss_aux": float(loss_aux.item()),
            "update_step": self.cost_update_count,
        }

    def predict_long_term_cost(self, laser_tensor, goal_tensor, geom_tensor) -> float:
        """推理阶段预测长期安全成本。"""

        if self.cost_sample_count < self.cost_config.min_buffer_size:
            return 0.0

        self.cost_critic.eval()
        with torch.no_grad():
            cost_pred = self.cost_critic(
                laser_tensor.unsqueeze(0), goal_tensor.unsqueeze(0), geom_tensor.unsqueeze(0)
            )
        return float(cost_pred.item())

    def predict_safety_risk(self, laser_tensor: torch.Tensor, goal_tensor: torch.Tensor, subgoal_geom: Sequence[float]) -> float:
        """基于当前状态与候选子目标预测未来风险距离值。"""

        if self.safety_sample_count < self.safety_config.min_buffer_size:
            return float("inf")  # 样本不足，返回无穷大风险

        geom_tensor = torch.as_tensor(subgoal_geom, dtype=torch.float32, device=self.device).unsqueeze(0)  # 几何信息张量

        with torch.no_grad():
            risk = self.safety_critic(
                laser_tensor.unsqueeze(0),  # 激光数据
                goal_tensor.unsqueeze(0),  # 目标信息
                geom_tensor,  # 几何信息
            )

        return float(risk.item())  # 返回风险值

    def _compute_progress_score(self, distance: float, angle: float, radius: float) -> float:
        """利用距离和角度估计候选子目标的进度收益。"""

        radius = max(radius, 1e-3)  # 确保半径不为零
        norm_distance = max(distance / radius, 0.0)  # 归一化距离
        distance_term = 1.0 / (1.0 + norm_distance)  # 距离项（距离越小收益越大）
        angle_term = (math.cos(angle) + 1.0) * 0.5  # 角度项（角度越小收益越大）

        return (
            self.safety_config.distance_weight * distance_term  # 距离权重
            + self.safety_config.angle_weight * angle_term  # 角度权重
        )

    def _convert_distance_to_risk_penalty(self, predicted_distance: float) -> float:
        """将预测的最小障碍距离转换为风险惩罚值。"""

        safe_distance = getattr(self.event_trigger, "safety_trigger_distance", 0.0)  # 安全距离
        if safe_distance <= 0:
            safe_distance = 1.0  # 默认安全距离

        if math.isinf(predicted_distance):  # 无穷大距离表示无风险
            return 0.0

        if math.isnan(predicted_distance):  # NaN值返回安全距离
            return safe_distance

        if predicted_distance <= 0:  # 负距离返回安全距离
            return safe_distance

        if predicted_distance >= safe_distance:  # 超过安全距离无风险
            return 0.0

        deficit = safe_distance - predicted_distance  # 计算安全距离缺口
        normalised = deficit / safe_distance  # 归一化缺口

        return deficit + normalised  # 返回风险惩罚值

    def _is_short_safe(self, predicted_distance: float, current_speed: float = 0.0) -> bool:
        """Check whether predicted clearance satisfies the dynamic SEN threshold."""

        threshold = self.safety_config.safe_distance_base + self.safety_config.safe_distance_kv * abs(
            float(current_speed)
        )
        if math.isnan(predicted_distance):
            return False
        if math.isinf(predicted_distance):
            return True
        return float(predicted_distance) >= threshold

    def check_triggers(
        self,
        laser_scan,
        robot_pose,
        goal_info,
        min_obstacle_dist=None,
        current_step: int = 0,
        window_metrics: Optional[dict] = None,
    ):
        """
        检查是否有任何事件触发器被激活

        Args:
            laser_scan: 当前激光雷达读数
            robot_pose: 当前机器人位姿 [x, y, theta]
            goal_info: 全局目标信息 [distance, cos, sin]
            min_obstacle_dist: 到最近障碍物的距离（如果为None，则从laser_scan计算）
            current_step: 当前时间步

        Returns:
            布尔值，指示是否应生成新子目标
        """
        # 检查时间间隔条件
        time_ready = self.event_trigger.time_based_trigger(current_step)

        # 提取目标距离信息
        goal_distance = float(goal_info[0]) if goal_info else float('inf')
        laser_scan = np.asarray(laser_scan, dtype=np.float32)  # 确保激光数据为numpy数组

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

        safe_trigger = self.event_trigger.safe_distance_trigger(min_obstacle_dist)
        progress_trigger = self.event_trigger.global_progress_stagnant(goal_distance, current_step)
        subgoal_trigger = self.event_trigger.subgoal_reached(dist_to_subgoal)

        # 最终触发决策：时间间隔满足 AND (安全/进度/到达 任一触发)
        trigger_new_subgoal = time_ready and (safe_trigger or progress_trigger or subgoal_trigger)

        # 如果触发，重置时间计数器
        if trigger_new_subgoal:
            self.event_trigger.reset_time(current_step)  # 重置触发时间

        return trigger_new_subgoal  # 返回触发结果

    def generate_subgoal(
        self,
        laser_scan,
        goal_distance,
        goal_cos,
        goal_sin,
        robot_pose=None,
        current_step: Optional[int] = None,
        waypoints=None,
        window_metrics: Optional[dict] = None,
        current_speed: Optional[float] = None,
    ):
        """
        基于当前状态生成新子目标

        Args:
            laser_scan: 处理后的激光雷达数据
            goal_distance: 到全局目标的距离
            goal_cos: 到全局目标角度的余弦值
            goal_sin: 到全局目标角度的正弦值
            robot_pose: 机器人位姿 [x, y, theta]（用于计算世界坐标）
            current_step: 当前全局时间步（用于进度重置）
            waypoints: 当前全局规划提供的候选航点
            window_metrics: 当前目标窗口的状态统计信息

        Returns:
            包含(子目标距离, 子目标角度)的元组
        """
        # 处理输入数据
        laser_tensor = self.process_laser_scan(laser_scan)  # 激光数据张量化
        waypoint_features = self.build_waypoint_features(waypoints, robot_pose)  # 构建航点特征
        goal_tensor = self.process_goal_info(goal_distance, goal_cos, goal_sin, waypoint_features)  # 处理目标信息

        # 使用网络生成子目标（不计算梯度）
        with torch.no_grad():  # 推理模式，不计算梯度
            hidden_in = self.subgoal_hidden
            if hidden_in is not None:
                hidden_in = hidden_in.detach()
                if hidden_in.device != self.device:
                    hidden_in = hidden_in.to(self.device)

            (
                distance_adjust_tensor,
                angle_offset_tensor,
                next_hidden,
            ) = self.subgoal_network(
                laser_tensor.unsqueeze(0),  # 增加批次维度：[1, belief_dim] -> [1, 1, belief_dim]
                goal_tensor.unsqueeze(0),  # 增加批次维度：[3] -> [1, 3]
                hidden_state=hidden_in,
            )
            self.subgoal_hidden = next_hidden.detach() if next_hidden is not None else None

        distance_adjust = float(distance_adjust_tensor.cpu().numpy().item())  # 距离调整量
        angle_offset = float(angle_offset_tensor.cpu().numpy().item())  # 角度偏移量

        candidate_info = []  # 候选航点信息
        active_window_index: Optional[int] = None  # 活动窗口索引
        active_window_radius: Optional[float] = None  # 活动窗口半径
        if robot_pose is not None:
            candidate_info, active_window_index, active_window_radius = self._generate_ogds_candidates(
                laser_scan=np.asarray(laser_scan, dtype=np.float32),
                goal_distance=float(goal_distance),
                goal_cos=float(goal_cos),
                goal_sin=float(goal_sin),
                robot_pose=np.asarray(robot_pose, dtype=np.float32),
            )

        selected_index = None  # 选择的索引
        final_distance: float  # 最终距离
        final_angle: float  # 最终角度
        world_target = None  # 世界坐标目标

        anchor_info = None  # 锚点信息
        scored_candidates: List[dict] = []  # 评分候选列表
        speed_for_safety = float(current_speed) if current_speed is not None else 0.0
        if candidate_info:
            reachable: List[dict] = []  # 可达候选
            fallback: List[dict] = []  # 回退候选
            for info in candidate_info:
                info_with_flags = dict(info)
                ray = self.event_trigger._ray_distance_to_angle(laser_scan, info["angle"])  # 射线距离
                info_with_flags["ray_distance"] = float(ray) if np.isfinite(ray) else float("inf")
                is_reachable = bool(
                    np.isfinite(ray)
                    and ray >= info["distance"] + 0.02  # 可达条件
                )
                info_with_flags["reachable"] = is_reachable
                if is_reachable:
                    reachable.append(info_with_flags)  # 添加到可达列表
                fallback.append(info_with_flags)  # 添加到回退列表

            evaluated = reachable if reachable else fallback  # 优先使用可达候选

            short_safe_candidates: List[dict] = []

            for info in evaluated:
                radius = max(info.get("radius", 0.0), 1e-3)  # 半径
                subgoal_geom = [float(info["distance"]), float(info["angle"]), float(radius)]  # 子目标几何信息
                predicted_distance = self.predict_safety_risk(
                    laser_tensor, goal_tensor, subgoal_geom  # 预测风险距离
                )
                is_short_safe = self._is_short_safe(predicted_distance, speed_for_safety)
                geom_tensor = torch.as_tensor(np.asarray(subgoal_geom, dtype=np.float32), device=self.device)
                cost_val = self.predict_long_term_cost(laser_tensor, goal_tensor, geom_tensor)
                risk_val = self._convert_distance_to_risk_penalty(predicted_distance)  # 风险值
                progress_val = self._compute_progress_score(
                    float(info["distance"]), float(info["angle"]), float(radius)  # 进度得分
                )
                score_val = (
                    self.safety_config.progress_weight * progress_val  # 进度权重
                    - self.safety_config.risk_weight * risk_val  # 风险权重
                    - self.cost_config.lambda_near * float(cost_val)
                )
                scored = dict(info)
                scored["risk"] = float(risk_val)  # 风险值
                scored["predicted_min_distance"] = float(predicted_distance)  # 预测最小距离
                scored["progress"] = float(progress_val)  # 进度值
                scored["score"] = float(score_val)  # 总分
                scored["short_safe"] = bool(is_short_safe)
                scored["long_cost"] = float(cost_val)
                scored_candidates.append(scored)  # 添加到评分候选列表
                if is_short_safe:
                    short_safe_candidates.append(scored)

            preferred = short_safe_candidates if short_safe_candidates else scored_candidates
            long_safe = [info for info in preferred if info.get("long_cost", 0.0) <= self.cost_config.safe_cost_threshold]
            candidate_pool = long_safe if long_safe else preferred

            if candidate_pool:
                anchor_info = max(candidate_pool, key=lambda entry: entry["score"])  # 选择得分最高的候选
                scored_candidates = candidate_pool

        # 找不到可行解就回退到原来的第一个候选
        if anchor_info is None and candidate_info:
            anchor_info = candidate_info[0]  # 回退到第一个候选

        if anchor_info is not None:
            if anchor_info["index"] is not None:
                selected_index = int(anchor_info["index"])  # 选择的索引
            base_distance = anchor_info["distance"]  # 基础距离
            base_angle = anchor_info["angle"]  # 基础角度
            anchor_radius = max(anchor_info["radius"], 0.3)  # 锚点半径
            active_window_radius = anchor_radius  # 活动窗口半径
        else:
            base_angle = math.atan2(goal_sin, goal_cos)  # 基础角度
            base_distance = max(0.5, min(float(goal_distance), 3.0))  # 基础距离
            anchor_radius = max(0.5, min(float(goal_distance), 2.0) * 0.5)  # 锚点半径
            if active_window_radius is None:
                active_window_radius = anchor_radius  # 活动窗口半径

        # 根据网络输出的调整量生成最终子目标
        min_distance = max(0.2, base_distance - anchor_radius)  # 最小距离
        max_distance = base_distance + anchor_radius  # 最大距离
        candidate_distance = base_distance + distance_adjust * anchor_radius  # 候选距离
        final_distance = float(np.clip(candidate_distance, min_distance, max_distance))  # 最终距离（裁剪后）
        final_angle = self._wrap_angle(base_angle + angle_offset)  # 最终角度

        # 记录实际应用的调整量（考虑裁剪后的效果）
        applied_distance_adjust = 0.0
        if anchor_radius > 1e-3:
            applied_distance_adjust = float(np.clip((final_distance - base_distance) / anchor_radius, -1.0, 1.0))  # 应用的距离调整
        applied_angle_offset = float(
            np.clip(self._wrap_angle(final_angle - base_angle), -math.pi / 4, math.pi / 4)  # 应用的角度偏移
        )

        if robot_pose is not None:
            world_target = self._relative_to_world(robot_pose, final_distance, final_angle)  # 计算世界坐标

        # 存储供将来参考
        self.current_subgoal = (final_distance, final_angle)  # 当前子目标
        self.last_goal_distance = float(goal_distance)  # 上次目标距离
        self.last_goal_direction = math.atan2(goal_sin, goal_cos)  # 上次目标方向
        if world_target is not None:
            self.current_subgoal_world = world_target  # 当前子目标世界坐标
            self.event_trigger.last_subgoal = world_target.tolist()  # 事件触发器中的子目标
        else:
            self.current_subgoal_world = None
            self.event_trigger.last_subgoal = None

        # 重置进度基准（新子目标意味着重新开始进度跟踪）
        progress_step = current_step if current_step is not None else 0
        self.event_trigger.reset_progress(goal_distance, progress_step)

        metadata = {
            "selected_waypoint": selected_index,  # 选择的航点
            "active_window_index": active_window_index,  # 活动窗口索引
            "active_window_radius": active_window_radius,  # 活动窗口半径
            "window_metrics": dict(window_metrics) if window_metrics else {},  # 窗口指标
            "anchor_distance": base_distance,  # 锚点距离
            "anchor_angle": base_angle,  # 锚点角度
            "anchor_radius": anchor_radius,  # 锚点半径
            "raw_distance_adjust": distance_adjust,  # 原始距离调整
            "raw_angle_offset": angle_offset,  # 原始角度偏移
            "distance_adjust_applied": applied_distance_adjust,  # 应用的距离调整
            "angle_offset_applied": applied_angle_offset,  # 应用的角度偏移
            "candidate_scores": [
                {
                    "index": entry.get("index"),  # 索引
                    "score": entry.get("score"),  # 得分
                    "risk": entry.get("risk"),  # 风险
                    "predicted_min_distance": entry.get("predicted_min_distance"),  # 预测最小距离
                    "progress": entry.get("progress"),  # 进度
                    "reachable": entry.get("reachable", False),  # 是否可达
                }
                for entry in scored_candidates  # 遍历评分候选
            ],
            "selected_risk": (float(anchor_info.get("risk")) if anchor_info and "risk" in anchor_info else None),  # 选择的风险
            "selected_predicted_min_distance": (
                float(anchor_info.get("predicted_min_distance"))  # 选择的预测最小距离
                if anchor_info and "predicted_min_distance" in anchor_info
                else None
            ),
            "selected_progress": (
                float(anchor_info.get("progress")) if anchor_info and "progress" in anchor_info else None  # 选择的进度
            ),
        }

        if metadata["selected_waypoint"] is None and active_window_index is not None:
            metadata["selected_waypoint"] = active_window_index  # 设置选择的航点

        return final_distance, final_angle, metadata  # 返回最终距离、角度和元数据

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

    def update_planner(self, states, actions, rewards, dones, next_states, batch_size=64):
        """
        使用收集的经验更新规划器的神经网络

        Args:
            states: 环境状态批次
            actions: 采取的动作批次（真实的子目标）
            rewards: 获得的奖励批次
            next_states: 结果状态批次
            dones: 完成标志批次
            batch_size: 训练批次大小

        Returns:
            训练指标字典
        """
        # 转换为PyTorch张量并移动到设备
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1)  # 增加维度便于广播
        dones = torch.FloatTensor(dones).to(self.device).unsqueeze(1)
        next_states = torch.FloatTensor(next_states).to(self.device)

        goal_feature_dim = self.goal_feature_dim  # 目标特征维度
        split_index = states.shape[1] - goal_feature_dim  # 分割索引
        laser_scans = states[:, :split_index]  # 激光数据
        goal_info = states[:, split_index:]  # 目标信息

        # 生成子目标（网络预测）
        distance_adjusts, angle_offsets, _ = self.subgoal_network(
            laser_scans,
            goal_info,
            hidden_state=None,  # 快速版训练：不维护跨样本的隐状态
        )
        # 合并调整量为完整动作 [distance_coeff, angle_offset]
        subgoals = torch.stack((distance_adjusts, angle_offsets), dim=1)

        # 基于软最大化的奖励加权，温度参数控制集中度
        with torch.no_grad():
            temperature = max(self.rwr_temperature, 1e-6)  # 温度
            scaled = (rewards - rewards.max()) / temperature  # 缩放奖励
            weights = torch.softmax(scaled.squeeze(1), dim=0).unsqueeze(1)  # 软最大化权重

        # 计算每个样本的MSE损失
        per_sample_loss = F.mse_loss(subgoals, actions, reduction='none').mean(dim=1, keepdim=True)
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
            updated_temp = self.rwr_temperature * self.rwr_temperature_decay  # 更新温度
            self.rwr_temperature = max(updated_temp, self.rwr_min_temperature)  # 确保不低于最小值

        # 记录指标到TensorBoard
        self.writer.add_scalar('planner/loss', loss.item(), self.iter_count)  # 损失值
        self.writer.add_scalar('planner/reward_weight_mean', weights.mean().item(), self.iter_count)  # 平均权重
        self.writer.add_scalar('planner/rwr_temperature', self.rwr_temperature, self.iter_count)  # RWR温度

        # 返回训练指标
        weight_entropy = float(-(weights * (weights.clamp_min(1e-8).log())).sum().item())  # 权重熵

        return {
            'loss': loss.item(),  # 损失值
            'avg_distance_adjust': distance_adjusts.mean().item(),  # 平均距离调整量
            'avg_angle_offset': angle_offsets.mean().item(),  # 平均角度偏移量
            'weight_mean': weights.mean().item(),  # 平均权重
            'weight_entropy': weight_entropy,  # 权重熵
            'temperature': self.rwr_temperature,  # 温度
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
