"""
奖励塑造工具函数，用于ETHSRL+GP导航栈。
提供高层规划器和低层控制器的奖励计算功能。
"""

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


@dataclass
class LowLevelRewardConfig:
    """低层控制器奖励配置容器类"""
    
    """低层控制器奖励配置容器类 - 【调试简化版】"""
    
    # === 核心驱动力 ===
    progress_weight: float = 5.0           # [保持] 唯一的核心正向奖励：鼓励朝向子目标移动。
    subgoal_bonus: float = 20.0            # [增加] 到达子目标是重要的阶段性成功，给予强信号。
    
    # === 绝对底线 ===
    collision_penalty: float = -40.0       # [保持] 碰撞是不可接受的。
    timeout_penalty: float = -25.0         # [保持] 超时是失败。
    goal_bonus: float = 50.0               # [保持] 到达最终目标是巨大成功。

    # === 行为素质 ===
    smoothness_weight: float = 1.0         # [核心-增加] 显著提高平滑度权重，鼓励稳定控制。
    
    # ===【调试期间禁用】以下为暂时关闭的参数 ===
    safety_weight: float = 0.0             # [禁用] 暂时关闭。碰撞惩罚已足够强大。
    efficiency_penalty: float = 0.0        # [禁用] 暂时不考虑细微的时间惩罚。
    window_progress_weight: float = 0.0    # [禁用] 暂时不让低层关心“窗口”，只关心子目标。
    window_entry_bonus: float = 0.0        # [禁用]
    window_inside_bonus: float = 0.0       # [禁用]
    window_outside_penalty: float = 0.0    # [禁用]
    window_timeout_penalty: float = 0.0    # [禁用]
    heading_alignment_weight: float = 0.0  # [禁用] progress_weight 已隐式包含航向对齐。
    
    # === 裁剪值 (保持不变) ===
    progress_clip: float = 0.2
    window_progress_clip: float = 0.3
    heading_alignment_clip: float = 0.5
    smoothness_clip: float = 0.6
    safe_distance: float = 0.7


@dataclass
class HighLevelRewardConfig:
    """高层规划器奖励配置容器类"""
    
    # 高层规划器奖励函数配置
    global_progress_weight: float = 3.5    # 鼓励朝向全局目标取得进展的权重
    path_progress_weight: float = 4.0      # 奖励沿全局路径推进的窗口索引权重
    path_regression_penalty: float = -6.0  # 倒退至先前窗口的惩罚
    window_convergence_weight: float = 3.0 # 奖励逼近当前目标窗口中心的权重
    window_convergence_clip: float = 1.5   # 距离改进裁剪阈值
    window_entry_bonus: float = 4.0        # 进入目标窗口的奖励
    window_persistence_bonus: float = 0.2  # 在窗口内保持的奖励系数
    window_persistence_cap: int = 10       # 在窗口内累计奖励的步数上限
    window_failure_penalty: float = -6.0   # 未能到达目标窗口的惩罚
    target_window_bonus: float = 6.0       # 成功稳定在目标窗口内的额外奖励
    low_level_return_scale: float = 0.03   # 低层累积奖励的缩放权重
    subgoal_completion_bonus: float = 10.0 # 成功完成一个子目标的奖励
    goal_bonus: float = 50.0               # 到达最终目的地的巨大奖励
    collision_penalty: float = -20.0       # 导致碰撞的巨大惩罚
    timeout_penalty: float = -10.0         # 导致超时的惩罚


def _clip_progress(delta: float, clip: float) -> float:
    """将进度项裁剪到指定范围，保持奖励数值稳定
    
    Args:
        delta: 原始进度值
        clip: 裁剪阈值
        
    Returns:
        裁剪后的进度值
    """
    
    if clip <= 0:
        return delta
    return float(np.clip(delta, -clip, clip))


def compute_low_level_reward(
    prev_subgoal_distance: Optional[float],
    current_subgoal_distance: Optional[float],
    min_obstacle_distance: float,
    reached_goal: bool,
    reached_subgoal: bool,
    collision: bool,
    timed_out: bool,
    config: LowLevelRewardConfig,
    *,
    window_entered: bool = False,
    window_inside: bool = False,
    window_limit_exceeded: bool = False,
    prev_window_distance: Optional[float] = None,
    current_window_distance: Optional[float] = None,
    window_radius: Optional[float] = None,
    current_subgoal_angle: Optional[float] = None,
    action_delta: Optional[Sequence[float]] = None,
) -> Tuple[float, Dict[str, float]]:
    """计算简化的、用于低层控制器的塑形奖励

    Args:
        prev_subgoal_distance: 动作执行前到当前子目标的距离
        current_subgoal_distance: 动作执行后到当前子目标的距离
        min_obstacle_distance: 动作执行后观测到的最小障碍物距离
        reached_goal: 是否在此步到达全局目标
        reached_subgoal: 是否在此步到达当前子目标
        collision: 是否发生碰撞
        timed_out: 是否因步数耗尽而终止
        config: 奖励塑造超参数
        window_entered: 是否刚进入窗口
        window_inside: 是否在窗口内
        window_limit_exceeded: 是否超过窗口停留时间限制
        prev_window_distance: 之前到窗口中心的距离
        current_window_distance: 当前到窗口中心的距离
        window_radius: 窗口半径
        current_subgoal_angle: 当前子目标角度
        action_delta: 动作变化量

    Returns:
        一个元组 ``(reward, components)``，其中 ``reward`` 是标量总奖励，
        ``components`` 提供了用于日志/调试的命名分解项
    """

    # 处理可能的空距离值
    prev_distance = float(prev_subgoal_distance) if prev_subgoal_distance is not None else 0.0
    curr_distance = float(current_subgoal_distance) if current_subgoal_distance is not None else prev_distance

    # 1. 进度奖励：奖励接近当前子目标
    progress_raw = prev_distance - curr_distance  # 计算距离减少量
    progress = _clip_progress(progress_raw, config.progress_clip)  # 裁剪进度值
    progress_reward = config.progress_weight * progress  # 计算进度奖励

    # 2. 窗口进度与奖励项
    window_progress_raw = 0.0
    # 如果有窗口距离信息，计算窗口进度
    if (
        prev_window_distance is not None
        and current_window_distance is not None
    ):
        window_progress_raw = prev_window_distance - current_window_distance
    window_progress = _clip_progress(window_progress_raw, config.window_progress_clip)  # 裁剪窗口进度
    window_progress_reward = config.window_progress_weight * window_progress  # 计算窗口进度奖励

    # 窗口进入奖励
    entry_bonus = config.window_entry_bonus if window_entered else 0.0
    # 窗口内/外奖励/惩罚
    inside_reward = config.window_inside_bonus if window_inside else -config.window_outside_penalty
    # 根据窗口半径缩放奖励
    if window_radius is not None and window_radius > 0:
        scale = float(np.clip(window_radius / 0.6, 0.5, 1.5))
        entry_bonus *= scale
        inside_reward *= scale
    # 窗口超时惩罚
    timeout_penalty = config.window_timeout_penalty if window_limit_exceeded else 0.0

    # 航向对齐奖励
    heading_reward = 0.0
    if current_subgoal_angle is not None:
        # 计算与子目标方向的对齐程度
        alignment = 1.0 - min(abs(current_subgoal_angle) / np.pi, 1.0)
        heading_reward = config.heading_alignment_weight * _clip_progress(alignment, config.heading_alignment_clip)

    # 平滑性惩罚
    smoothness_penalty = 0.0
    if action_delta is not None:
        # 计算动作变化幅度
        delta = np.asarray(action_delta, dtype=np.float32)
        magnitude = float(np.linalg.norm(delta))
        smoothness_penalty = -config.smoothness_weight * _clip_progress(magnitude, config.smoothness_clip)

    # 3. 安全惩罚：惩罚离障碍物太近
    safety_penalty = 0.0
    if min_obstacle_distance < config.safe_distance:
        # 当智能体进入安全距离内，惩罚会二次方增长
        violation = (config.safe_distance - min_obstacle_distance) / config.safe_distance
        safety_penalty = -config.safety_weight * (violation ** 2)

    # 4. 效率惩罚：对每一步施加一个小的固定惩罚
    efficiency_penalty = -config.efficiency_penalty

    # 5. 终止状态奖励/惩罚
    terminal_reward = 0.0
    if reached_goal:
        terminal_reward += config.goal_bonus  # 到达目标奖励
    elif reached_subgoal:
        terminal_reward += config.subgoal_bonus  # 到达子目标奖励
    elif collision:
        terminal_reward += config.collision_penalty  # 碰撞惩罚
    elif timed_out:
        terminal_reward += config.timeout_penalty  # 超时惩罚

    # 计算总奖励
    total_reward = (
        progress_reward
        + window_progress_reward
        + entry_bonus
        + inside_reward
        + timeout_penalty
        + heading_reward
        + smoothness_penalty
        + safety_penalty
        + efficiency_penalty
        + terminal_reward
    )

    # 用于分析的奖励分量字典
    components = {
        "progress": progress_reward,
        "window_progress": window_progress_reward,
        "window_entry": entry_bonus,
        "window_presence": inside_reward,
        "window_timeout": timeout_penalty,
        "heading_alignment": heading_reward,
        "smoothness": smoothness_penalty,
        "safety": safety_penalty,
        "efficiency": efficiency_penalty,
        "terminal": terminal_reward,
    }

    return float(total_reward), components


def compute_high_level_reward(
    start_goal_distance: float,
    end_goal_distance: float,
    subgoal_completed: bool,
    reached_goal: bool,
    collision: bool,
    timed_out: bool,
    config: HighLevelRewardConfig,
    *,
    start_window_index: Optional[int] = None,
    end_window_index: Optional[int] = None,
    start_window_distance: Optional[float] = None,
    best_window_distance: Optional[float] = None,
    end_window_distance: Optional[float] = None,
    window_entered: bool = False,
    window_inside_steps: int = 0,
    target_window_index: Optional[int] = None,
    target_window_reached: bool = False,
    low_level_return: float = 0.0,
) -> Tuple[float, Dict[str, float]]:
    """计算简化的、用于高层规划器的奖励

    Args:
        start_goal_distance: 发出子目标时，机器人到全局目标的距离
        end_goal_distance: 子目标终止时，机器人到全局目标的距离
        subgoal_completed: 子目标在终止前是否被完成
        reached_goal: 回合是否因到达全局目标而结束
        collision: 回合是否因碰撞而结束
        timed_out: 回合是否超时
        config: 规划器的奖励塑造超参数
        start_window_index: 开始时的窗口索引
        end_window_index: 结束时的窗口索引
        start_window_distance: 开始时的窗口距离
        best_window_distance: 最佳窗口距离
        end_window_distance: 结束时的窗口距离
        window_entered: 是否进入窗口
        window_inside_steps: 在窗口内的步数
        target_window_index: 目标窗口索引
        target_window_reached: 是否到达目标窗口
        low_level_return: 低层累积奖励

    Returns:
        与 :func:`compute_low_level_reward` 类似的元组 ``(reward, components)``
    """

    # 1. 全局进度奖励：奖励机器人到最终目标的距离减少量
    goal_progress = start_goal_distance - end_goal_distance
    global_progress_reward = config.global_progress_weight * goal_progress

    # 2. 路径索引进度：鼓励沿全局规划窗口前进
    path_progress_reward = 0.0
    path_regression = 0.0
    if start_window_index is not None and end_window_index is not None:
        index_delta = end_window_index - start_window_index  # 计算窗口索引变化
        if index_delta > 0:
            path_progress_reward = config.path_progress_weight * index_delta  # 前进奖励
        elif index_delta < 0:
            path_regression = config.path_regression_penalty * abs(index_delta)  # 倒退惩罚

    # 3. 逼近窗口中心：度量子目标选择质量
    window_convergence_reward = 0.0
    if start_window_distance is not None:
        baseline = start_window_distance
        candidate = best_window_distance
        if candidate is None:
            candidate = end_window_distance
        if candidate is not None:
            distance_gain = baseline - candidate  # 计算距离改进
            distance_gain = _clip_progress(distance_gain, config.window_convergence_clip)  # 裁剪改进值
            window_convergence_reward = config.window_convergence_weight * distance_gain

    # 4. 进入与停留奖励
    entry_bonus = config.window_entry_bonus if window_entered else 0.0  # 进入窗口奖励
    persistence_steps = min(window_inside_steps, config.window_persistence_cap)  # 限制最大持续步数
    persistence_bonus = config.window_persistence_bonus * float(max(persistence_steps, 0))  # 持续停留奖励
    target_bonus = config.target_window_bonus if target_window_reached else 0.0  # 目标窗口奖励

    # 5. 子目标完成奖励
    completion_bonus = config.subgoal_completion_bonus if subgoal_completed else 0.0

    # 6. 低层回报共享
    shared_return = config.low_level_return_scale * float(low_level_return)

    # 7. 终止状态奖励/惩罚
    terminal_reward = 0.0
    if reached_goal:
        terminal_reward += config.goal_bonus  # 到达目标奖励
    elif collision:
        terminal_reward += config.collision_penalty  # 碰撞惩罚
    elif timed_out:
        terminal_reward += config.timeout_penalty  # 超时惩罚

    # 8. 未能到达目标窗口的惩罚
    failure_penalty = 0.0
    if target_window_index is not None:
        # 检查是否未能到达目标窗口
        if end_window_index is None or end_window_index < target_window_index:
            failure_penalty += config.window_failure_penalty  # 完全失败惩罚
        elif not target_window_reached:
            failure_penalty += 0.5 * config.window_failure_penalty  # 部分失败惩罚

    # 计算高层总奖励
    total_reward = (
        global_progress_reward
        + path_progress_reward
        + path_regression
        + window_convergence_reward
        + entry_bonus
        + persistence_bonus
        + target_bonus
        + completion_bonus
        + shared_return
        + terminal_reward
        + failure_penalty
    )

    # 用于分析的奖励分量字典
    components = {
        "global_progress": global_progress_reward,
        "path_progress": path_progress_reward,
        "path_regression": path_regression,
        "window_convergence": window_convergence_reward,
        "window_entry": entry_bonus,
        "window_persistence": persistence_bonus,
        "target_bonus": target_bonus,
        "completion_bonus": completion_bonus,
        "shared_low_level": shared_return,
        "terminal": terminal_reward,
        "failure": failure_penalty,
    }

    return float(total_reward), components


# 模块导出列表
__all__ = [
    "LowLevelRewardConfig",      # 低层奖励配置类
    "HighLevelRewardConfig",     # 高层奖励配置类
    "compute_low_level_reward",  # 低层奖励计算函数
    "compute_high_level_reward", # 高层奖励计算函数
]
