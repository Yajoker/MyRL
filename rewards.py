"""Reward shaping utilities for the ETHSRL+GP navigation stack."""

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


@dataclass
class LowLevelRewardConfig:
    """Configuration container for the low-level (controller) reward."""
    
    # 奖励函数各项权重参数配置
    progress_weight: float = 5.0           # 进度奖励权重
    action_weight: float = 0.2             # 动作惩罚权重（控制能耗）
    smooth_weight: float = 0.1             # 平滑性惩罚权重
    safety_weight: float = 2.5             # 安全性惩罚权重
    efficiency_penalty: float = 0.02       # 效率惩罚（鼓励快速完成）
    goal_bonus: float = 150.0              # 到达最终目标奖励
    subgoal_bonus: float = 15.0            # 到达子目标奖励
    collision_penalty: float = 200.0       # 碰撞惩罚
    timeout_penalty: float = 40.0          # 超时惩罚
    safe_distance: float = 0.5             # 安全距离阈值
    safety_margin: float = 0.3             # 安全裕度
    linear_velocity_scale: float = 0.5     # 线速度缩放因子
    angular_velocity_scale: float = 1.0    # 角速度缩放因子
    progress_clip: float = 0.2             # 进度奖励裁剪值（米/步）


@dataclass
class HighLevelRewardConfig:
    """Configuration container for the high-level (planner) reward."""
    
    # 高层规划器奖励函数配置
    global_progress_weight: float = 6.0           # 全局进度奖励权重
    subgoal_completion_bonus: float = 20.0        # 子目标完成奖励
    collision_penalty: float = 220.0              # 碰撞惩罚
    goal_bonus: float = 300.0                     # 最终目标奖励
    timeout_penalty: float = 60.0                 # 超时惩罚


def _clip_progress(delta: float, clip: float) -> float:
    """Clamp the progress term to keep rewards numerically stable.
    
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
    action: Sequence[float],
    prev_action: Optional[Sequence[float]],
    min_obstacle_distance: float,
    reached_goal: bool,
    reached_subgoal: bool,
    collision: bool,
    timed_out: bool,
    config: LowLevelRewardConfig,
) -> Tuple[float, Dict[str, float]]:
    """Compute the shaped reward for the low-level controller.

    Args:
        prev_subgoal_distance: 执行动作前到活动子目标的距离
        current_subgoal_distance: 执行动作后到活动子目标的距离
        action: 执行的控制命令 [线速度, 角速度]
        prev_action: 上一步的控制命令
        min_obstacle_distance: 动作后观测到的最短障碍物距离
        reached_goal: 是否在此步骤到达全局目标
        reached_subgoal: 是否在此步骤到达活动子目标
        collision: 是否发生碰撞
        timed_out: 是否因步数预算而终止回合
        config: 奖励塑造超参数

    Returns:
        元组 ``(reward, components)``，其中 ``reward`` 是标量总奖励，
        ``components`` 提供命名的分解项用于日志记录/调试
    """

    # 处理可能的空距离值
    prev_distance = float(prev_subgoal_distance) if prev_subgoal_distance is not None else 0.0
    curr_distance = float(current_subgoal_distance) if current_subgoal_distance is not None else prev_distance

    # 计算向当前子目标的进度
    progress_raw = prev_distance - curr_distance  # 距离减少表示正向进度
    progress = _clip_progress(progress_raw, config.progress_clip)  # 裁剪进度值保持数值稳定
    progress_reward = config.progress_weight * progress  # 进度奖励

    # 控制能耗和平滑性惩罚
    action_arr = np.asarray(action, dtype=np.float32)
    prev_action_arr = np.asarray(prev_action, dtype=np.float32) if prev_action is not None else np.zeros_like(action_arr)

    # 计算动作能耗惩罚（基于当前动作）
    lin_scaled = action_arr[0] / max(config.linear_velocity_scale, 1e-6)
    ang_scaled = action_arr[1] / max(config.angular_velocity_scale, 1e-6)
    effort_penalty = -config.action_weight * (lin_scaled ** 2 + ang_scaled ** 2)

    # 计算动作平滑性惩罚（基于动作变化）
    delta_lin = (action_arr[0] - prev_action_arr[0]) / max(config.linear_velocity_scale, 1e-6)
    delta_ang = (action_arr[1] - prev_action_arr[1]) / max(config.angular_velocity_scale, 1e-6)
    smoothness_penalty = -config.smooth_weight * (delta_lin ** 2 + delta_ang ** 2)

    # 基于最近障碍物测量的安全性塑造
    safety_term = 0.0
    safe_buffer = config.safe_distance + config.safety_margin  # 安全缓冲区距离
    
    if min_obstacle_distance < config.safe_distance:
        # 在安全距离内：严重违反安全约束
        violation = (config.safe_distance - min_obstacle_distance) / max(config.safe_distance, 1e-6)
        safety_term -= config.safety_weight * (1.0 + violation ** 2)  # 二次惩罚
    elif min_obstacle_distance < safe_buffer:
        # 在安全缓冲区内：轻微违反
        buffer_fraction = (safe_buffer - min_obstacle_distance) / max(config.safety_margin, 1e-6)
        safety_term -= 0.5 * config.safety_weight * (buffer_fraction ** 2)  # 较轻的二次惩罚

    # 时间效率惩罚（鼓励快速完成子目标）
    efficiency_penalty = -config.efficiency_penalty

    # 终止状态奖励/惩罚
    terminal_bonus = 0.0
    if reached_goal:
        terminal_bonus += config.goal_bonus  # 到达最终目标奖励
    elif reached_subgoal:
        terminal_bonus += config.subgoal_bonus  # 到达子目标奖励

    terminal_penalty = 0.0
    if collision:
        terminal_penalty -= config.collision_penalty  # 碰撞惩罚
    if timed_out and not reached_goal and not collision:
        terminal_penalty -= config.timeout_penalty  # 超时惩罚

    # 计算总奖励（所有分量的加权和）
    total_reward = (
        progress_reward
        + effort_penalty
        + smoothness_penalty
        + safety_term
        + efficiency_penalty
        + terminal_bonus
        + terminal_penalty
    )

    # 构建奖励分量字典用于分析和调试
    components = {
        "progress": progress_reward,
        "effort": effort_penalty,
        "smoothness": smoothness_penalty,
        "safety": safety_term,
        "efficiency": efficiency_penalty,
        "terminal_bonus": terminal_bonus,
        "terminal_penalty": terminal_penalty,
    }

    return float(total_reward), components


def compute_high_level_reward(
    accumulated_low_level_reward: float,
    start_goal_distance: float,
    end_goal_distance: float,
    subgoal_completed: bool,
    reached_goal: bool,
    collision: bool,
    timed_out: bool,
    config: HighLevelRewardConfig,
) -> Tuple[float, Dict[str, float]]:
    """Compute the reward assigned to the high-level planner.

    Args:
        accumulated_low_level_reward: 子目标活动期间收集的低层奖励总和
        start_goal_distance: 发出子目标时的机器人到目标距离
        end_goal_distance: 子目标终止时的机器人到目标距离
        subgoal_completed: 子目标在终止前是否完成
        reached_goal: 回合终止时是否到达全局目标
        collision: 回合是否以碰撞结束
        timed_out: 回合是否超时
        config: 规划器的奖励塑造超参数

    Returns:
        与 :func:`compute_low_level_reward` 类似的元组 ``(reward, components)``
    """

    # 计算全局目标进度
    goal_progress = start_goal_distance - end_goal_distance  # 距离减少表示正向进度
    global_progress_reward = config.global_progress_weight * goal_progress
    
    # 子目标完成奖励
    completion_bonus = config.subgoal_completion_bonus if subgoal_completed else 0.0

    # 终止状态奖励/惩罚
    terminal_bonus = config.goal_bonus if reached_goal else 0.0
    terminal_penalty = 0.0
    if collision:
        terminal_penalty -= config.collision_penalty
    if timed_out and not reached_goal and not collision:
        terminal_penalty -= config.timeout_penalty

    # 计算高层规划器总奖励
    total_reward = (
        accumulated_low_level_reward    # 累积的低层奖励
        + global_progress_reward        # 全局进度奖励
        + completion_bonus              # 子目标完成奖励
        + terminal_bonus                # 终止奖励
        + terminal_penalty              # 终止惩罚
    )

    # 构建奖励分量字典
    components = {
        "low_level": accumulated_low_level_reward,
        "global_progress": global_progress_reward,
        "completion_bonus": completion_bonus,
        "terminal_bonus": terminal_bonus,
        "terminal_penalty": terminal_penalty,
    }

    return float(total_reward), components


# 模块导出列表
__all__ = [
    "LowLevelRewardConfig",
    "HighLevelRewardConfig",
    "compute_low_level_reward",
    "compute_high_level_reward",
]