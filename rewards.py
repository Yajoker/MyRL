"""Reward shaping utilities for the ETHSRL+GP navigation stack."""

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


@dataclass
class LowLevelRewardConfig:
    """Configuration container for the low-level (controller) reward."""
    
    # 奖励函数各项权重参数配置
    progress_weight: float = 2.5           # 鼓励朝向子目标移动
    safety_weight: float = 2.5             # 惩罚过于靠近障碍物的行为
    efficiency_penalty: float = 0.05       # 每走一步的轻微时间惩罚，鼓励效率
    goal_bonus: float = 50.0               # 到达最终目标的巨大奖励
    subgoal_bonus: float = 5.0             # 到达当前子目标的奖励（降低重复刷分收益）
    collision_penalty: float = -20.0       # 碰撞的巨大惩罚
    timeout_penalty: float = -10.0         # 超时的惩罚
    safe_distance: float = 1               # 安全距离阈值（米），小于此距离将触发安全惩罚
    progress_clip: float = 0.2             # 每步奖励的进度裁剪值（米），用于稳定训练


@dataclass
class HighLevelRewardConfig:
    """Configuration container for the high-level (planner) reward."""
    
    # 高层规划器奖励函数配置
    global_progress_weight: float = 5.0    # 鼓励朝向全局目标取得进展
    subgoal_completion_bonus: float = 10.0 # 成功完成一个子目标的奖励
    goal_bonus: float = 50.0               # 到达最终目的地的巨大奖励
    collision_penalty: float = -20.0       # 导致碰撞的巨大惩罚
    timeout_penalty: float = -10.0         # 导致超时的惩罚


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
    min_obstacle_distance: float,
    reached_goal: bool,
    reached_subgoal: bool,
    collision: bool,
    timed_out: bool,
    config: LowLevelRewardConfig,
) -> Tuple[float, Dict[str, float]]:
    """计算简化的、用于低层控制器的塑形奖励。

    Args:
        prev_subgoal_distance: 动作执行前到当前子目标的距离
        current_subgoal_distance: 动作执行后到当前子目标的距离
        min_obstacle_distance: 动作执行后观测到的最小障碍物距离
        reached_goal: 是否在此步到达全局目标
        reached_subgoal: 是否在此步到达当前子目标
        collision: 是否发生碰撞
        timed_out: 是否因步数耗尽而终止
        config: 奖励塑造超参数

    Returns:
        一个元组 ``(reward, components)``，其中 ``reward`` 是标量总奖励，
        ``components`` 提供了用于日志/调试的命名分解项。
    """

    # 处理可能的空距离值
    prev_distance = float(prev_subgoal_distance) if prev_subgoal_distance is not None else 0.0
    curr_distance = float(current_subgoal_distance) if current_subgoal_distance is not None else prev_distance

    # 1. 进度奖励：奖励接近当前子目标
    progress_raw = prev_distance - curr_distance
    progress = _clip_progress(progress_raw, config.progress_clip)
    progress_reward = config.progress_weight * progress

    # 2. 安全惩罚：惩罚离障碍物太近
    safety_penalty = 0.0
    if min_obstacle_distance < config.safe_distance:
        # 当智能体进入安全距离内，惩罚会二次方增长
        violation = (config.safe_distance - min_obstacle_distance) / config.safe_distance
        safety_penalty = -config.safety_weight * (violation ** 2)

    # 3. 效率惩罚：对每一步施加一个小的固定惩罚
    efficiency_penalty = -config.efficiency_penalty

    # 4. 终止状态奖励/惩罚
    terminal_reward = 0.0
    if reached_goal:
        terminal_reward += config.goal_bonus
    elif reached_subgoal:
        terminal_reward += config.subgoal_bonus
    elif collision:
        terminal_reward += config.collision_penalty
    elif timed_out:
        terminal_reward += config.timeout_penalty

    # 计算总奖励
    total_reward = (
        progress_reward
        + safety_penalty
        + efficiency_penalty
        + terminal_reward
    )

    # 用于分析的奖励分量字典
    components = {
        "progress": progress_reward,
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
) -> Tuple[float, Dict[str, float]]:
    """计算简化的、用于高层规划器的奖励。

    Args:
        start_goal_distance: 发出子目标时，机器人到全局目标的距离
        end_goal_distance: 子目标终止时，机器人到全局目标的距离
        subgoal_completed: 子目标在终止前是否被完成
        reached_goal: 回合是否因到达全局目标而结束
        collision: 回合是否因碰撞而结束
        timed_out: 回合是否超时
        config: 规划器的奖励塑造超参数

    Returns:
        与 :func:`compute_low_level_reward` 类似的元组 ``(reward, components)``
    """

    # 1. 全局进度奖励：奖励机器人到最终目标的距离减少量
    goal_progress = start_goal_distance - end_goal_distance
    global_progress_reward = config.global_progress_weight * goal_progress
    
    # 2. 子目标完成奖励
    completion_bonus = config.subgoal_completion_bonus if subgoal_completed else 0.0

    # 3. 终止状态奖励/惩罚
    terminal_reward = 0.0
    if reached_goal:
        terminal_reward += config.goal_bonus
    elif collision:
        terminal_reward += config.collision_penalty
    elif timed_out:
        terminal_reward += config.timeout_penalty

    # 计算高层总奖励
    total_reward = (
        global_progress_reward
        + completion_bonus
        + terminal_reward
    )

    # 用于分析的奖励分量字典
    components = {
        "global_progress": global_progress_reward,
        "completion_bonus": completion_bonus,
        "terminal": terminal_reward,
    }

    return float(total_reward), components


# 模块导出列表
__all__ = [
    "LowLevelRewardConfig",
    "HighLevelRewardConfig",
    "compute_low_level_reward",
    "compute_high_level_reward",
]
