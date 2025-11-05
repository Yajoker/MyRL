"""
奖励塑造工具函数，用于ETHSRL+GP导航栈。
提供高层规划器和低层控制器的奖励计算功能。

[最终优化版] - 基于Yajoker用户的最终设计方案实现。
该版本具有逻辑清晰、职责分离、数学表达精炼的优点。
"""

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple
import math
import numpy as np


# --- 1. 低层奖励函数 ---

@dataclass
class LowLevelRewardConfig:
    """
    [最终优化版] 低层控制器奖励配置。
    根据用户方案：R_low = R_progress + R_safety + R_terminal
    """
    # 进展奖励 (R_progress)
    progress_weight: float = 12.0         # w_p: 进展权重
    efficiency_penalty: float = 0.12      # w_eff: 每步的效率惩罚（按0.3s步长缩放）

    # 安全惩罚 (R_safety)
    safety_weight: float = 1.8            # w_s: 安全权重
    safety_sensitivity: float = 2.0       # σ: 指数惩罚的敏感度参数
    safety_clearance: float = 1.0         # 清晰距离阈值，超过该距离不再惩罚

    # 终局奖励 (R_terminal)
    goal_bonus: float = 60.0              # 到达最终目标的奖励
    subgoal_bonus: float = 18.0           # 到达子目标的奖励
    collision_penalty: float = -50.0      # 碰撞惩罚
    timeout_penalty: float = -15.0        # 超时惩罚


@dataclass
class HighLevelRewardConfig:
    """
    [最终优化版] 高层规划器奖励配置。
    根据用户方案：R_high = R_strategic + R_decision + R_terminal
    """
    # 战略进展奖励 (R_strategic)
    path_progress_weight: float = 5.0    # w_path: 沿路径窗口索引推进的权重
    global_progress_weight: float = 2.0   # w_global: 朝向最终目标进展的权重
    # (注意：路径后退惩罚将在计算中通过 path_progress_weight * (负的Δi) 实现)

    # 决策质量奖励 (R_decision)
    low_level_return_scale: float = 0.02  # κ: 低层累积回报的共享权重
    subgoal_completion_bonus: float = 4.0 # w_comp: 子目标完成奖励
    low_level_failure_penalty: float = -10.0 # w_fail: 低层执行失败的惩罚

    # 终局奖励 (R_terminal)
    goal_bonus: float = 60.0             # 到达最终目标的巨大奖励
    collision_penalty: float = -50.0     # 导致碰撞的决策的巨大惩罚
    timeout_penalty: float = -25.0        # 导致超时的决策的惩罚


def compute_low_level_reward(
    # --- 核心输入 ---
    prev_subgoal_distance: Optional[float],
    current_subgoal_distance: Optional[float],
    min_obstacle_distance: float,
    # --- 终局状态 ---
    reached_goal: bool,
    reached_subgoal: bool,
    collision: bool,
    timed_out: bool,
    # --- 配置 ---
    config: LowLevelRewardConfig,
    **kwargs,  # 忽略所有其他高层输入
) -> Tuple[float, Dict[str, float]]:
    """
    [最终优化版] 计算低层控制器的奖励。
    """
    components: Dict[str, float] = {}

    # 1. 进展奖励 (R_progress)
    # 公式: R_progress = w_p * (d_prev - d_current) - w_eff
    progress_reward = 0.0
    step_penalty = 0.0 if reached_goal else config.efficiency_penalty
    if reached_subgoal and not reached_goal:
        step_penalty *= 0.5  # 达到子目标时仅保留一半效率惩罚
    if prev_subgoal_distance is not None and current_subgoal_distance is not None:
        progress_delta = prev_subgoal_distance - current_subgoal_distance
        progress_reward = config.progress_weight * progress_delta - step_penalty
    else:
        # 如果没有有效的距离信息，只施加效率惩罚
        progress_reward = -step_penalty
    components["progress"] = progress_reward

    # 2. 安全惩罚 (R_safety)
    # 公式: R_safety = -w_s * exp(-σ * d_min)
    # 使用 math.exp 来精确实现指数形式
    # 为了防止 d_min 过大导致 exp 结果为0，或 d_min 过小导致 exp 溢出，可以做适当处理
    # 但在典型机器人场景下，d_min 通常在合理范围内
    if min_obstacle_distance >= config.safety_clearance:
        safety_penalty = 0.0
    else:
        safety_penalty = -config.safety_weight * math.exp(-config.safety_sensitivity * min_obstacle_distance)
    components["safety"] = safety_penalty

    # 3. 终局奖励/惩罚 (R_terminal)
    terminal_reward = 0.0
    if reached_goal:
        terminal_reward = config.goal_bonus
    elif reached_subgoal:
        terminal_reward = config.subgoal_bonus
    elif collision:
        terminal_reward = config.collision_penalty
    elif timed_out:
        terminal_reward = config.timeout_penalty
    components["terminal"] = terminal_reward

    # 计算总奖励: R_low = R_progress + R_safety + R_terminal
    total_reward = sum(components.values())

    # 碰撞时强制奖励为负值（避免进展项抵消惩罚）
    if collision and not reached_goal:
        total_reward = min(total_reward, config.collision_penalty)

    return float(total_reward), components


def compute_high_level_reward(
    # --- 战略进展输入 ---
    start_goal_distance: float,
    end_goal_distance: float,
    start_window_index: Optional[int],
    end_window_index: Optional[int],
    # --- 决策质量输入 ---
    subgoal_completed: bool,
    low_level_return: float,
    # --- 终局状态 ---
    reached_goal: bool,
    collision: bool,
    timed_out: bool,
    # --- 配置 ---
    config: HighLevelRewardConfig,
    **kwargs,  # 忽略所有其他低层输入
) -> Tuple[float, Dict[str, float]]:
    """
    [最终优化版] 计算高层规划器的奖励。
    """
    components: Dict[str, float] = {}
    
    # 1. 战略进展奖励 (R_strategic)
    # 公式: R_strategic = w_path * Δi + w_global * (d_start - d_end)
    strategic_reward = 0.0
    
    # 路径索引进展
    path_progress = 0.0
    if start_window_index is not None and end_window_index is not None:
        index_delta = end_window_index - start_window_index
        # Δi 可以是正（前进）或负（后退），惩罚已包含在内
        path_progress = config.path_progress_weight * index_delta

    # 全局目标进展
    global_progress = config.global_progress_weight * (start_goal_distance - end_goal_distance)
    
    strategic_reward = path_progress + global_progress
    components["strategic_progress"] = strategic_reward

    # 2. 决策质量奖励 (R_decision)
    # 公式: R_decision = κ * ∑R_low + w_comp * I_complete + w_fail * I_fail
    
    # 共享回报
    shared_return = config.low_level_return_scale * low_level_return
    
    # 子目标完成奖励 (I_complete = 1)
    completion_bonus = config.subgoal_completion_bonus if subgoal_completed else 0.0
    
    # 低层失败惩罚 (I_fail = 1)
    # 失败定义为：在子目标执行期间发生碰撞或超时
    low_level_failed = not reached_goal and (collision or timed_out)
    failure_penalty = config.low_level_failure_penalty if low_level_failed else 0.0

    decision_reward = shared_return + completion_bonus + failure_penalty
    components["decision_quality"] = decision_reward

    # 3. 终局奖励/惩罚 (R_terminal)
    terminal_reward = 0.0
    if reached_goal:
        terminal_reward = config.goal_bonus
    elif collision:
        terminal_reward = config.collision_penalty
    elif timed_out:
        terminal_reward = config.timeout_penalty
    components["terminal"] = terminal_reward

    # 计算总奖励: R_high = R_strategic + R_decision + R_terminal
    total_reward = sum(components.values())

    # 碰撞时强制奖励为负，防止进展带来正值
    if collision and not reached_goal:
        total_reward = min(total_reward, config.collision_penalty)
    
    return float(total_reward), components


# 模块导出列表
__all__ = [
    "LowLevelRewardConfig",
    "HighLevelRewardConfig",
    "compute_low_level_reward",
    "compute_high_level_reward",
]
