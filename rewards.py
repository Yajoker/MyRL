from typing import Dict, Optional, Tuple
import numpy as np
from typing import Dict, Optional, Tuple

from config import HighLevelRewardConfig, LowLevelRewardConfig


# ================================================================
#  低层奖励：TD3 控制器（“怎么走到子目标”）
# ================================================================
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
    """计算低层控制器的奖励（供 TD3 使用）。

    设计目标：
    - 告诉低层：应当“快速、平滑、安全”地接近当前子目标；
    - 终点 / 子目标成功 / 超时的语义交给高层处理；
    - 只保留一个“轻微的局部碰撞惩罚”，提醒低层本身也要讨厌撞墙。

    返回:
        total_reward: 本步低层总奖励（标量）
        components:   各个组成部分的字典，方便日志分析
    """
    components: Dict[str, float] = {}

    # ------------------------------------------------------------
    # 1. 子目标进展奖励 R_progress
    #    R_progress = w_p * (d_prev - d_cur) - step_penalty
    # ------------------------------------------------------------
    progress_reward = 0.0

    # 步数惩罚：表达“时间成本”
    # 终点那一步可以不惩罚；到子目标那一步则减半，避免穿越子目标时被过度惩罚。
    step_penalty = 0.0 if reached_goal else config.efficiency_penalty
    if reached_subgoal and not reached_goal:
        step_penalty *= 0.5

    if prev_subgoal_distance is not None and current_subgoal_distance is not None:
        progress_delta = prev_subgoal_distance - current_subgoal_distance
        progress_reward = config.progress_weight * progress_delta - step_penalty
    else:
        # 没有距离信息时，只保留时间惩罚
        progress_reward = -step_penalty

    components["progress"] = float(progress_reward)

    # ------------------------------------------------------------
    # 2. 安全塑形奖励 R_safety
    #    离障碍越近，惩罚越大；超过安全距离则不惩罚。
    # ------------------------------------------------------------
    if min_obstacle_distance < config.collision_distance:
        safety_raw = -1.0
    elif min_obstacle_distance < config.safety_clearance:
        # 在线性区间 [collision_distance, safety_clearance] 内插值
        ratio = (config.safety_clearance - min_obstacle_distance) / (
            config.safety_clearance - config.collision_distance
        )
        safety_raw = -0.3 - 0.7 * ratio  # 从 -0.3 到 -1.0
    else:
        safety_raw = 0.0

    safety_reward = config.safety_weight * safety_raw
    components["safety"] = float(safety_reward)

    # ------------------------------------------------------------
    # 3. 低层局部终局项 R_terminal
    #    只在发生碰撞时给一个小的额外惩罚。
    #    终点 / 子目标成功 / 超时都由高层奖励统一处理。
    # ------------------------------------------------------------
    terminal_reward = 0.0

    if collision and not reached_goal:
        # 将碰撞惩罚缩小到局部尺度：避免与高层的大失败惩罚重复。
        # 建议在 config 里把 collision_penalty 设得比高层小很多，
        terminal_reward = config.collision_penalty
    elif timed_out and not reached_goal:
        # 低层也对超时有一个小的局部惩罚
        terminal_reward = config.timeout_penalty

    components["terminal"] = float(terminal_reward)

    # ------------------------------------------------------------
    # 4. 总奖励汇总 & 碰撞时做一次下限裁剪
    # ------------------------------------------------------------
    total_reward = progress_reward + safety_reward + terminal_reward

    # 确保碰撞时 reward 不会被进度项刷成正值
    #if collision and not reached_goal:
    #    total_reward = min(total_reward, terminal_reward)

    return float(total_reward), components


# ================================================================
#  高层奖励：子目标规划器 RWR（“往哪设子目标”）
# ================================================================
def compute_high_level_reward(
    start_goal_distance: float,
    end_goal_distance: float,
    subgoal_step_count: int,
    collision: bool,
    *,
    config: HighLevelRewardConfig,
    **kwargs,  # 忽略所有其他低层输入
) -> Tuple[float, Dict[str, float]]:
    """计算高层规划器的奖励（供 RWR 作为样本权重使用）。

    设计目标：
    - 按简化公式 Gi = alpha * Δd_global - beta_col * I[collision] - beta_time * T
    - 仅聚焦全局进展、碰撞与时间成本，便于 Advantage-RWR 直接利用。
    """
    components: Dict[str, float] = {}

    # 全局距离进展
    delta_global = start_goal_distance - end_goal_distance
    progress_reward = config.alpha_global_progress * float(delta_global)
    components["global_progress"] = float(progress_reward)

    # 碰撞惩罚（子目标执行期间是否发生碰撞）
    collision_penalty = config.beta_collision if collision else 0.0
    components["collision_penalty"] = float(-collision_penalty)

    # 时间成本（子目标执行步数）
    time_penalty = config.beta_time * float(max(subgoal_step_count, 0))
    components["time_penalty"] = float(-time_penalty)

    total_reward = progress_reward - collision_penalty - time_penalty
    return float(total_reward), components


__all__ = [
    "LowLevelRewardConfig",
    "HighLevelRewardConfig",
    "compute_low_level_reward",
    "compute_high_level_reward",
]
