import math
from typing import Dict, Optional, Tuple

import numpy as np

try:
    # 当以包形式导入（例如 `from myrl.rewards import ...`）时使用相对导入
    from .config import HighLevelRewardConfig, LowLevelRewardConfig
except ImportError:  # pragma: no cover
    # 兼容旧的脚本运行方式（在 myrl 目录下直接运行 train.py 等）
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
    - 使用方向有效速度替代距离差分奖励，避免背对目标刷分；
    - 终局时直接给出 option 语义奖励/惩罚，避免叠加偏置。

    返回:
        total_reward: 本步低层总奖励（标量）
        components:   各个组成部分的字典，方便日志分析
    """
    components: Dict[str, float] = {}

    terminal_reward = 0.0
    if collision:
        terminal_reward = config.collision_penalty
    elif timed_out:
        terminal_reward = config.timeout_penalty
    elif reached_subgoal:
        terminal_reward = config.subgoal_bonus
    elif reached_goal:
        terminal_reward = config.goal_bonus

    if terminal_reward != 0.0:
        components["progress_v"] = 0.0
        components["turn"] = 0.0
        components["obs"] = 0.0
        components["living"] = 0.0
        components["terminal"] = float(terminal_reward)
        components["total"] = float(terminal_reward)
        return float(terminal_reward), components

    # ------------------------------------------------------------
    # 1. 方向有效速度奖励 R_progress
    #    R_progress = w_f * max(0, v) * cos(angle)
    # ------------------------------------------------------------
    action = kwargs.get("action")
    angle_to_subgoal = kwargs.get("angle_to_subgoal", 0.0)
    dt = kwargs.get("dt", 1.0)

    v = float(action[0]) if action is not None else 0.0
    w = float(action[1]) if action is not None else 0.0
    if angle_to_subgoal is None:
        angle_to_subgoal = 0.0
    cos_angle = math.cos(float(angle_to_subgoal))
    if config.direction_clip != 0.0:
        cos_angle = max(cos_angle, config.direction_clip)

    progress_v = max(0.0, v) * cos_angle
    if config.use_directional_velocity:
        progress_reward = config.forward_weight * progress_v
    else:
        if prev_subgoal_distance is not None and current_subgoal_distance is not None:
            progress_delta = prev_subgoal_distance - current_subgoal_distance
            progress_reward = config.progress_weight * progress_delta
        else:
            progress_reward = 0.0
    components["progress_v"] = float(progress_v)

    # ------------------------------------------------------------
    # 2. 转向惩罚 R_turn
    # ------------------------------------------------------------
    turn_scale = 0.3 + 0.7 * max(0.0, cos_angle)
    turn_reward = -config.turn_weight * abs(w) * turn_scale
    components["turn"] = float(turn_reward)

    # ------------------------------------------------------------
    # 3. 靠近障碍惩罚 R_obs
    # ------------------------------------------------------------
    r3 = max(0.0, config.safe_distance - float(min_obstacle_distance))
    obs_reward = -config.obstacle_weight * r3
    components["obs"] = float(obs_reward)

    # ------------------------------------------------------------
    # 4. 生存成本 R_living
    # ------------------------------------------------------------
    if config.living_cost_per_sec > 0:
        living_reward = -config.living_cost_per_sec * float(dt)
    elif config.living_cost_per_step > 0:
        living_reward = -config.living_cost_per_step
    else:
        living_reward = 0.0
    components["living"] = float(living_reward)

    # ------------------------------------------------------------
    # 5. 总奖励汇总
    # ------------------------------------------------------------
    total_reward = progress_reward + turn_reward + obs_reward + living_reward
    components["terminal"] = 0.0
    components["total"] = float(total_reward)

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
    short_cost_sum: float = 0.0,
    near_obstacle_steps: int = 0,
    **kwargs,
) -> Tuple[Tuple[float, float], Dict[str, float]]:
    """计算高层规划器的效率回报与安全成本。

    返回 (r_eff, c_safe) 二元组：
    - r_eff：效率回报，越大越好；
    - c_safe：安全成本，正值且越大越不安全（训练时作为监督信号）。
    """
    components: Dict[str, float] = {}

    # 全局距离进展
    delta_global = float(start_goal_distance - end_goal_distance)
    progress_reward = float(config.alpha_global_progress * delta_global)
    components["global_progress"] = float(progress_reward)

    # 时间成本（作为效率回报中的惩罚项）
    time_penalty = float(config.beta_time * float(max(subgoal_step_count, 0)))
    components["time_penalty"] = float(-time_penalty)

    # 效率回报：进展奖励 - 时间惩罚
    reward_eff = progress_reward - time_penalty
    components["eff_reward"] = float(reward_eff)

    # 安全成本：碰撞 + 短距离风险 + 近障步数
    collision_cost = float(config.beta_collision) if collision else 0.0
    short_cost_term = float(config.gamma_short_cost * max(short_cost_sum, 0.0))
    near_cost_term = float(config.gamma_near_steps * max(near_obstacle_steps, 0))
    safety_cost = collision_cost + short_cost_term + near_cost_term

    components["collision_cost"] = float(-collision_cost)
    components["short_cost_term"] = float(-short_cost_term)
    components["near_cost_term"] = float(-near_cost_term)
    components["safety_cost_total"] = float(-safety_cost)

    total_reward = reward_eff - safety_cost
    components["total_reward"] = float(total_reward)

    return (float(reward_eff), float(safety_cost)), components


# ================================================================
#  短期安全成本计算（供高层成本/安全统计使用）
# ================================================================
def compute_step_safety_cost(risk_index: float, collision: bool, *, config: HighLevelRewardConfig) -> float:
    """Compute the per-step safety cost using the unified risk index.

    每步成本会累计到高层的安全成本监督信号中。
    """

    cost = float(config.lambda_near) * float(max(risk_index, 0.0))
    if collision:
        cost += float(config.lambda_col)
    return float(cost)


__all__ = [
    "LowLevelRewardConfig",
    "HighLevelRewardConfig",
    "compute_low_level_reward",
    "compute_high_level_reward",
    "compute_step_safety_cost",
]
