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
    steps_taken: int,
    max_steps: int,
    subgoal_replanned: bool,
    goal_distance: float,
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

    # 基准锚点：拖满一局的时间成本
    R_base = config.efficiency_penalty * max_steps

    # ------------------------------------------------------------
    # 1. 子目标进展奖励（防刷分 + 跳变裁剪）
    # ------------------------------------------------------------
    progress_delta = 0.0
    if (
        not subgoal_replanned
        and prev_subgoal_distance is not None
        and current_subgoal_distance is not None
    ):
        raw_delta = prev_subgoal_distance - current_subgoal_distance
        progress_delta = float(
            np.clip(raw_delta, -config.collision_distance, config.collision_distance)
        )

    r_prog = config.progress_weight * progress_delta
    components["progress"] = float(r_prog)

    # ------------------------------------------------------------
    # 2. 时间项（隐式防抖动）
    # ------------------------------------------------------------
    r_time = -config.efficiency_penalty
    if progress_delta <= 0:
        r_time -= config.efficiency_penalty  # 无进展时加倍惩罚
    components["time"] = float(r_time)

    # ------------------------------------------------------------
    # 3. 安全塑形奖励（线性归一化）
    # ------------------------------------------------------------
    r_safety = 0.0
    if min_obstacle_distance < config.safety_clearance:
        ratio = (config.safety_clearance - min_obstacle_distance) / config.safety_clearance
        r_safety = -config.safety_weight * ratio
    components["safety"] = float(r_safety)

    # ------------------------------------------------------------
    # 4. 终局强锚点
    # ------------------------------------------------------------
    r_terminal = 0.0
    steps_taken = int(max(0, steps_taken))

    if reached_goal:
        # 成功：基准大奖 + 剩余时间分
        r_terminal = R_base + config.efficiency_penalty * max(0, max_steps - steps_taken)
    elif collision:
        # 碰撞：基准倒扣 - 剩余距离罚 - 提前结束补偿
        r_terminal = (
            -R_base
            - (config.progress_weight * max(goal_distance, 0.0))
            - (config.efficiency_penalty * max(0, max_steps - steps_taken))
        )
    elif timed_out:
        # 超时：基准倒扣 - 剩余距离罚
        r_terminal = -R_base - (config.progress_weight * max(goal_distance, 0.0))

    components["terminal"] = float(r_terminal)

    # ------------------------------------------------------------
    # 5. 总奖励汇总（再做全局缩放）
    # ------------------------------------------------------------
    total_reward_raw = r_prog + r_time + r_safety + r_terminal

    # 全局缩放奖励，压低TD目标尺度但不改变最优策略
    total_reward = float(total_reward_raw * config.reward_scale)
    components["reward_raw"] = float(total_reward_raw)

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
