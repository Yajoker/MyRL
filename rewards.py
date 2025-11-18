"""奖励塑造工具函数（重构版），用于 ETHSRL+GP 分层导航栈。

本文件只负责把「环境状态与事件」映射成两种标量奖励：
- 低层奖励：给 TD3 actor–critic，用于学习“如何跟随当前子目标并避障”；
- 高层奖励：给子目标网络的 RWR，用于学习“在什么状态下应该选什么子目标”。

设计原则：
1. 低层专注局部控制：子目标进展 + 避障 + （可选）轻微碰撞惩罚，不再自己定义终点 / 子目标的大奖励。
2. 高层负责任务成败：全局目标进展 + 路径进展 + 低层执行质量 + 终局事件（成功/碰撞/超时）。
3. 同一个语义只在一个层级被定义一次，避免冗余和相互打架。
"""

from typing import Dict, Optional, Tuple
import numpy as np

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
        # 这里再乘一个系数进一步减弱。
        terminal_reward = 0.2 * config.collision_penalty

    components["terminal"] = float(terminal_reward)

    # ------------------------------------------------------------
    # 4. 总奖励汇总 & 碰撞时做一次下限裁剪
    # ------------------------------------------------------------
    total_reward = progress_reward + safety_reward + terminal_reward

    # 确保碰撞时 reward 不会被进度项刷成正值
    if collision and not reached_goal:
        total_reward = min(total_reward, terminal_reward)

    return float(total_reward), components


# ================================================================
#  高层奖励：子目标规划器 RWR（“往哪设子目标”）
# ================================================================
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
    """计算高层规划器的奖励（供 RWR 作为样本权重使用）。

    设计目标：
    - 为「这一次子目标决策 (state, subgoal)」提供一个可排序的标量得分；
    - 高得分 ≈ 子目标让机器人明显朝最终目标前进，且执行顺利、不导致失败；
    - 低得分 ≈ 子目标让机器人走向死胡同、危险区域或直接导致碰撞/超时。
    """
    components: Dict[str, float] = {}

    # ------------------------------------------------------------
    # 1. 战略几何进展 R_strategic
    #    R_strategic = w_path * Δindex + w_global * (d_start - d_end)
    # ------------------------------------------------------------
    # 路径索引进展：窗口 index 的变化（允许为负，对“走回头路”进行惩罚）
    path_progress = 0.0
    if start_window_index is not None and end_window_index is not None:
        index_delta = end_window_index - start_window_index
        path_progress = config.path_progress_weight * index_delta

    # 到最终目标的距离变化：朝目标靠近则为正，远离则为负
    global_progress = config.global_progress_weight * (
        start_goal_distance - end_goal_distance
    )

    strategic_reward = path_progress + global_progress
    components["strategic_progress"] = float(strategic_reward)

    # ------------------------------------------------------------
    # 2. 决策质量 R_decision
    #    R_decision = κ * norm(low_level_return) + b_complete + b_fail
    # ------------------------------------------------------------
    # 为了避免 low_level_return 尺度太大，让它进入一个大致 [-2, 2] 的区间
    # 再乘上 config.low_level_return_scale。
    # 这里的 40.0 是一个经验值，可根据日志中 low_level_return 的典型范围调整。
    norm_return = np.clip(low_level_return / 40.0, -2.0, 2.0)
    shared_return = config.low_level_return_scale * float(norm_return)

    # 子目标完成的小奖励（高层视角）
    completion_bonus = config.subgoal_completion_bonus if subgoal_completed else 0.0

    # 低层在执行这个子目标时是否发生失败（碰撞 / 超时）
    low_level_failed = (collision or timed_out) and (not reached_goal)
    failure_penalty = config.low_level_failure_penalty if low_level_failed else 0.0

    decision_reward = shared_return + completion_bonus + failure_penalty
    components["decision_quality"] = float(decision_reward)

    # ------------------------------------------------------------
    # 3. 终局事件 R_terminal
    #    只在 episode 结束时起作用，由高层统一定义任务成败的语义。
    # ------------------------------------------------------------
    terminal_reward = 0.0
    if reached_goal:
        terminal_reward = config.goal_bonus
    elif collision:
        terminal_reward = config.collision_penalty
    elif timed_out:
        terminal_reward = config.timeout_penalty

    components["terminal"] = float(terminal_reward)

    # ------------------------------------------------------------
    # 4. 总奖励汇总 & 碰撞时做一次下限裁剪
    # ------------------------------------------------------------
    total_reward = strategic_reward + decision_reward + terminal_reward

    # 确保碰撞导致的总 reward 不会被其它正项“洗白”
    if collision and not reached_goal:
        total_reward = min(total_reward, config.collision_penalty)

    return float(total_reward), components


__all__ = [
    "LowLevelRewardConfig",
    "HighLevelRewardConfig",
    "compute_low_level_reward",
    "compute_high_level_reward",
]
