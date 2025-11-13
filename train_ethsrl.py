"""
ETHSRL+GP分层导航系统的训练入口点

该脚本遵循原始``robot_nav/rl_train.py``的结构，
同时集成了新实现的高层规划器和低层控制器。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

# 导入自定义模块
from config import ConfigBundle, HighLevelRewardConfig, LowLevelRewardConfig, TrainingConfig
from integration import HierarchicalNavigationSystem
from rewards import compute_high_level_reward, compute_low_level_reward
from robot_nav.SIM_ENV.sim import SIM
from robot_nav.replay_buffer import ReplayBuffer


@dataclass
class SubgoalContext:
    """高层子目标生命周期内的统计上下文"""

    start_state: np.ndarray  # 子目标开始时的状态
    action: np.ndarray  # 选择的子目标调整量 [距离系数, 角度偏移]
    world_target: np.ndarray  # 子目标的全局坐标
    start_goal_distance: float  # 开始时的目标距离
    last_goal_distance: float  # 最后的目标距离
    low_level_return: float = 0.0  # 累积的低层奖励
    steps: int = 0  # 子目标执行的步数
    subgoal_completed: bool = False  # 子目标是否完成
    last_state: Optional[np.ndarray] = None  # 最后的状态
    start_window_index: Optional[int] = None  # 子目标开始时的活动窗口索引
    target_window_index: Optional[int] = None  # 高层选择的目标窗口索引
    start_window_distance: Optional[float] = None  # 初始窗口中心距离
    last_window_index: Optional[int] = None  # 最近一次记录的窗口索引
    last_window_distance: Optional[float] = None  # 最近一次记录的窗口距离
    best_window_distance: Optional[float] = None  # 子目标执行期间达到的最小窗口距离
    window_entered: bool = False  # 是否首次进入目标窗口
    window_inside_steps: int = 0  # 在目标窗口内累计的步数
    target_window_reached: bool = False  # 是否稳定到达目标窗口
    min_dmin: float = float("inf")  # 子目标执行期间观测到的最近障碍距离
    collision_occurred: bool = False  # 执行期间是否发生碰撞
    subgoal_angle_at_start: Optional[float] = None  # 子目标生成时的角度
    base_distance: Optional[float] = None  # Safety-Critic几何：锚点距离
    base_angle: Optional[float] = None  # Safety-Critic几何：锚点角度
    anchor_radius: Optional[float] = None  # Safety-Critic几何：窗口半径


def compute_subgoal_world(robot_pose: Tuple[float, float, float], distance: float, angle: float) -> np.ndarray:
    """将相对子目标 (r, θ) 转换为全局坐标.

    Args:
        robot_pose: 机器人位姿 (x, y, theta)
        distance: 子目标相对距离
        angle: 子目标相对角度
        
    Returns:
        子目标的全局坐标 [x, y]
    """

    # 计算子目标在世界坐标系中的位置
    world_x = robot_pose[0] + distance * np.cos(robot_pose[2] + angle)  # x坐标计算
    world_y = robot_pose[1] + distance * np.sin(robot_pose[2] + angle)  # y坐标计算
    return np.array([world_x, world_y], dtype=np.float32)  # 返回世界坐标数组


def finalize_subgoal_transition(
    context: Optional[SubgoalContext],
    buffer: List[Tuple[np.ndarray, np.ndarray, float, np.ndarray, float]],
    high_cfg: HighLevelRewardConfig,
    done: bool,
    reached_goal: bool,
    collision: bool,
    timed_out: bool,
) -> Optional[Tuple[dict, Optional[Tuple[np.ndarray, np.ndarray, float]]]]:
    """结束当前子目标并生成高层训练样本.

    Args:
        context: 子目标上下文
        buffer: 高层经验回放缓冲区
        high_cfg: 高层奖励配置
        done: 是否终止
        reached_goal: 是否到达目标
        collision: 是否碰撞
        timed_out: 是否超时
        
    Returns:
        包含奖励分量字典及可选风险样本的元组，或None
    """

    # 检查上下文有效性
    if context is None or context.steps == 0:  # 上下文为空或步数为0
        return None  # 返回None

    # 确定最后状态
    last_state = context.last_state if context.last_state is not None else context.start_state  # 使用最后状态或开始状态

    # 计算高层奖励
    reward, components = compute_high_level_reward(  # 调用高层奖励计算函数
        start_goal_distance=context.start_goal_distance,  # 开始目标距离
        end_goal_distance=context.last_goal_distance,  # 结束目标距离
        subgoal_completed=context.subgoal_completed,  # 子目标完成标志
        reached_goal=reached_goal,  # 是否到达目标
        collision=collision,  # 是否碰撞
        timed_out=timed_out,  # 是否超时
        config=high_cfg,  # 高层奖励配置
        start_window_index=context.start_window_index,  # 开始窗口索引
        end_window_index=context.last_window_index,  # 结束窗口索引
        start_window_distance=context.start_window_distance,  # 开始窗口距离
        best_window_distance=context.best_window_distance,  # 最佳窗口距离
        end_window_distance=context.last_window_distance,  # 结束窗口距离
        window_entered=context.window_entered,  # 窗口进入标志
        window_inside_steps=context.window_inside_steps,  # 窗口内步数
        target_window_index=context.target_window_index,  # 目标窗口索引
        target_window_reached=context.target_window_reached,  # 目标窗口到达标志
        low_level_return=context.low_level_return,  # 低层回报
    )

    # 将经验添加到缓冲区
    buffer.append(  # 向缓冲区添加经验元组
        (
            context.start_state.astype(np.float32, copy=False),  # 开始状态
            context.action.astype(np.float32, copy=False),  # 子目标动作
            float(reward),  # 奖励值
            last_state.astype(np.float32, copy=False),  # 结束状态
            float(done),  # 终止标志
        )
    )

    risk_sample: Optional[Tuple[np.ndarray, np.ndarray, float]] = None  # 风险样本初始化为None

    target_distance = context.min_dmin  # 目标距离为最小障碍距离
    if collision or context.collision_occurred:  # 如果发生碰撞
        target_distance = 0.0  # 目标距离设为0

    if np.isfinite(target_distance):  # 如果目标距离是有限值
        base_distance = float(context.base_distance) if context.base_distance is not None else 0.0  # 基础距离
        base_angle = (
            float(context.base_angle)
            if context.base_angle is not None
            else float(context.subgoal_angle_at_start or 0.0)  # 基础角度
        )
        anchor_radius = float(context.anchor_radius) if context.anchor_radius is not None else 0.0  # 锚点半径
        subgoal_geom = np.array([base_distance, base_angle, anchor_radius], dtype=np.float32)  # 子目标几何信息
        risk_sample = (  # 创建风险样本
            context.start_state.astype(np.float32, copy=False),  # 开始状态
            subgoal_geom,  # 子目标几何信息
            float(target_distance),  # 目标距离
        )

    return components, risk_sample  # 返回奖励组件和风险样本


def maybe_train_high_level(
    planner,
    buffer: List[Tuple[np.ndarray, np.ndarray, float, np.ndarray, float]],
    batch_size: int,
) -> Optional[dict]:
    """当缓存样本足够时触发一次高层更新.

    Args:
        planner: 高层规划器
        buffer: 高层经验缓冲区
        batch_size: 批次大小
        
    Returns:
        训练指标字典或None
    """

    # 检查缓冲区是否足够
    if len(buffer) < batch_size:  # 如果缓冲区样本数小于批次大小
        return None  # 返回None

    # 提取批次数据
    batch = buffer[:batch_size]  # 取前batch_size个样本
    del buffer[:batch_size]  # 移除已使用的样本

    # 组织批次数据
    states = np.stack([entry[0] for entry in batch])  # 堆叠状态
    actions = np.stack([entry[1] for entry in batch])  # 堆叠动作
    rewards = np.array([entry[2] for entry in batch], dtype=np.float32)  # 奖励数组
    next_states = np.stack([entry[3] for entry in batch])  # 下一状态数组
    dones = np.array([entry[4] for entry in batch], dtype=np.float32)  # 终止标志数组

    # 更新规划器
    metrics = planner.update_planner(states, actions, rewards, dones, next_states, batch_size=batch_size)  # 更新高层规划器
    return metrics  # 返回训练指标


class TD3ReplayAdapter:
    """匹配控制器期望的回放缓冲区API的薄包装器"""

    def __init__(self, buffer_size: int, random_seed: int = 666) -> None:
        """初始化回放缓冲区适配器"""
        self._buffer = ReplayBuffer(buffer_size=buffer_size, random_seed=random_seed)  # 创建回放缓冲区

    def add(self, state, action, reward, done, next_state) -> None:
        """向缓冲区添加经验"""
        state_arr = np.asarray(state, dtype=np.float32)  # 状态数组
        action_arr = np.asarray(action, dtype=np.float32)  # 动作数组
        next_state_arr = np.asarray(next_state, dtype=np.float32)  # 下一状态数组
        reward_val = float(reward)  # 奖励值
        done_val = float(done)  # 终止标志
        self._buffer.add(state_arr, action_arr, reward_val, done_val, next_state_arr)  # 添加到缓冲区

    def size(self) -> int:
        """返回缓冲区当前大小"""
        return self._buffer.size()  # 返回缓冲区大小

    def sample(self, batch_size: int):
        """从缓冲区采样批次数据"""
        states, actions, rewards, dones, next_states = self._buffer.sample_batch(batch_size)  # 采样批次数据
        return states, actions, rewards, dones, next_states  # 返回采样数据

    def clear(self) -> None:
        """清空缓冲区"""
        self._buffer.clear()  # 清空缓冲区


def get_robot_pose(sim: SIM) -> Tuple[float, float, float]:
    """从IR-Sim包装器中提取机器人位姿并返回(x, y, theta)

    Args:
        sim: 仿真环境实例
        
    Returns:
        机器人位姿 (x, y, theta)
    """

    robot_state = sim.env.get_robot_state()  # 获取机器人状态
    return (
        float(robot_state[0].item()),  # x坐标
        float(robot_state[1].item()),  # y坐标
        float(robot_state[2].item()),  # 航向角theta
    )


def get_goal_pose(sim: SIM) -> Tuple[float, float, float]:
    """返回仿真环境中当前目标位姿 (x, y, theta)."""

    goal = sim.env.robot.goal  # 获取目标状态
    return (
        float(goal[0].item()),  # 目标x坐标
        float(goal[1].item()),  # 目标y坐标
        float(goal[2].item()) if len(goal) > 2 else 0.0,  # 目标角度（如果存在）
    )


def evaluate(
    system: HierarchicalNavigationSystem,
    sim: SIM,
    config: TrainingConfig,
    epoch: int,
    low_cfg: LowLevelRewardConfig,
) -> None:
    """运行无探索噪声的评估 rollout 并记录汇总统计信息.

    Args:
        system: 分层导航系统
        sim: 仿真环境
        config: 训练配置
        epoch: 当前轮数
        low_cfg: 低层奖励配置
    """

    print("\n" + "=" * 60)  # 打印分隔线
    print(f"🎯 EPOCH {epoch:03d} EVALUATION")  # 打印评估标题
    print("=" * 60)

    # 初始化评估统计
    total_reward = 0.0  # 总奖励
    total_steps = 0  # 总步数
    collision_count = 0  # 碰撞次数
    goal_count = 0  # 到达目标次数
    timeout_count = 0  # 超时次数
    episode_rewards: List[float] = []  # 情节奖励列表
    episode_lengths: List[int] = []  # 情节长度列表
    episode_success_flags: List[bool] = []  # 情节成功标志列表

    # 运行评估情节
    for ep_idx in range(config.eval_episodes):  # 遍历每个评估情节
        system.reset()  # 重置系统状态
        latest_scan, distance, cos, sin, collision, goal, prev_action, _ = sim.reset()  # 重置仿真环境
        prev_action = [0.0, 0.0]  # 初始化动作
        current_subgoal_world: Optional[np.ndarray] = None  # 当前子目标世界坐标
        robot_pose = get_robot_pose(sim)  # 获取机器人位姿
        eval_goal_pose = get_goal_pose(sim)  # 获取评估目标位姿
        system.plan_global_route(robot_pose, eval_goal_pose, force=True)  # 强制规划全局路径
        done = False  # 终止标志
        steps = 0  # 步数计数器
        episode_reward = 0.0  # 情节奖励
        current_subgoal_completed = False  # 当前子目标完成标志

        # 单次评估情节循环
        while not done and steps < config.max_steps:  # 当未终止且未超时时循环
            robot_pose = get_robot_pose(sim)  # 获取机器人位姿
            system.plan_global_route(robot_pose, eval_goal_pose)  # 规划全局路径
            active_waypoints = system.get_active_waypoints(robot_pose, include_indices=True)  # 获取活动航点
            window_metrics = system.update_window_state(robot_pose, active_waypoints)  # 更新窗口状态
            goal_info = [distance, cos, sin]  # 目标信息

            # 检查是否需要重新规划
            should_replan = (
                system.high_level_planner.current_subgoal_world is None  # 没有当前子目标
                or system.high_level_planner.check_triggers(  # 或触发器条件满足
                    latest_scan,  # 最新激光数据
                    robot_pose,  # 机器人位姿
                    goal_info,  # 目标信息
                    prev_action=prev_action,  # 上次动作
                    current_step=steps,  # 当前步数
                    window_metrics=window_metrics,  # 窗口指标
                )
            )
            if window_metrics.get("limit_exceeded", False):  # 如果窗口限制超限
                should_replan = True  # 需要重新规划

            subgoal_distance: Optional[float] = None  # 子目标距离
            subgoal_angle: Optional[float] = None  # 子目标角度
            metadata = {}  # 元数据字典

            if should_replan:  # 如果需要重新规划
                # 生成新子目标
                subgoal_distance, subgoal_angle, metadata = system.high_level_planner.generate_subgoal(
                    latest_scan,  # 激光数据
                    distance,  # 目标距离
                    cos,  # 目标余弦
                    sin,  # 目标正弦
                    prev_action=prev_action,  # 上次动作
                    robot_pose=robot_pose,  # 机器人位姿
                    current_step=steps,  # 当前步数
                    waypoints=active_waypoints,  # 活动航点
                    window_metrics=window_metrics,  # 窗口指标
                )
                system.reset_window_tracking()  # 重置窗口跟踪
                system.update_selected_waypoint(metadata.get("selected_waypoint"))  # 更新选择的航点
                planner_world = system.high_level_planner.current_subgoal_world  # 规划器子目标世界坐标
                current_subgoal_world = np.asarray(planner_world, dtype=np.float32) if planner_world is not None else None  # 当前子目标世界坐标
                system.high_level_planner.event_trigger.reset_time(steps)  # 重置事件触发器时间
                if current_subgoal_world is None:  # 如果没有子目标世界坐标
                    current_subgoal_world = compute_subgoal_world(robot_pose, subgoal_distance, subgoal_angle)  # 计算子目标世界坐标
                current_subgoal_completed = False  # 重置子目标完成标志
            else:
                planner_world = system.high_level_planner.current_subgoal_world  # 规划器子目标世界坐标
                if planner_world is not None:  # 如果存在子目标世界坐标
                    current_subgoal_world = np.asarray(planner_world, dtype=np.float32)  # 更新当前子目标世界坐标

            system.current_subgoal_world = current_subgoal_world  # 设置系统当前子目标世界坐标

            relative_geometry = system.high_level_planner.get_relative_subgoal(robot_pose)  # 获取相对子目标
            if relative_geometry[0] is None:  # 如果没有相对几何信息
                if should_replan and subgoal_distance is not None and subgoal_angle is not None:  # 如果需要重新规划且有子目标信息
                    relative_geometry = (subgoal_distance, subgoal_angle)  # 使用新生成的子目标
                elif system.current_subgoal is not None:  # 如果有当前子目标
                    relative_geometry = system.current_subgoal  # 使用当前子目标
                else:
                    relative_geometry = (0.0, 0.0)  # 默认值

            subgoal_distance, subgoal_angle = float(relative_geometry[0]), float(relative_geometry[1])  # 更新子目标距离和角度
            system.current_subgoal = (subgoal_distance, subgoal_angle)  # 设置系统当前子目标

            # 计算子目标距离
            prev_subgoal_distance = None  # 前一个子目标距离
            if current_subgoal_world is not None:  # 如果有当前子目标世界坐标
                prev_pos = np.array(robot_pose[:2], dtype=np.float32)  # 前一个位置
                prev_subgoal_distance = float(np.linalg.norm(prev_pos - current_subgoal_world))  # 计算前一个子目标距离

            # 处理低层观测
            state = system.low_level_controller.process_observation(  # 处理低层观测
                latest_scan,  # 激光数据
                subgoal_distance,  # 子目标距离
                subgoal_angle,  # 子目标角度
                prev_action,  # 上次动作
            )

            # 预测动作（无探索噪声）
            action = system.low_level_controller.predict_action(state, add_noise=False)  # 预测动作（无噪声）
            lin_cmd = float(np.clip((action[0] + 1.0) / 4.0, 0.0, config.max_lin_velocity))  # 线性速度命令
            ang_cmd = float(np.clip(action[1], -config.max_ang_velocity, config.max_ang_velocity))  # 角速度命令
            lin_cmd, ang_cmd = system.apply_velocity_shielding(lin_cmd, ang_cmd, latest_scan)  # 应用速度屏蔽

            # 执行动作
            latest_scan, distance, cos, sin, collision, goal, _, _ = sim.step(  # 执行一步仿真
                lin_velocity=lin_cmd,  # 线性速度
                ang_velocity=ang_cmd,  # 角速度
            )

            # 更新子目标距离
            next_pose = get_robot_pose(sim)  # 获取下一时刻机器人位姿
            system.plan_global_route(next_pose, eval_goal_pose)  # 规划全局路径
            next_waypoints = system.get_active_waypoints(next_pose, include_indices=True)  # 获取下一时刻活动航点
            post_window_metrics = system.update_window_state(next_pose, next_waypoints)  # 更新窗口状态
            current_subgoal_distance = None  # 当前子目标距离
            if current_subgoal_world is not None:  # 如果有当前子目标世界坐标
                next_pos = np.array(next_pose[:2], dtype=np.float32)  # 下一时刻位置
                current_subgoal_distance = float(np.linalg.norm(next_pos - current_subgoal_world))  # 计算当前子目标距离

            relative_after = system.high_level_planner.get_relative_subgoal(next_pose)  # 获取下一时刻相对子目标
            subgoal_alignment_angle: Optional[float] = None  # 子目标对齐角度
            if relative_after[0] is not None:  # 如果有相对几何信息
                subgoal_alignment_angle = float(relative_after[1])  # 子目标对齐角度
                if current_subgoal_distance is None:  # 如果没有当前子目标距离
                    current_subgoal_distance = float(relative_after[0])  # 使用相对距离

            action_delta: Optional[List[float]] = None  # 动作变化量
            if prev_action is not None:  # 如果有上次动作
                delta_lin = float(lin_cmd - prev_action[0])  # 线性速度变化
                delta_ang = float(ang_cmd - prev_action[1])  # 角速度变化
                action_delta = [delta_lin, delta_ang]  # 动作变化量

            # 计算最小障碍物距离
            scan_arr = np.asarray(latest_scan, dtype=np.float32)  # 激光数据数组
            finite_scan = scan_arr[np.isfinite(scan_arr)]  # 有限值扫描
            min_obstacle_distance = float(finite_scan.min()) if finite_scan.size else 8.0  # 最小障碍距离
            # 检查终止条件
            just_reached_subgoal = False  # 刚刚到达子目标标志
            if not current_subgoal_completed:  # 如果当前子目标未完成
                if (
                    current_subgoal_distance is not None  # 如果有当前子目标距离
                    and current_subgoal_distance <= config.subgoal_radius  # 且距离小于子目标半径
                ):
                    if prev_subgoal_distance is None:  # 如果前一个子目标距离为None
                        just_reached_subgoal = True  # 标记为刚刚到达
                    elif prev_subgoal_distance > config.subgoal_radius:  # 如果前一个距离大于半径
                        just_reached_subgoal = True  # 标记为刚刚到达
            else:
                just_reached_subgoal = False  # 否则未到达
            if just_reached_subgoal:  # 如果刚刚到达子目标
                current_subgoal_completed = True  # 标记子目标完成
            timed_out = steps == config.max_steps - 1 and not (goal or collision)  # 超时判断

            # 计算低层奖励
            low_reward, _ = compute_low_level_reward(  # 计算低层奖励
                prev_subgoal_distance=prev_subgoal_distance,  # 前一个子目标距离
                current_subgoal_distance=current_subgoal_distance,  # 当前子目标距离
                min_obstacle_distance=min_obstacle_distance,  # 最小障碍距离
                reached_goal=goal,  # 是否到达目标
                reached_subgoal=just_reached_subgoal,  # 是否到达子目标
                collision=collision,  # 是否碰撞
                timed_out=timed_out,  # 是否超时
                window_entered=post_window_metrics.get("entered", False),  # 窗口进入标志
                window_inside=post_window_metrics.get("inside", False),  # 窗口内部标志
                window_limit_exceeded=post_window_metrics.get("limit_exceeded", False),  # 窗口限制超限
                prev_window_distance=post_window_metrics.get("prev_distance"),  # 前一个窗口距离
                current_window_distance=post_window_metrics.get("distance"),  # 当前窗口距离
                window_radius=post_window_metrics.get("radius"),  # 窗口半径
                current_subgoal_angle=subgoal_alignment_angle,  # 当前子目标角度
                action_delta=action_delta,  # 动作变化量
                config=low_cfg,  # 低层奖励配置
            )

            # 更新统计
            episode_reward += low_reward  # 累加情节奖励
            steps += 1  # 步数加1
            prev_action = [lin_cmd, ang_cmd]  # 更新上次动作

            # 检查终止
            if collision:  # 如果碰撞
                collision_count += 1  # 碰撞计数加1
                done = True  # 标记终止
            elif goal:  # 如果到达目标
                goal_count += 1  # 目标计数加1
                done = True  # 标记终止
            elif steps >= config.max_steps:  # 如果达到最大步数
                timeout_count += 1  # 超时计数加1
                done = True  # 标记终止

        # 记录情节结果
        episode_rewards.append(episode_reward)  # 添加情节奖励
        episode_lengths.append(steps)  # 添加情节长度
        episode_success_flags.append(goal)  # 添加成功标志
        total_reward += episode_reward  # 累加总奖励
        total_steps += steps  # 累加总步数

        status = "🎯" if goal else "💥" if collision else "⏰"  # 状态表情
        print(
            f"   Evaluation Episode {ep_idx + 1:2d}/{config.eval_episodes}: {status} | "
            f"Steps: {steps:3d} | Reward: {episode_reward:7.1f}"  # 打印评估结果
        )

    # 计算汇总统计
    avg_reward = total_reward / config.eval_episodes  # 平均奖励
    avg_steps = total_steps / config.eval_episodes  # 平均步数
    success_rate = goal_count / config.eval_episodes * 100  # 成功率
    collision_rate = collision_count / config.eval_episodes * 100  # 碰撞率
    timeout_rate = timeout_count / config.eval_episodes * 100  # 超时率

    reward_std = np.std(episode_rewards) if config.eval_episodes > 1 else 0.0  # 奖励标准差
    steps_std = np.std(episode_lengths) if config.eval_episodes > 1 else 0.0  # 步数标准差

    # 输出评估结果
    print("\n📈 Performance Summary:")  # 性能总结标题
    print(f"   • Success Rate:      {success_rate:6.1f}% ({goal_count:2d}/{config.eval_episodes:2d})")  # 成功率
    print(f"   • Collision Rate:    {collision_rate:6.1f}% ({collision_count:2d}/{config.eval_episodes:2d})")  # 碰撞率
    print(f"   • Timeout Rate:      {timeout_rate:6.1f}% ({timeout_count:2d}/{config.eval_episodes:2d})")  # 超时率
    print(f"   • Average Reward:    {avg_reward:8.2f} ± {reward_std:.2f}")  # 平均奖励
    print(f"   • Average Steps:     {avg_steps:8.1f} ± {steps_std:.1f}")  # 平均步数

    if goal_count > 0:  # 如果有成功的情节
        successful_rewards = [r for r, success in zip(episode_rewards, episode_success_flags) if success]  # 成功情节奖励
        avg_success_reward = np.mean(successful_rewards) if successful_rewards else 0.0  # 平均成功奖励
        print(f"   • Avg Success Reward: {avg_success_reward:8.2f}")  # 打印平均成功奖励

    print("-" * 60)  # 分隔线
    print(f"⏰ Evaluation completed: {config.eval_episodes} episodes")  # 评估完成信息
    print("=" * 60)  # 分隔线

    # 记录到TensorBoard
    writer = system.low_level_controller.writer  # 获取TensorBoard写入器
    writer.add_scalar("eval/success_rate", success_rate, epoch)  # 记录成功率
    writer.add_scalar("eval/collision_rate", collision_rate, epoch)  # 记录碰撞率
    writer.add_scalar("eval/timeout_rate", timeout_rate, epoch)  # 记录超时率
    writer.add_scalar("eval/avg_reward", avg_reward, epoch)  # 记录平均奖励
    writer.add_scalar("eval/avg_steps", avg_steps, epoch)  # 记录平均步数
    writer.add_scalar("eval/reward_std", reward_std, epoch)  # 记录奖励标准差
    writer.add_scalar("eval_raw/success_count", goal_count, epoch)  # 记录成功计数
    writer.add_scalar("eval_raw/collision_count", collision_count, epoch)  # 记录碰撞计数


def main(args=None):
    """ETHSRL+GP的主要训练循环"""

    # ========== 训练配置与设备初始化 ==========
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设置设备
    bundle = ConfigBundle()  # 配置包
    config = bundle.training  # 训练配置
    integration_config = bundle.integration  # 集成配置
    safety_cfg = bundle.safety_critic  # 安全评估配置

    raw_world = Path(config.world_file)  # 世界文件路径
    base_dir = Path(__file__).resolve().parent  # 基础目录
    candidate_paths: List[Path] = []  # 候选路径列表
    if raw_world.is_absolute():  # 如果是绝对路径
        candidate_paths.append(raw_world)  # 添加绝对路径
    else:
        candidate_paths.extend(  # 添加相对路径候选
            [
                base_dir / raw_world,  # 基础目录下的路径
                base_dir / "worlds" / raw_world,  # worlds目录下的路径
                base_dir.parent / "robot_nav" / "worlds" / raw_world,  # 父目录下的路径
            ]
        )

    world_path: Optional[Path] = None  # 世界文件路径
    for candidate in candidate_paths:  # 遍历候选路径
        if candidate.exists():  # 如果路径存在
            world_path = candidate.resolve()  # 设置世界文件路径
            break

    if world_path is None:  # 如果未找到世界文件
        search_list = ", ".join(str(p) for p in candidate_paths)  # 搜索列表
        raise FileNotFoundError(  # 抛出文件未找到异常
            f"Unable to locate world file '{config.world_file}'. Checked: {search_list}"
        )

    world_path_str = str(world_path)  # 世界文件路径字符串

    # ========== 训练初始化日志 ==========
    print("\n" + "="*60)  # 分隔线
    print("🚀 Starting ETHSRL+GP Hierarchical Navigation Training")  # 训练开始标题
    print("="*60)
    print(f"📋 Training Configuration:")  # 训练配置标题
    print(f"   • Device: {device}")  # 设备信息
    print(
        f"   • Max epochs: {config.max_epochs}, Episodes per epoch: {config.episodes_per_epoch}"  # 最大轮次和每轮情节数
    )
    print(
        f"   • Training iterations: {config.training_iterations}, Batch size: {config.batch_size}"  # 训练迭代次数和批次大小
    )
    print(f"   • Max steps per episode: {config.max_steps}")  # 每情节最大步数
    print(f"   • Train every {config.train_every_n_episodes} episodes")  # 训练频率
    print(f"   • World file: {world_path}")  # 世界文件路径
    print(
        "   • Global planner: res={:.2f} m, margin={:.2f} m, lookahead={}".format(  # 全局规划器参数
            config.global_plan_resolution,  # 分辨率
            config.global_plan_margin,  # 安全边界
            config.waypoint_lookahead,  # 前瞻航点数
        )
    )
    if config.save_every > 0:  # 如果设置了保存频率
        print(f"   • Save models every {config.save_every} episodes")  # 保存模型频率
    else:
        print("   • Save models at end of training only")  # 仅在训练结束时保存
    print("="*60)

    # ========== 系统初始化 ==========
    print("🔄 Initializing ETHSRL+GP system...")  # 系统初始化信息
    system = HierarchicalNavigationSystem(  # 创建分层导航系统
        device=device,  # 设备
        subgoal_threshold=config.subgoal_radius,  # 子目标阈值
        world_file=world_path,  # 世界文件
        global_plan_resolution=config.global_plan_resolution,  # 全局规划分辨率
        global_plan_margin=config.global_plan_margin,  # 全局规划安全边界
        waypoint_lookahead=config.waypoint_lookahead,  # 航点前瞻数量
        integration_config=integration_config,  # 集成配置
    )
    replay_buffer = TD3ReplayAdapter(  # 创建回放缓冲区适配器
        buffer_size=config.buffer_size,  # 缓冲区大小
        random_seed=config.random_seed or 666,  # 随机种子
    )
    print("✅ System initialization completed")  # 系统初始化完成

    # ========== 环境初始化 ==========
    print("🔄 Initializing simulation environment...")  # 环境初始化信息
    sim = SIM(world_file=world_path_str, disable_plotting=False)  # 创建仿真环境
    print("✅ Environment initialization completed")  # 环境初始化完成

    # ========== 训练统计变量初始化 ==========
    episode_reward = 0.0  # 情节奖励
    epoch_total_reward = 0.0  # 轮次总奖励
    epoch_total_steps = 0  # 轮次总步数
    epoch_goal_count = 0  # 轮次目标计数
    epoch_collision_count = 0  # 轮次碰撞计数

    # 训练计数器初始化
    episode = 0  # 情节计数器
    epoch = 0  # 轮次计数器

    print("\n🎬 Starting main training loop...")  # 开始主训练循环
    print("-" * 50)  # 分隔线

    # 奖励配置初始化
    low_reward_cfg = bundle.low_level_reward  # 低层奖励配置
    high_reward_cfg = bundle.high_level_reward  # 高层奖励配置
    high_level_buffer: List[Tuple[np.ndarray, np.ndarray, float, np.ndarray, float]] = []  # 高层缓冲区
    current_subgoal_context: Optional[SubgoalContext] = None  # 当前子目标上下文

    # ========== 主训练循环 ==========
    while epoch < config.max_epochs:  # 当轮次小于最大轮次时循环
        # 重置环境和系统状态
        system.reset()  # 重置系统
        current_subgoal_context = None  # 重置子目标上下文
        system.current_subgoal = None  # 重置当前子目标

        latest_scan, distance, cos, sin, collision, goal, prev_action, _ = sim.reset()  # 重置仿真环境
        prev_action = [0.0, 0.0]  # 重置动作
        current_subgoal_world: Optional[np.ndarray] = None  # 当前子目标世界坐标

        robot_pose = get_robot_pose(sim)  # 获取机器人位姿
        episode_goal_pose = get_goal_pose(sim)  # 获取情节目标位姿
        system.plan_global_route(robot_pose, episode_goal_pose, force=True)  # 强制规划全局路径

        steps = 0  # 步数计数器
        episode_reward = 0.0  # 情节奖励
        done = False  # 终止标志
        current_subgoal_completed = False  # 当前子目标完成标志

        # ========== 单次情节循环 ==========
        while not done and steps < config.max_steps:  # 当未终止且未超时时循环
            robot_pose = get_robot_pose(sim)  # 获取机器人位姿
            system.plan_global_route(robot_pose, episode_goal_pose)  # 规划全局路径
            active_waypoints = system.get_active_waypoints(robot_pose, include_indices=True)  # 获取活动航点
            window_metrics = system.update_window_state(robot_pose, active_waypoints)  # 更新窗口状态
            waypoint_sequence = active_waypoints  # 航点序列
            goal_info = [distance, cos, sin]  # 目标信息

            # 检查是否需要重新规划子目标
            should_replan = (
                system.high_level_planner.current_subgoal_world is None  # 没有当前子目标
                or system.high_level_planner.check_triggers(  # 或触发器条件满足
                    latest_scan,  # 最新激光数据
                    robot_pose,  # 机器人位姿
                    goal_info,  # 目标信息
                    prev_action=prev_action,  # 上次动作
                    current_step=steps,  # 当前步数
                    window_metrics=window_metrics,  # 窗口指标
                )
            )
            if window_metrics.get("limit_exceeded", False):  # 如果窗口限制超限
                should_replan = True  # 需要重新规划

            metadata = {}  # 元数据字典
            subgoal_distance = None  # 子目标距离
            subgoal_angle = None  # 子目标角度

            if should_replan:  # 如果需要重新规划
                # 完成当前子目标并训练
                finalize_result = finalize_subgoal_transition(  # 完成子目标转换
                    current_subgoal_context,  # 当前子目标上下文
                    high_level_buffer,  # 高层缓冲区
                    high_reward_cfg,  # 高层奖励配置
                    done=False,  # 未终止
                    reached_goal=False,  # 未到达目标
                    collision=False,  # 未碰撞
                    timed_out=False,  # 未超时
                )
                if finalize_result is not None:  # 如果有结果
                    finalize_components, risk_sample = finalize_result  # 解包结果
                    if risk_sample is not None:  # 如果有风险样本
                        system.high_level_planner.store_safety_sample(*risk_sample)  # 存储安全样本
                        system.high_level_planner.maybe_update_safety_critic(  # 可能更新安全评估器
                            batch_size=safety_cfg.update_batch_size  # 批次大小
                        )
                    metrics = maybe_train_high_level(  # 可能训练高层
                        system.high_level_planner,  # 高层规划器
                        high_level_buffer,  # 高层缓冲区
                        config.batch_size,  # 批次大小
                    )
                    if metrics:  # 如果有训练指标
                        # 记录训练指标
                        for key, value in metrics.items():  # 遍历指标
                            system.high_level_planner.writer.add_scalar(  # 记录标量
                                f"planner/{key}",  # 指标名称
                                value,  # 指标值
                                system.high_level_planner.iter_count,  # 迭代计数
                            )

                # 生成新子目标
                subgoal_distance, subgoal_angle, metadata = system.high_level_planner.generate_subgoal(  # 生成子目标
                    latest_scan,  # 激光数据
                    distance,  # 目标距离
                    cos,  # 目标余弦
                    sin,  # 目标正弦
                    prev_action=prev_action,  # 上次动作
                    robot_pose=robot_pose,  # 机器人位姿
                    current_step=steps,  # 当前步数
                    waypoints=active_waypoints,  # 活动航点
                    window_metrics=window_metrics,  # 窗口指标
                )
                system.reset_window_tracking()  # 重置窗口跟踪
                system.update_selected_waypoint(metadata.get("selected_waypoint"))  # 更新选择的航点
                planner_world = system.high_level_planner.current_subgoal_world  # 规划器子目标世界坐标
                current_subgoal_world = np.asarray(planner_world, dtype=np.float32) if planner_world is not None else None  # 当前子目标世界坐标
                system.high_level_planner.event_trigger.reset_time(steps)  # 重置事件触发器时间
                if current_subgoal_world is None:  # 如果没有子目标世界坐标
                    current_subgoal_world = compute_subgoal_world(robot_pose, subgoal_distance, subgoal_angle)  # 计算子目标世界坐标

                # 构建高层状态向量
                start_state = system.high_level_planner.build_state_vector(  # 构建状态向量
                    latest_scan,  # 激光数据
                    distance,  # 目标距离
                    cos,  # 目标余弦
                    sin,  # 目标正弦
                    prev_action,  # 上次动作
                    waypoints=waypoint_sequence,  # 航点序列
                    robot_pose=robot_pose,  # 机器人位姿
                )

                # 创建新的子目标上下文
                meta_metrics = metadata.get("window_metrics", {}) if metadata else {}  # 元数据指标
                start_window_index = meta_metrics.get("index")  # 开始窗口索引
                start_window_distance = meta_metrics.get("distance")  # 开始窗口距离
                target_window_index = metadata.get("selected_waypoint")  # 目标窗口索引
                distance_adjust_action = float(metadata.get("distance_adjust_applied", 0.0)) if metadata else 0.0  # 距离调整动作
                angle_offset_action = float(metadata.get("angle_offset_applied", 0.0)) if metadata else 0.0  # 角度偏移动作
                anchor_distance = metadata.get("anchor_distance", subgoal_distance)  # 锚点距离
                anchor_angle = metadata.get("anchor_angle", subgoal_angle)  # 锚点角度
                anchor_radius = metadata.get("anchor_radius") if metadata else None  # 锚点半径

                current_subgoal_context = SubgoalContext(  # 创建子目标上下文
                    start_state=start_state.astype(np.float32, copy=False),  # 开始状态
                    action=np.array([distance_adjust_action, angle_offset_action], dtype=np.float32),  # 动作
                    world_target=current_subgoal_world,  # 世界目标
                    start_goal_distance=distance,  # 开始目标距离
                    last_goal_distance=distance,  # 最后目标距离
                    low_level_return=0.0,  # 低层回报
                    steps=0,  # 步数
                    subgoal_completed=False,  # 子目标完成标志
                    last_state=start_state.astype(np.float32, copy=False),  # 最后状态
                    start_window_index=int(start_window_index) if start_window_index is not None else None,  # 开始窗口索引
                    target_window_index=int(target_window_index) if target_window_index is not None else None,  # 目标窗口索引
                    start_window_distance=float(start_window_distance) if start_window_distance is not None else None,  # 开始窗口距离
                    last_window_index=int(start_window_index) if start_window_index is not None else None,  # 最后窗口索引
                    last_window_distance=float(start_window_distance) if start_window_distance is not None else None,  # 最后窗口距离
                    best_window_distance=float(start_window_distance) if start_window_distance is not None else None,  # 最佳窗口距离
                    subgoal_angle_at_start=float(subgoal_angle) if subgoal_angle is not None else None,  # 子目标开始角度
                    base_distance=float(anchor_distance) if anchor_distance is not None else None,  # 基础距离
                    base_angle=float(anchor_angle) if anchor_angle is not None else None,  # 基础角度
                    anchor_radius=float(anchor_radius) if anchor_radius is not None else None,  # 锚点半径
                )
                scan_arr = np.asarray(latest_scan, dtype=np.float32)  # 激光数据数组
                finite_scan = scan_arr[np.isfinite(scan_arr)]  # 有限值扫描
                if finite_scan.size:  # 如果有有限值
                    current_subgoal_context.min_dmin = float(min(current_subgoal_context.min_dmin, finite_scan.min()))  # 更新最小障碍距离
                current_subgoal_completed = False  # 重置子目标完成标志
            else:
                planner_world = system.high_level_planner.current_subgoal_world  # 规划器子目标世界坐标
                if planner_world is not None:  # 如果存在子目标世界坐标
                    current_subgoal_world = np.asarray(planner_world, dtype=np.float32)  # 更新当前子目标世界坐标

            system.current_subgoal_world = current_subgoal_world  # 设置系统当前子目标世界坐标

            relative_geometry = system.high_level_planner.get_relative_subgoal(robot_pose)  # 获取相对子目标
            if relative_geometry[0] is None:  # 如果没有相对几何信息
                if should_replan and subgoal_distance is not None and subgoal_angle is not None:  # 如果需要重新规划且有子目标信息
                    relative_geometry = (subgoal_distance, subgoal_angle)  # 使用新生成的子目标
                elif system.current_subgoal is not None:  # 如果有当前子目标
                    relative_geometry = system.current_subgoal  # 使用当前子目标
                else:
                    relative_geometry = (0.0, 0.0)  # 默认值

            subgoal_distance, subgoal_angle = float(relative_geometry[0]), float(relative_geometry[1])  # 更新子目标距离和角度
            system.current_subgoal = (subgoal_distance, subgoal_angle)  # 设置系统当前子目标

            # 计算子目标距离
            prev_subgoal_distance = None  # 前一个子目标距离
            if current_subgoal_world is not None:  # 如果有当前子目标世界坐标
                prev_pos = np.array(robot_pose[:2], dtype=np.float32)  # 前一个位置
                prev_subgoal_distance = float(np.linalg.norm(prev_pos - current_subgoal_world))  # 计算前一个子目标距离

            # 处理低层观测
            state = system.low_level_controller.process_observation(  # 处理低层观测
                latest_scan,  # 激光数据
                subgoal_distance,  # 子目标距离
                subgoal_angle,  # 子目标角度
                prev_action,  # 上次动作
            )

            # 预测动作（带探索噪声）
            action = system.low_level_controller.predict_action(  # 预测动作
                state,
                add_noise=True,  # 添加噪声
                noise_scale=config.exploration_noise,  # 噪声尺度
            )
            action = np.clip(action, -1.0, 1.0)  # 裁剪动作

            # 转换为实际控制命令（未屏蔽的环境动作）
            env_lin_cmd = float(np.clip((action[0] + 1.0) / 4.0, 0.0, config.max_lin_velocity))  # 线性速度命令
            env_ang_cmd = float(np.clip(action[1], -config.max_ang_velocity, config.max_ang_velocity))  # 角速度命令
            lin_cmd, ang_cmd = system.apply_velocity_shielding(env_lin_cmd, env_ang_cmd, latest_scan)  # 应用速度屏蔽

            # 执行动作
            latest_scan, distance, cos, sin, collision, goal, executed_action, _ = sim.step(  # 执行一步仿真
                lin_velocity=lin_cmd,  # 线性速度
                ang_velocity=ang_cmd,  # 角速度
            )

            # 更新子目标距离
            next_pose = get_robot_pose(sim)  # 获取下一时刻机器人位姿
            system.plan_global_route(next_pose, episode_goal_pose)  # 规划全局路径
            next_active_waypoints = system.get_active_waypoints(next_pose, include_indices=True)  # 获取下一时刻活动航点
            post_window_metrics = system.update_window_state(next_pose, next_active_waypoints)  # 更新窗口状态
            current_subgoal_distance = None  # 当前子目标距离
            if current_subgoal_world is not None:  # 如果有当前子目标世界坐标
                next_pos = np.array(next_pose[:2], dtype=np.float32)  # 下一时刻位置
                current_subgoal_distance = float(np.linalg.norm(next_pos - current_subgoal_world))  # 计算当前子目标距离

            relative_after = system.high_level_planner.get_relative_subgoal(next_pose)  # 获取下一时刻相对子目标
            subgoal_alignment_angle: Optional[float] = None  # 子目标对齐角度
            if relative_after[0] is not None:  # 如果有相对几何信息
                subgoal_alignment_angle = float(relative_after[1])  # 子目标对齐角度
                if current_subgoal_distance is None:  # 如果没有当前子目标距离
                    current_subgoal_distance = float(relative_after[0])  # 使用相对距离

            # Refresh subgoal geometry using post-step pose so replay stores t+1 state.
            if system.current_subgoal is not None:  # 如果有系统当前子目标
                post_subgoal_distance = float(system.current_subgoal[0])  # 后子目标距离
                post_subgoal_angle = float(system.current_subgoal[1])  # 后子目标角度
            else:
                post_subgoal_distance = float(subgoal_distance) if subgoal_distance is not None else 0.0  # 后子目标距离
                post_subgoal_angle = float(subgoal_angle) if subgoal_angle is not None else 0.0  # 后子目标角度

            if relative_after[0] is not None:  # 如果有相对几何信息
                post_subgoal_distance = float(relative_after[0])  # 使用相对距离
                post_subgoal_angle = float(relative_after[1])  # 使用相对角度
            else:
                if current_subgoal_distance is not None:  # 如果有当前子目标距离
                    post_subgoal_distance = float(current_subgoal_distance)  # 使用当前子目标距离
                if subgoal_alignment_angle is not None:  # 如果有子目标对齐角度
                    post_subgoal_angle = float(subgoal_alignment_angle)  # 使用子目标对齐角度

            system.current_subgoal = (post_subgoal_distance, post_subgoal_angle)  # 设置系统当前子目标

            action_delta: Optional[List[float]] = None  # 动作变化量
            if executed_action is not None and prev_action is not None:  # 如果有执行动作和上次动作
                delta_lin = float(executed_action[0] - prev_action[0])  # 线性速度变化
                delta_ang = float(executed_action[1] - prev_action[1])  # 角速度变化
                action_delta = [delta_lin, delta_ang]  # 动作变化量

            # 计算最小障碍物距离
            scan_arr = np.asarray(latest_scan, dtype=np.float32)  # 激光数据数组
            finite_scan = scan_arr[np.isfinite(scan_arr)]  # 有限值扫描
            min_obstacle_distance = float(finite_scan.min()) if finite_scan.size else 8.0  # 最小障碍距离
            if current_subgoal_context is not None:  # 如果有当前子目标上下文
                current_subgoal_context.min_dmin = min(  # 更新最小障碍距离
                    current_subgoal_context.min_dmin,
                    min_obstacle_distance,
                )
                if collision:  # 如果碰撞
                    current_subgoal_context.collision_occurred = True  # 标记碰撞发生

            # 检查终止条件
            just_reached_subgoal = False  # 刚刚到达子目标标志
            if not current_subgoal_completed:  # 如果当前子目标未完成
                if (
                    current_subgoal_distance is not None  # 如果有当前子目标距离
                    and current_subgoal_distance <= config.subgoal_radius  # 且距离小于子目标半径
                ):
                    if prev_subgoal_distance is None:  # 如果前一个子目标距离为None
                        just_reached_subgoal = True  # 标记为刚刚到达
                    elif prev_subgoal_distance > config.subgoal_radius:  # 如果前一个距离大于半径
                        just_reached_subgoal = True  # 标记为刚刚到达
            else:
                just_reached_subgoal = False  # 否则未到达
            if just_reached_subgoal:  # 如果刚刚到达子目标
                current_subgoal_completed = True  # 标记子目标完成
            timed_out = steps == config.max_steps - 1 and not (goal or collision)  # 超时判断

            # 计算低层奖励
            low_reward, _ = compute_low_level_reward(  # 计算低层奖励
                prev_subgoal_distance=prev_subgoal_distance,  # 前一个子目标距离
                current_subgoal_distance=current_subgoal_distance,  # 当前子目标距离
                min_obstacle_distance=min_obstacle_distance,  # 最小障碍距离
                reached_goal=goal,  # 是否到达目标
                reached_subgoal=just_reached_subgoal,  # 是否到达子目标
                collision=collision,  # 是否碰撞
                timed_out=timed_out,  # 是否超时
                window_entered=post_window_metrics.get("entered", False),  # 窗口进入标志
                window_inside=post_window_metrics.get("inside", False),  # 窗口内部标志
                window_limit_exceeded=post_window_metrics.get("limit_exceeded", False),  # 窗口限制超限
                prev_window_distance=post_window_metrics.get("prev_distance"),  # 前一个窗口距离
                current_window_distance=post_window_metrics.get("distance"),  # 当前窗口距离
                window_radius=post_window_metrics.get("radius"),  # 窗口半径
                current_subgoal_angle=subgoal_alignment_angle,  # 当前子目标角度
                action_delta=action_delta,  # 动作变化量
                config=low_reward_cfg,  # 低层奖励配置
            )

            # 更新奖励统计
            episode_reward += low_reward  # 累加情节奖励
            epoch_total_reward += low_reward  # 累加轮次总奖励
            epoch_total_steps += 1  # 累加轮次总步数

            # 更新子目标上下文
            if current_subgoal_context is not None:  # 如果有当前子目标上下文
                current_subgoal_context.low_level_return += low_reward  # 累加低层回报
                current_subgoal_context.steps += 1  # 累加步数
                current_subgoal_context.subgoal_completed |= just_reached_subgoal  # 更新子目标完成标志
                current_subgoal_context.last_goal_distance = distance  # 更新最后目标距离
                # 构建下一状态向量
                next_active_waypoints = system.get_active_waypoints(next_pose, include_indices=True)  # 获取下一时刻活动航点
                next_state_vector = system.high_level_planner.build_state_vector(  # 构建下一状态向量
                    latest_scan,  # 激光数据
                    distance,  # 目标距离
                    cos,  # 目标余弦
                    sin,  # 目标正弦
                    executed_action,  # 执行动作
                    waypoints=next_active_waypoints,  # 下一时刻活动航点
                    robot_pose=next_pose,  # 下一时刻机器人位姿
                )
                current_subgoal_context.last_state = next_state_vector.astype(np.float32, copy=False)  # 更新最后状态
                idx_metric = post_window_metrics.get("index") if post_window_metrics else None  # 索引指标
                dist_metric = post_window_metrics.get("distance") if post_window_metrics else None  # 距离指标
                if idx_metric is not None:  # 如果有索引指标
                    idx_val = int(idx_metric)  # 索引值
                    current_subgoal_context.last_window_index = idx_val  # 更新最后窗口索引
                    if current_subgoal_context.start_window_index is None:  # 如果开始窗口索引为None
                        current_subgoal_context.start_window_index = idx_val  # 设置开始窗口索引
                    target_idx = current_subgoal_context.target_window_index  # 目标窗口索引
                    if (
                        target_idx is not None  # 如果有目标窗口索引
                        and idx_val >= target_idx  # 且当前索引大于等于目标索引
                        and post_window_metrics.get("inside", False)  # 且在窗口内部
                    ):
                        current_subgoal_context.target_window_reached = True  # 标记目标窗口到达
                if dist_metric is not None:  # 如果有距离指标
                    dist_val = float(dist_metric)  # 距离值
                    current_subgoal_context.last_window_distance = dist_val  # 更新最后窗口距离
                    best = current_subgoal_context.best_window_distance  # 最佳窗口距离
                    if best is None or dist_val < best:  # 如果当前距离更小
                        current_subgoal_context.best_window_distance = dist_val  # 更新最佳窗口距离
                if post_window_metrics.get("entered", False):  # 如果进入窗口
                    current_subgoal_context.window_entered = True  # 标记窗口进入
                if post_window_metrics.get("inside", False):  # 如果在窗口内部
                    current_subgoal_context.window_inside_steps += 1  # 累加窗口内部步数

            # 准备下一状态
            next_prev_action = [executed_action[0], executed_action[1]]  # 下一时刻上次动作
            next_state = system.low_level_controller.process_observation(  # 处理下一状态观测
                latest_scan,  # 激光数据
                post_subgoal_distance,  # 后子目标距离
                post_subgoal_angle,  # 后子目标角度
                next_prev_action,  # 下一时刻上次动作
            )

            # 检查终止条件
            done = collision or goal or steps == config.max_steps - 1  # 终止条件

            # 添加经验到回放缓冲区（存储未屏蔽的环境动作）
            scaled_env_action = np.array([env_lin_cmd, env_ang_cmd], dtype=np.float32)
            replay_buffer.add(state, scaled_env_action, low_reward, float(done), next_state)  # 添加到回放缓冲区

            # 定期输出回放缓冲区大小与奖励
            if steps % 50 == 0:  # 每50步输出一次
                buffer_size = replay_buffer.size()  # 缓冲区大小
                print(
                    f"🏃 Training | Epoch {epoch:2d}/{config.max_epochs} | "  # 训练信息
                    f"Episode {episode:3d}/{config.max_epochs*config.episodes_per_epoch} | "
                    f"Step {steps:3d}/{config.max_steps} | "
                    f"Reward: {low_reward:7.2f} | Buffer: {buffer_size:6d}"
                )

            prev_action = next_prev_action  # 更新上次动作
            steps += 1  # 步数加1

        # ========== 情节结束处理 ==========
        timed_out_episode = not goal and not collision and steps >= config.max_steps  # 超时情节判断

        # 完成最后一个子目标
        finalize_result = finalize_subgoal_transition(  # 完成子目标转换
            current_subgoal_context,  # 当前子目标上下文
            high_level_buffer,  # 高层缓冲区
            high_reward_cfg,  # 高层奖励配置
            done=True,  # 终止
            reached_goal=goal,  # 到达目标
            collision=collision,  # 碰撞
            timed_out=timed_out_episode,  # 超时
        )
        if finalize_result is not None:  # 如果有结果
            finalize_components, risk_sample = finalize_result  # 解包结果
            if risk_sample is not None:  # 如果有风险样本
                system.high_level_planner.store_safety_sample(*risk_sample)  # 存储安全样本
                system.high_level_planner.maybe_update_safety_critic(  # 可能更新安全评估器
                    batch_size=safety_cfg.update_batch_size  # 批次大小
                )
            metrics = maybe_train_high_level(  # 可能训练高层
                system.high_level_planner,  # 高层规划器
                high_level_buffer,  # 高层缓冲区
                config.batch_size,  # 批次大小
            )
            if metrics:  # 如果有训练指标
                for key, value in metrics.items():  # 遍历指标
                    system.high_level_planner.writer.add_scalar(  # 记录标量
                        f"planner/{key}",  # 指标名称
                        value,  # 指标值
                        system.high_level_planner.iter_count,  # 迭代计数
                    )
        
        # 重置子目标上下文
        current_subgoal_context = None  # 重置子目标上下文
        current_subgoal_world = None  # 重置子目标世界坐标

        # 更新统计
        if goal:  # 如果到达目标
            epoch_goal_count += 1  # 轮次目标计数加1
        if collision:  # 如果碰撞
            epoch_collision_count += 1  # 轮次碰撞计数加1

        # 输出情节结果
        status = "🎯 GOAL" if goal else "💥 COLLISION" if collision else "⏰ TIMEOUT"  # 状态信息
        print(
            f"   Episode {episode:3d} finished: {status} | "  # 情节完成信息
            f"Steps: {steps:3d} | Total Reward: {episode_reward:7.1f}"
        )

        # 记录到TensorBoard
        writer = system.low_level_controller.writer  # 获取TensorBoard写入器
        writer.add_scalar("train/episode_reward", episode_reward, episode)  # 记录情节奖励

        # ========== 训练低层控制器 ==========
        if (
            replay_buffer.size() >= config.min_buffer_size  # 如果缓冲区大小达到最小值
            and episode % config.train_every_n_episodes == 0  # 且达到训练频率
        ):
            current_buffer_size = replay_buffer.size()  # 当前缓冲区大小
            print(f"   🔄 Training model... (Buffer: {current_buffer_size} samples)")  # 训练模型信息

            # 执行多次训练迭代
            for _ in range(config.training_iterations):  # 训练迭代次数
                system.low_level_controller.update(  # 更新低层控制器
                    replay_buffer,  # 回放缓冲区
                    batch_size=config.batch_size,  # 批次大小
                    discount=0.99,  # 折扣因子
                    tau=0.005,    # 软更新参数
                    policy_noise=0.2,  # 策略噪声
                    noise_clip=0.5,  # 噪声裁剪
                    policy_freq=2,   # 策略频率
                )
            print("   ✅ Training completed")  # 训练完成信息

        episode += 1  # 情节计数器加1

        # ========== 模型保存 ==========
        if config.save_every > 0 and episode % config.save_every == 0:  # 如果达到保存频率
            print(f"   💾 Saving checkpoints after episode {episode}")  # 保存检查点信息
            system.high_level_planner.save_model(  # 保存高层规划器模型
                filename=system.high_level_planner.model_name,  # 模型名称
                directory=system.high_level_planner.save_directory,  # 保存目录
            )
            system.low_level_controller.save_model(  # 保存低层控制器模型
                filename=system.low_level_controller.model_name,  # 模型名称
                directory=system.low_level_controller.save_directory,  # 保存目录
            )

        # ========== 轮次结束处理 ==========
        if episode % config.episodes_per_epoch == 0:  # 如果达到轮次情节数
            # 计算轮次统计
            epoch_avg_reward = epoch_total_reward / config.episodes_per_epoch  # 轮次平均奖励
            epoch_success_rate = epoch_goal_count / config.episodes_per_epoch * 100  # 轮次成功率
            epoch_collision_rate = epoch_collision_count / config.episodes_per_epoch * 100  # 轮次碰撞率

            # 输出轮次总结
            print("\n" + "=" * 60)  # 分隔线
            print(f"📊 EPOCH {epoch:03d} TRAINING SUMMARY")  # 轮次总结标题
            print("=" * 60)
            print(
                f"   • Success Rate:    {epoch_success_rate:6.1f}% "  # 成功率
                f"({epoch_goal_count:2d}/{config.episodes_per_epoch:2d})"
            )
            print(
                f"   • Collision Rate:  {epoch_collision_rate:6.1f}% "  # 碰撞率
                f"({epoch_collision_count:2d}/{config.episodes_per_epoch:2d})"
            )
            print(f"   • Average Reward:  {epoch_avg_reward:8.2f}")  # 平均奖励
            print(f"   • Total Steps:     {epoch_total_steps:8d}")  # 总步数
            print(f"   • Buffer Size:     {replay_buffer.size():8d}")  # 缓冲区大小
            print("=" * 60)

            # 重置轮次统计
            epoch_total_reward = 0.0  # 重置轮次总奖励
            epoch_total_steps = 0  # 重置轮次总步数
            epoch_goal_count = 0  # 重置轮次目标计数
            epoch_collision_count = 0  # 重置轮次碰撞计数

            epoch += 1  # 轮次计数器加1

            # 执行评估
            evaluate(system, sim, config, epoch, low_reward_cfg)  # 执行评估

    # ========== 训练完成处理 ==========
    print("\n💾 Saving final checkpoints...")  # 保存最终检查点
    system.high_level_planner.save_model(  # 保存高层规划器模型
        filename=system.high_level_planner.model_name,  # 模型名称
        directory=system.high_level_planner.save_directory,  # 保存目录
    )
    system.low_level_controller.save_model(  # 保存低层控制器模型
        filename=system.low_level_controller.model_name,  # 模型名称
        directory=system.low_level_controller.save_directory,  # 保存目录
    )

    print("\n" + "="*60)  # 分隔线
    print("🎉 ETHSRL+GP Training Completed!")  # 训练完成信息
    print("="*60)
    print(f"📈 Final performance after {config.max_epochs} epochs")  # 最终性能信息
    print("="*60)


if __name__ == "__main__":
    main()  # 运行主函数
