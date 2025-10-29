"""
ETHSRL+GP分层导航系统的训练入口点

该脚本遵循原始``robot_nav/rl_train.py``的结构，
同时集成了新实现的高层规划器和低层控制器。
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch

# 导入自定义模块
from ethsrl.core.integration import HierarchicalNavigationSystem
from ethsrl.core.rewards import (
    HighLevelRewardConfig,
    LowLevelRewardConfig,
    compute_high_level_reward,
    compute_low_level_reward,
)
from robot_nav.SIM_ENV.sim import SIM
from robot_nav.replay_buffer import ReplayBuffer


@dataclass
class TrainingConfig:
    """训练超参数配置容器"""

    buffer_size: int = 100_000  # 经验回放缓冲区大小
    batch_size: int = 64  # 训练批次大小
    max_epochs: int = 60  # 最大训练轮数
    episodes_per_epoch: int = 70  # 每轮训练的情节数
    max_steps: int = 300  # 每个情节的最大步数
    train_every_n_episodes: int = 2  # 每N个情节训练一次
    training_iterations: int = 80  # 每次训练的迭代次数
    exploration_noise: float = 0.2  # 探索噪声强度
    min_buffer_size: int = 1_000  # 开始训练的最小缓冲区大小
    max_lin_velocity: float = 0.5  # 最大线速度
    max_ang_velocity: float = 1.0  # 最大角速度
    eval_episodes: int = 10  # 评估时使用的情节数
    subgoal_radius: float = 0.5  # 判定子目标达成的距离阈值
    save_every: int = 5  # 每隔多少个情节保存一次模型（<=0 表示仅最终保存）


@dataclass
class SubgoalContext:
    """高层子目标生命周期内的统计上下文"""

    start_state: np.ndarray  # 子目标开始时的状态
    action: np.ndarray  # 选择的子目标动作 [距离, 角度]
    world_target: np.ndarray  # 子目标的全局坐标
    start_goal_distance: float  # 开始时的目标距离
    last_goal_distance: float  # 最后的目标距离
    low_level_return: float = 0.0  # 累积的低层奖励
    steps: int = 0  # 子目标执行的步数
    subgoal_completed: bool = False  # 子目标是否完成
    last_state: Optional[np.ndarray] = None  # 最后的状态


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
    world_x = robot_pose[0] + distance * np.cos(robot_pose[2] + angle)
    world_y = robot_pose[1] + distance * np.sin(robot_pose[2] + angle)
    return np.array([world_x, world_y], dtype=np.float32)


def finalize_subgoal_transition(
    context: Optional[SubgoalContext],
    buffer: List[Tuple[np.ndarray, np.ndarray, float, np.ndarray, float]],
    high_cfg: HighLevelRewardConfig,
    done: bool,
    reached_goal: bool,
    collision: bool,
    timed_out: bool,
) -> Optional[dict]:
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
        奖励分量字典或None
    """

    # 检查上下文有效性
    if context is None or context.steps == 0:
        return None

    # 确定最后状态
    last_state = context.last_state if context.last_state is not None else context.start_state

    # 计算高层奖励
    reward, components = compute_high_level_reward(
        start_goal_distance=context.start_goal_distance,
        end_goal_distance=context.last_goal_distance,
        subgoal_completed=context.subgoal_completed,
        reached_goal=reached_goal,
        collision=collision,
        timed_out=timed_out,
        config=high_cfg,
    )

    # 将经验添加到缓冲区
    buffer.append(
        (
            context.start_state.astype(np.float32, copy=False),  # 开始状态
            context.action.astype(np.float32, copy=False),  # 子目标动作
            float(reward),  # 奖励
            last_state.astype(np.float32, copy=False),  # 结束状态
            float(done),  # 终止标志
        )
    )

    return components


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
    if len(buffer) < batch_size:
        return None

    # 提取批次数据
    batch = buffer[:batch_size]
    del buffer[:batch_size]  # 移除已使用的样本

    # 组织批次数据
    states = np.stack([entry[0] for entry in batch])
    actions = np.stack([entry[1] for entry in batch])
    rewards = np.array([entry[2] for entry in batch], dtype=np.float32)
    next_states = np.stack([entry[3] for entry in batch])
    dones = np.array([entry[4] for entry in batch], dtype=np.float32)

    # 更新规划器
    metrics = planner.update_planner(states, actions, rewards, dones, next_states, batch_size=batch_size)
    return metrics


class TD3ReplayAdapter:
    """匹配控制器期望的回放缓冲区API的薄包装器"""

    def __init__(self, buffer_size: int, random_seed: int = 666) -> None:
        """初始化回放缓冲区适配器"""
        self._buffer = ReplayBuffer(buffer_size=buffer_size, random_seed=random_seed)

    def add(self, state, action, reward, done, next_state) -> None:
        """向缓冲区添加经验"""
        self._buffer.add(state, action, reward, done, next_state)

    def size(self) -> int:
        """返回缓冲区当前大小"""
        return self._buffer.size()

    def sample(self, batch_size: int):
        """从缓冲区采样批次数据"""
        states, actions, rewards, dones, next_states = self._buffer.sample_batch(batch_size)
        return states, actions, rewards, dones, next_states

    def clear(self) -> None:
        """清空缓冲区"""
        self._buffer.clear()


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

    print("\n" + "=" * 60)
    print(f"🎯 EPOCH {epoch:03d} EVALUATION")
    print("=" * 60)

    # 初始化评估统计
    total_reward = 0.0
    total_steps = 0
    collision_count = 0
    goal_count = 0
    timeout_count = 0
    episode_rewards: List[float] = []
    episode_lengths: List[int] = []
    episode_success_flags: List[bool] = []

    # 运行评估情节
    for ep_idx in range(config.eval_episodes):
        system.reset()  # 重置系统状态
        latest_scan, distance, cos, sin, collision, goal, prev_action, _ = sim.reset()
        prev_action = [0.0, 0.0]  # 初始化动作
        current_subgoal_world: Optional[np.ndarray] = None
        done = False
        steps = 0
        episode_reward = 0.0

        # 单次评估情节循环
        while not done and steps < config.max_steps:
            robot_pose = get_robot_pose(sim)
            goal_info = [distance, cos, sin]

            # 检查是否需要重新规划
            should_replan = (
                system.high_level_planner.current_subgoal_world is None
                or system.high_level_planner.check_triggers(
                    latest_scan,
                    robot_pose,
                    goal_info,
                    prev_action=prev_action,
                    current_step=steps,
                )
            )

            subgoal_distance: Optional[float] = None
            subgoal_angle: Optional[float] = None

            subgoal_distance: Optional[float] = None
            subgoal_angle: Optional[float] = None

            if should_replan:
                # 生成新子目标
                subgoal_distance, subgoal_angle = system.high_level_planner.generate_subgoal(
                    latest_scan,
                    distance,
                    cos,
                    sin,
                    prev_action=prev_action,
                    robot_pose=robot_pose,
                    current_step=steps,
                )
                planner_world = system.high_level_planner.current_subgoal_world
                current_subgoal_world = np.asarray(planner_world, dtype=np.float32) if planner_world is not None else None
                system.high_level_planner.event_trigger.reset_time(steps)
                if current_subgoal_world is None:
                    current_subgoal_world = compute_subgoal_world(robot_pose, subgoal_distance, subgoal_angle)
            else:
                planner_world = system.high_level_planner.current_subgoal_world
                if planner_world is not None:
                    current_subgoal_world = np.asarray(planner_world, dtype=np.float32)

            system.current_subgoal_world = current_subgoal_world

            relative_geometry = system.high_level_planner.get_relative_subgoal(robot_pose)
            if relative_geometry[0] is None:
                if should_replan and subgoal_distance is not None and subgoal_angle is not None:
                    relative_geometry = (subgoal_distance, subgoal_angle)
                elif system.current_subgoal is not None:
                    relative_geometry = system.current_subgoal
                else:
                    relative_geometry = (0.0, 0.0)

            subgoal_distance, subgoal_angle = float(relative_geometry[0]), float(relative_geometry[1])
            system.current_subgoal = (subgoal_distance, subgoal_angle)

            # 计算子目标距离
            prev_subgoal_distance = None
            if current_subgoal_world is not None:
                prev_pos = np.array(robot_pose[:2], dtype=np.float32)
                prev_subgoal_distance = float(np.linalg.norm(prev_pos - current_subgoal_world))

            # 处理低层观测
            state = system.low_level_controller.process_observation(
                latest_scan,
                subgoal_distance,
                subgoal_angle,
                prev_action,
            )

            # 预测动作（无探索噪声）
            action = system.low_level_controller.predict_action(state, add_noise=False)
            lin_cmd = float(np.clip((action[0] + 1.0) / 4.0, 0.0, config.max_lin_velocity))
            ang_cmd = float(np.clip(action[1], -config.max_ang_velocity, config.max_ang_velocity))

            # 执行动作
            latest_scan, distance, cos, sin, collision, goal, _, _ = sim.step(
                lin_velocity=lin_cmd,
                ang_velocity=ang_cmd,
            )

            # 更新子目标距离
            next_pose = get_robot_pose(sim)
            current_subgoal_distance = None
            if current_subgoal_world is not None:
                next_pos = np.array(next_pose[:2], dtype=np.float32)
                current_subgoal_distance = float(np.linalg.norm(next_pos - current_subgoal_world))

            # 计算最小障碍物距离
            scan_arr = np.asarray(latest_scan, dtype=np.float32)
            finite_scan = scan_arr[np.isfinite(scan_arr)]
            min_obstacle_distance = float(finite_scan.min()) if finite_scan.size else 8.0

            # 检查终止条件
            just_reached_subgoal = False
            if (
                current_subgoal_distance is not None
                and current_subgoal_distance <= config.subgoal_radius
            ):
                if prev_subgoal_distance is None:
                    just_reached_subgoal = True
                elif prev_subgoal_distance > config.subgoal_radius:
                    just_reached_subgoal = True
            timed_out = steps == config.max_steps - 1 and not (goal or collision)

            # 计算低层奖励
            low_reward, _ = compute_low_level_reward(
                prev_subgoal_distance=prev_subgoal_distance,
                current_subgoal_distance=current_subgoal_distance,
                min_obstacle_distance=min_obstacle_distance,
                reached_goal=goal,
                reached_subgoal=just_reached_subgoal,
                collision=collision,
                timed_out=timed_out,
                config=low_cfg,
            )

            # 更新统计
            episode_reward += low_reward
            steps += 1
            prev_action = [lin_cmd, ang_cmd]

            # 检查终止
            if collision:
                collision_count += 1
                done = True
            elif goal:
                goal_count += 1
                done = True
            elif steps >= config.max_steps:
                timeout_count += 1
                done = True

        # 记录情节结果
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        episode_success_flags.append(goal)
        total_reward += episode_reward
        total_steps += steps

        status = "🎯" if goal else "💥" if collision else "⏰"
        print(
            f"   Evaluation Episode {ep_idx + 1:2d}/{config.eval_episodes}: {status} | "
            f"Steps: {steps:3d} | Reward: {episode_reward:7.1f}"
        )

    # 计算汇总统计
    avg_reward = total_reward / config.eval_episodes
    avg_steps = total_steps / config.eval_episodes
    success_rate = goal_count / config.eval_episodes * 100
    collision_rate = collision_count / config.eval_episodes * 100
    timeout_rate = timeout_count / config.eval_episodes * 100

    reward_std = np.std(episode_rewards) if config.eval_episodes > 1 else 0.0
    steps_std = np.std(episode_lengths) if config.eval_episodes > 1 else 0.0

    # 输出评估结果
    print("\n📈 Performance Summary:")
    print(f"   • Success Rate:      {success_rate:6.1f}% ({goal_count:2d}/{config.eval_episodes:2d})")
    print(f"   • Collision Rate:    {collision_rate:6.1f}% ({collision_count:2d}/{config.eval_episodes:2d})")
    print(f"   • Timeout Rate:      {timeout_rate:6.1f}% ({timeout_count:2d}/{config.eval_episodes:2d})")
    print(f"   • Average Reward:    {avg_reward:8.2f} ± {reward_std:.2f}")
    print(f"   • Average Steps:     {avg_steps:8.1f} ± {steps_std:.1f}")

    if goal_count > 0:
        successful_rewards = [r for r, success in zip(episode_rewards, episode_success_flags) if success]
        avg_success_reward = np.mean(successful_rewards) if successful_rewards else 0.0
        print(f"   • Avg Success Reward: {avg_success_reward:8.2f}")

    print("-" * 60)
    print(f"⏰ Evaluation completed: {config.eval_episodes} episodes")
    print("=" * 60)

    # 记录到TensorBoard
    writer = system.low_level_controller.writer
    writer.add_scalar("eval/success_rate", success_rate, epoch)
    writer.add_scalar("eval/collision_rate", collision_rate, epoch)
    writer.add_scalar("eval/timeout_rate", timeout_rate, epoch)
    writer.add_scalar("eval/avg_reward", avg_reward, epoch)
    writer.add_scalar("eval/avg_steps", avg_steps, epoch)
    writer.add_scalar("eval/reward_std", reward_std, epoch)
    writer.add_scalar("eval_raw/success_count", goal_count, epoch)
    writer.add_scalar("eval_raw/collision_count", collision_count, epoch)


def main(args=None):
    """ETHSRL+GP的主要训练循环"""

    # ========== 训练配置与设备初始化 ==========
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = TrainingConfig()

    # ========== 训练初始化日志 ==========
    print("\n" + "="*60)
    print("🚀 Starting ETHSRL+GP Hierarchical Navigation Training")
    print("="*60)
    print(f"📋 Training Configuration:")
    print(f"   • Device: {device}")
    print(
        f"   • Max epochs: {config.max_epochs}, Episodes per epoch: {config.episodes_per_epoch}"
    )
    print(
        f"   • Training iterations: {config.training_iterations}, Batch size: {config.batch_size}"
    )
    print(f"   • Max steps per episode: {config.max_steps}")
    print(f"   • Train every {config.train_every_n_episodes} episodes")
    if config.save_every > 0:
        print(f"   • Save models every {config.save_every} episodes")
    else:
        print("   • Save models at end of training only")
    print("="*60)

    # ========== 系统初始化 ==========
    print("🔄 Initializing ETHSRL+GP system...")
    system = HierarchicalNavigationSystem(device=device, subgoal_threshold=config.subgoal_radius)
    replay_buffer = TD3ReplayAdapter(buffer_size=config.buffer_size)
    print("✅ System initialization completed")

    # ========== 环境初始化 ==========
    print("🔄 Initializing simulation environment...")
    sim = SIM(world_file="worlds/env_b_none.yaml", disable_plotting=False)
    print("✅ Environment initialization completed")

    # ========== 训练统计变量初始化 ==========
    episode_reward = 0.0
    epoch_total_reward = 0.0
    epoch_total_steps = 0
    epoch_goal_count = 0
    epoch_collision_count = 0

    # 训练计数器初始化
    episode = 0
    epoch = 0

    print("\n🎬 Starting main training loop...")
    print("-" * 50)

    # 奖励配置初始化
    low_reward_cfg = LowLevelRewardConfig()
    high_reward_cfg = HighLevelRewardConfig()
    high_level_buffer: List[Tuple[np.ndarray, np.ndarray, float, np.ndarray, float]] = []
    current_subgoal_context: Optional[SubgoalContext] = None

    # ========== 主训练循环 ==========
    while epoch < config.max_epochs:
        # 重置环境和系统状态
        system.reset()
        current_subgoal_context = None
        system.current_subgoal = None

        latest_scan, distance, cos, sin, collision, goal, prev_action, _ = sim.reset()
        prev_action = [0.0, 0.0]  # 重置动作
        current_subgoal_world: Optional[np.ndarray] = None

        steps = 0
        episode_reward = 0.0
        done = False

        # ========== 单次情节循环 ==========
        while not done and steps < config.max_steps:
            robot_pose = get_robot_pose(sim)
            goal_info = [distance, cos, sin]

            # 检查是否需要重新规划子目标
            should_replan = (
                system.high_level_planner.current_subgoal_world is None
                or system.high_level_planner.check_triggers(
                    latest_scan,
                    robot_pose,
                    goal_info,
                    prev_action=prev_action,
                    current_step=steps,
                )
            )

            if should_replan:
                # 完成当前子目标并训练
                finalize_components = finalize_subgoal_transition(
                    current_subgoal_context,
                    high_level_buffer,
                    high_reward_cfg,
                    done=False,
                    reached_goal=False,
                    collision=False,
                    timed_out=False,
                )
                if finalize_components is not None:
                    metrics = maybe_train_high_level(
                        system.high_level_planner,
                        high_level_buffer,
                        config.batch_size,
                    )
                    if metrics:
                        # 记录训练指标
                        for key, value in metrics.items():
                            system.high_level_planner.writer.add_scalar(
                                f"planner/{key}",
                                value,
                                system.high_level_planner.iter_count,
                            )

                # 生成新子目标
                subgoal_distance, subgoal_angle = system.high_level_planner.generate_subgoal(
                    latest_scan,
                    distance,
                    cos,
                    sin,
                    prev_action=prev_action,
                    robot_pose=robot_pose,
                    current_step=steps,
                )
                planner_world = system.high_level_planner.current_subgoal_world
                current_subgoal_world = np.asarray(planner_world, dtype=np.float32) if planner_world is not None else None
                system.high_level_planner.event_trigger.reset_time(steps)
                if current_subgoal_world is None:
                    current_subgoal_world = compute_subgoal_world(robot_pose, subgoal_distance, subgoal_angle)

                # 构建高层状态向量
                start_state = system.high_level_planner.build_state_vector(
                    latest_scan,
                    distance,
                    cos,
                    sin,
                    prev_action,
                )

                # 创建新的子目标上下文
                current_subgoal_context = SubgoalContext(
                    start_state=start_state.astype(np.float32, copy=False),
                    action=np.array([subgoal_distance, subgoal_angle], dtype=np.float32),
                    world_target=current_subgoal_world,
                    start_goal_distance=distance,
                    last_goal_distance=distance,
                    low_level_return=0.0,
                    steps=0,
                    subgoal_completed=False,
                    last_state=start_state.astype(np.float32, copy=False),
                )
            else:
                planner_world = system.high_level_planner.current_subgoal_world
                if planner_world is not None:
                    current_subgoal_world = np.asarray(planner_world, dtype=np.float32)

            system.current_subgoal_world = current_subgoal_world

            relative_geometry = system.high_level_planner.get_relative_subgoal(robot_pose)
            if relative_geometry[0] is None:
                if should_replan and subgoal_distance is not None and subgoal_angle is not None:
                    relative_geometry = (subgoal_distance, subgoal_angle)
                elif system.current_subgoal is not None:
                    relative_geometry = system.current_subgoal
                else:
                    relative_geometry = (0.0, 0.0)

            subgoal_distance, subgoal_angle = float(relative_geometry[0]), float(relative_geometry[1])
            system.current_subgoal = (subgoal_distance, subgoal_angle)

            # 计算子目标距离
            prev_subgoal_distance = None
            if current_subgoal_world is not None:
                prev_pos = np.array(robot_pose[:2], dtype=np.float32)
                prev_subgoal_distance = float(np.linalg.norm(prev_pos - current_subgoal_world))

            # 处理低层观测
            state = system.low_level_controller.process_observation(
                latest_scan,
                subgoal_distance,
                subgoal_angle,
                prev_action,
            )

            # 预测动作（带探索噪声）
            action = system.low_level_controller.predict_action(
                state,
                add_noise=True,
                noise_scale=config.exploration_noise,
            )
            action = np.clip(action, -1.0, 1.0)  # 裁剪动作

            # 转换为实际控制命令
            lin_cmd = float(np.clip((action[0] + 1.0) / 4.0, 0.0, config.max_lin_velocity))
            ang_cmd = float(np.clip(action[1], -config.max_ang_velocity, config.max_ang_velocity))

            # 执行动作
            latest_scan, distance, cos, sin, collision, goal, executed_action, _ = sim.step(
                lin_velocity=lin_cmd,
                ang_velocity=ang_cmd,
            )

            # 更新子目标距离
            next_pose = get_robot_pose(sim)
            current_subgoal_distance = None
            if current_subgoal_world is not None:
                next_pos = np.array(next_pose[:2], dtype=np.float32)
                current_subgoal_distance = float(np.linalg.norm(next_pos - current_subgoal_world))

            # 计算最小障碍物距离
            scan_arr = np.asarray(latest_scan, dtype=np.float32)
            finite_scan = scan_arr[np.isfinite(scan_arr)]
            min_obstacle_distance = float(finite_scan.min()) if finite_scan.size else 8.0

            # 检查终止条件
            just_reached_subgoal = False
            if (
                current_subgoal_distance is not None
                and current_subgoal_distance <= config.subgoal_radius
            ):
                if prev_subgoal_distance is None:
                    just_reached_subgoal = True
                elif prev_subgoal_distance > config.subgoal_radius:
                    just_reached_subgoal = True
            if (
                current_subgoal_context is not None
                and current_subgoal_context.subgoal_completed
            ):
                just_reached_subgoal = False
            timed_out = steps == config.max_steps - 1 and not (goal or collision)

            # 计算低层奖励
            low_reward, _ = compute_low_level_reward(
                prev_subgoal_distance=prev_subgoal_distance,
                current_subgoal_distance=current_subgoal_distance,
                min_obstacle_distance=min_obstacle_distance,
                reached_goal=goal,
                reached_subgoal=just_reached_subgoal,
                collision=collision,
                timed_out=timed_out,
                config=low_reward_cfg,
            )

            # 更新奖励统计
            episode_reward += low_reward
            epoch_total_reward += low_reward
            epoch_total_steps += 1

            # 定期输出训练进度
            if steps % 50 == 0:
                print(
                    f"🏃 Training | Epoch {epoch:2d}/{config.max_epochs} | "
                    f"Episode {episode:3d}/{config.episodes_per_epoch} | "
                    f"Step {steps:3d}/{config.max_steps} | "
                    f"Reward: {low_reward:7.2f}"
                )

            # 更新子目标上下文
            if current_subgoal_context is not None:
                current_subgoal_context.low_level_return += low_reward
                current_subgoal_context.steps += 1
                current_subgoal_context.subgoal_completed |= just_reached_subgoal
                current_subgoal_context.last_goal_distance = distance
                # 构建下一状态向量
                next_state_vector = system.high_level_planner.build_state_vector(
                    latest_scan,
                    distance,
                    cos,
                    sin,
                    executed_action,
                )
                current_subgoal_context.last_state = next_state_vector.astype(np.float32, copy=False)

            # 准备下一状态
            next_prev_action = [executed_action[0], executed_action[1]]
            next_state = system.low_level_controller.process_observation(
                latest_scan,
                system.current_subgoal[0] if system.current_subgoal else subgoal_distance,
                system.current_subgoal[1] if system.current_subgoal else subgoal_angle,
                next_prev_action,
            )

            # 检查终止条件
            done = collision or goal or steps == config.max_steps - 1

            # 添加经验到回放缓冲区
            replay_buffer.add(state, action, low_reward, float(done), next_state)

            prev_action = next_prev_action
            steps += 1

        # ========== 情节结束处理 ==========
        timed_out_episode = not goal and not collision and steps >= config.max_steps

        # 完成最后一个子目标
        finalize_components = finalize_subgoal_transition(
            current_subgoal_context,
            high_level_buffer,
            high_reward_cfg,
            done=True,
            reached_goal=goal,
            collision=collision,
            timed_out=timed_out_episode,
        )
        if finalize_components is not None:
            metrics = maybe_train_high_level(
                system.high_level_planner,
                high_level_buffer,
                config.batch_size,
            )
            if metrics:
                for key, value in metrics.items():
                    system.high_level_planner.writer.add_scalar(
                        f"planner/{key}",
                        value,
                        system.high_level_planner.iter_count,
                    )
        
        # 重置子目标上下文
        current_subgoal_context = None
        current_subgoal_world = None

        # 更新统计
        if goal:
            epoch_goal_count += 1
        if collision:
            epoch_collision_count += 1

        # 输出情节结果
        status = "🎯 GOAL" if goal else "💥 COLLISION" if collision else "⏰ TIMEOUT"
        print(
            f"   Episode {episode:3d} finished: {status} | "
            f"Steps: {steps:3d} | Total Reward: {episode_reward:7.1f}"
        )

        # 记录到TensorBoard
        writer = system.low_level_controller.writer
        writer.add_scalar("train/episode_reward", episode_reward, episode)

        # ========== 训练低层控制器 ==========
        if (
            replay_buffer.size() >= config.min_buffer_size
            and episode % config.train_every_n_episodes == 0
        ):
            current_buffer_size = replay_buffer.size()
            print(f"   🔄 Training model... (Buffer: {current_buffer_size} samples)")

            # 执行多次训练迭代
            for _ in range(config.training_iterations):
                system.low_level_controller.update(
                    replay_buffer,
                    batch_size=config.batch_size,
                    discount=0.99,
                    tau=0.005,
                    policy_noise=0.2,
                    noise_clip=0.5,
                    policy_freq=2,
                )
            print("   ✅ Training completed")

        episode += 1

        # ========== 模型保存 ==========
        if config.save_every > 0 and episode % config.save_every == 0:
            print(f"   💾 Saving checkpoints after episode {episode}")
            system.high_level_planner.save_model(
                filename=system.high_level_planner.model_name,
                directory=system.high_level_planner.save_directory,
            )
            system.low_level_controller.save_model(
                filename=system.low_level_controller.model_name,
                directory=system.low_level_controller.save_directory,
            )

        # ========== 轮次结束处理 ==========
        if episode % config.episodes_per_epoch == 0:
            # 计算轮次统计
            epoch_avg_reward = epoch_total_reward / config.episodes_per_epoch
            epoch_success_rate = epoch_goal_count / config.episodes_per_epoch * 100
            epoch_collision_rate = epoch_collision_count / config.episodes_per_epoch * 100

            # 输出轮次总结
            print("\n" + "=" * 60)
            print(f"📊 EPOCH {epoch:03d} TRAINING SUMMARY")
            print("=" * 60)
            print(
                f"   • Success Rate:    {epoch_success_rate:6.1f}% "
                f"({epoch_goal_count:2d}/{config.episodes_per_epoch:2d})"
            )
            print(
                f"   • Collision Rate:  {epoch_collision_rate:6.1f}% "
                f"({epoch_collision_count:2d}/{config.episodes_per_epoch:2d})"
            )
            print(f"   • Average Reward:  {epoch_avg_reward:8.2f}")
            print(f"   • Total Steps:     {epoch_total_steps:8d}")
            print(f"   • Buffer Size:     {replay_buffer.size():8d}")
            print("=" * 60)

            # 重置轮次统计
            epoch_total_reward = 0.0
            epoch_total_steps = 0
            epoch_goal_count = 0
            epoch_collision_count = 0

            epoch += 1

            # 执行评估
            evaluate(system, sim, config, epoch, low_reward_cfg)

    # ========== 训练完成处理 ==========
    print("\n💾 Saving final checkpoints...")
    system.high_level_planner.save_model(
        filename=system.high_level_planner.model_name,
        directory=system.high_level_planner.save_directory,
    )
    system.low_level_controller.save_model(
        filename=system.low_level_controller.model_name,
        directory=system.low_level_controller.save_directory,
    )

    print("\n" + "="*60)
    print("🎉 ETHSRL+GP Training Completed!")
    print("="*60)
    print(f"📈 Final performance after {config.max_epochs} epochs")
    print("="*60)


if __name__ == "__main__":
    main()
