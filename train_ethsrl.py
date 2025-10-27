"""
ETHSRL+GP分层导航系统的训练入口点

该脚本遵循原始``robot_nav/rl_train.py``的结构，
同时集成了新实现的高层规划器和低层控制器。
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch

from ethsrl.core.integration import HierarchicalNavigationSystem
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
    eval_episodes: int = 6  # 评估时使用的情节数


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
        return states, actions, rewards, next_states, dones

    def clear(self) -> None:
        """清空缓冲区"""
        self._buffer.clear()


def get_robot_pose(sim: SIM) -> Tuple[float, float, float]:
    """从IR-Sim包装器中提取机器人位姿并返回(x, y, theta)"""

    robot_state = sim.env.get_robot_state()  # 获取机器人状态
    return (
        float(robot_state[0].item()),  # x坐标
        float(robot_state[1].item()),  # y坐标
        float(robot_state[2].item()),  # 航向角theta
    )


def evaluate(system: HierarchicalNavigationSystem, sim: SIM, config: TrainingConfig, epoch: int) -> None:
    """运行无探索噪声的评估 rollout 并记录汇总统计信息"""

    print("\n" + "="*60)
    print(f"🎯 EPOCH {epoch:03d} EVALUATION")
    print("="*60)
    
    # 初始化评估指标
    total_reward = 0.0
    total_steps = 0
    collision_count = 0
    goal_count = 0
    timeout_count = 0
    episode_rewards = []
    episode_lengths = []

    # 运行多个评估回合
    for ep_idx in range(config.eval_episodes):
        system.reset()  # 重置系统状态
        # 重置模拟环境并获取初始观测
        latest_scan, distance, cos, sin, collision, goal, prev_action, reward = sim.reset()
        prev_action = [0.0, 0.0]  # 初始化动作为零
        done = False
        steps = 0
        episode_reward = 0.0

        # 单个评估回合循环
        while not done and steps < config.max_steps:
            robot_pose = get_robot_pose(sim)  # 获取机器人位姿
            goal_info = [distance, cos, sin]  # 目标信息

            # 检查是否需要更新子目标
            if system.current_subgoal is None or system.high_level_planner.check_triggers(
                latest_scan, robot_pose, goal_info
            ):
                # 生成新的子目标
                subgoal_distance, subgoal_angle = system.high_level_planner.generate_subgoal(
                    latest_scan, distance, cos, sin
                )
            else:
                # 使用当前子目标
                subgoal_distance, subgoal_angle = system.current_subgoal

            # 处理观测为状态向量
            state = system.low_level_controller.process_observation(
                latest_scan, subgoal_distance, subgoal_angle, prev_action
            )

            # 预测动作（无探索噪声）
            action = system.low_level_controller.predict_action(state, add_noise=False)
            # 将动作转换为控制命令
            lin_cmd = float(np.clip((action[0] + 1.0) / 4.0, 0.0, config.max_lin_velocity))
            ang_cmd = float(np.clip(action[1], -config.max_ang_velocity, config.max_ang_velocity))

            # 在模拟环境中执行动作
            latest_scan, distance, cos, sin, collision, goal, _, reward = sim.step(
                lin_velocity=lin_cmd, ang_velocity=ang_cmd
            )

            prev_action = [lin_cmd, ang_cmd]  # 更新历史动作
            episode_reward += reward
            steps += 1

            # 检查终止条件
            if collision:
                collision_count += 1
                done = True
            elif goal:
                goal_count += 1
                done = True
            elif steps >= config.max_steps:
                timeout_count += 1
                done = True

        # 记录单回合数据
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        total_reward += episode_reward
        total_steps += steps
        
        # 显示单回合评估进度
        status = "🎯" if goal else "💥" if collision else "⏰"
        print(f"   Evaluation Episode {ep_idx+1:2d}/{config.eval_episodes}: {status} | "
              f"Steps: {steps:3d} | Reward: {episode_reward:7.1f}")

    # ========== 计算统计指标 ==========
    avg_reward = total_reward / config.eval_episodes
    avg_steps = total_steps / config.eval_episodes
    success_rate = goal_count / config.eval_episodes * 100
    collision_rate = collision_count / config.eval_episodes * 100
    timeout_rate = timeout_count / config.eval_episodes * 100
    
    # 计算标准差
    reward_std = np.std(episode_rewards) if config.eval_episodes > 1 else 0
    steps_std = np.std(episode_lengths) if config.eval_episodes > 1 else 0
    
    # ========== 格式化输出 ==========
    print("\n📈 Performance Summary:")
    print(f"   • Success Rate:      {success_rate:6.1f}% ({goal_count:2d}/{config.eval_episodes:2d})")
    print(f"   • Collision Rate:    {collision_rate:6.1f}% ({collision_count:2d}/{config.eval_episodes:2d})")
    print(f"   • Timeout Rate:      {timeout_rate:6.1f}% ({timeout_count:2d}/{config.eval_episodes:2d})")
    print(f"   • Average Reward:    {avg_reward:8.2f} ± {reward_std:.2f}")
    print(f"   • Average Steps:     {avg_steps:8.1f} ± {steps_std:.1f}")
    
    # 额外指标
    if goal_count > 0:
        successful_episodes_reward = sum(r for i, r in enumerate(episode_rewards) 
                                       if episode_lengths[i] < config.max_steps and not collision)
        avg_success_reward = successful_episodes_reward / goal_count
        print(f"   • Avg Success Reward: {avg_success_reward:8.2f}")
    
    print("-" * 60)
    print(f"⏰ Evaluation completed: {config.eval_episodes} episodes")
    print("=" * 60)
    
    # ========== TensorBoard记录 ==========
    writer = system.low_level_controller.writer
    writer.add_scalar("eval/success_rate", success_rate, epoch)
    writer.add_scalar("eval/collision_rate", collision_rate, epoch)
    writer.add_scalar("eval/timeout_rate", timeout_rate, epoch)
    writer.add_scalar("eval/avg_reward", avg_reward, epoch)
    writer.add_scalar("eval/avg_steps", avg_steps, epoch)
    writer.add_scalar("eval/reward_std", reward_std, epoch)
    
    # 记录原始计数
    writer.add_scalar("eval_raw/success_count", goal_count, epoch)
    writer.add_scalar("eval_raw/collision_count", collision_count, epoch)


def main(args=None):
    """ETHSRL+GP的主要训练循环"""

    # ========== 训练初始化日志 ==========
    print("\n" + "="*60)
    print("🚀 Starting ETHSRL+GP Hierarchical Navigation Training")
    print("="*60)
    print(f"📋 Training Configuration:")
    print(f"   • Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print(f"   • Max epochs: {60}, Episodes per epoch: {70}")
    print(f"   • Training iterations: {80}, Batch size: {64}")
    print(f"   • Max steps per episode: {300}")
    print(f"   • Train every {2} episodes")
    print("="*60)

    # 设备和系统初始化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = TrainingConfig()

    # ========== 系统初始化 ==========
    print("🔄 Initializing ETHSRL+GP system...")
    system = HierarchicalNavigationSystem(device=device)
    replay_buffer = TD3ReplayAdapter(buffer_size=config.buffer_size)
    print("✅ System initialization completed")

    # ========== 环境初始化 ==========
    print("🔄 Initializing simulation environment...")
    sim = SIM(world_file="worlds/env_a.yaml", disable_plotting=False)
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

    # 主训练循环
    while epoch < config.max_epochs:
        system.reset()
        # 重置模拟环境并获取初始观测
        latest_scan, distance, cos, sin, collision, goal, prev_action, reward = sim.reset()
        prev_action = [0.0, 0.0]

        # 单个情节变量初始化
        steps = 0
        episode_reward = 0.0
        done = False

        # 单个情节循环
        while not done and steps < config.max_steps:
            robot_pose = get_robot_pose(sim)
            goal_info = [distance, cos, sin]

            # 确定是否需要新的子目标
            if system.current_subgoal is None or system.high_level_planner.check_triggers(
                latest_scan, robot_pose, goal_info
            ):
                subgoal_distance, subgoal_angle = system.high_level_planner.generate_subgoal(
                    latest_scan, distance, cos, sin
                )
            else:
                subgoal_distance, subgoal_angle = system.current_subgoal

            # 处理观测为状态向量
            state = system.low_level_controller.process_observation(
                latest_scan, subgoal_distance, subgoal_angle, prev_action
            )

            # 预测动作（带探索噪声）
            action = system.low_level_controller.predict_action(
                state, add_noise=True, noise_scale=config.exploration_noise
            )
            action = np.clip(action, -1.0, 1.0)

            # 将动作转换为控制命令
            lin_cmd = float(np.clip((action[0] + 1.0) / 4.0, 0.0, config.max_lin_velocity))
            ang_cmd = float(np.clip(action[1], -config.max_ang_velocity, config.max_ang_velocity))

            # 在模拟环境中执行动作
            latest_scan, distance, cos, sin, collision, goal, executed_action, reward = sim.step(
                lin_velocity=lin_cmd, ang_velocity=ang_cmd
            )

            episode_reward += reward
            epoch_total_reward += reward
            epoch_total_steps += 1

            # 每50步显示一次训练进度
            if steps % 50 == 0:
                print(f"🏃 Training | Epoch {epoch:2d}/{config.max_epochs} | "
                      f"Episode {episode:3d}/{config.episodes_per_epoch} | "
                      f"Step {steps:3d}/{config.max_steps} | "
                      f"Reward: {reward:7.2f}")

            # 准备下一个状态
            next_prev_action = [executed_action[0], executed_action[1]]
            next_state = system.low_level_controller.process_observation(
                latest_scan,
                system.current_subgoal[0] if system.current_subgoal else subgoal_distance,
                system.current_subgoal[1] if system.current_subgoal else subgoal_angle,
                next_prev_action,
            )

            # 检查终止条件
            done = collision or goal or steps == config.max_steps - 1
            
            # 将经验添加到回放缓冲区
            replay_buffer.add(state, action, reward, float(done), next_state)

            prev_action = next_prev_action
            steps += 1

        # 更新统计
        if goal:
            epoch_goal_count += 1
        if collision:
            epoch_collision_count += 1

        # 显示回合结束信息
        status = "🎯 GOAL" if goal else "💥 COLLISION" if collision else "⏰ TIMEOUT"
        print(f"   Episode {episode:3d} finished: {status} | "
              f"Steps: {steps:3d} | Total Reward: {episode_reward:7.1f}")

        # 记录情节奖励到TensorBoard
        writer = system.low_level_controller.writer
        writer.add_scalar("train/episode_reward", episode_reward, episode)

        # 检查是否应该进行训练
        if (
            replay_buffer.size() >= config.min_buffer_size
            and episode % config.train_every_n_episodes == 0
        ):
            current_buffer_size = replay_buffer.size()
            print(f"   🔄 Training model... (Buffer: {current_buffer_size} samples)")
            
            # 执行训练迭代
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
            print(f"   ✅ Training completed")

        episode += 1

        # 检查是否完成一个训练轮次
        if episode % config.episodes_per_epoch == 0:
            # 训练阶段统计
            epoch_avg_reward = epoch_total_reward / config.episodes_per_epoch
            epoch_success_rate = epoch_goal_count / config.episodes_per_epoch * 100
            epoch_collision_rate = epoch_collision_count / config.episodes_per_epoch * 100
            
            print("\n" + "="*60)
            print(f"📊 EPOCH {epoch:03d} TRAINING SUMMARY")
            print("="*60)
            print(f"   • Success Rate:    {epoch_success_rate:6.1f}% ({epoch_goal_count:2d}/{config.episodes_per_epoch:2d})")
            print(f"   • Collision Rate:  {epoch_collision_rate:6.1f}% ({epoch_collision_count:2d}/{config.episodes_per_epoch:2d})")
            print(f"   • Average Reward:  {epoch_avg_reward:8.2f}")
            print(f"   • Total Steps:     {epoch_total_steps:8d}")
            print(f"   • Buffer Size:     {replay_buffer.size():8d}")
            print("="*60)
            
            # 重置epoch统计
            epoch_total_reward = 0.0
            epoch_total_steps = 0
            epoch_goal_count = 0
            epoch_collision_count = 0
            
            epoch += 1
            
            # 运行评估
            evaluate(system, sim, config, epoch)

    # ========== 训练完成日志 ==========
    print("\n" + "="*60)
    print("🎉 ETHSRL+GP Training Completed!")
    print("="*60)
    print(f"📈 Final performance after {config.max_epochs} epochs")
    print("="*60)


if __name__ == "__main__":
    main()
