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
    max_epochs: int = 40  # 最大训练轮数
    episodes_per_epoch: int = 60  # 每轮训练的情节数
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

    print("-" * 46)  # 打印分隔线
    print(f"Epoch {epoch}: 运行评估轨迹")

    # 初始化评估指标
    avg_reward = 0.0  # 平均奖励
    collisions = 0    # 碰撞次数
    successes = 0     # 成功到达目标次数

    # 运行指定次数的评估情节
    for _ in range(config.eval_episodes):
        system.reset()  # 重置系统状态
        # 重置模拟环境并获取初始观测
        latest_scan, distance, cos, sin, collision, goal, prev_action, reward = sim.reset()
        prev_action = [0.0, 0.0]  # 初始化动作为零
        done = False  # 情节完成标志
        step = 0      # 步数计数器

        # 运行单个评估情节
        while not done and step < config.max_steps:
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
            avg_reward += reward  # 累计奖励
            step += 1  # 增加步数

            # 检查终止条件
            if collision:
                collisions += 1  # 记录碰撞
                done = True
            if goal:
                successes += 1  # 记录成功
                done = True

    # 计算平均指标
    avg_reward /= max(config.eval_episodes, 1)
    avg_collision_rate = collisions / max(config.eval_episodes, 1)
    avg_success_rate = successes / max(config.eval_episodes, 1)

    # 打印评估结果
    print(f"平均奖励        : {avg_reward:.2f}")
    print(f"碰撞率          : {avg_collision_rate:.2f}")
    print(f"目标到达率      : {avg_success_rate:.2f}")
    print("-" * 46)

    # 记录到TensorBoard
    writer = system.low_level_controller.writer
    writer.add_scalar("eval/avg_reward", avg_reward, epoch)
    writer.add_scalar("eval/collision_rate", avg_collision_rate, epoch)
    writer.add_scalar("eval/goal_rate", avg_success_rate, epoch)


def main(args=None):
    """ETHSRL+GP的主要训练循环"""

    # 设备和系统初始化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 自动选择设备
    config = TrainingConfig()  # 创建训练配置

    system = HierarchicalNavigationSystem(device=device)  # 初始化分层导航系统
    replay_buffer = TD3ReplayAdapter(buffer_size=config.buffer_size)  # 初始化回放缓冲区

    # 初始化模拟环境
    sim = SIM(world_file="worlds/env_a.yaml", disable_plotting=False)

    # 训练计数器初始化
    episode = 0  # 情节计数器
    epoch = 0    # 训练轮数计数器

    # 主训练循环
    while epoch < config.max_epochs:
        system.reset()  # 重置系统状态
        # 重置模拟环境并获取初始观测
        latest_scan, distance, cos, sin, collision, goal, prev_action, reward = sim.reset()
        prev_action = [0.0, 0.0]  # 初始化动作为零

        # 单个情节变量初始化
        step = 0  # 步数计数器
        episode_reward = 0.0  # 情节累计奖励
        done = False  # 情节完成标志

        # 单个情节循环
        while not done and step < config.max_steps:
            robot_pose = get_robot_pose(sim)  # 获取机器人位姿
            goal_info = [distance, cos, sin]  # 目标信息

            # 确定是否需要新的子目标
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

            # 预测动作（带探索噪声）
            action = system.low_level_controller.predict_action(
                state, add_noise=True, noise_scale=config.exploration_noise
            )
            action = np.clip(action, -1.0, 1.0)  # 裁剪动作到合法范围

            # 将动作转换为控制命令
            lin_cmd = float(np.clip((action[0] + 1.0) / 4.0, 0.0, config.max_lin_velocity))
            ang_cmd = float(np.clip(action[1], -config.max_ang_velocity, config.max_ang_velocity))

            # 在模拟环境中执行动作
            latest_scan, distance, cos, sin, collision, goal, executed_action, reward = sim.step(
                lin_velocity=lin_cmd, ang_velocity=ang_cmd
            )

            episode_reward += reward  # 累计情节奖励

            # 准备下一个状态
            next_prev_action = [executed_action[0], executed_action[1]]
            next_state = system.low_level_controller.process_observation(
                latest_scan,
                system.current_subgoal[0] if system.current_subgoal else subgoal_distance,
                system.current_subgoal[1] if system.current_subgoal else subgoal_angle,
                next_prev_action,
            )

            # 检查终止条件
            done = collision or goal or step == config.max_steps - 1
            # 将经验添加到回放缓冲区
            replay_buffer.add(state, action, reward, float(done), next_state)

            prev_action = next_prev_action  # 更新历史动作
            step += 1  # 增加步数

        episode += 1  # 增加情节计数

        # 记录情节奖励到TensorBoard
        writer = system.low_level_controller.writer
        writer.add_scalar("train/episode_reward", episode_reward, episode)

        # 检查是否应该进行训练
        if (
            replay_buffer.size() >= config.min_buffer_size  # 缓冲区有足够数据
            and episode % config.train_every_n_episodes == 0  # 达到训练间隔
        ):
            # 执行训练迭代
            for _ in range(config.training_iterations):
                system.low_level_controller.update(
                    replay_buffer,
                    batch_size=config.batch_size,
                    discount=0.99,  # 折扣因子
                    tau=0.005,  # 目标网络软更新系数
                    policy_noise=0.2,  # 策略噪声
                    noise_clip=0.5,  # 噪声裁剪
                    policy_freq=2,  # 策略更新频率
                )

        # 检查是否完成一个训练轮次
        if episode % config.episodes_per_epoch == 0:
            epoch += 1  # 增加轮次计数
            evaluate(system, sim, config, epoch)  # 运行评估


if __name__ == "__main__":
    main()  # 运行主函数
