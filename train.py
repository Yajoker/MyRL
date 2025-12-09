from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

# 导入自定义模块
from config import ConfigBundle, HighLevelRewardConfig, LowLevelRewardConfig, TrainingConfig
from integration import HierarchicalNavigationSystem
from rewards import compute_high_level_reward, compute_low_level_reward, compute_step_safety_cost
from robot_nav.SIM_ENV.sim import SIM
from replay_buffer import HighLevelReplayBuffer, ReplayBuffer


@dataclass
class SubgoalContext:
    """高层子目标生命周期内的统计上下文"""

    start_state: np.ndarray  # 子目标开始时的状态
    action: np.ndarray  # 选择的子目标几何 [距离, 角度]
    world_target: np.ndarray  # 子目标的全局坐标
    start_goal_distance: float  # 开始时的目标距离
    last_goal_distance: float  # 最后的目标距离
    low_level_return: float = 0.0  # 累积的低层奖励
    steps: int = 0  # 子目标执行的步数
    subgoal_completed: bool = False  # 子目标是否完成
    last_state: Optional[np.ndarray] = None  # 最后的状态
    min_dmin: float = float("inf")  # 子目标执行期间观测到的最近障碍距离
    collision_occurred: bool = False  # 执行期间是否发生碰撞
    subgoal_angle_at_start: Optional[float] = None  # 子目标生成时的角度
    short_cost_sum: float = 0.0  # 短期安全成本累计
    near_obstacle_steps: int = 0  # 近障碍步数


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
    buffer,
    high_cfg: HighLevelRewardConfig,
    done: bool,
    reached_goal: bool,
    collision: bool,
    timed_out: bool,
) -> Optional[Tuple[dict, None]]:
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
        包含奖励分量字典的元组，或None
    """

    # 检查上下文有效性
    if context is None or context.steps == 0:  # 上下文为空或步数为0
        return None  # 返回None

    # 确定最后状态
    last_state = context.last_state if context.last_state is not None else context.start_state  # 使用最后状态或开始状态

    collision_flag = collision or context.collision_occurred

    # 计算高层奖励
    (reward_eff, safety_cost), components = compute_high_level_reward(  # 调用高层奖励计算函数
        start_goal_distance=context.start_goal_distance,  # 开始目标距离
        end_goal_distance=context.last_goal_distance,  # 结束目标距离
        subgoal_step_count=context.steps,  # 子目标步数
        collision=collision_flag,  # 是否碰撞
        config=high_cfg,  # 高层奖励配置
        short_cost_sum=context.short_cost_sum,
        near_obstacle_steps=context.near_obstacle_steps,
    )

    components.update(
        {
            "start_global_distance": float(context.start_goal_distance),
            "end_global_distance": float(context.last_goal_distance),
            "subgoal_steps": float(context.steps),
            "collision_flag": float(collision_flag),
            "subgoal_completed": float(context.subgoal_completed),
            "timed_out": float(timed_out),
            "reached_goal": float(reached_goal),
        }
    )

    # 将经验添加到缓冲区
    buffer.add(
        context.start_state.astype(np.float32, copy=False),
        context.action.astype(np.float32, copy=False),
        float(reward_eff),
        float(safety_cost),
        float(done),
        last_state.astype(np.float32, copy=False),
    )

    return components, None


def maybe_train_high_level(
    planner,
    buffer: HighLevelReplayBuffer,
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

    if buffer.size() < batch_size:
        return None

    states, actions, rewards_eff, safety_costs, dones, next_states = buffer.sample(batch_size)

    # 更新规划器
    metrics = planner.update_planner(
        states,
        actions,
        rewards_eff,
        safety_costs,
        dones,
        next_states,
        batch_size=batch_size,
    )  # 更新高层规划器
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
        latest_scan, distance, cos, sin, collision, goal, _, _ = sim.reset()  # 重置仿真环境
        prev_policy_action = np.zeros(2, dtype=np.float32)  # 初始化策略动作
        prev_env_action = [0.0, 0.0]  # 初始化物理动作
        current_subgoal_world: Optional[np.ndarray] = None  # 当前子目标世界坐标
        robot_pose = get_robot_pose(sim)  # 获取机器人位姿
        eval_goal_pose = get_goal_pose(sim)  # 获取评估目标位姿
        done = False  # 终止标志
        steps = 0  # 步数计数器
        episode_reward = 0.0  # 情节奖励
        current_subgoal_completed = False  # 当前子目标完成标志

        # 单次评估情节循环
        while not done and steps < config.max_steps:  # 当未终止且未超时时循环
            robot_pose = get_robot_pose(sim)  # 获取机器人位姿
            active_waypoints: list = []
            window_metrics: dict = {}
            goal_info = [distance, cos, sin]  # 目标信息

            scan_arr = np.asarray(latest_scan, dtype=np.float32)
            risk_index, d_min, d_percentile = system.high_level_planner.compute_risk_index(scan_arr)

            trigger_flags = system.high_level_planner.check_triggers(
                latest_scan,  # 最新激光数据
                robot_pose,  # 机器人位姿
                goal_info,  # 目标信息
                risk_index=risk_index,
                current_step=steps,  # 当前步数
                window_metrics=None,  # 窗口指标
            )
            # 检查是否需要重新规划
            should_replan = (
                system.high_level_planner.current_subgoal_world is None  # 没有当前子目标
                or system.high_level_planner.should_replan(trigger_flags)  # 或触发器条件满足
            )

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
                    robot_pose=robot_pose,  # 机器人位姿
                    current_step=steps,  # 当前步数
                    waypoints=None,  # 活动航点
                    window_metrics=None,  # 窗口指标
                )
                planner_world = system.high_level_planner.current_subgoal_world  # 规划器子目标世界坐标
                current_subgoal_world = np.asarray(planner_world, dtype=np.float32) if planner_world is not None else None  # 当前子目标世界坐标
                # 仅在成功生成新子目标后重置事件触发时间
                system.high_level_planner.event_trigger.reset_time(steps)
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
                prev_policy_action,  # 上次策略动作
            )

            # 预测动作（无探索噪声）
            action = system.low_level_controller.predict_action(state, add_noise=False)  # 预测动作（无噪声）
            policy_action = np.clip(action, -1.0, 1.0)
            env_action = system.low_level_controller.scale_action_for_env(policy_action)
            lin_cmd = float(env_action[0])
            ang_cmd = float(env_action[1])
            lin_cmd, ang_cmd = system.apply_velocity_shielding(lin_cmd, ang_cmd, latest_scan)  # 应用速度屏蔽

            # 执行动作
            latest_scan, distance, cos, sin, collision, goal, _, _ = sim.step(  # 执行一步仿真
                lin_velocity=lin_cmd,  # 线性速度
                ang_velocity=ang_cmd,  # 角速度
            )

            # 使用动作后的激光数据计算奖励所需的最小障碍距离
            post_scan = np.asarray(latest_scan, dtype=np.float32)
            finite_scan = post_scan[np.isfinite(post_scan)]
            if finite_scan.size > 0:
                risk_percentile = getattr(
                    system.high_level_planner.event_trigger, "risk_percentile", 10.0
                )
                min_obstacle_distance = float(
                    np.percentile(finite_scan, risk_percentile)
                )
            else:
                min_obstacle_distance = 8.0

            # 更新子目标距离
            next_pose = get_robot_pose(sim)  # 获取下一时刻机器人位姿
            post_window_metrics: dict = {}
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
            if prev_env_action is not None:  # 如果有上次动作
                delta_lin = float(lin_cmd - prev_env_action[0])  # 线性速度变化
                delta_ang = float(ang_cmd - prev_env_action[1])  # 角速度变化
                action_delta = [delta_lin, delta_ang]  # 动作变化量

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
                config=low_cfg,  # 低层奖励配置
            )

            # 更新统计
            episode_reward += low_reward  # 累加情节奖励
            steps += 1  # 步数加1
            prev_env_action = [lin_cmd, ang_cmd]
            prev_policy_action = policy_action.astype(np.float32, copy=False)

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
    print("   • Global planner: disabled (mapless mode)")  # 全局规划器关闭，启用无图导航
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
        waypoint_lookahead=config.waypoint_lookahead,  # 航点前瞻数量（对 mapless 分支无影响）
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
    trigger_cfg = integration_config.trigger
    high_level_buffer = HighLevelReplayBuffer(buffer_size=config.buffer_size, random_seed=config.random_seed or 666)
    current_subgoal_context: Optional[SubgoalContext] = None  # 当前子目标上下文

    # ========== 主训练循环 ==========
    while epoch < config.max_epochs:  # 当轮次小于最大轮次时循环
        # 重置环境和系统状态
        system.reset()  # 重置系统
        current_subgoal_context = None  # 重置子目标上下文
        system.current_subgoal = None  # 重置当前子目标

        latest_scan, distance, cos, sin, collision, goal, _, _ = sim.reset()  # 重置仿真环境
        prev_policy_action = np.zeros(2, dtype=np.float32)  # 归一化动作历史
        prev_env_action = [0.0, 0.0]  # 物理动作历史
        current_subgoal_world: Optional[np.ndarray] = None  # 当前子目标世界坐标

        robot_pose = get_robot_pose(sim)  # 获取机器人位姿
        episode_goal_pose = get_goal_pose(sim)  # 获取情节目标位姿

        steps = 0  # 步数计数器
        episode_reward = 0.0  # 情节奖励
        done = False  # 终止标志
        current_subgoal_completed = False  # 当前子目标完成标志

        # ========== 单次情节循环 ==========
        while not done and steps < config.max_steps:  # 当未终止且未超时时循环
            robot_pose = get_robot_pose(sim)  # 获取机器人位姿
            window_metrics: dict = {}
            waypoint_sequence: list = []
            goal_info = [distance, cos, sin]  # 目标信息

            scan_arr = np.asarray(latest_scan, dtype=np.float32)
            risk_index, d_min, d_percentile = system.high_level_planner.compute_risk_index(scan_arr)

            trigger_flags = system.high_level_planner.check_triggers(
                latest_scan,  # 最新激光数据
                robot_pose,  # 机器人位姿
                goal_info,  # 目标信息
                risk_index=risk_index,
                current_step=steps,  # 当前步数
                window_metrics=None,  # 窗口指标
            )
            # 检查是否需要重新规划子目标
            should_replan = (
                system.high_level_planner.current_subgoal_world is None  # 没有当前子目标
                or system.high_level_planner.should_replan(trigger_flags)  # 或触发器条件满足
            )

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
                    finalize_components, _ = finalize_result  # 解包结果
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
                    robot_pose=robot_pose,  # 机器人位姿
                    current_step=steps,  # 当前步数
                    waypoints=None,  # 活动航点
                    window_metrics=None,  # 窗口指标
                )
                planner_world = system.high_level_planner.current_subgoal_world  # 规划器子目标世界坐标
                current_subgoal_world = np.asarray(planner_world, dtype=np.float32) if planner_world is not None else None  # 当前子目标世界坐标
                # 生成新的子目标后统一重置事件触发时间
                system.high_level_planner.event_trigger.reset_time(steps)
                if current_subgoal_world is None:  # 如果没有子目标世界坐标
                    current_subgoal_world = compute_subgoal_world(robot_pose, subgoal_distance, subgoal_angle)  # 计算子目标世界坐标

                # 构建高层状态向量
                start_state = system.high_level_planner.build_state_vector(  # 构建状态向量
                    latest_scan,  # 激光数据
                    distance,  # 目标距离
                    cos,  # 目标余弦
                    sin,  # 目标正弦
                    waypoints=waypoint_sequence,  # 航点序列
                    robot_pose=robot_pose,  # 机器人位姿
                )

                # 创建新的子目标上下文
                current_subgoal_context = SubgoalContext(  # 创建子目标上下文
                    start_state=start_state.astype(np.float32, copy=False),  # 开始状态
                    action=np.array([subgoal_distance, subgoal_angle], dtype=np.float32),  # 子目标几何
                    world_target=current_subgoal_world,  # 世界目标
                    start_goal_distance=distance,  # 开始目标距离
                    last_goal_distance=distance,  # 最后目标距离
                    low_level_return=0.0,  # 低层回报
                    steps=0,  # 步数
                    subgoal_completed=False,  # 子目标完成标志
                    last_state=start_state.astype(np.float32, copy=False),  # 最后状态
                    subgoal_angle_at_start=float(subgoal_angle) if subgoal_angle is not None else None,  # 子目标开始角度
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
                prev_policy_action,  # 上次归一化策略动作
            )

            # 预测动作（带探索噪声）
            raw_action = system.low_level_controller.predict_action(  # 预测动作
                state,
                add_noise=True,  # 添加噪声
                noise_scale=config.exploration_noise,  # 噪声尺度
            )
            policy_action = np.clip(raw_action, -1.0, 1.0)  # 归一化策略动作

            # 转换为实际控制命令（未屏蔽的环境动作）
            env_action = system.low_level_controller.scale_action_for_env(policy_action)
            env_lin_cmd = float(env_action[0])  # 线性速度命令
            env_ang_cmd = float(env_action[1])  # 角速度命令
            lin_cmd, ang_cmd = system.apply_velocity_shielding(env_lin_cmd, env_ang_cmd, latest_scan)  # 应用速度屏蔽

            # 执行动作
            latest_scan, distance, cos, sin, collision, goal, executed_action, _ = sim.step(  # 执行一步仿真
                lin_velocity=lin_cmd,  # 线性速度
                ang_velocity=ang_cmd,  # 角速度
            )

            # 使用动作后的激光数据刷新奖励所需的最小障碍距离
            post_scan = np.asarray(latest_scan, dtype=np.float32)
            finite_scan = post_scan[np.isfinite(post_scan)]
            if finite_scan.size > 0:
                risk_percentile = getattr(
                    system.high_level_planner.event_trigger, "risk_percentile", 10.0
                )
                min_obstacle_distance = float(
                    np.percentile(finite_scan, risk_percentile)
                )
            else:
                min_obstacle_distance = 8.0

            # 更新子目标距离
            next_pose = get_robot_pose(sim)  # 获取下一时刻机器人位姿
            post_window_metrics: dict = {}
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
            if executed_action is not None and prev_env_action is not None:  # 如果有执行动作和上次动作
                delta_lin = float(executed_action[0] - prev_env_action[0])  # 线性速度变化
                delta_ang = float(executed_action[1] - prev_env_action[1])  # 角速度变化
                action_delta = [delta_lin, delta_ang]  # 动作变化量

            if current_subgoal_context is not None:  # 如果有当前子目标上下文
                current_subgoal_context.min_dmin = min(  # 更新最小障碍距离
                    current_subgoal_context.min_dmin,
                    min_obstacle_distance,
                )
                step_cost = compute_step_safety_cost(
                    risk_index,
                    collision,
                    config=high_reward_cfg,
                )
                current_subgoal_context.short_cost_sum += step_cost
                if risk_index >= trigger_cfg.risk_near_threshold:
                    current_subgoal_context.near_obstacle_steps += 1
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
                # 构建下一状态向量（mapless 模式不再使用全局航点）
                next_state_vector = system.high_level_planner.build_state_vector(  # 构建下一状态向量
                    latest_scan,  # 激光数据
                    distance,  # 目标距离
                    cos,  # 目标余弦
                    sin,  # 目标正弦
                    waypoints=None,  # mapless: 不再提供活动航点
                    robot_pose=next_pose,  # 下一时刻机器人位姿
                )
                current_subgoal_context.last_state = next_state_vector.astype(np.float32, copy=False)  # 更新最后状态

            # 准备下一状态
            next_policy_action = policy_action.astype(np.float32, copy=False)
            next_state = system.low_level_controller.process_observation(  # 处理下一状态观测
                latest_scan,  # 激光数据
                post_subgoal_distance,  # 后子目标距离
                post_subgoal_angle,  # 后子目标角度
                next_policy_action,  # 下一时刻策略动作
            )

            # 检查终止条件
            done = collision or goal or steps == config.max_steps - 1  # 终止条件

            # 添加经验到回放缓冲区（存储未屏蔽的环境动作）
            scaled_env_action = np.array([env_lin_cmd, env_ang_cmd], dtype=np.float32)

            #low_reward=0.2*low_reward  # 奖励缩放
            #replay_buffer.add(state, scaled_env_action, low_reward, float(done), next_state)  # 添加到回放缓冲区
            # ✅ 用 policy_action 作为 replay buffer 里的动作
            replay_buffer.add(state, policy_action, low_reward, float(done), next_state)  # 添加到回放缓冲区

            # 定期输出回放缓冲区大小与奖励
            if steps % 50 == 0:  # 每50步输出一次
                buffer_size = replay_buffer.size()  # 缓冲区大小
                print(
                    f"🏃 Training | Epoch {epoch:2d}/{config.max_epochs} | "  # 训练信息
                    f"Episode {episode:3d}/{config.max_epochs*config.episodes_per_epoch} | "
                    f"Step {steps:3d}/{config.max_steps} | "
                    f"Reward: {low_reward:7.2f} | Buffer: {buffer_size:6d}"
                )

            prev_policy_action = next_policy_action  # 更新策略动作
            prev_env_action = [executed_action[0], executed_action[1]]  # 更新物理动作
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
            finalize_components, _ = finalize_result  # 解包结果
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
