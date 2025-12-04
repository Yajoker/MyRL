from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch

from config import IntegrationConfig
# 导入系统内部模块：低层控制器和高层规划器
from low_level_controller import LowLevelController
from high_level_planner import HighLevelPlanner


class HierarchicalNavigationSystem:
    """
    分层导航系统类（Hierarchical Navigation System）

    该系统将事件触发的高层规划与反应式低层控制相结合，
    以实现高效、安全的自主机器人导航。
    """

    def __init__(
            self,
            laser_dim: int = 180,
            action_dim: int = 2,
            max_action: float = 1.0,
            device=None,
            load_models: bool = False,
            models_directory: Path = Path("ethsrl/models"),
            step_duration: Optional[float] = None,  # 与yaml中的保持一致
            trigger_min_interval: Optional[float] = None,
            subgoal_threshold: Optional[float] = None,
            waypoint_lookahead: Optional[int] = None,
            integration_config: Optional[IntegrationConfig] = None,
    ) -> None:
        """
        初始化分层导航系统。

        参数:
            laser_dim: 激光雷达输入维度（例如180个角度的测距）
            action_dim: 动作维度（一般为 [线速度, 角速度]）
            max_action: 动作的最大幅值
            device: 计算设备（CPU 或 GPU）
            load_models: 是否加载已训练好的模型
            models_directory: 模型文件所在目录
        """
        # 提取集中配置
        self._integration_config = integration_config or IntegrationConfig()  # 集成配置
        motion_cfg = self._integration_config.motion  # 运动配置
        trigger_cfg = self._integration_config.trigger  # 触发器配置
        planner_cfg = self._integration_config.planner  # 规划器配置

        # 将显式参数与集中配置结合
        if step_duration is None:
            step_duration = motion_cfg.dt  # 步长时间
        if trigger_min_interval is None:
            trigger_min_interval = trigger_cfg.min_interval if trigger_cfg.min_interval > 0 else trigger_cfg.min_step_interval * step_duration  # 最小触发间隔
        if subgoal_threshold is None:
            subgoal_threshold = trigger_cfg.subgoal_reach_threshold  # 子目标到达阈值
        if waypoint_lookahead is None:
            waypoint_lookahead = planner_cfg.waypoint_lookahead  # 航点前瞻数量

        # 设置计算设备：若未指定则自动检测 GPU，否则使用 CPU
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 计算状态维度
        low_level_state_dim = laser_dim + 4  # 低层控制器状态维度 = 激光维度 + 4个额外特征

        # 初始化高层规划器（负责生成子目标）
        self.high_level_planner = HighLevelPlanner(
            belief_dim=laser_dim,  # 输入维度（激光雷达特征）
            device=self.device,  # 使用的计算设备
            save_directory=models_directory / "high_level",  # 模型保存路径
            model_name="high_level_planner",  # 模型名称
            load_model=load_models,  # 是否加载已有模型
            step_duration=step_duration,  # 步长时间
            min_interval=trigger_min_interval,  # 最小触发间隔
            subgoal_reach_threshold=subgoal_threshold,  # 子目标到达阈值
            waypoint_lookahead=waypoint_lookahead,  # 航点前瞻数量
            trigger_config=trigger_cfg,  # 触发器配置
            planner_config=planner_cfg,  # 规划器配置
        )

        # 初始化低层控制器（负责执行动作）
        self.low_level_controller = LowLevelController(
            state_dim=low_level_state_dim,  # 低层输入维度
            action_dim=action_dim,  # 输出动作维度
            max_action=max_action,  # 最大动作值
            device=self.device,  # 计算设备
            save_directory=models_directory / "low_level",  # 模型保存路径
            model_name="low_level_controller",  # 模型名称
            load_model=load_models,  # 是否加载已有模型
            max_lin_velocity=self._integration_config.training.max_lin_velocity,
            max_ang_velocity=self._integration_config.training.max_ang_velocity,
        )

        # 系统运行状态变量
        self.current_subgoal = None  # 当前子目标（距离、角度）
        self.current_subgoal_world: Optional[np.ndarray] = None  # 当前子目标在世界坐标系中的位置
        self.prev_policy_action = np.zeros(2, dtype=np.float32)  # 上一步策略输出（归一化）
        self.prev_env_action = [0.0, 0.0]  # 上一步执行的物理动作 [线速度, 角速度]
        self.step_count = 0  # 总步数计数
        self.last_replanning_step = 0  # 上次重新规划的步数（用于事件触发判断）
        self.step_duration = step_duration  # 步长时间
        self.subgoal_threshold = subgoal_threshold  # 子目标到达阈值
        self.waypoint_lookahead = waypoint_lookahead  # 航点前瞻数量
        self._cached_window_info: Dict[str, object] = {}  # 占位，保持接口兼容
        self.last_linear_velocity: float = 0.0

    def step(self, laser_scan, goal_distance, goal_cos, goal_sin, robot_pose, goal_position=None):
        """
        执行导航系统的单步操作（Step）。

        参数:
            laser_scan: 当前激光雷达读数（numpy 数组）
            goal_distance: 机器人到全局目标的距离
            goal_cos: 到全局目标方向的余弦值
            goal_sin: 到全局目标方向的正弦值
            robot_pose: 当前机器人位姿 [x, y, θ]
            goal_position: 全局目标在世界坐标系中的位置 [x, y]

        返回:
            动作 [线速度, 角速度]
        """
        # 步数加一
        self.step_count += 1

        waypoint_candidates = None
        window_metrics: dict = {}

        goal_info = [goal_distance, goal_cos, goal_sin]  # 目标信息

        trigger_flags = self.high_level_planner.check_triggers(
            laser_scan,  # 激光数据
            robot_pose,  # 机器人位姿
            goal_info,  # 目标信息
            current_step=self.step_count,  # 当前步数
            window_metrics=None,  # 窗口指标
        )

        # 标志位：是否需要重新生成子目标
        should_replan = (
            self.current_subgoal_world is None  # 没有子目标时需要重新规划
            or self.high_level_planner.should_replan(trigger_flags)
        )

        subgoal_distance: Optional[float] = None  # 子目标距离
        subgoal_angle: Optional[float] = None  # 子目标角度
        decision_meta: dict = {}  # 决策元数据

        if should_replan:
            # 生成新的子目标
            subgoal_distance, subgoal_angle, decision_meta = self.high_level_planner.generate_subgoal(
                laser_scan,  # 激光数据
                goal_distance,  # 目标距离
                goal_cos,  # 目标余弦
                goal_sin,  # 目标正弦
                robot_pose=robot_pose,  # 机器人位姿
                current_step=self.step_count,  # 当前步数
                waypoints=None,  # 航点候选（mapless模式）
                window_metrics=None,  # 窗口指标
                current_speed=self.last_linear_velocity,  # 近期执行的线速度
            )
            planner_world = self.high_level_planner.current_subgoal_world  # 规划器中的子目标世界坐标
            self.current_subgoal_world = None if planner_world is None else np.asarray(planner_world,
                                                                                       dtype=np.float32)  # 更新当前子目标世界坐标
            self.last_replanning_step = self.step_count  # 记录上次重新规划步数
            # 仅在成功生成新子目标后重置事件触发时间
            self.high_level_planner.event_trigger.reset_time(self.step_count)
        else:
            planner_world = self.high_level_planner.current_subgoal_world  # 规划器中的子目标世界坐标
            if planner_world is not None:
                self.current_subgoal_world = np.asarray(planner_world, dtype=np.float32)  # 更新当前子目标世界坐标

        # 使用最新的机器人姿态计算子目标在机器人坐标系下的表示
        relative_geometry = self.high_level_planner.get_relative_subgoal(robot_pose)  # 获取相对子目标
        if relative_geometry[0] is None:
            if subgoal_distance is not None and subgoal_angle is not None:
                relative_geometry = (float(subgoal_distance), float(subgoal_angle))  # 使用新生成的子目标
            elif self.current_subgoal is not None:
                relative_geometry = self.current_subgoal  # 使用当前子目标
            else:
                relative_geometry = (0.0, 0.0)  # 默认值

        self.current_subgoal = (float(relative_geometry[0]), float(relative_geometry[1]))  # 更新当前子目标

        if should_replan:
            selected_wp = decision_meta.get("selected_waypoint")  # 选择的航点
            if selected_wp is not None:
                print(
                    "New subgoal (wp {}): distance={:.2f}m, angle={:.2f}rad".format(
                        int(selected_wp), self.current_subgoal[0], self.current_subgoal[1]  # 打印新子目标信息
                    )
                )
            else:
                print(
                    "New subgoal: distance={:.2f}m, angle={:.2f}rad".format(
                        self.current_subgoal[0], self.current_subgoal[1]  # 打印新子目标信息
                    )
                )

        # 生成低层输入状态，用于控制器决策
        low_level_state = self.low_level_controller.process_observation(
            laser_scan,  # 激光数据
            self.current_subgoal[0],  # 子目标距离
            self.current_subgoal[1],  # 子目标角度
            self.prev_policy_action  # 上一步的策略动作（用于平滑控制）
        )

        # 通过低层控制器预测下一步动作（网络输出）
        action = self.low_level_controller.predict_action(low_level_state)  # 预测动作
        policy_action = np.clip(action, -1.0, 1.0)

        # 将网络输出映射为实际机器人可执行的速度命令
        env_action = self.low_level_controller.scale_action_for_env(policy_action)
        linear_velocity = float(env_action[0])
        angular_velocity = float(env_action[1])

        # 应用速度缩放屏蔽以提高安全性
        linear_velocity, angular_velocity = self._apply_velocity_shielding(
            linear_velocity,  # 线性速度
            angular_velocity,  # 角速度
            laser_scan,  # 激光数据
        )

        self.last_linear_velocity = float(linear_velocity)

        # 记录当前动作（用于下一次输入）
        self.prev_env_action = [linear_velocity, angular_velocity]
        self.prev_policy_action = policy_action.astype(np.float32, copy=False)

        # 返回控制命令
        return [linear_velocity, angular_velocity]  # 返回动作

    def apply_velocity_shielding(
            self,
            linear_velocity: float,
            angular_velocity: float,
            laser_scan,
    ) -> Tuple[float, float]:
        """公开接口：对外提供零侵入式速度屏蔽能力。"""

        return self._apply_velocity_shielding(linear_velocity, angular_velocity, laser_scan)  # 调用内部速度屏蔽方法

    def _apply_velocity_shielding(
            self,
            linear_velocity: float,
            angular_velocity: float,
            laser_scan,
    ) -> Tuple[float, float]:
        """按最近障碍距离对速度进行单调缩放，实现零侵入式屏蔽。"""

        motion_cfg = self._integration_config.motion  # 运动配置
        shield_cfg = motion_cfg.shielding  # 屏蔽配置
        if not shield_cfg.enabled:
            return float(linear_velocity), float(angular_velocity)  # 屏蔽未启用，直接返回原速度

        scan_arr = np.asarray(laser_scan, dtype=np.float32)  # 转换为numpy数组
        finite_scan = scan_arr[np.isfinite(scan_arr)]  # 过滤有效值
        if finite_scan.size == 0:
            return float(linear_velocity), float(angular_velocity)  # 无有效激光数据，直接返回

        d_min = float(finite_scan.min())  # 最小障碍距离

        # Logistic 缩放确保线速度在安全距离附近平滑衰减
        sigma_input = float(np.clip(shield_cfg.gain * (d_min - shield_cfg.safe_distance), -60.0, 60.0))  # 计算缩放输入
        linear_scale = float(1.0 / (1.0 + np.exp(-sigma_input)))  # 线性缩放因子

        scaled_linear = float(linear_velocity * linear_scale)  # 缩放后的线性速度
        scaled_linear = float(np.clip(scaled_linear, 0.0, motion_cfg.v_max))  # 裁剪到最大速度

        scaled_angular = float(angular_velocity)  # 角速度
        if d_min <= shield_cfg.safe_distance:  # 如果距离小于安全距离
            scaled_angular = float(scaled_angular * shield_cfg.angular_gain)  # 缩放角速度

        scaled_angular = float(np.clip(scaled_angular, -motion_cfg.omega_max, motion_cfg.omega_max))  # 裁剪角速度

        return scaled_linear, scaled_angular  # 返回缩放后的速度

    def reset(self):
        """
        重置整个导航系统的内部状态。
        （例如在新仿真回合开始时调用）
        """
        self.current_subgoal = None  # 清空子目标
        self.current_subgoal_world = None  # 清空子目标世界坐标
        self.prev_env_action = [0.0, 0.0]  # 重置上一步动作
        self.prev_policy_action = np.zeros(2, dtype=np.float32)
        self.step_count = 0  # 步数归零
        self.last_replanning_step = 0  # 清除上次规划记录
        self.high_level_planner.current_subgoal = None  # 重置高层规划器子目标
        self.high_level_planner.current_subgoal_world = None  # 重置高层规划器子目标世界坐标
        self.high_level_planner.event_trigger.reset_state()  # 重置事件触发器状态
        self.high_level_planner.reset_subgoal_hidden()  # 清空子目标网络隐状态
        self._cached_window_info = {}

    # ------------------------------------------------------------------
    # 兼容旧接口的空实现（mapless 模式下不会使用全局航点/窗口）
    # ------------------------------------------------------------------
    def plan_global_route(self, *_, **__):
        """为兼容性保留的占位方法：mapless 模式下不执行任何操作。"""
        return []

    def get_active_waypoints(self, *_, **__):
        """为兼容性保留的占位方法：返回空列表以避免 AttributeError。"""
        return []

    def update_window_state(self, *_, **__):
        """为兼容性保留的占位方法：返回空 dict 以保持调用方稳定。"""
        return {}


def create_navigation_system(
        load_models: bool = False,
        subgoal_threshold: float = 0.5,
        waypoint_lookahead: int = 3,
):
    """
    工厂函数：创建一个完整的分层导航系统实例。

    参数:
        load_models: 是否加载预训练模型

    返回:
        初始化好的 HierarchicalNavigationSystem 对象
    """
    return HierarchicalNavigationSystem(
        laser_dim=180,  # 激光维度
        action_dim=2,  # 动作维度
        max_action=1.0,  # 最大动作幅值
        load_models=load_models,  # 是否加载模型
        subgoal_threshold=subgoal_threshold,  # 子目标阈值
        waypoint_lookahead=waypoint_lookahead,  # 航点前瞻数量
    )
