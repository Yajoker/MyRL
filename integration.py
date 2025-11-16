from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch

from config import IntegrationConfig
# 导入系统内部模块：低层控制器和高层规划器
from low_level_controller import LowLevelController
from global_planner import GlobalPlanner, WaypointWindow
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
            world_file: Optional[Path] = None,
            global_plan_resolution: Optional[float] = None,
            global_plan_margin: Optional[float] = None,
            waypoint_lookahead: Optional[int] = None,
            window_step_limit: Optional[int] = None,
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
        safety_cfg = self._integration_config.safety_critic  # 安全评估配置
        window_cfg = self._integration_config.window  # 窗口配置

        # 将显式参数与集中配置结合
        if step_duration is None:
            step_duration = motion_cfg.dt  # 步长时间
        if trigger_min_interval is None:
            trigger_min_interval = trigger_cfg.min_interval if trigger_cfg.min_interval > 0 else trigger_cfg.min_step_interval * step_duration  # 最小触发间隔
        if subgoal_threshold is None:
            subgoal_threshold = trigger_cfg.subgoal_reach_threshold  # 子目标到达阈值
        if global_plan_resolution is None:
            global_plan_resolution = planner_cfg.resolution  # 全局规划分辨率
        if global_plan_margin is None:
            global_plan_margin = planner_cfg.safety_margin  # 全局规划安全边界
        if waypoint_lookahead is None:
            waypoint_lookahead = planner_cfg.waypoint_lookahead  # 航点前瞻数量
        if window_step_limit is None:
            window_step_limit = window_cfg.step_limit  # 窗口步数限制

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
            safety_config=safety_cfg,  # 安全评估配置
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
        self.prev_action = [0.0, 0.0]  # 上一步执行的动作 [线速度, 角速度]
        self.step_count = 0  # 总步数计数
        self.last_replanning_step = 0  # 上次重新规划的步数（用于事件触发判断）
        self.step_duration = step_duration  # 步长时间
        self.subgoal_threshold = subgoal_threshold  # 子目标到达阈值
        self.waypoint_lookahead = waypoint_lookahead  # 航点前瞻数量
        self.window_step_limit = max(1, int(window_step_limit))  # 窗口步数限制
        self.window_step_count = 0  # 当前窗口步数计数
        self.steps_inside_window = 0  # 在窗口内的步数
        self.window_last_index: Optional[int] = None  # 上次窗口索引
        self.last_window_distance: Optional[float] = None  # 上次窗口距离
        self.window_limit_exceeded = False  # 窗口限制是否超限
        self.window_within = False  # 是否在窗口内
        self.last_window_update_step: int = -1  # 上次窗口更新步数
        self._cached_window_info: Dict[str, object] = {}  # 缓存的窗口信息
        self._printed_plan_overview = False  # 是否已打印规划概览

        self.global_planner: Optional[GlobalPlanner] = None  # 全局规划器
        if world_file is not None:
            try:
                self.global_planner = GlobalPlanner(
                    world_file=world_file,  # 世界文件
                    resolution=global_plan_resolution,  # 分辨率
                    safety_margin=global_plan_margin,  # 安全边界
                )
            except FileNotFoundError as exc:
                print(f"[GlobalPlanner] {exc}. Global planning disabled.")  # 全局规划禁用
        self.global_waypoints: List[WaypointWindow] = []  # 全局航点列表
        self.current_waypoint_index: int = 0  # 当前航点索引
        self.global_goal: Optional[np.ndarray] = None  # 全局目标位置

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

        # 如果提供了最新的目标位置，则确保全局路径已更新
        if goal_position is not None:
            self.plan_global_route(robot_pose, goal_position)  # 规划全局路径
        elif self.global_waypoints:
            self._advance_waypoints(robot_pose)  # 推进航点

        waypoint_candidates = self.get_active_waypoints(
            robot_pose, include_indices=True  # 获取活动航点候选
        )
        window_metrics = self.update_window_state(robot_pose, waypoint_candidates)  # 更新窗口状态

        goal_info = [goal_distance, goal_cos, goal_sin]  # 目标信息

        # 标志位：是否需要重新生成子目标
        if self.current_subgoal_world is None:
            should_replan = True  # 没有子目标时需要重新规划
        else:
            should_replan = self.high_level_planner.check_triggers(
                laser_scan,  # 激光数据
                robot_pose,  # 机器人位姿
                goal_info,  # 目标信息
                current_step=self.step_count,  # 当前步数
                window_metrics=window_metrics,  # 窗口指标
            )
        if window_metrics.get("limit_exceeded", False):
            should_replan = True  # 窗口限制超限时需要重新规划

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
                waypoints=waypoint_candidates,  # 航点候选
                window_metrics=window_metrics,  # 窗口指标
            )
            self.reset_window_tracking()  # 重置窗口跟踪
            self.update_selected_waypoint(decision_meta.get("selected_waypoint"))  # 更新选择的航点
            planner_world = self.high_level_planner.current_subgoal_world  # 规划器中的子目标世界坐标
            self.current_subgoal_world = None if planner_world is None else np.asarray(planner_world,
                                                                                       dtype=np.float32)  # 更新当前子目标世界坐标
            self.last_replanning_step = self.step_count  # 记录上次重新规划步数
            self.high_level_planner.event_trigger.reset_time(self.step_count)  # 重置事件触发器时间
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
            self.prev_action  # 上一步的动作（用于平滑控制）
        )

        # 通过低层控制器预测下一步动作（网络输出）
        action = self.low_level_controller.predict_action(low_level_state)  # 预测动作

        # 将网络输出映射为实际机器人可执行的速度命令
        # 将线速度从 [-1,1] 映射为 [0,0.5]
        linear_velocity = (action[0] + 1) / 4  # 线性速度
        # 保持角速度在 [-1,1] 范围内
        angular_velocity = action[1]  # 角速度

        # 应用速度缩放屏蔽以提高安全性
        linear_velocity, angular_velocity = self._apply_velocity_shielding(
            linear_velocity,  # 线性速度
            angular_velocity,  # 角速度
            laser_scan,  # 激光数据
        )

        # 记录当前动作（用于下一次输入）
        self.prev_action = [linear_velocity, angular_velocity]  # 更新上次动作

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

    def plan_global_route(self, robot_pose, goal_position, force: bool = False):
        """计算（或刷新）全局航点序列。"""

        if self.global_planner is None:
            return []  # 没有全局规划器，返回空列表

        start_xy = np.asarray(robot_pose[:2], dtype=np.float32)  # 起始位置
        goal_vec = np.asarray(goal_position[:2], dtype=np.float32)  # 目标位置

        if (
                not force
                and self.global_goal is not None
                and self.global_waypoints
                and np.linalg.norm(goal_vec - self.global_goal) < 1e-4  # 目标未改变
        ):
            self._advance_waypoints(robot_pose)  # 推进航点
            return self.global_waypoints  # 返回现有航点

        self.global_goal = goal_vec  # 更新全局目标

        try:
            raw_path = self.global_planner.plan(start_xy, goal_vec)  # 规划路径
        except RuntimeError as exc:
            print(f"[GlobalPlanner] {exc}. Using direct segment to goal.")  # 规划失败，使用直接路径
            raw_path = [WaypointWindow(center=goal_vec.copy(), radius=self.global_planner.window_radius)]  # 创建目标窗口

        filtered: List[WaypointWindow] = []  # 过滤后的路径
        for window in raw_path:
            centre = np.asarray(window.center, dtype=np.float32)  # 窗口中心
            if np.linalg.norm(centre - start_xy) <= 0.5 * self.global_planner.resolution:  # 距离太近则跳过
                continue
            filtered.append(WaypointWindow(center=centre.copy(), radius=float(window.radius)))  # 添加到过滤列表

        if not filtered:
            filtered = [
                WaypointWindow(center=goal_vec.copy(), radius=self.global_planner.window_radius)]  # 如果没有航点，添加目标窗口

        if not self._printed_plan_overview:
            goal_dists = [float(np.linalg.norm(goal_vec - wp.center)) for wp in filtered]  # 计算各航点到目标的距离
            print(f"[GlobalPlanner] Waypoints in world frame ({len(filtered)} total):")  # 打印航点信息
            for idx, (wp, dist) in enumerate(zip(filtered, goal_dists)):
                print(
                    f"  #{idx:02d} {wp.center.tolist()} | dist_to_goal={dist:.3f} m | radius={wp.radius:.2f} m")  # 打印每个航点

            if len(goal_dists) > 1:
                # 检查距离是否单调递减
                monotonic = all(goal_dists[i + 1] <= goal_dists[i] + 1e-6 for i in range(len(goal_dists) - 1))
                if not monotonic:
                    print(
                        "[GlobalPlanner] Warning: waypoint distance does not consistently decrease toward the goal.")  # 警告：距离不单调递减

            self._printed_plan_overview = True  # 标记已打印规划概览

        self.global_waypoints = filtered  # 更新全局航点
        self.current_waypoint_index = 0  # 重置当前航点索引
        self.reset_window_tracking()  # 重置窗口跟踪
        self._advance_waypoints(robot_pose)  # 推进航点
        return self.global_waypoints  # 返回全局航点

    def reset_window_tracking(self) -> None:
        """重置窗口跟踪状态。"""
        self.window_last_index = None  # 上次窗口索引
        self.window_step_count = 0  # 窗口步数计数
        self.steps_inside_window = 0  # 窗口内步数
        self.last_window_distance = None  # 上次窗口距离
        self.window_limit_exceeded = False  # 窗口限制超限标志
        self.window_within = False  # 是否在窗口内
        self.last_window_update_step = -1  # 上次窗口更新步数
        self._cached_window_info = {}  # 清空缓存窗口信息

    def _advance_waypoints(self, robot_pose) -> None:
        """根据机器人当前位置推进航点索引。"""
        if not self.global_waypoints:
            return  # 没有航点，直接返回
        if robot_pose is None:
            return  # 没有机器人位姿，直接返回

        position = np.asarray(robot_pose[:2], dtype=np.float32)  # 机器人位置
        base_threshold = max(0.1, self.subgoal_threshold * 0.8)  # 基础阈值
        if self.global_planner is not None:
            base_threshold = max(base_threshold, 0.5 * self.global_planner.resolution)  # 考虑规划器分辨率

        window_changed = False  # 窗口是否改变标志
        while self.current_waypoint_index < len(self.global_waypoints):
            window = self.global_waypoints[self.current_waypoint_index]  # 当前窗口
            centre = np.asarray(window.center, dtype=np.float32)  # 窗口中心
            radius = float(window.radius)  # 窗口半径
            threshold = max(base_threshold, radius * 0.9)  # 计算阈值
            distance = float(np.linalg.norm(centre - position))  # 计算距离

            if distance <= threshold and self.current_waypoint_index < len(
                    self.global_waypoints) - 1:  # 如果距离小于阈值且不是最后一个航点
                self.current_waypoint_index += 1  # 推进到下一个航点
                window_changed = True  # 标记窗口改变
                continue
            break

        if window_changed:
            self.reset_window_tracking()  # 重置窗口跟踪

    def _update_window_metrics(
            self,
            robot_pose,
            waypoint_candidates: List[Tuple[int, WaypointWindow]],
    ) -> None:
        """更新窗口指标。"""
        if not waypoint_candidates:
            self.reset_window_tracking()  # 没有候选航点，重置跟踪
            return

        first = waypoint_candidates[0]  # 第一个候选
        if isinstance(first, tuple) and len(first) == 2:
            index, window = int(first[0]), first[1]  # 提取索引和窗口
        else:
            index = self.current_waypoint_index  # 使用当前航点索引
            window = first  # 使用第一个窗口
        robot_xy = np.asarray(robot_pose[:2], dtype=np.float32)  # 机器人位置
        centre = np.asarray(window.center, dtype=np.float32)  # 窗口中心
        distance = float(np.linalg.norm(centre - robot_xy))  # 计算距离
        radius = float(window.radius)  # 窗口半径

        changed = index != self.window_last_index  # 检查索引是否改变
        prev_distance = self.last_window_distance if not changed else None  # 上次距离
        prev_inside = self.window_within if not changed else False  # 上次是否在窗口内

        if changed:
            self.window_last_index = index  # 更新窗口索引
            self.window_step_count = 0  # 重置窗口步数
            self.steps_inside_window = 0  # 重置窗口内步数
            self.window_limit_exceeded = False  # 重置限制超限标志
            prev_inside = False  # 重置上次在窗口内标志

        self.window_step_count += 1  # 增加窗口步数

        inside = distance <= radius  # 检查是否在窗口内
        if inside:
            self.steps_inside_window += 1  # 增加窗口内步数
        else:
            self.steps_inside_window = 0  # 重置窗口内步数

        entered = inside and not prev_inside  # 检查是否刚进入窗口

        self.window_within = inside  # 更新是否在窗口内
        self.window_limit_exceeded = self.steps_inside_window >= self.window_step_limit  # 检查是否超限
        self.last_window_distance = distance  # 更新上次窗口距离
        self.last_window_update_step = self.step_count  # 更新上次窗口更新步数

        margin = (distance - radius) / max(radius, 1e-3)  # 计算边界
        margin = float(np.clip(margin, -1.0, 1.0))  # 裁剪边界值

        self._cached_window_info = {  # 更新缓存窗口信息
            "index": index,  # 索引
            "radius": radius,  # 半径
            "distance": distance,  # 距离
            "prev_distance": prev_distance,  # 上次距离
            "inside": inside,  # 是否在窗口内
            "entered": entered,  # 是否刚进入
            "steps_inside": self.steps_inside_window,  # 窗口内步数
            "step_count": self.window_step_count,  # 窗口步数计数
            "step_limit": self.window_step_limit,  # 窗口步数限制
            "limit_exceeded": self.window_limit_exceeded,  # 是否超限
            "margin": margin,  # 边界
        }

    def update_window_state(
            self,
            robot_pose,
            waypoint_candidates: List[Tuple[int, WaypointWindow]],
    ) -> Dict[str, object]:
        """更新窗口状态并返回指标。"""
        self._update_window_metrics(robot_pose, waypoint_candidates)  # 更新窗口指标
        return self.get_window_metrics()  # 返回窗口指标

    def get_window_metrics(self) -> Dict[str, object]:
        """获取窗口指标。"""
        return dict(self._cached_window_info)  # 返回缓存的窗口信息

    def get_active_waypoints(self, robot_pose, lookahead: Optional[int] = None, include_indices: bool = False):
        """返回离机器人最近的航点窗口。"""

        if not self.global_waypoints:
            return []  # 没有全局航点，返回空列表

        self._advance_waypoints(robot_pose)  # 推进航点

        horizon = lookahead or self.waypoint_lookahead  # 前瞻范围
        start_idx = min(self.current_waypoint_index, len(self.global_waypoints) - 1)  # 起始索引
        end_idx = min(len(self.global_waypoints), start_idx + max(1, horizon))  # 结束索引

        indices = range(start_idx, end_idx)  # 索引范围
        if include_indices:
            return [(idx, self.global_waypoints[idx].clone()) for idx in indices]  # 返回带索引的航点
        return [self.global_waypoints[idx].clone() for idx in indices]  # 返回航点

    def update_selected_waypoint(self, selected_index: Optional[int]) -> None:
        """记录高层规划器选择的全局航点。"""

        if selected_index is None or not self.global_waypoints:
            return  # 没有选择索引或没有航点，直接返回

        idx = int(selected_index)  # 转换为整数
        if idx < 0:
            idx = 0  # 确保不小于0
        if idx >= len(self.global_waypoints):
            idx = len(self.global_waypoints) - 1  # 确保不超过最大值

        if idx >= self.current_waypoint_index:
            if idx != self.current_waypoint_index:
                self.current_waypoint_index = idx  # 更新当前航点索引
                self.reset_window_tracking()  # 重置窗口跟踪
            else:
                self.current_waypoint_index = idx  # 更新当前航点索引

    def clear_global_route(self) -> None:
        """重置存储的航点和目标信息。"""

        self.global_waypoints = []  # 清空全局航点
        self.current_waypoint_index = 0  # 重置当前航点索引
        self.global_goal = None  # 清空全局目标
        self.reset_window_tracking()  # 重置窗口跟踪

    def reset(self):
        """
        重置整个导航系统的内部状态。
        （例如在新仿真回合开始时调用）
        """
        self.current_subgoal = None  # 清空子目标
        self.current_subgoal_world = None  # 清空子目标世界坐标
        self.prev_action = [0.0, 0.0]  # 重置上一步动作
        self.step_count = 0  # 步数归零
        self.last_replanning_step = 0  # 清除上次规划记录
        self.high_level_planner.current_subgoal = None  # 重置高层规划器子目标
        self.high_level_planner.current_subgoal_world = None  # 重置高层规划器子目标世界坐标
        self.high_level_planner.event_trigger.last_subgoal = None  # 重置事件触发器上次子目标
        self.high_level_planner.event_trigger.reset_state()  # 重置事件触发器状态
        self.high_level_planner.reset_subgoal_hidden()  # 清空子目标网络隐状态
        if hasattr(self.low_level_controller, "reset_hidden_state"):
            self.low_level_controller.reset_hidden_state()
        self.clear_global_route()  # 清空全局路径


def create_navigation_system(
        load_models: bool = False,
        subgoal_threshold: float = 0.5,
        world_file: Optional[Path] = None,
        global_plan_resolution: float = 0.25,
        global_plan_margin: float = 0.35,
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
        world_file=world_file,  # 世界文件
        global_plan_resolution=global_plan_resolution,  # 全局规划分辨率
        global_plan_margin=global_plan_margin,  # 全局规划安全边界
        waypoint_lookahead=waypoint_lookahead,  # 航点前瞻数量
    )
