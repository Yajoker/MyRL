from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch

# 导入系统内部模块：低层控制器和高层规划器
from ethsrl.core.control.low_level_controller import LowLevelController
from ethsrl.core.planning.global_planner import GlobalPlanner, WaypointWindow
from ethsrl.core.planning.high_level_planner import HighLevelPlanner


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
        step_duration: float = 0.1,
        trigger_min_interval: float = 1.0,
        subgoal_threshold: float = 0.5,
        world_file: Optional[Path] = None,
        global_plan_resolution: float = 0.25,
        global_plan_margin: float = 0.35,
        waypoint_lookahead: int = 3,
        window_step_limit: int = 80,
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
        # 设置计算设备：若未指定则自动检测 GPU，否则使用 CPU
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 计算状态维度
        low_level_state_dim = laser_dim + 4

        # 初始化高层规划器（负责生成子目标）
        self.high_level_planner = HighLevelPlanner(
            belief_dim=laser_dim,  # 输入维度（激光雷达特征）
            device=self.device,  # 使用的计算设备
            save_directory=models_directory / "high_level",  # 模型保存路径
            model_name="high_level_planner",  # 模型名称
            load_model=load_models,  # 是否加载已有模型
            step_duration=step_duration,
            min_interval=trigger_min_interval,
            subgoal_reach_threshold=subgoal_threshold,
            waypoint_lookahead=waypoint_lookahead,
        )

        # 初始化低层控制器（负责执行动作）
        self.low_level_controller = LowLevelController(
            state_dim=low_level_state_dim,  # 低层输入维度
            action_dim=action_dim,  # 输出动作维度
            max_action=max_action,  # 最大动作值
            device=self.device,  # 计算设备
            save_directory=models_directory / "low_level",  # 模型保存路径
            model_name="low_level_controller",  # 模型名称
            load_model=load_models  # 是否加载已有模型
        )

        # 系统运行状态变量
        self.current_subgoal = None  # 当前子目标（距离、角度）
        self.current_subgoal_world: Optional[np.ndarray] = None  # 当前子目标在世界坐标系中的位置
        self.prev_action = [0.0, 0.0]  # 上一步执行的动作 [线速度, 角速度]
        self.step_count = 0  # 总步数计数
        self.last_replanning_step = 0  # 上次重新规划的步数（用于事件触发判断）
        self.step_duration = step_duration
        self.subgoal_threshold = subgoal_threshold
        self.waypoint_lookahead = waypoint_lookahead
        self.window_step_limit = max(1, int(window_step_limit))
        self.window_step_count = 0
        self.steps_inside_window = 0
        self.window_last_index: Optional[int] = None
        self.last_window_distance: Optional[float] = None
        self.window_limit_exceeded = False
        self.window_within = False
        self.last_window_update_step: int = -1
        self._cached_window_info: Dict[str, object] = {}

        self.global_planner: Optional[GlobalPlanner] = None
        if world_file is not None:
            try:
                self.global_planner = GlobalPlanner(
                    world_file=world_file,
                    resolution=global_plan_resolution,
                    safety_margin=global_plan_margin,
                )
            except FileNotFoundError as exc:
                print(f"[GlobalPlanner] {exc}. Global planning disabled.")
        self.global_waypoints: List[WaypointWindow] = []
        self.current_waypoint_index: int = 0
        self.global_goal: Optional[np.ndarray] = None

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
            self.plan_global_route(robot_pose, goal_position)
        elif self.global_waypoints:
            self._advance_waypoints(robot_pose)

        waypoint_candidates = self.get_active_waypoints(
            robot_pose, include_indices=True
        )
        window_metrics = self.update_window_state(robot_pose, waypoint_candidates)

        goal_info = [goal_distance, goal_cos, goal_sin]

        # 标志位：是否需要重新生成子目标
        if self.current_subgoal_world is None:
            should_replan = True
        else:
            should_replan = self.high_level_planner.check_triggers(
                laser_scan,
                robot_pose,
                goal_info,
                prev_action=self.prev_action,
                current_step=self.step_count,
                window_metrics=window_metrics,
            )
        if window_metrics.get("limit_exceeded", False):
            should_replan = True

        subgoal_distance: Optional[float] = None
        subgoal_angle: Optional[float] = None
        decision_meta: dict = {}

        if should_replan:
            subgoal_distance, subgoal_angle, decision_meta = self.high_level_planner.generate_subgoal(
                laser_scan,
                goal_distance,
                goal_cos,
                goal_sin,
                prev_action=self.prev_action,
                robot_pose=robot_pose,
                current_step=self.step_count,
                waypoints=waypoint_candidates,
                window_metrics=window_metrics,
            )
            self.reset_window_tracking()
            self.update_selected_waypoint(decision_meta.get("selected_waypoint"))
            planner_world = self.high_level_planner.current_subgoal_world
            self.current_subgoal_world = None if planner_world is None else np.asarray(planner_world, dtype=np.float32)
            self.last_replanning_step = self.step_count
            self.high_level_planner.event_trigger.reset_time(self.step_count)
        else:
            planner_world = self.high_level_planner.current_subgoal_world
            if planner_world is not None:
                self.current_subgoal_world = np.asarray(planner_world, dtype=np.float32)

        # 使用最新的机器人姿态计算子目标在机器人坐标系下的表示
        relative_geometry = self.high_level_planner.get_relative_subgoal(robot_pose)
        if relative_geometry[0] is None:
            if subgoal_distance is not None and subgoal_angle is not None:
                relative_geometry = (float(subgoal_distance), float(subgoal_angle))
            elif self.current_subgoal is not None:
                relative_geometry = self.current_subgoal
            else:
                relative_geometry = (0.0, 0.0)

        self.current_subgoal = (float(relative_geometry[0]), float(relative_geometry[1]))

        if should_replan:
            selected_wp = decision_meta.get("selected_waypoint")
            if selected_wp is not None:
                print(
                    "New subgoal (wp {}): distance={:.2f}m, angle={:.2f}rad".format(
                        int(selected_wp), self.current_subgoal[0], self.current_subgoal[1]
                    )
                )
            else:
                print(
                    "New subgoal: distance={:.2f}m, angle={:.2f}rad".format(
                        self.current_subgoal[0], self.current_subgoal[1]
                    )
                )

        # 生成低层输入状态，用于控制器决策
        low_level_state = self.low_level_controller.process_observation(
            laser_scan,
            self.current_subgoal[0],  # 子目标距离
            self.current_subgoal[1],  # 子目标角度
            self.prev_action  # 上一步的动作（用于平滑控制）
        )

        # 通过低层控制器预测下一步动作（网络输出）
        action = self.low_level_controller.predict_action(low_level_state)

        # 将网络输出映射为实际机器人可执行的速度命令
        # 将线速度从 [-1,1] 映射为 [0,0.5]
        linear_velocity = (action[0] + 1) / 4
        # 保持角速度在 [-1,1] 范围内
        angular_velocity = action[1]

        # 记录当前动作（用于下一次输入）
        self.prev_action = [linear_velocity, angular_velocity]

        # 返回控制命令
        return [linear_velocity, angular_velocity]

    def plan_global_route(self, robot_pose, goal_position, force: bool = False):
        """Compute (or refresh) the global waypoint sequence."""

        if self.global_planner is None:
            return []

        start_xy = np.asarray(robot_pose[:2], dtype=np.float32)
        goal_vec = np.asarray(goal_position[:2], dtype=np.float32)

        if (
            not force
            and self.global_goal is not None
            and self.global_waypoints
            and np.linalg.norm(goal_vec - self.global_goal) < 1e-4
        ):
            self._advance_waypoints(robot_pose)
            return self.global_waypoints

        self.global_goal = goal_vec

        try:
            raw_path = self.global_planner.plan(start_xy, goal_vec)
        except RuntimeError as exc:
            print(f"[GlobalPlanner] {exc}. Using direct segment to goal.")
            raw_path = [WaypointWindow(center=goal_vec.copy(), radius=self.global_planner.window_radius)]

        filtered: List[WaypointWindow] = []
        for window in raw_path:
            centre = np.asarray(window.center, dtype=np.float32)
            if np.linalg.norm(centre - start_xy) <= 0.5 * self.global_planner.resolution:
                continue
            filtered.append(WaypointWindow(center=centre.copy(), radius=float(window.radius)))

        if not filtered:
            filtered = [WaypointWindow(center=goal_vec.copy(), radius=self.global_planner.window_radius)]

        goal_dists = [float(np.linalg.norm(goal_vec - wp.center)) for wp in filtered]
        print(f"[GlobalPlanner] Waypoints in world frame ({len(filtered)} total):")
        for idx, (wp, dist) in enumerate(zip(filtered, goal_dists)):
            print(f"  #{idx:02d} {wp.center.tolist()} | dist_to_goal={dist:.3f} m | radius={wp.radius:.2f} m")

        if len(goal_dists) > 1:
            monotonic = all(goal_dists[i + 1] <= goal_dists[i] + 1e-6 for i in range(len(goal_dists) - 1))
            if not monotonic:
                print("[GlobalPlanner] Warning: waypoint distance does not consistently decrease toward the goal.")

        self.global_waypoints = filtered
        self.current_waypoint_index = 0
        self.reset_window_tracking()
        self._advance_waypoints(robot_pose)
        return self.global_waypoints

    def reset_window_tracking(self) -> None:
        self.window_last_index = None
        self.window_step_count = 0
        self.steps_inside_window = 0
        self.last_window_distance = None
        self.window_limit_exceeded = False
        self.window_within = False
        self.last_window_update_step = -1
        self._cached_window_info = {}

    def _advance_waypoints(self, robot_pose) -> None:
        if not self.global_waypoints:
            return
        if robot_pose is None:
            return

        position = np.asarray(robot_pose[:2], dtype=np.float32)
        base_threshold = max(0.1, self.subgoal_threshold * 0.8)
        if self.global_planner is not None:
            base_threshold = max(base_threshold, 0.5 * self.global_planner.resolution)

        window_changed = False
        while self.current_waypoint_index < len(self.global_waypoints):
            window = self.global_waypoints[self.current_waypoint_index]
            centre = np.asarray(window.center, dtype=np.float32)
            radius = float(window.radius)
            threshold = max(base_threshold, radius * 0.9)
            distance = float(np.linalg.norm(centre - position))

            if distance <= threshold and self.current_waypoint_index < len(self.global_waypoints) - 1:
                self.current_waypoint_index += 1
                window_changed = True
                continue
            break

        if window_changed:
            self.reset_window_tracking()

    def _update_window_metrics(
        self,
        robot_pose,
        waypoint_candidates: List[Tuple[int, WaypointWindow]],
    ) -> None:
        if not waypoint_candidates:
            self.reset_window_tracking()
            return

        first = waypoint_candidates[0]
        if isinstance(first, tuple) and len(first) == 2:
            index, window = int(first[0]), first[1]
        else:
            index = self.current_waypoint_index
            window = first
        robot_xy = np.asarray(robot_pose[:2], dtype=np.float32)
        centre = np.asarray(window.center, dtype=np.float32)
        distance = float(np.linalg.norm(centre - robot_xy))
        radius = float(window.radius)

        changed = index != self.window_last_index
        prev_distance = self.last_window_distance if not changed else None
        prev_inside = self.window_within if not changed else False

        if changed:
            self.window_last_index = index
            self.window_step_count = 0
            self.steps_inside_window = 0
            self.window_limit_exceeded = False
            prev_inside = False

        self.window_step_count += 1

        inside = distance <= radius
        if inside:
            self.steps_inside_window += 1
        else:
            self.steps_inside_window = 0

        entered = inside and not prev_inside

        self.window_within = inside
        self.window_limit_exceeded = self.steps_inside_window >= self.window_step_limit
        self.last_window_distance = distance
        self.last_window_update_step = self.step_count

        margin = (distance - radius) / max(radius, 1e-3)
        margin = float(np.clip(margin, -1.0, 1.0))

        self._cached_window_info = {
            "index": index,
            "radius": radius,
            "distance": distance,
            "prev_distance": prev_distance,
            "inside": inside,
            "entered": entered,
            "steps_inside": self.steps_inside_window,
            "step_count": self.window_step_count,
            "step_limit": self.window_step_limit,
            "limit_exceeded": self.window_limit_exceeded,
            "margin": margin,
        }

    def update_window_state(
        self,
        robot_pose,
        waypoint_candidates: List[Tuple[int, WaypointWindow]],
    ) -> Dict[str, object]:
        self._update_window_metrics(robot_pose, waypoint_candidates)
        return self.get_window_metrics()

    def get_window_metrics(self) -> Dict[str, object]:
        return dict(self._cached_window_info)

    def get_active_waypoints(self, robot_pose, lookahead: Optional[int] = None, include_indices: bool = False):
        """Return the waypoint window closest to the robot."""

        if not self.global_waypoints:
            return []

        self._advance_waypoints(robot_pose)

        horizon = lookahead or self.waypoint_lookahead
        start_idx = min(self.current_waypoint_index, len(self.global_waypoints) - 1)
        end_idx = min(len(self.global_waypoints), start_idx + max(1, horizon))

        indices = range(start_idx, end_idx)
        if include_indices:
            return [(idx, self.global_waypoints[idx].clone()) for idx in indices]
        return [self.global_waypoints[idx].clone() for idx in indices]

    def update_selected_waypoint(self, selected_index: Optional[int]) -> None:
        """Record the global waypoint chosen by the high-level planner."""

        if selected_index is None or not self.global_waypoints:
            return

        idx = int(selected_index)
        if idx < 0:
            idx = 0
        if idx >= len(self.global_waypoints):
            idx = len(self.global_waypoints) - 1

        if idx >= self.current_waypoint_index:
            if idx != self.current_waypoint_index:
                self.current_waypoint_index = idx
                self.reset_window_tracking()
            else:
                self.current_waypoint_index = idx

    def clear_global_route(self) -> None:
        """Reset stored waypoints and goal information."""

        self.global_waypoints = []
        self.current_waypoint_index = 0
        self.global_goal = None
        self.reset_window_tracking()

    def reset(self):
        """
        重置整个导航系统的内部状态。
        （例如在新仿真回合开始时调用）
        """
        self.current_subgoal = None  # 清空子目标
        self.current_subgoal_world = None
        self.prev_action = [0.0, 0.0]  # 重置上一步动作
        self.step_count = 0  # 步数归零
        self.last_replanning_step = 0  # 清除上次规划记录
        self.high_level_planner.current_subgoal = None
        self.high_level_planner.current_subgoal_world = None
        self.high_level_planner.prev_action = [0.0, 0.0]
        self.high_level_planner.event_trigger.last_subgoal = None
        self.high_level_planner.event_trigger.reset_state()
        self.clear_global_route()


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
        subgoal_threshold=subgoal_threshold,
        world_file=world_file,
        global_plan_resolution=global_plan_resolution,
        global_plan_margin=global_plan_margin,
        waypoint_lookahead=waypoint_lookahead,
    )
