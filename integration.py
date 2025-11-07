from pathlib import Path
import logging
import math
import time
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch

# 导入系统内部模块：低层控制器和高层规划器
from ethsrl.config import IntegrationConfig, MotionConfig
from ethsrl.low_level_controller import LowLevelController
from ethsrl.global_planner import GlobalPlanner, WaypointWindow
from ethsrl.high_level_planner import HighLevelPlanner


logger = logging.getLogger(__name__)


def _clip_unit_interval(value: float) -> float:
    """Clamp a scalar to the closed interval [-1, 1]."""

    return max(-1.0, min(1.0, float(value)))


def _wrap_angle(angle: float) -> float:
    """Wrap angle to (-π, π]."""

    wrapped = (float(angle) + math.pi) % (2.0 * math.pi) - math.pi
    if wrapped <= -math.pi:
        wrapped += 2.0 * math.pi
    return wrapped


def map_linear_speed(actor_linear: float, motion: MotionConfig) -> float:
    """Scale low-level actor linear output to the configured velocity range."""

    clipped = _clip_unit_interval(actor_linear)
    return 0.5 * (clipped + 1.0) * motion.v_max


def map_angular_speed(actor_angular: float, motion: MotionConfig) -> float:
    """Scale low-level actor angular output to the configured rotational range."""

    clipped = _clip_unit_interval(actor_angular)
    return clipped * motion.omega_max


def map_actor_to_commands(action: Sequence[float], motion: MotionConfig) -> Tuple[float, float]:
    """Convert actor outputs in [-1, 1]² to executable (v, ω) commands."""

    if len(action) < 2:
        raise ValueError("Expected at least two elements for (a_lin, a_ang).")

    raw_lin = float(action[0])
    raw_ang = float(action[1])

    linear_velocity = map_linear_speed(raw_lin, motion)
    angular_velocity = map_angular_speed(raw_ang, motion)

    if logger.isEnabledFor(logging.DEBUG):
        clipped_lin = _clip_unit_interval(raw_lin)
        clipped_ang = _clip_unit_interval(raw_ang)
        logger.debug(
            "Actor scaling: raw=(%.3f, %.3f) clipped=(%.3f, %.3f) -> command=(%.3f, %.3f); limits=(v_max=%.3f, ω_max=%.3f)",
            raw_lin,
            raw_ang,
            clipped_lin,
            clipped_ang,
            linear_velocity,
            angular_velocity,
            motion.v_max,
            motion.omega_max,
        )

    return linear_velocity, angular_velocity


def map_commands_to_actor(linear_velocity: float, angular_velocity: float, motion: MotionConfig) -> Tuple[float, float]:
    """Project executed (v, ω) commands back into the actor's normalized space."""

    v_max = max(motion.v_max, 1e-6)
    omega_max = max(motion.omega_max, 1e-6)

    lin_clamped = float(np.clip(linear_velocity, 0.0, motion.v_max))
    ang_clamped = float(np.clip(angular_velocity, -motion.omega_max, motion.omega_max))

    norm_lin = (lin_clamped / v_max) * 2.0 - 1.0
    norm_ang = ang_clamped / omega_max

    return _clip_unit_interval(norm_lin), _clip_unit_interval(norm_ang)


class HierarchicalNavigationSystem:
    """
    分层导航系统类（Hierarchical Navigation System）

    该系统将事件触发的高层规划与反应式低层控制相结合，
    以实现高效、安全的自主机器人导航。
    """

    def __init__(
        self,
        *,
        laser_dim: int = 180,
        action_dim: int = 2,
        max_action: float = 1.0,
        device=None,
        load_models: bool = False,
        models_directory: Path = Path("ethsrl/models"),
       
        world_file: Optional[Path] = None,
        config: Optional[IntegrationConfig] = None,        
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

        # 统一读取系统配置
        self.config = config or IntegrationConfig()
        self.motion_limits = self.config.motion

        # 计算状态维度
        low_level_state_dim = laser_dim + 4

        # 初始化高层规划器（负责生成子目标）
        self.high_level_planner = HighLevelPlanner(
            belief_dim=laser_dim,  # 输入维度（激光雷达特征）
            device=self.device,  # 使用的计算设备
            save_directory=models_directory / "high_level",  # 模型保存路径
            model_name="high_level_planner",  # 模型名称
            load_model=load_models,  # 是否加载已有模型
            waypoint_lookahead=self.config.planner.waypoint_lookahead,
            trigger_config=self.config.trigger,
            motion_config=self.motion_limits,
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
            distance_normalizer=self.config.planner.subgoal_distance_normalizer,
        )

        # 系统运行状态变量
        self.current_subgoal = None  # 当前子目标（距离、角度）
        self.current_subgoal_world: Optional[np.ndarray] = None  # 当前子目标在世界坐标系中的位置
        self.prev_action = [0.0, 0.0]  # 上一步执行的归一化动作值
        self.prev_command = [0.0, 0.0]  # 上一步执行的实际控制命令 [线速度, 角速度]
        self.step_count = 0  # 总步数计数
        min_step_interval = getattr(self.high_level_planner.event_trigger, "min_step_interval", 1)
        self.trigger_step_interval = max(1, int(min_step_interval))
        self.last_replanning_step = -self.trigger_step_interval  # 上次重新规划的步数（用于节流）

        self.step_duration = self.motion_limits.dt
        self.subgoal_threshold = self.config.trigger.subgoal_reach_threshold
        self.waypoint_lookahead = self.config.planner.waypoint_lookahead
        self.window_step_limit = max(1, int(self.config.window.step_limit))
        self.clip_to_window = bool(getattr(self.config.window, "clip_to_window", False))
        self.use_path_tangent = bool(getattr(self.config.planner, "use_path_tangent", True))
        self.window_margin = float(self.config.window.margin)

        self.window_step_count = 0
        self.steps_inside_window = 0
        self.window_last_index: Optional[int] = None
        self.last_window_distance: Optional[float] = None
        self.window_limit_exceeded = False
        self.window_within = False
        self.last_window_update_step: int = -1
        self._cached_window_info: Dict[str, object] = {}
        self._printed_plan_overview = False

        self.global_planner: Optional[GlobalPlanner] = None
        if world_file is not None:
            try:
                self.global_planner = GlobalPlanner(
                    world_file=world_file,
                    resolution=self.config.planner.resolution,
                    safety_margin=self.config.planner.safety_margin,
                    window_spacing=self.config.planner.window_spacing,
                    window_radius=self.config.planner.window_radius,
                )
            except FileNotFoundError as exc:
                print(f"[GlobalPlanner] {exc}. Global planning disabled.")
        self.global_waypoints: List[WaypointWindow] = []
        self.current_waypoint_index: int = 0
        self.global_goal: Optional[np.ndarray] = None

    def _resolve_anchor_window(
        self,
        robot_pose: Sequence[float],
        waypoint_candidates: List[Tuple[int, WaypointWindow]],
        goal_distance: float,
        goal_cos: float,
        goal_sin: float,
    ) -> Dict[str, object]:
        """Determine reference window geometry for subgoal synthesis."""

        robot_xy = np.asarray(robot_pose[:2], dtype=np.float32)
        yaw = float(robot_pose[2])

        anchor_window: Optional[WaypointWindow] = None
        anchor_index: Optional[int] = None

        if waypoint_candidates:
            entry = waypoint_candidates[0]
            if isinstance(entry, tuple) and len(entry) == 2:
                anchor_index = int(entry[0]) if entry[0] is not None else None
                anchor_window = entry[1]
            elif isinstance(entry, WaypointWindow):
                anchor_window = entry
            if anchor_window is not None and not isinstance(anchor_window, WaypointWindow):
                anchor_window = None

        if anchor_window is not None:
            centre = np.asarray(anchor_window.center, dtype=np.float32)
            offset = centre - robot_xy
            base_distance = float(np.linalg.norm(offset))
            world_heading = math.atan2(float(offset[1]), float(offset[0])) if base_distance > 1e-6 else yaw
            base_angle = _wrap_angle(world_heading - yaw)
            radius = float(anchor_window.radius)
            tangent_raw = getattr(anchor_window, "tangent_heading", None)
            tangent_heading = float(tangent_raw) if tangent_raw is not None else None
            return {
                "index": anchor_index,
                "centre": centre,
                "radius": radius,
                "tangent_heading": tangent_heading,
                "base_distance": base_distance,
                "base_angle": base_angle,
                "robot_distance": base_distance,
                "robot_bearing": base_angle,
                "world_heading": world_heading,
            }

        base_angle_rel = _wrap_angle(math.atan2(goal_sin, goal_cos))
        world_heading = _wrap_angle(base_angle_rel + yaw)
        base_distance = float(max(goal_distance, 0.0))

        return {
            "index": None,
            "centre": None,
            "radius": None,
            "base_distance": base_distance,
            "base_angle": base_angle_rel,
            "tangent_heading": None,
            "robot_distance": base_distance,
            "robot_bearing": base_angle_rel,
            "world_heading": world_heading,
        }

    def _synthesise_subgoal(
        self,
        *,
        distance_scale: float,
        angle_offset: float,
        robot_pose: Sequence[float],
        goal_distance: float,
        goal_cos: float,
        goal_sin: float,
        waypoint_candidates: List[Tuple[int, WaypointWindow]],
        window_metrics: Optional[Dict[str, object]],
        metadata: Optional[Dict[str, object]] = None,
    ) -> Tuple[float, float, Dict[str, object], np.ndarray]:
        """Combine high-level outputs with geometry to produce executable subgoals."""

        decision_meta: Dict[str, object] = dict(metadata or {})

        anchor = self._resolve_anchor_window(
            robot_pose,
            waypoint_candidates,
            goal_distance,
            goal_cos,
            goal_sin,
        )

        anchor_index = anchor.get("index")
        window_radius = anchor.get("radius")
        anchor_centre = anchor.get("centre")
        anchor_tangent = anchor.get("tangent_heading")
        robot_base_distance = float(anchor.get("robot_distance", anchor.get("base_distance", 0.0)))
        robot_base_angle = float(anchor.get("robot_bearing", anchor.get("base_angle", 0.0)))
        world_heading = float(anchor.get("world_heading", robot_pose[2]))

        distance_scale = float(np.clip(distance_scale, 0.0, 1.0))
        margin = self.window_margin

        robot_xy = np.asarray(robot_pose[:2], dtype=np.float32)
        yaw = float(robot_pose[2])

        los_attempted = False
        los_applied = False
        los_checks = 0
        los_success: Optional[bool] = None
        los_backoff_step = 0.1
        los_min_distance = 0.1

        final_distance: float
        final_angle: float
        world_target: np.ndarray
        distance_scale_window_applied: Optional[float] = None
        window_retreat: Optional[float] = None
        angle_offset_applied: float
        phi_base: Optional[float] = None
        phi: float = 0.0
        window_distance: Optional[float] = None
        window_radius_limit: Optional[float] = None
        target_radius: Optional[float] = None

        if anchor_centre is not None and window_radius is not None and float(window_radius) > 0.0:
            centre = np.asarray(anchor_centre, dtype=np.float32)
            radius_val = float(window_radius)
            window_radius_limit = max(0.0, radius_val - margin)
            window_epsilon = 1e-3
            span = max(0.0, window_radius_limit - window_epsilon)

            if window_radius_limit <= window_epsilon:
                target_radius = window_radius_limit
            else:
                target_radius = window_epsilon + distance_scale * span

            phi_base = float(anchor_tangent) if (self.use_path_tangent and anchor_tangent is not None) else None
            if phi_base is not None and not math.isfinite(phi_base):
                phi_base = None

            if phi_base is None:
                centre_to_robot = robot_xy - centre
                if np.linalg.norm(centre_to_robot) > 1e-6:
                    phi_base = math.atan2(float(centre_to_robot[1]), float(centre_to_robot[0])) + math.pi
                else:
                    phi_base = world_heading

            phi = _wrap_angle(phi_base + angle_offset)

            initial_radius = target_radius

            planner = None if self.clip_to_window else self.global_planner
            if planner is not None and target_radius > 1e-6:
                los_attempted = True
                r_floor = max(0.0, min(window_radius_limit, los_min_distance))
                if window_radius_limit <= window_epsilon:
                    r_floor = window_radius_limit
                r_floor = max(window_epsilon if window_radius_limit > window_epsilon else 0.0, r_floor)

                current_radius = target_radius

                def build_target(radius_val: float) -> np.ndarray:
                    return np.array(
                        [
                            centre[0] + radius_val * math.cos(phi),
                            centre[1] + radius_val * math.sin(phi),
                        ],
                        dtype=np.float32,
                    )

                while current_radius > 1e-6:
                    candidate_target = build_target(current_radius)
                    los_checks += 1
                    if planner.has_line_of_sight(robot_xy, candidate_target):
                        los_success = True
                        target_radius = current_radius
                        break

                    los_applied = True
                    next_radius = max(r_floor, current_radius - los_backoff_step)
                    if next_radius >= current_radius - 1e-6:
                        los_success = False
                        target_radius = next_radius
                        break

                    current_radius = next_radius

                if los_success is None:
                    los_success = False

                if los_applied and logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "LoS retreat (window) applied: r=%.3f -> %.3f (checks=%d, success=%s)",
                        initial_radius,
                        target_radius,
                        los_checks,
                        los_success,
                    )

            world_target = np.array(
                [
                    centre[0] + target_radius * math.cos(phi),
                    centre[1] + target_radius * math.sin(phi),
                ],
                dtype=np.float32,
            )

            delta = world_target - robot_xy
            final_distance = float(np.linalg.norm(delta))
            final_angle = _wrap_angle(math.atan2(float(delta[1]), float(delta[0])) - yaw)

            window_distance = float(np.linalg.norm(world_target - centre))
            if window_radius_limit is not None and window_radius_limit > 1e-6:
                distance_scale_window_applied = float(np.clip(target_radius / window_radius_limit, 0.0, 1.0))
            elif window_radius_limit is not None:
                distance_scale_window_applied = 0.0
            window_retreat = float(max(0.0, initial_radius - target_radius))
            angle_offset_applied = _wrap_angle(phi - (phi_base if phi_base is not None else 0.0))

        else:
            # 回退到机器人锚点几何，保持兼容性
            proposed_angle = _wrap_angle(robot_base_angle + angle_offset)
            scaled_distance = float(distance_scale * robot_base_distance)
            final_distance = max(0.0, scaled_distance)
            clip_limit = None

            if self.clip_to_window and window_radius is not None:
                clip_limit = max(0.0, float(window_radius) - margin)
                if final_distance > clip_limit:
                    final_distance = clip_limit

            planner = None if self.clip_to_window else self.global_planner
            if planner is not None and final_distance > 1e-6:
                los_attempted = True
                los_distance = final_distance
                while los_distance > 1e-6:
                    candidate_target = np.array(
                        [
                            robot_pose[0] + los_distance * math.cos(yaw + proposed_angle),
                            robot_pose[1] + los_distance * math.sin(yaw + proposed_angle),
                        ],
                        dtype=np.float32,
                    )
                    los_checks += 1
                    if planner.has_line_of_sight(robot_xy, candidate_target):
                        los_success = True
                        final_distance = los_distance
                        break

                    los_applied = True
                    next_distance = max(los_min_distance, los_distance - los_backoff_step)
                    if next_distance >= los_distance - 1e-6:
                        los_success = False
                        final_distance = next_distance
                        break

                    los_distance = next_distance

                if los_success is None:
                    los_success = False

            final_angle = proposed_angle
            world_target = np.array(
                [
                    robot_pose[0] + final_distance * math.cos(yaw + final_angle),
                    robot_pose[1] + final_distance * math.sin(yaw + final_angle),
                ],
                dtype=np.float32,
            )
            angle_offset_applied = _wrap_angle(final_angle - robot_base_angle)
            target_radius = None

        applied_scale_robot = distance_scale
        if robot_base_distance > 1e-6:
            applied_scale_robot = float(np.clip(final_distance / robot_base_distance, 0.0, 1.0))

        meta_updates: Dict[str, object] = {
            "anchor_index": anchor_index,
            "anchor_base_distance": robot_base_distance,
            "anchor_base_angle": robot_base_angle,
            "anchor_tangent_heading": anchor_tangent,
            "anchor_world_heading": world_heading,
            "distance_scale": distance_scale,
            "distance_scale_applied": applied_scale_robot,
            "distance_scale_window_applied": distance_scale_window_applied,
            "angle_offset_applied": angle_offset_applied,
            "final_distance": final_distance,
            "final_angle": final_angle,
            "window_margin": margin if window_radius is not None else None,
            "window_radius": float(window_radius) if window_radius is not None else None,
            "window_radius_limit": window_radius_limit,
            "window_centre": anchor_centre.tolist() if anchor_centre is not None else None,
            "window_radius_applied": float(target_radius) if target_radius is not None else None,
            "clip_to_window_enabled": self.clip_to_window,
            "los_attempted": los_attempted,
            "los_backoff_step": los_backoff_step,
            "los_min_distance": los_min_distance,
            "window_distance": window_distance,
            "distance_retreat": window_retreat,
            "phi_base": phi_base,
            "phi_final": phi if anchor_centre is not None else None,
        }

        if los_attempted:
            meta_updates.update(
                {
                    "los_retreat_applied": los_applied,
                    "los_success": bool(los_success),
                    "los_checks": los_checks,
                }
            )

        decision_meta.update(meta_updates)

        if decision_meta.get("selected_waypoint") is None and anchor_index is not None:
            decision_meta["selected_waypoint"] = anchor_index

        if decision_meta.get("active_window_index") is None and window_metrics:
            window_idx = window_metrics.get("index")
            if window_idx is not None:
                decision_meta["active_window_index"] = window_idx

        return final_distance, final_angle, decision_meta, world_target

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

        throttle_ready = (self.step_count - self.last_replanning_step) >= self.trigger_step_interval
        force_trigger = window_metrics.get("limit_exceeded", False)
        effective_throttle = throttle_ready or force_trigger

        # 标志位：是否需要重新生成子目标
        if self.current_subgoal_world is None:
            should_replan = True
        elif effective_throttle:
            should_replan = self.high_level_planner.check_triggers(
                laser_scan,
                robot_pose,
                goal_info,
                prev_action=self.prev_action,
                current_step=self.step_count,
                window_metrics=window_metrics,
                throttle_ready=effective_throttle,
            )
        else:
            should_replan = False

        if force_trigger:
            should_replan = True

        subgoal_distance: Optional[float] = None
        subgoal_angle: Optional[float] = None
        decision_meta: dict = {}

        if should_replan:
            distance_scale, angle_offset, decision_meta = self.high_level_planner.generate_subgoal(
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

            final_distance, final_angle, decision_meta, world_target = self._synthesise_subgoal(
                distance_scale=distance_scale,
                angle_offset=angle_offset,
                robot_pose=robot_pose,
                goal_distance=goal_distance,
                goal_cos=goal_cos,
                goal_sin=goal_sin,
                waypoint_candidates=waypoint_candidates,
                window_metrics=window_metrics,
                metadata=decision_meta,
            )

            if decision_meta.get("window_centre") is not None and decision_meta.get("window_radius") is not None:
                centre_vec = np.asarray(decision_meta["window_centre"], dtype=np.float32)
                radius_val = float(decision_meta["window_radius"])
                permitted = max(0.0, radius_val - self.window_margin)
                distance_to_centre = float(np.linalg.norm(world_target - centre_vec))
                if distance_to_centre - permitted > 1e-3 and logger.isEnabledFor(logging.WARNING):
                    logger.warning(
                        "Subgoal outside window: dist=%.3f, limit=%.3f (margin=%.3f)",
                        distance_to_centre,
                        permitted,
                        self.window_margin,
                    )

            goal_direction_rel = math.atan2(goal_sin, goal_cos)
            self.high_level_planner.commit_subgoal(
                distance=final_distance,
                angle=final_angle,
                world_target=world_target,
                goal_distance=goal_distance,
                goal_direction=goal_direction_rel,
                prev_action=self.prev_action,
                current_step=self.step_count,
                robot_pose=robot_pose,
            )

            planner_current = self.high_level_planner.current_subgoal_world
            if planner_current is not None:
                self.current_subgoal_world = np.asarray(planner_current, dtype=np.float32)
            else:
                self.current_subgoal_world = None
            self.last_replanning_step = self.step_count
            self.high_level_planner.event_trigger.reset_time(self.step_count)

            if self.high_level_planner.current_subgoal is not None:
                smoothed_distance, smoothed_angle = self.high_level_planner.current_subgoal
                self.current_subgoal = (smoothed_distance, smoothed_angle)
                subgoal_distance = smoothed_distance
                subgoal_angle = smoothed_angle
            else:
                self.current_subgoal = None
                subgoal_distance = None
                subgoal_angle = None
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
            self.prev_action  # 上一步的归一化动作（用于平滑控制）
        )

        # 通过低层控制器预测下一步动作（网络输出）
        action = self.low_level_controller.predict_action(low_level_state)

        # 将网络输出映射为实际机器人可执行的速度命令
        linear_velocity, angular_velocity = map_actor_to_commands(action, self.motion_limits)

        # 记录当前动作（用于下一次输入）
        norm_lin, norm_ang = map_commands_to_actor(linear_velocity, angular_velocity, self.motion_limits)
        self.prev_action = [norm_lin, norm_ang]
        self.prev_command = [linear_velocity, angular_velocity]

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
            direction = goal_vec - start_xy
            heading = float(math.atan2(direction[1], direction[0])) if np.linalg.norm(direction) > 1e-6 else 0.0
            raw_path = [WaypointWindow(center=goal_vec.copy(), radius=self.global_planner.window_radius, tangent_heading=heading)]

        filtered: List[WaypointWindow] = []
        for window in raw_path:
            centre = np.asarray(window.center, dtype=np.float32)
            if np.linalg.norm(centre - start_xy) <= 0.5 * self.global_planner.resolution:
                continue
            tangent = float(getattr(window, "tangent_heading", 0.0))
            filtered.append(
                WaypointWindow(
                    center=centre.copy(),
                    radius=float(window.radius),
                    tangent_heading=tangent,
                )
            )

        if not filtered:
            direction = goal_vec - start_xy
            heading = float(math.atan2(direction[1], direction[0])) if np.linalg.norm(direction) > 1e-6 else 0.0
            filtered = [WaypointWindow(center=goal_vec.copy(), radius=self.global_planner.window_radius, tangent_heading=heading)]

        if not self._printed_plan_overview:
            goal_dists = [float(np.linalg.norm(goal_vec - wp.center)) for wp in filtered]
            print(f"[GlobalPlanner] Waypoints in world frame ({len(filtered)} total):")
            for idx, (wp, dist) in enumerate(zip(filtered, goal_dists)):
                print(f"  #{idx:02d} {wp.center.tolist()} | dist_to_goal={dist:.3f} m | radius={wp.radius:.2f} m")

            if len(goal_dists) > 1:
                monotonic = all(goal_dists[i + 1] <= goal_dists[i] + 1e-6 for i in range(len(goal_dists) - 1))
                if not monotonic:
                    print("[GlobalPlanner] Warning: waypoint distance does not consistently decrease toward the goal.")

            self._printed_plan_overview = True

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
            "centre": centre.tolist(),
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
        self.high_level_planner.reset_smoothing()

    def reset(self):
        """
        重置整个导航系统的内部状态。
        （例如在新仿真回合开始时调用）
        """
        self.current_subgoal = None  # 清空子目标
        self.current_subgoal_world = None
        self.prev_action = [0.0, 0.0]  # 重置上一步动作
        self.prev_command = [0.0, 0.0]
        self.step_count = 0  # 步数归零
        self.last_replanning_step = -self.trigger_step_interval  # 清除上次规划记录
        self.high_level_planner.current_subgoal = None
        self.high_level_planner.current_subgoal_world = None
        self.high_level_planner.prev_action = [0.0, 0.0]
        self.high_level_planner.event_trigger.last_subgoal = None
        self.high_level_planner.event_trigger.reset_state()
        self.high_level_planner.reset_smoothing()
        self.clear_global_route()


def create_navigation_system(
    *,
    load_models: bool = False,

    world_file: Optional[Path] = None,
    config: Optional[IntegrationConfig] = None,
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

        world_file=world_file,
        config=config,
    )
