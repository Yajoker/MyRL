"""Integration layer for the hierarchical mapless navigation method.

The simulator owns raw observations, while this module owns the single
episode-local temporal LiDAR processor.  Every planner/controller consumer
therefore sees the same two-frame observation and the same risk estimate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch

from ablation_config import AblationConfig
from ablation_system import HighLevelPlannerPeriodic
from config import IntegrationConfig
from high_level_planner import HighLevelPlanner
from high_level_planner_scalar import (
    HighLevelPlannerScalar,
    HighLevelPlannerScalarPeriodic,
)
from low_level_controller import LowLevelController
from robot_nav.SIM_ENV.sensor_metadata import LidarMetadata, RawLidarObservation
from temporal_lidar import TemporalLidarObservation, TemporalLidarProcessor


PreparedObservation = Union[RawLidarObservation, TemporalLidarObservation]


class HierarchicalNavigationSystem:
    """Combine temporal LiDAR, event-triggered planning, and low-level TD3."""

    def __init__(
        self,
        *,
        lidar_metadata: LidarMetadata,
        action_dim: int = 2,
        max_action: float = 1.0,
        device=None,
        load_models: bool = False,
        models_directory: Path = Path("myrl/models"),
        subgoal_threshold: Optional[float] = None,
        integration_config: Optional[IntegrationConfig] = None,
        ablation_config: Optional[AblationConfig] = None,
    ) -> None:
        if not isinstance(lidar_metadata, LidarMetadata):
            raise TypeError("lidar_metadata must be a LidarMetadata instance")

        self._integration_config = integration_config or IntegrationConfig()
        motion_cfg = self._integration_config.motion
        trigger_cfg = self._integration_config.trigger
        planner_cfg = self._integration_config.planner

        if not np.isclose(
            float(motion_cfg.dt),
            float(lidar_metadata.sample_period_s),
            rtol=0.0,
            atol=1.0e-9,
        ):
            raise ValueError(
                "MotionConfig.dt must match the simulator LiDAR sample period "
                f"({motion_cfg.dt} != {lidar_metadata.sample_period_s})"
            )
        if subgoal_threshold is None:
            subgoal_threshold = trigger_cfg.subgoal_reach_threshold
        if float(subgoal_threshold) <= 0.0:
            raise ValueError("subgoal_threshold must be positive")

        self.lidar_metadata = lidar_metadata
        self.num_beams = int(lidar_metadata.beam_count)
        self.step_duration = float(lidar_metadata.sample_period_s)
        self.subgoal_threshold = float(subgoal_threshold)
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # This is the only temporal processor in a navigation system.  Raw
        # observations may be prepared once and then shared by every consumer.
        self.temporal_lidar = TemporalLidarProcessor(
            lidar_metadata,
            trigger_cfg,
        )

        ab_cfg = ablation_config or AblationConfig.from_env()
        is_baseline = (
            ab_cfg.replan_mode == "event" and ab_cfg.value_mode == "dual"
        )
        if not is_baseline:
            models_directory = Path(models_directory) / ab_cfg.exp_tag()

        planner_cls = HighLevelPlanner
        planner_kwargs = {}
        if (
            ab_cfg.replan_mode == "periodic"
            and ab_cfg.value_mode == "scalar"
        ):
            planner_cls = HighLevelPlannerScalarPeriodic
            planner_kwargs = {
                "replan_k": int(ab_cfg.replan_k),
                "allow_subgoal_immediate": bool(
                    ab_cfg.allow_subgoal_immediate
                ),
                "scalar_lambda": float(ab_cfg.scalar_lambda),
            }
        elif ab_cfg.replan_mode == "periodic":
            planner_cls = HighLevelPlannerPeriodic
            planner_kwargs = {
                "replan_k": int(ab_cfg.replan_k),
                "allow_subgoal_immediate": bool(
                    ab_cfg.allow_subgoal_immediate
                ),
            }
        elif ab_cfg.value_mode == "scalar":
            planner_cls = HighLevelPlannerScalar
            planner_kwargs = {
                "scalar_lambda": float(ab_cfg.scalar_lambda),
            }

        self.high_level_planner = planner_cls(
            belief_dim=self.num_beams,
            device=self.device,
            save_directory=Path(models_directory) / "high_level",
            model_name="high_level_planner",
            load_model=load_models,
            step_duration=self.step_duration,
            subgoal_reach_threshold=self.subgoal_threshold,
            lidar_metadata=lidar_metadata,
            max_angular_velocity=motion_cfg.omega_max,
            trigger_config=trigger_cfg,
            planner_config=planner_cfg,
            **planner_kwargs,
        )

        low_level_state_dim = 2 * self.num_beams + 5
        self.low_level_controller = LowLevelController(
            state_dim=low_level_state_dim,
            action_dim=action_dim,
            max_action=max_action,
            device=self.device,
            save_directory=Path(models_directory) / "low_level",
            model_name="low_level_controller",
            load_model=load_models,
            num_beams=self.num_beams,
            max_lin_velocity=self._integration_config.training.max_lin_velocity,
            max_ang_velocity=self._integration_config.training.max_ang_velocity,
        )

        self.current_subgoal: Optional[Tuple[float, float]] = None
        self.current_subgoal_world: Optional[np.ndarray] = None
        self.prev_policy_action = np.zeros(2, dtype=np.float32)
        self.prev_env_action = [0.0, 0.0]
        self.step_count = 0
        self.last_replanning_step = 0
        self.last_linear_velocity = 0.0

    def prepare_observation(
        self,
        observation: PreparedObservation,
    ) -> TemporalLidarObservation:
        """Return the canonical temporal packet for one simulator observation.

        Passing an already prepared packet is permitted only when it was built
        for this system's beam geometry.  Passing the same raw observation more
        than once is idempotent by the processor contract.
        """

        if isinstance(observation, RawLidarObservation):
            return self.temporal_lidar.process(observation)
        if isinstance(observation, TemporalLidarObservation):
            expected_shape = (2, self.num_beams)
            if observation.lidar_channels.shape != expected_shape:
                raise ValueError(
                    "prepared lidar_channels do not match this system: "
                    f"expected {expected_shape}, got "
                    f"{observation.lidar_channels.shape}"
                )
            return observation
        raise TypeError(
            "observation must be RawLidarObservation or "
            "TemporalLidarObservation"
        )

    def step(
        self,
        observation: PreparedObservation,
        goal_distance: float,
        goal_cos: float,
        goal_sin: float,
        goal_position=None,
    ):
        """Choose one command from an atomic current observation.

        This convenience path is intended for inference.  Training uses the
        same ``prepare_observation`` method but keeps transition bookkeeping in
        ``train.py``.
        """

        packet = self.prepare_observation(observation)
        robot_pose = packet.pose_xytheta
        self.step_count = int(packet.observation_id)

        trigger_flags = self.high_level_planner.check_triggers(
            packet,
            robot_pose,
            current_step=self.step_count,
        )
        should_replan = (
            self.high_level_planner.current_subgoal_world is None
            or self.high_level_planner.should_replan(trigger_flags)
        )

        subgoal_distance: Optional[float] = None
        subgoal_angle: Optional[float] = None
        decision_meta: dict = {}
        if should_replan:
            (
                subgoal_distance,
                subgoal_angle,
                decision_meta,
            ) = self.high_level_planner.generate_subgoal(
                packet,
                goal_distance,
                goal_cos,
                goal_sin,
                robot_pose=robot_pose,
                current_step=self.step_count,
                waypoints=None,
                window_metrics=None,
                current_speed=self.last_linear_velocity,
            )
            planner_world = self.high_level_planner.current_subgoal_world
            self.current_subgoal_world = (
                None
                if planner_world is None
                else np.asarray(planner_world, dtype=np.float32)
            )
            self.last_replanning_step = self.step_count
        else:
            planner_world = self.high_level_planner.current_subgoal_world
            if planner_world is not None:
                self.current_subgoal_world = np.asarray(
                    planner_world, dtype=np.float32
                )

        relative_geometry = self.high_level_planner.get_relative_subgoal(
            robot_pose
        )
        if relative_geometry[0] is None:
            if subgoal_distance is not None and subgoal_angle is not None:
                relative_geometry = (subgoal_distance, subgoal_angle)
            elif self.current_subgoal is not None:
                relative_geometry = self.current_subgoal
            else:
                relative_geometry = (0.0, 0.0)
        self.current_subgoal = (
            float(relative_geometry[0]),
            float(relative_geometry[1]),
        )

        if should_replan:
            selected_wp = decision_meta.get("selected_waypoint")
            if selected_wp is None:
                print(
                    "New subgoal: distance={:.2f}m, angle={:.2f}rad".format(
                        self.current_subgoal[0],
                        self.current_subgoal[1],
                    )
                )
            else:
                print(
                    "New subgoal (wp {}): distance={:.2f}m, "
                    "angle={:.2f}rad".format(
                        int(selected_wp),
                        self.current_subgoal[0],
                        self.current_subgoal[1],
                    )
                )

        low_level_state = self.low_level_controller.process_observation(
            packet,
            self.current_subgoal[0],
            self.current_subgoal[1],
            self.prev_policy_action,
        )
        policy_action = np.clip(
            self.low_level_controller.predict_action(low_level_state),
            -1.0,
            1.0,
        )
        env_action = self.low_level_controller.scale_action_for_env(
            policy_action
        )
        linear_velocity, angular_velocity = self._apply_velocity_shielding(
            float(env_action[0]),
            float(env_action[1]),
            packet.current_scan_m,
        )

        self.last_linear_velocity = float(linear_velocity)
        self.prev_env_action = [linear_velocity, angular_velocity]
        self.prev_policy_action = policy_action.astype(
            np.float32, copy=False
        )
        return [linear_velocity, angular_velocity]

    def apply_velocity_shielding(
        self,
        linear_velocity: float,
        angular_velocity: float,
        laser_scan,
    ) -> Tuple[float, float]:
        """Apply the frozen scan-based velocity shielding rule."""

        return self._apply_velocity_shielding(
            linear_velocity,
            angular_velocity,
            laser_scan,
        )

    def _apply_velocity_shielding(
        self,
        linear_velocity: float,
        angular_velocity: float,
        laser_scan,
    ) -> Tuple[float, float]:
        motion_cfg = self._integration_config.motion
        shield_cfg = motion_cfg.shielding
        if not shield_cfg.enabled:
            return float(linear_velocity), float(angular_velocity)

        scan_arr = np.asarray(laser_scan, dtype=np.float32)
        finite_scan = scan_arr[np.isfinite(scan_arr)]
        if finite_scan.size == 0:
            return float(linear_velocity), float(angular_velocity)

        d_min = float(finite_scan.min())
        sigma_input = float(
            np.clip(
                shield_cfg.gain * (d_min - shield_cfg.safe_distance),
                -60.0,
                60.0,
            )
        )
        linear_scale = float(1.0 / (1.0 + np.exp(-sigma_input)))
        scaled_linear = float(
            np.clip(
                linear_velocity * linear_scale,
                0.0,
                motion_cfg.v_max,
            )
        )

        scaled_angular = float(angular_velocity)
        if d_min <= shield_cfg.safe_distance:
            scaled_angular *= shield_cfg.angular_gain
        scaled_angular = float(
            np.clip(
                scaled_angular,
                -motion_cfg.omega_max,
                motion_cfg.omega_max,
            )
        )
        return scaled_linear, scaled_angular

    def reset(self) -> None:
        """Reset every episode-local temporal, planner, and action state."""

        self.temporal_lidar.reset()
        self.high_level_planner.reset_runtime_state()
        self.current_subgoal = None
        self.current_subgoal_world = None
        self.prev_env_action = [0.0, 0.0]
        self.prev_policy_action = np.zeros(2, dtype=np.float32)
        self.step_count = 0
        self.last_replanning_step = 0
        self.last_linear_velocity = 0.0

    # Mapless compatibility surfaces retained for callers outside this module.
    def plan_global_route(self, *_, **__):
        return []

    def get_active_waypoints(self, *_, **__):
        return []

    def update_window_state(self, *_, **__):
        return {}


def create_navigation_system(
    *,
    lidar_metadata: LidarMetadata,
    load_models: bool = False,
    subgoal_threshold: Optional[float] = None,
    **kwargs,
) -> HierarchicalNavigationSystem:
    """Create a navigation system from simulator-provided sensor metadata."""

    return HierarchicalNavigationSystem(
        lidar_metadata=lidar_metadata,
        action_dim=2,
        max_action=1.0,
        load_models=load_models,
        subgoal_threshold=subgoal_threshold,
        **kwargs,
    )
