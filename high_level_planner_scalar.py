from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from ablation_system import HighLevelPlannerPeriodic
from config import PlannerConfig, TriggerConfig
from high_level_planner import HighLevelPlanner
from high_level_value_scalar import HighLevelValueNetScalar
from robot_nav.SIM_ENV.sensor_metadata import LidarMetadata


def _without_double_q(planner_config: Optional[PlannerConfig]) -> PlannerConfig:
    """Return the scalar-ablation planner configuration."""

    config = planner_config or PlannerConfig()
    if config.high_level_double_q_enabled:
        config = replace(config, high_level_double_q_enabled=False)
    return config


class _ScalarPlannerMixin:
    """Shared scalar-Q implementation for event and periodic replanning."""

    scalar_lambda: float
    belief_dim: int
    goal_feature_dim: int

    def _install_scalar_networks(self, scalar_lambda: float) -> None:
        scalar_lambda = float(scalar_lambda)
        if not math.isfinite(scalar_lambda) or scalar_lambda < 0.0:
            raise ValueError("scalar_lambda must be finite and non-negative")
        self.scalar_lambda = scalar_lambda
        self.high_level_double_q_enabled = False

        self.value_net = HighLevelValueNetScalar(
            belief_dim=self.belief_dim,
            goal_info_dim=self.goal_feature_dim,
            geom_dim=2,
            hidden_dim=192,
        ).to(self.device)
        self.target_value_net = HighLevelValueNetScalar(
            belief_dim=self.belief_dim,
            goal_info_dim=self.goal_feature_dim,
            geom_dim=2,
            hidden_dim=192,
        ).to(self.device)
        self.target_value_net.load_state_dict(
            self.value_net.state_dict(), strict=True
        )
        for parameter in self.target_value_net.parameters():
            parameter.requires_grad = False

        self.value_optimizer = torch.optim.Adam(
            self.value_net.parameters(), lr=1e-3
        )
        self.value_loss_fn = nn.MSELoss()

    def _select_best_subgoal(
        self,
        lidar_channels: np.ndarray,
        goal_info: Tuple[float, float, float],
        candidates: List[Tuple[float, float]],
        robot_pose: Optional[Sequence[float]] = None,
    ) -> Tuple[float, float]:
        """Select a candidate by scalar Q only.

        ``robot_pose`` remains in the signature for compatibility with the
        main planner, but no network-external continuity bonus is applied.
        """

        del robot_pose
        if not candidates:
            goal_distance, goal_cos, goal_sin = goal_info
            goal_direction = math.atan2(goal_sin, goal_cos)
            distance = max(
                self.frontier_min_distance,
                min(goal_distance, self.frontier_max_distance),
            )
            return float(distance), float(goal_direction)

        channels = np.asarray(lidar_channels, dtype=np.float32)
        expected_shape = (2, self.belief_dim)
        if channels.shape != expected_shape:
            raise ValueError(
                f"lidar_channels must have shape {expected_shape}, "
                f"got {channels.shape}"
            )
        if not np.all(np.isfinite(channels)):
            raise ValueError("lidar_channels must contain only finite values")

        candidate_array = np.asarray(candidates, dtype=np.float32)
        if candidate_array.ndim != 2 or candidate_array.shape[1] != 2:
            raise ValueError(
                "candidates must have shape [num_candidates, 2], "
                f"got {candidate_array.shape}"
            )

        self.value_net.eval()
        candidate_count = candidate_array.shape[0]
        lidar_batch = torch.as_tensor(
            channels,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0).repeat(candidate_count, 1, 1)
        goal_single = self.process_goal_info(*goal_info)
        goal_batch = goal_single.unsqueeze(0).repeat(candidate_count, 1)
        subgoal_batch = torch.as_tensor(
            candidate_array,
            dtype=torch.float32,
            device=self.device,
        )

        with torch.no_grad():
            q_values = self.value_net(
                lidar_batch,
                goal_batch,
                subgoal_batch,
            )

        best_index = int(torch.argmax(q_values).item())
        best_distance, best_angle = candidates[best_index]
        return float(best_distance), float(best_angle)

    def update_planner(
        self,
        states,
        actions,
        rewards_eff,
        safety_costs,
        dones,
        next_states,
        batch_size: int = 64,
    ):
        """Update scalar Q using the frozen ``2N+3`` high-level state."""

        del batch_size
        states_t = torch.as_tensor(
            states, dtype=torch.float32, device=self.device
        )
        actions_t = torch.as_tensor(
            actions, dtype=torch.float32, device=self.device
        )
        rewards_eff_t = torch.as_tensor(
            rewards_eff, dtype=torch.float32, device=self.device
        ).reshape(-1)
        safety_costs_t = torch.as_tensor(
            safety_costs, dtype=torch.float32, device=self.device
        ).reshape(-1)
        dones_t = torch.as_tensor(
            dones, dtype=torch.float32, device=self.device
        ).reshape(-1)
        next_states_t = torch.as_tensor(
            next_states, dtype=torch.float32, device=self.device
        )

        expected_state_dim = 2 * self.belief_dim + self.goal_feature_dim
        if states_t.ndim != 2 or states_t.shape[1] != expected_state_dim:
            raise ValueError(
                f"high-level states must have shape [B, {expected_state_dim}], "
                f"got {tuple(states_t.shape)}"
            )
        if next_states_t.shape != states_t.shape:
            raise ValueError(
                f"next_states must match states shape {tuple(states_t.shape)}, "
                f"got {tuple(next_states_t.shape)}"
            )
        batch_count = states_t.shape[0]
        if actions_t.shape != (batch_count, 2):
            raise ValueError(
                f"actions must have shape [{batch_count}, 2], "
                f"got {tuple(actions_t.shape)}"
            )
        for name, values in (
            ("rewards_eff", rewards_eff_t),
            ("safety_costs", safety_costs_t),
            ("dones", dones_t),
        ):
            if values.shape != (batch_count,):
                raise ValueError(
                    f"{name} must contain {batch_count} values, "
                    f"got {tuple(values.shape)}"
                )

        lidar_end = 2 * self.belief_dim
        lidar_t = states_t[:, :lidar_end].reshape(
            batch_count, 2, self.belief_dim
        )
        goal_t = states_t[:, lidar_end:]
        lidar_next_t = next_states_t[:, :lidar_end].reshape(
            batch_count, 2, self.belief_dim
        )
        goal_next_t = next_states_t[:, lidar_end:]

        self.value_net.train()
        scalar_reward = (
            rewards_eff_t - self.scalar_lambda * safety_costs_t
        )
        not_done = 1.0 - dones_t

        with torch.no_grad():
            self.target_value_net.eval()
            next_goal = goal_next_t.cpu().numpy().astype(
                np.float32, copy=False
            )
            next_goal_distance = next_goal[:, 0] * 30.0
            q_next_values: List[float] = []

            for index in range(batch_count):
                # Candidate geometry is determined from the current-range
                # channel only. The signed closure channel remains a value-net
                # input and never changes frontier geometry.
                next_scan_m = (
                    lidar_next_t[index, 0].cpu().numpy()
                    * float(self.lidar_metadata.range_max_m)
                ).astype(np.float32, copy=False)
                candidates = self._generate_frontier_candidates(
                    next_scan_m,
                    float(next_goal_distance[index]),
                    float(next_goal[index, 1]),
                    float(next_goal[index, 2]),
                )
                if not candidates:
                    q_next_values.append(0.0)
                    continue

                subgoal_batch = torch.as_tensor(
                    np.asarray(candidates, dtype=np.float32),
                    dtype=torch.float32,
                    device=self.device,
                )
                candidate_count = subgoal_batch.shape[0]
                lidar_batch = lidar_next_t[index].unsqueeze(0).repeat(
                    candidate_count, 1, 1
                )
                goal_batch = goal_next_t[index].unsqueeze(0).repeat(
                    candidate_count, 1
                )
                candidate_q = self.target_value_net(
                    lidar_batch,
                    goal_batch,
                    subgoal_batch,
                )
                q_next_values.append(float(torch.max(candidate_q).item()))

            q_next = torch.as_tensor(
                q_next_values,
                dtype=torch.float32,
                device=self.device,
            )
            target = (
                scalar_reward
                + self.gamma_high * not_done * q_next
            )

        q_prediction = self.value_net(lidar_t, goal_t, actions_t)
        loss = self.value_loss_fn(q_prediction, target)

        self.value_optimizer.zero_grad()
        loss.backward()
        self.value_optimizer.step()
        self._soft_update_target(self.value_net, self.target_value_net)
        self.iter_count += 1

        return {
            "loss_total": float(loss.item()),
            "q_mean": float(q_prediction.mean().item()),
            "r_eff_mean": float(rewards_eff_t.mean().item()),
            "c_safe_mean": float(safety_costs_t.mean().item()),
            "r_scalar_mean": float(scalar_reward.mean().item()),
            "q_next_mean": float(q_next.mean().item()),
        }

    def save_model(self, filename, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{filename}.pth"
        torch.save(self.value_net.state_dict(), path)
        print(f"模型已保存到 {path}")

    def load_model(self, filename, directory):
        path = Path(directory) / f"{filename}.pth"
        if not path.is_file():
            raise FileNotFoundError(path)

        state_dict = torch.load(
            path,
            map_location=self.device,
            weights_only=True,
        )
        expected = self.value_net.state_dict()
        if set(state_dict) != set(expected):
            missing_keys = sorted(set(expected) - set(state_dict))
            unexpected_keys = sorted(set(state_dict) - set(expected))
            raise RuntimeError(
                "Incompatible scalar checkpoint keys; "
                f"missing={missing_keys}, unexpected={unexpected_keys}"
            )
        shape_mismatches = {
            key: (tuple(state_dict[key].shape), tuple(expected[key].shape))
            for key in expected
            if state_dict[key].shape != expected[key].shape
        }
        if shape_mismatches:
            raise RuntimeError(
                "Incompatible scalar checkpoint tensor shapes: "
                f"{shape_mismatches}"
            )

        self.value_net.load_state_dict(state_dict, strict=True)
        self.target_value_net.load_state_dict(
            self.value_net.state_dict(), strict=True
        )
        print(f"模型已从 {path} 加载")


class HighLevelPlannerScalar(_ScalarPlannerMixin, HighLevelPlanner):
    """Scalar-Q ablation with event-triggered replanning."""

    def __init__(
        self,
        belief_dim: Optional[int] = None,
        device=None,
        save_directory: Path = Path("models/high_level"),
        model_name: str = "high_level_planner",
        load_model: bool = False,
        load_directory=None,
        step_duration: float = 0.3,
        subgoal_reach_threshold: Optional[float] = None,
        *,
        lidar_metadata: LidarMetadata,
        max_angular_velocity: float,
        trigger_config: Optional[TriggerConfig] = None,
        planner_config: Optional[PlannerConfig] = None,
        scalar_lambda: float = 1.0,
    ) -> None:
        super().__init__(
            belief_dim=belief_dim,
            device=device,
            save_directory=save_directory,
            model_name=model_name,
            load_model=False,
            load_directory=load_directory,
            step_duration=step_duration,
            subgoal_reach_threshold=subgoal_reach_threshold,
            lidar_metadata=lidar_metadata,
            max_angular_velocity=max_angular_velocity,
            trigger_config=trigger_config,
            planner_config=_without_double_q(planner_config),
        )
        self._install_scalar_networks(scalar_lambda)

        if load_model:
            load_directory = (
                load_directory if load_directory is not None
                else save_directory
            )
            self.load_model(filename=model_name, directory=load_directory)


class HighLevelPlannerScalarPeriodic(
    _ScalarPlannerMixin,
    HighLevelPlannerPeriodic,
):
    """Scalar-Q ablation with fixed-period replanning."""

    def __init__(
        self,
        belief_dim: Optional[int] = None,
        device=None,
        save_directory: Path = Path("models/high_level"),
        model_name: str = "high_level_planner",
        load_model: bool = False,
        load_directory=None,
        step_duration: float = 0.3,
        subgoal_reach_threshold: Optional[float] = None,
        *,
        lidar_metadata: LidarMetadata,
        max_angular_velocity: float,
        trigger_config: Optional[TriggerConfig] = None,
        planner_config: Optional[PlannerConfig] = None,
        scalar_lambda: float = 1.0,
        replan_k: int = 10,
        allow_subgoal_immediate: bool = True,
    ) -> None:
        super().__init__(
            belief_dim=belief_dim,
            device=device,
            save_directory=save_directory,
            model_name=model_name,
            load_model=False,
            load_directory=load_directory,
            step_duration=step_duration,
            subgoal_reach_threshold=subgoal_reach_threshold,
            lidar_metadata=lidar_metadata,
            max_angular_velocity=max_angular_velocity,
            trigger_config=trigger_config,
            planner_config=_without_double_q(planner_config),
            replan_k=replan_k,
            allow_subgoal_immediate=allow_subgoal_immediate,
        )
        self._install_scalar_networks(scalar_lambda)

        if load_model:
            load_directory = (
                load_directory if load_directory is not None
                else save_directory
            )
            self.load_model(filename=model_name, directory=load_directory)
