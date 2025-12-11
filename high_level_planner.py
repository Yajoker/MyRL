"""高层规划器：事件触发的前沿候选 + 统一价值网络 (E-FVQ)。"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from config import PlannerConfig, TriggerConfig


class HighLevelValueNet(nn.Module):
    """统一高层值函数 Q_H(s, g) with dual heads."""

    def __init__(self, belief_dim: int = 90, goal_info_dim: int = 3, geom_dim: int = 2, hidden_dim: int = 192):
        super().__init__()

        self.cnn1 = nn.Conv1d(1, 8, kernel_size=5, stride=2)
        self.cnn2 = nn.Conv1d(8, 16, kernel_size=3, stride=2)
        self.cnn3 = nn.Conv1d(16, 8, kernel_size=3, stride=1)

        self.goal_embed = nn.Linear(goal_info_dim, 64)
        self.subgoal_embed = nn.Linear(geom_dim, 16)

        cnn_out_dim = self._get_cnn_output_dim(belief_dim)
        concat_dim = cnn_out_dim + 64 + 16

        self.shared_fc1 = nn.Linear(concat_dim, hidden_dim)
        self.shared_fc2 = nn.Linear(hidden_dim, hidden_dim)

        branch_hidden_dim = hidden_dim // 2
        self.eff_fc = nn.Linear(hidden_dim, branch_hidden_dim)
        self.q_eff_head = nn.Linear(branch_hidden_dim, 1)

        self.safe_fc = nn.Linear(hidden_dim, branch_hidden_dim)
        self.q_safe_head = nn.Linear(branch_hidden_dim, 1)

    def _get_cnn_output_dim(self, belief_dim: int) -> int:
        dummy = torch.zeros(1, 1, belief_dim)
        x = self.cnn1(dummy)
        x = self.cnn2(x)
        x = self.cnn3(x)
        return x.view(1, -1).shape[1]

    def forward(
        self,
        laser: torch.Tensor,
        goal_info: torch.Tensor,
        subgoal_geom: torch.Tensor,
        return_heads: bool = False,
    ):
        x = laser.unsqueeze(1)
        x = F.relu(self.cnn1(x))
        x = F.relu(self.cnn2(x))
        x = F.relu(self.cnn3(x))
        x = x.view(x.size(0), -1)

        g = F.relu(self.goal_embed(goal_info))
        geom = F.relu(self.subgoal_embed(subgoal_geom))

        h = torch.cat([x, g, geom], dim=1)
        h_shared = F.relu(self.shared_fc1(h))
        h_shared = F.relu(self.shared_fc2(h_shared))

        h_eff = F.relu(self.eff_fc(h_shared))
        q_eff = self.q_eff_head(h_eff).squeeze(-1)

        h_safe = F.relu(self.safe_fc(h_shared))
        q_safe = self.q_safe_head(h_safe).squeeze(-1)

        if return_heads:
            return q_eff, q_safe

        lambda_q = getattr(self, "safety_q_weight", 1.0)
        q_total = q_eff - lambda_q * q_safe
        return q_total


class EventTrigger:
    """事件触发器：决定何时重新生成子目标。"""

    def __init__(
        self,
        *,
        config: TriggerConfig,
        step_duration: float,
        min_interval: Optional[float] = None,
        subgoal_reach_threshold: Optional[float] = None,
    ) -> None:
        self._config = config
        self.safety_trigger_distance = config.safety_trigger_distance
        self.subgoal_reach_threshold = (
            subgoal_reach_threshold if subgoal_reach_threshold is not None else config.subgoal_reach_threshold
        )
        self.stagnation_steps = max(1, int(config.stagnation_steps))
        self.step_duration = float(step_duration)
        self.progress_abs = float(config.progress_epsilon_abs)
        self.progress_rel = float(config.progress_epsilon_rel)
        self.risk_alpha = float(config.risk_alpha)
        self.risk_trigger_threshold = float(config.risk_trigger_threshold)
        self.risk_near_threshold = float(config.risk_near_threshold)
        self.risk_percentile = float(config.risk_percentile)

        if min_interval is not None and min_interval > 0:
            self.min_interval = float(min_interval)
        elif getattr(config, "min_interval", 0) and config.min_interval > 0:
            self.min_interval = float(config.min_interval)
        else:
            steps_cfg = max(1, int(getattr(config, "min_step_interval", 1)))
            self.min_interval = float(steps_cfg * self.step_duration) if self.step_duration > 0 else 0.0

        if self.step_duration > 0:
            self.min_step_interval = max(1, int(math.ceil(self.min_interval / self.step_duration)))
            self.max_step_interval = max(1, int(math.ceil(config.max_interval / self.step_duration)))
        else:
            self.min_step_interval = max(1, int(getattr(config, "min_step_interval", 1)))
            self.max_step_interval = max(1, int(getattr(config, "max_step_interval", 1)))

        self.last_trigger_step = -self.min_step_interval
        self.best_goal_distance: Optional[float] = None
        self.last_progress_step = 0

    def _delta_progress_min(self, reference_distance: float) -> float:
        return self.progress_abs + self.progress_rel * max(reference_distance, 0.0)

    def risk_trigger(self, risk_index: float) -> bool:
        return risk_index >= self.risk_trigger_threshold

    def subgoal_reached(self, dist_to_subgoal: Optional[float]) -> bool:
        return dist_to_subgoal is not None and dist_to_subgoal <= self.subgoal_reach_threshold

    def global_progress_stagnant(self, goal_distance: float, current_step: int) -> bool:
        if not np.isfinite(goal_distance):
            return False

        if self.best_goal_distance is None:
            self.best_goal_distance = goal_distance
            self.last_progress_step = current_step
            return False

        improvement = self.best_goal_distance - goal_distance
        threshold = self._delta_progress_min(self.best_goal_distance)

        if improvement >= threshold:
            self.best_goal_distance = goal_distance
            self.last_progress_step = current_step
            return False

        return (current_step - self.last_progress_step) >= self.stagnation_steps

    def reset_progress(self, goal_distance: float, current_step: int) -> None:
        if not np.isfinite(goal_distance):
            self.best_goal_distance = None
            self.last_progress_step = current_step
            return
        self.best_goal_distance = goal_distance
        self.last_progress_step = current_step

    def update_progress(self, goal_distance: float, current_step: int) -> None:
        if not np.isfinite(goal_distance):
            return
        if self.best_goal_distance is None:
            self.best_goal_distance = goal_distance
            self.last_progress_step = current_step
            return
        improvement = self.best_goal_distance - goal_distance
        if improvement >= self._delta_progress_min(self.best_goal_distance):
            self.best_goal_distance = goal_distance
            self.last_progress_step = current_step

    def is_stagnated(self, current_step: int) -> bool:
        if self.best_goal_distance is None:
            return False
        return (current_step - self.last_progress_step) >= self.stagnation_steps

    def can_replan(self, current_step: int) -> bool:
        return current_step - self.last_trigger_step >= self.min_step_interval

    def time_upper_bound(self, current_step: int) -> bool:
        return current_step - self.last_trigger_step >= self.max_step_interval

    def reset_time(self, current_step: int) -> None:
        self.last_trigger_step = current_step

    def reset_state(self) -> None:
        self.last_trigger_step = -self.min_step_interval
        self.best_goal_distance = None
        self.last_progress_step = 0


@dataclass
class TriggerFlags:
    time_ready: bool
    time_over: bool
    progress_stagnant: bool
    risk: bool
    subgoal_reached: bool


class HighLevelPlanner:
    """事件触发 + 前沿候选 + 高层值函数的规划器。"""

    def __init__(
        self,
        belief_dim: int = 90,
        device=None,
        save_directory: Path = Path("models/high_level"),
        model_name: str = "high_level_planner",
        load_model: bool = False,
        load_directory=None,
        step_duration: float = 0.3,
        min_interval: Optional[float] = None,
        subgoal_reach_threshold: Optional[float] = None,
        waypoint_lookahead: Optional[int] = None,
        *,
        trigger_config: Optional[TriggerConfig] = None,
        planner_config: Optional[PlannerConfig] = None,
    ) -> None:
        trigger_cfg = trigger_config or TriggerConfig()
        planner_cfg = planner_config or PlannerConfig()
        self.planner_config = planner_cfg
        self.safety_q_weight = float(planner_cfg.safety_q_weight)
        self.safety_loss_weight = float(planner_cfg.safety_loss_weight)

        if subgoal_reach_threshold is None:
            subgoal_reach_threshold = trigger_cfg.subgoal_reach_threshold
        if waypoint_lookahead is None:
            waypoint_lookahead = planner_cfg.waypoint_lookahead
        if min_interval is None:
            min_interval = trigger_cfg.min_interval if trigger_cfg.min_interval > 0 else trigger_cfg.min_step_interval * step_duration

        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_directory = Path(save_directory)
        self.save_directory.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name

        self.waypoint_lookahead = max(1, int(waypoint_lookahead))
        self.active_window_feature_dim = 6
        self.per_window_feature_dim = 4
        self.goal_feature_dim = 3 + self.active_window_feature_dim + self.per_window_feature_dim * self.waypoint_lookahead
        self.belief_dim = belief_dim

        self.value_net = HighLevelValueNet(belief_dim=belief_dim, goal_info_dim=self.goal_feature_dim, geom_dim=2).to(self.device)
        self.value_net.safety_q_weight = self.safety_q_weight
        self.target_value_net = HighLevelValueNet(
            belief_dim=belief_dim, goal_info_dim=self.goal_feature_dim, geom_dim=2
        ).to(self.device)
        self.target_value_net.load_state_dict(self.value_net.state_dict())
        self.target_value_net.safety_q_weight = self.safety_q_weight
        for p in self.target_value_net.parameters():
            p.requires_grad = False
        self.gamma_high = getattr(self.planner_config, "high_level_gamma", 0.99)
        self.tau_high = getattr(self.planner_config, "high_level_tau", 0.005)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=1e-3)
        self.value_loss_fn = nn.MSELoss()

        self.event_trigger = EventTrigger(
            config=trigger_cfg,
            step_duration=step_duration,
            min_interval=min_interval,
            subgoal_reach_threshold=subgoal_reach_threshold,
        )
        self.step_duration = step_duration

        self.writer = SummaryWriter(log_dir=self.save_directory)
        self.iter_count = 0

        self.current_subgoal: Optional[Tuple[float, float]] = None
        self.current_subgoal_world: Optional[np.ndarray] = None
        self.last_goal_distance: float = float("inf")
        self.last_goal_direction: float = 0.0
        self.subgoal_hidden = None

        self.frontier_min_distance = planner_cfg.frontier_min_dist
        self.frontier_max_distance = planner_cfg.frontier_max_dist
        self.frontier_gap_min_width = planner_cfg.frontier_gap_min_width
        self.frontier_num_candidates = planner_cfg.frontier_num_candidates
        self.consistency_lambda = planner_cfg.consistency_lambda
        self.consistency_sigma_r = planner_cfg.consistency_sigma_r
        self.consistency_sigma_theta = planner_cfg.consistency_sigma_theta

        if load_model:
            load_dir = load_directory if load_directory else save_directory
            self.load_model(filename=model_name, directory=load_dir)

    # ------------------------- 基础工具 -------------------------
    def reset_subgoal_hidden(self) -> None:
        self.subgoal_hidden = None

    def _wrap_angle(self, angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    def _world_to_relative(self, robot_pose, waypoint) -> Tuple[float, float]:
        if robot_pose is None:
            return 0.0, 0.0
        waypoint_vec = np.asarray(waypoint, dtype=np.float32)
        dx = float(waypoint_vec[0] - robot_pose[0])
        dy = float(waypoint_vec[1] - robot_pose[1])
        distance = math.hypot(dx, dy)
        angle = self._wrap_angle(math.atan2(dy, dx) - robot_pose[2])
        return distance, angle

    def _relative_to_world(self, robot_pose, distance: float, angle: float) -> np.ndarray:
        if robot_pose is None:
            return np.zeros(2, dtype=np.float32)
        world_x = robot_pose[0] + distance * math.cos(robot_pose[2] + angle)
        world_y = robot_pose[1] + distance * math.sin(robot_pose[2] + angle)
        return np.array([world_x, world_y], dtype=np.float32)

    def get_relative_subgoal(self, robot_pose: Optional[Sequence[float]]) -> Tuple[Optional[float], Optional[float]]:
        if robot_pose is None or self.current_subgoal_world is None:
            return None, None

        robot_xy = np.asarray(robot_pose[:2], dtype=np.float32)
        subgoal_world = np.asarray(self.current_subgoal_world, dtype=np.float32)
        delta = subgoal_world - robot_xy
        distance = float(np.linalg.norm(delta))
        if distance <= 1e-6:
            return 0.0, 0.0

        heading = float(robot_pose[2])
        angle = math.atan2(float(delta[1]), float(delta[0])) - heading
        angle = math.atan2(math.sin(angle), math.cos(angle))
        return distance, angle

    # ------------------------- 状态处理 -------------------------
    def process_laser_scan(self, laser_scan):
        laser_scan = np.array(laser_scan)
        inf_mask = np.isinf(laser_scan)
        laser_scan[inf_mask] = 7.0
        laser_scan = laser_scan / 7.0
        return torch.FloatTensor(laser_scan).to(self.device)

    def process_goal_info(self, distance, cos_angle, sin_angle, waypoint_features=None):
        norm_distance = min(float(distance) / 30.0, 1.0)
        goal_tensor = torch.tensor([norm_distance, float(cos_angle), float(sin_angle)], device=self.device, dtype=torch.float32)
        if waypoint_features is not None:
            goal_tensor = torch.cat((goal_tensor, torch.tensor(waypoint_features, device=self.device, dtype=torch.float32)))
        return goal_tensor

    def build_waypoint_features(self, waypoints, robot_pose) -> List[float]:
        active_features = [0.0] * self.active_window_feature_dim
        sequence_features = [0.0] * (self.per_window_feature_dim * self.waypoint_lookahead)
        return active_features + sequence_features

    def build_state_vector(self, laser_scan, distance, cos_angle, sin_angle, waypoints=None, robot_pose=None):
        with torch.no_grad():
            laser_tensor = self.process_laser_scan(laser_scan)
            waypoint_features = self.build_waypoint_features(waypoints, robot_pose)
            goal_tensor = self.process_goal_info(distance, cos_angle, sin_angle, waypoint_features)
            state_tensor = torch.cat((laser_tensor, goal_tensor))
        return state_tensor.cpu().numpy()

    def compute_risk_index(self, laser_scan: np.ndarray) -> Tuple[float, float, float]:
        scan = np.asarray(laser_scan, dtype=np.float32)
        finite_scan = scan[np.isfinite(scan)]
        if finite_scan.size == 0:
            return 0.0, float("inf"), float("inf")

        d_min = float(np.min(finite_scan))
        percentile = float(np.percentile(finite_scan, self.event_trigger.risk_percentile))
        safe_d = max(self.event_trigger.safety_trigger_distance, 1e-6)

        r_min = max(0.0, (safe_d - d_min) / safe_d)
        r_p = max(0.0, (safe_d - percentile) / safe_d)
        alpha = self.event_trigger.risk_alpha
        risk_index = min(1.0, alpha * r_min + (1.0 - alpha) * r_p)
        return risk_index, d_min, percentile

    # ------------------------- 事件触发 -------------------------
    def check_triggers(
        self,
        laser_scan,
        robot_pose,
        goal_info,
        risk_index: float,
        current_step: int = 0,
        window_metrics: Optional[dict] = None,
    ) -> TriggerFlags:
        goal_distance = float(goal_info[0]) if goal_info else float("inf")
        self.event_trigger.update_progress(goal_distance, current_step)

        dist_to_subgoal, _ = self.get_relative_subgoal(robot_pose)

        time_ready = self.event_trigger.can_replan(current_step)
        time_event = self.event_trigger.time_upper_bound(current_step)
        risk_event = self.event_trigger.risk_trigger(risk_index)
        progress_event = self.event_trigger.global_progress_stagnant(goal_distance, current_step)
        subgoal_event = self.event_trigger.subgoal_reached(dist_to_subgoal)

        return TriggerFlags(
            time_ready=time_ready,
            time_over=time_event,
            progress_stagnant=progress_event,
            risk=risk_event,
            subgoal_reached=subgoal_event,
        )

    def should_replan(self, flags: TriggerFlags) -> bool:
        if flags.risk or flags.subgoal_reached:
            return True
        if not flags.time_ready:
            return False
        return flags.time_over or flags.progress_stagnant

    # ------------------------- 子目标生成 -------------------------
    def _generate_frontier_candidates(self, laser_scan: np.ndarray, goal_distance: float, goal_cos: float, goal_sin: float) -> List[Tuple[float, float]]:
        scan = np.asarray(laser_scan, dtype=np.float32)
        scan = np.nan_to_num(
            scan,
            nan=self.frontier_max_distance,
            posinf=self.frontier_max_distance,
            neginf=0.0,
        )
        scan = np.clip(scan, 0.0, self.frontier_max_distance)

        n = scan.shape[0]
        angles = np.linspace(-math.pi, math.pi, n, endpoint=False)

        safe_dist = max(float(self.event_trigger.safety_trigger_distance), float(self.frontier_min_distance))
        valid = scan[scan > safe_dist]
        if valid.size > 0:
            perc = float(np.percentile(valid, 70.0))
            raw_frontier_dist = max(safe_dist + 0.25, perc)
        else:
            raw_frontier_dist = safe_dist + 0.25

        frontier_max = float(self.frontier_max_distance)
        frontier_dist = min(frontier_max, raw_frontier_dist)
        mask_frontier = scan >= frontier_dist

        diffs = np.abs(np.diff(scan, append=scan[0]))
        delta_d = max(0.4, 0.5 * safe_dist)
        jump_mask = np.zeros_like(scan, dtype=bool)
        for i_diff in range(n):
            if diffs[i_diff] > delta_d and max(scan[i_diff], scan[(i_diff + 1) % n]) > safe_dist:
                jump_mask[i_diff] = True

        combined_mask = mask_frontier | jump_mask

        candidates: List[Tuple[float, float]] = []
        i = 0
        while i < n:
            if not combined_mask[i]:
                i += 1
                continue
            start = i
            while i + 1 < n and combined_mask[i + 1]:
                i += 1
            end = i
            width = angles[end] - angles[start]
            if abs(width) >= self.frontier_gap_min_width:
                mid = (start + end) // 2
                dist = float(scan[mid])
                theta = float(angles[mid])
                r = float(np.clip(dist * 0.8, self.frontier_min_distance, self.frontier_max_distance))
                candidates.append((r, theta))
            i += 1

        goal_dir = math.atan2(goal_sin, goal_cos)
        r_goal = float(np.clip(goal_distance, self.frontier_min_distance, self.frontier_max_distance))
        candidates.append((r_goal, goal_dir))

        if len(candidates) > self.frontier_num_candidates:
            candidates.sort(key=lambda g: -math.cos(g[1] - goal_dir))
            candidates = candidates[: self.frontier_num_candidates]

        return candidates

    def _select_best_subgoal(
        self,
        laser_scan,
        goal_info: Tuple[float, float, float],
        candidates: List[Tuple[float, float]],
        robot_pose: Optional[Sequence[float]] = None,
    ) -> Tuple[float, float]:
        if not candidates:
            goal_distance, goal_cos, goal_sin = goal_info
            goal_dir = math.atan2(goal_sin, goal_cos)
            r = max(self.frontier_min_distance, min(goal_distance, self.frontier_max_distance))
            return r, goal_dir

        self.value_net.eval()
        scan = np.asarray(laser_scan, dtype=np.float32)
        scan = np.nan_to_num(scan, nan=0.0, posinf=self.frontier_max_distance, neginf=0.0)

        laser_t = torch.as_tensor(scan[None, :], dtype=torch.float32, device=self.device)
        dummy_waypoints = self.build_waypoint_features(waypoints=None, robot_pose=None)
        goal_t_single = self.process_goal_info(goal_info[0], goal_info[1], goal_info[2], dummy_waypoints)
        goal_t = goal_t_single.unsqueeze(0)
        geom_t = torch.as_tensor(np.asarray(candidates, dtype=np.float32), dtype=torch.float32, device=self.device)

        laser_batch = laser_t.repeat(geom_t.shape[0], 1)
        goal_batch = goal_t.repeat(geom_t.shape[0], 1)

        with torch.no_grad():
            q_eff, q_safe = self.value_net(laser_batch, goal_batch, geom_t, return_heads=True)
            q_eff_np = q_eff.cpu().numpy()
            q_safe_np = q_safe.cpu().numpy()
            q_vals = q_eff_np - self.safety_q_weight * q_safe_np

        scores = q_vals

        if robot_pose is not None and self.current_subgoal_world is not None:
            last_r, last_theta = self.get_relative_subgoal(robot_pose)
            if last_r is not None:
                lambda_cons = self.consistency_lambda
                sigma_r = max(self.consistency_sigma_r, 1e-6)
                sigma_theta = max(self.consistency_sigma_theta, 1e-6)

                bonuses: List[float] = []
                for (r, theta) in candidates:
                    dr = (r - last_r) / sigma_r
                    dtheta = (theta - last_theta) / sigma_theta
                    bonus = math.exp(-0.5 * (dr * dr + dtheta * dtheta))
                    bonuses.append(lambda_cons * bonus)

                scores = q_vals + np.asarray(bonuses, dtype=np.float32)

        best_idx = int(np.argmax(scores))
        best_r, best_theta = candidates[best_idx]
        return float(best_r), float(best_theta)

    def generate_subgoal(
        self,
        laser_scan,
        goal_distance,
        goal_cos,
        goal_sin,
        robot_pose=None,
        current_step: Optional[int] = None,
        waypoints=None,
        window_metrics: Optional[dict] = None,
        current_speed: Optional[float] = None,
    ):
        goal_info = (float(goal_distance), float(goal_cos), float(goal_sin))

        candidates = self._generate_frontier_candidates(laser_scan, *goal_info)
        final_distance, final_angle = self._select_best_subgoal(
            laser_scan,
            goal_info,
            candidates,
            robot_pose=robot_pose,
        )

        world_target = None
        if robot_pose is not None:
            world_target = self._relative_to_world(robot_pose, final_distance, final_angle)

        self.current_subgoal = (final_distance, final_angle)
        self.last_goal_distance = float(goal_distance)
        self.last_goal_direction = math.atan2(goal_sin, goal_cos)
        if world_target is not None:
            self.current_subgoal_world = world_target
        else:
            self.current_subgoal_world = None

        progress_step = current_step if current_step is not None else 0
        self.event_trigger.reset_progress(goal_distance, progress_step)

        metadata = {"num_candidates": len(candidates)}
        return final_distance, final_angle, metadata

    # ------------------------- 训练 -------------------------
    def _soft_update_target(self) -> None:
        tau = self.tau_high
        for param, target_param in zip(self.value_net.parameters(), self.target_value_net.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

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
        self.value_net.train()

        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        rewards_eff_t = torch.as_tensor(rewards_eff, dtype=torch.float32, device=self.device)
        safety_costs_t = torch.as_tensor(safety_costs, dtype=torch.float32, device=self.device)
        next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device)
        not_done = 1.0 - dones_t

        laser_dim = states_t.shape[1] - self.goal_feature_dim
        laser_t = states_t[:, :laser_dim]
        goal_t = states_t[:, laser_dim:]

        laser_next_t = next_states_t[:, :laser_dim]
        goal_next_t = next_states_t[:, laser_dim:]

        with torch.no_grad():
            self.target_value_net.eval()
            laser_next_np = (laser_next_t.cpu().numpy() * 7.0).astype(np.float32)
            goal_next_np = goal_next_t.cpu().numpy().astype(np.float32)

            norm_dist = goal_next_np[:, 0]
            cos_next = goal_next_np[:, 1]
            sin_next = goal_next_np[:, 2]
            goal_dist_next = norm_dist * 30.0

            q_eff_next_list = []
            q_safe_next_list = []

            for i in range(states_t.shape[0]):
                scan_next = laser_next_np[i]
                gd = float(goal_dist_next[i])
                gc = float(cos_next[i])
                gs = float(sin_next[i])

                candidates = self._generate_frontier_candidates(scan_next, gd, gc, gs)
                if not candidates:
                    q_eff_next_list.append(0.0)
                    q_safe_next_list.append(0.0)
                    continue

                subgoals = torch.tensor(candidates, dtype=torch.float32, device=self.device)
                laser_i = torch.tensor(scan_next / 7.0, dtype=torch.float32, device=self.device).unsqueeze(0)
                laser_i = laser_i.repeat(subgoals.size(0), 1)
                goal_i = torch.tensor(goal_next_np[i], dtype=torch.float32, device=self.device).unsqueeze(0)
                goal_i = goal_i.repeat(subgoals.size(0), 1)

                q_eff_cand, q_safe_cand = self.target_value_net(laser_i, goal_i, subgoals, return_heads=True)
                q_total_cand = q_eff_cand - self.safety_q_weight * q_safe_cand
                idx_best = torch.argmax(q_total_cand).item()

                q_eff_next_list.append(float(q_eff_cand[idx_best].item()))
                q_safe_next_list.append(float(q_safe_cand[idx_best].item()))

            q_eff_next = torch.tensor(q_eff_next_list, device=self.device, dtype=torch.float32)
            q_safe_next = torch.tensor(q_safe_next_list, device=self.device, dtype=torch.float32)

            target_eff = rewards_eff_t + self.gamma_high * not_done * q_eff_next
            target_safe = safety_costs_t + self.gamma_high * not_done * q_safe_next

        q_eff_pred, q_safe_pred = self.value_net(laser_t, goal_t, actions_t, return_heads=True)
        loss_eff = self.value_loss_fn(q_eff_pred, target_eff.detach())
        loss_safe = self.value_loss_fn(q_safe_pred, target_safe.detach())
        loss = loss_eff + self.safety_loss_weight * loss_safe

        self.value_optimizer.zero_grad()
        loss.backward()
        self.value_optimizer.step()

        self._soft_update_target()

        self.iter_count += 1

        return {
            "loss_eff": float(loss_eff.item()),
            "loss_safe": float(loss_safe.item()),
            "loss_total": float(loss.item()),
            "q_eff_mean": float(q_eff_pred.mean().item()),
            "q_safe_mean": float(q_safe_pred.mean().item()),
            "r_eff_mean": float(rewards_eff_t.mean().item()),
            "c_safe_mean": float(safety_costs_t.mean().item()),
        }

    # ------------------------- 模型存储 -------------------------
    def save_model(self, filename, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        torch.save(self.value_net.state_dict(), f"{directory}/{filename}.pth")
        print(f"模型已保存到 {directory}/{filename}.pth")

    def load_model(self, filename, directory):
        try:
            self.value_net.load_state_dict(torch.load(f"{directory}/{filename}.pth", map_location=self.device))
            if hasattr(self, "target_value_net"):
                self.target_value_net.load_state_dict(self.value_net.state_dict())
            print(f"模型已从 {directory}/{filename}.pth 加载")
        except FileNotFoundError as e:
            print(f"加载模型时出错: {e}")
