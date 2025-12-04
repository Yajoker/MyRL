"""高层规划器：事件触发的前沿候选 + 统一价值网络 (E-FVQ)。"""

import math
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from config import PlannerConfig, TriggerConfig


class HighLevelValueNet(nn.Module):
    """统一高层值函数 Q_H(s, g)。"""

    def __init__(self, belief_dim: int = 90, goal_info_dim: int = 3, geom_dim: int = 2, hidden_dim: int = 192):
        super().__init__()

        self.cnn1 = nn.Conv1d(1, 8, kernel_size=5, stride=2)
        self.cnn2 = nn.Conv1d(8, 16, kernel_size=3, stride=2)
        self.cnn3 = nn.Conv1d(16, 8, kernel_size=3, stride=1)

        self.goal_embed = nn.Linear(goal_info_dim, 64)
        self.subgoal_embed = nn.Linear(geom_dim, 16)

        cnn_out_dim = self._get_cnn_output_dim(belief_dim)
        self.fc1 = nn.Linear(cnn_out_dim + 64 + 16, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.out = nn.Linear(hidden_dim // 2, 1)

    def _get_cnn_output_dim(self, belief_dim: int) -> int:
        dummy = torch.zeros(1, 1, belief_dim)
        x = self.cnn1(dummy)
        x = self.cnn2(x)
        x = self.cnn3(x)
        return x.view(1, -1).shape[1]

    def forward(self, laser: torch.Tensor, goal_info: torch.Tensor, subgoal_geom: torch.Tensor) -> torch.Tensor:
        x = laser.unsqueeze(1)
        x = F.relu(self.cnn1(x))
        x = F.relu(self.cnn2(x))
        x = F.relu(self.cnn3(x))
        x = x.view(x.size(0), -1)

        g = F.relu(self.goal_embed(goal_info))
        geom = F.relu(self.subgoal_embed(subgoal_geom))

        h = torch.cat([x, g, geom], dim=1)
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        return self.out(h).squeeze(-1)


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
        current_step: int = 0,
        window_metrics: Optional[dict] = None,
    ) -> bool:
        if not self.event_trigger.can_replan(current_step):
            return False

        goal_distance = float(goal_info[0]) if goal_info else float("inf")
        laser_scan = np.asarray(laser_scan, dtype=np.float32)

        risk_index, _, _ = self.compute_risk_index(laser_scan)
        self.event_trigger.update_progress(goal_distance, current_step)

        dist_to_subgoal, _ = self.get_relative_subgoal(robot_pose)

        time_event = self.event_trigger.time_upper_bound(current_step)
        risk_event = self.event_trigger.risk_trigger(risk_index)
        progress_event = self.event_trigger.global_progress_stagnant(goal_distance, current_step)
        subgoal_event = self.event_trigger.subgoal_reached(dist_to_subgoal)

        trigger_new_subgoal = time_event or risk_event or progress_event or subgoal_event
        if trigger_new_subgoal:
            self.event_trigger.reset_time(current_step)
        return trigger_new_subgoal

    # ------------------------- 子目标生成 -------------------------
    def _generate_frontier_candidates(self, laser_scan: np.ndarray, goal_distance: float, goal_cos: float, goal_sin: float) -> List[Tuple[float, float]]:
        scan = np.asarray(laser_scan, dtype=np.float32)
        scan = np.nan_to_num(scan, nan=0.0, posinf=self.frontier_max_distance, neginf=0.0)

        n = scan.shape[0]
        angles = np.linspace(-math.pi, math.pi, n, endpoint=False)

        frontier_dist = max(self.frontier_min_distance, 0.8 * self.frontier_max_distance)
        mask = scan >= frontier_dist

        candidates: List[Tuple[float, float]] = []
        i = 0
        while i < n:
            if not mask[i]:
                i += 1
                continue
            start = i
            while i + 1 < n and mask[i + 1]:
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
        self, laser_scan, goal_info: Tuple[float, float, float], candidates: List[Tuple[float, float]]
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
            q_vals = self.value_net(laser_batch, goal_batch, geom_t).cpu().numpy()

        if self.current_subgoal is not None:
            last_r, last_theta = self.current_subgoal
            lambda_cons = self.consistency_lambda
            sigma_r = self.consistency_sigma_r
            sigma_theta = self.consistency_sigma_theta
            bonuses = []
            for (r, theta) in candidates:
                dr = (r - last_r) / sigma_r
                dtheta = (theta - last_theta) / sigma_theta
                bonus = math.exp(-0.5 * (dr * dr + dtheta * dtheta))
                bonuses.append(lambda_cons * bonus)
            scores = q_vals + np.asarray(bonuses, dtype=np.float32)
        else:
            scores = q_vals

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
        final_distance, final_angle = self._select_best_subgoal(laser_scan, goal_info, candidates)

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
    def update_planner(self, states, actions, rewards, dones, next_states, batch_size: int = 64):
        self.value_net.train()

        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)

        laser_dim = states_t.shape[1] - self.goal_feature_dim
        laser_t = states_t[:, :laser_dim]
        goal_t = states_t[:, laser_dim:]

        q_pred = self.value_net(laser_t, goal_t, actions_t)
        loss = self.value_loss_fn(q_pred, rewards_t)

        self.value_optimizer.zero_grad()
        loss.backward()
        self.value_optimizer.step()

        self.iter_count += 1

        return {
            "planner_loss": float(loss.item()),
            "planner_q_mean": float(q_pred.mean().item()),
            "planner_reward_mean": float(rewards_t.mean().item()),
        }

    # ------------------------- 模型存储 -------------------------
    def save_model(self, filename, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        torch.save(self.value_net.state_dict(), f"{directory}/{filename}.pth")
        print(f"模型已保存到 {directory}/{filename}.pth")

    def load_model(self, filename, directory):
        try:
            self.value_net.load_state_dict(torch.load(f"{directory}/{filename}.pth", map_location=self.device))
            print(f"模型已从 {directory}/{filename}.pth 加载")
        except FileNotFoundError as e:
            print(f"加载模型时出错: {e}")
