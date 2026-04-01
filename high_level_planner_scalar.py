from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from config import PlannerConfig, TriggerConfig
from ablation_system import HighLevelPlannerPeriodic
from high_level_planner import HighLevelPlanner
from high_level_value_scalar import HighLevelValueNetScalar


class HighLevelPlannerScalar(HighLevelPlanner):
    """A2 ablation: scalar value function Q(s, g) trained on r_eff - lambda * c_safe.

    Keeps the same public interfaces as `HighLevelPlanner`:
    - `check_triggers / should_replan`
    - `generate_subgoal` (inherits)
    - `_select_best_subgoal` (overridden)
    - `update_planner` (overridden)

    Note:
        This planner disables the internal high-level Double-Q option for simplicity
        of the ablation.
    """

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
        scalar_lambda: float = 1.0,
    ) -> None:
        planner_cfg = planner_config or PlannerConfig()
        if getattr(planner_cfg, "high_level_double_q_enabled", False):
            # enforce disable, to avoid mixing changes in the ablation
            planner_cfg = PlannerConfig(**{**planner_cfg.__dict__, "high_level_double_q_enabled": False})

        super().__init__(
            belief_dim=belief_dim,
            device=device,
            save_directory=save_directory,
            model_name=model_name,
            load_model=False,  # defer loading until scalar nets are installed
            load_directory=load_directory,
            step_duration=step_duration,
            min_interval=min_interval,
            subgoal_reach_threshold=subgoal_reach_threshold,
            waypoint_lookahead=waypoint_lookahead,
            trigger_config=trigger_config,
            planner_config=planner_cfg,
        )

        self.scalar_lambda = float(scalar_lambda)

        # Replace dual-head nets with scalar nets.
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
        self.target_value_net.load_state_dict(self.value_net.state_dict())
        for p in self.target_value_net.parameters():
            p.requires_grad = False

        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=1e-3)
        self.value_loss_fn = nn.MSELoss()

        if load_model:
            load_dir = load_directory if load_directory else save_directory
            self.load_model(filename=model_name, directory=load_dir)

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
            return float(r), float(goal_dir)

        self.value_net.eval()

        scan = np.asarray(laser_scan, dtype=np.float32)
        scan = np.nan_to_num(scan, nan=self.frontier_max_distance, posinf=self.frontier_max_distance, neginf=0.0)
        scan = np.clip(scan, 0.0, self.frontier_max_distance)

        laser_t = torch.as_tensor(scan[None, :], dtype=torch.float32, device=self.device)
        dummy_waypoints = self.build_waypoint_features(waypoints=None, robot_pose=None)
        goal_t_single = self.process_goal_info(goal_info[0], goal_info[1], goal_info[2], dummy_waypoints)
        goal_t = goal_t_single.unsqueeze(0)
        geom_t = torch.as_tensor(np.asarray(candidates, dtype=np.float32), dtype=torch.float32, device=self.device)

        laser_batch = laser_t.repeat(geom_t.shape[0], 1)
        goal_batch = goal_t.repeat(geom_t.shape[0], 1)

        with torch.no_grad():
            q_vals = self.value_net(laser_batch, goal_batch, geom_t).cpu().numpy()

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

        self.value_net.train()
        r_scalar = rewards_eff_t - self.scalar_lambda * safety_costs_t

        with torch.no_grad():
            self.target_value_net.eval()

            laser_next_np = (laser_next_t.cpu().numpy() * 9.0).astype(np.float32)
            goal_next_np = goal_next_t.cpu().numpy().astype(np.float32)

            norm_dist = goal_next_np[:, 0]
            cos_next = goal_next_np[:, 1]
            sin_next = goal_next_np[:, 2]
            goal_dist_next = norm_dist * 30.0

            q_next_list: List[float] = []
            for i in range(states_t.shape[0]):
                scan_next = laser_next_np[i]
                gd = float(goal_dist_next[i])
                gc = float(cos_next[i])
                gs = float(sin_next[i])

                candidates = self._generate_frontier_candidates(scan_next, gd, gc, gs)
                if not candidates:
                    q_next_list.append(0.0)
                    continue

                subgoals = torch.tensor(candidates, dtype=torch.float32, device=self.device)
                laser_i = torch.tensor(scan_next / 9.0, dtype=torch.float32, device=self.device).unsqueeze(0)
                laser_i = laser_i.repeat(subgoals.size(0), 1)
                goal_i = torch.tensor(goal_next_np[i], dtype=torch.float32, device=self.device).unsqueeze(0)
                goal_i = goal_i.repeat(subgoals.size(0), 1)

                q_vals = self.target_value_net(laser_i, goal_i, subgoals)
                q_next_list.append(float(torch.max(q_vals).item()))

            q_next = torch.tensor(q_next_list, device=self.device, dtype=torch.float32)
            target = r_scalar + self.gamma_high * not_done * q_next

        q_pred = self.value_net(laser_t, goal_t, actions_t)
        loss = self.value_loss_fn(q_pred, target.detach())

        self.value_optimizer.zero_grad()
        loss.backward()
        self.value_optimizer.step()

        self._soft_update_target(self.value_net, self.target_value_net)
        self.iter_count += 1

        return {
            "loss_total": float(loss.item()),
            "q_mean": float(q_pred.mean().item()),
            "r_eff_mean": float(rewards_eff_t.mean().item()),
            "c_safe_mean": float(safety_costs_t.mean().item()),
            "r_scalar_mean": float(r_scalar.mean().item()),
            "q_next_mean": float(q_next.mean().item()),
        }

    def save_model(self, filename, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        torch.save(self.value_net.state_dict(), f"{directory}/{filename}.pth")
        print(f"模型已保存到 {directory}/{filename}.pth")

    def load_model(self, filename, directory):
        path = Path(directory) / f"{filename}.pth"
        try:
            state_dict = torch.load(path, map_location=self.device)
            self.value_net.load_state_dict(state_dict)
            self.target_value_net.load_state_dict(self.value_net.state_dict())
            print(f"模型已从 {path} 加载")
        except FileNotFoundError as e:
            print(f"加载模型时出错: {e}")
        except RuntimeError as e:
            print(f"加载模型参数不匹配: {e}")


class HighLevelPlannerScalarPeriodic(HighLevelPlannerPeriodic):
    """A1+A2: periodic replanning with scalar Q(s,g) training/selection."""

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
        scalar_lambda: float = 1.0,
        replan_k: int = 10,
        allow_subgoal_immediate: bool = True,
    ) -> None:
        planner_cfg = planner_config or PlannerConfig()
        if getattr(planner_cfg, "high_level_double_q_enabled", False):
            planner_cfg = PlannerConfig(**{**planner_cfg.__dict__, "high_level_double_q_enabled": False})

        super().__init__(
            belief_dim=belief_dim,
            device=device,
            save_directory=save_directory,
            model_name=model_name,
            load_model=False,
            load_directory=load_directory,
            step_duration=step_duration,
            min_interval=min_interval,
            subgoal_reach_threshold=subgoal_reach_threshold,
            waypoint_lookahead=waypoint_lookahead,
            trigger_config=trigger_config,
            planner_config=planner_cfg,
            replan_k=replan_k,
            allow_subgoal_immediate=allow_subgoal_immediate,
        )

        self.scalar_lambda = float(scalar_lambda)
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
        self.target_value_net.load_state_dict(self.value_net.state_dict())
        for p in self.target_value_net.parameters():
            p.requires_grad = False

        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=1e-3)
        self.value_loss_fn = nn.MSELoss()

        if load_model:
            load_dir = load_directory if load_directory else save_directory
            self.load_model(filename=model_name, directory=load_dir)

    def _select_best_subgoal(
        self,
        laser_scan,
        goal_info: Tuple[float, float, float],
        candidates: List[Tuple[float, float]],
        robot_pose: Optional[Sequence[float]] = None,
    ) -> Tuple[float, float]:
        # same as scalar variant
        if not candidates:
            goal_distance, goal_cos, goal_sin = goal_info
            goal_dir = math.atan2(goal_sin, goal_cos)
            r = max(self.frontier_min_distance, min(goal_distance, self.frontier_max_distance))
            return float(r), float(goal_dir)

        self.value_net.eval()

        scan = np.asarray(laser_scan, dtype=np.float32)
        scan = np.nan_to_num(scan, nan=self.frontier_max_distance, posinf=self.frontier_max_distance, neginf=0.0)
        scan = np.clip(scan, 0.0, self.frontier_max_distance)

        laser_t = torch.as_tensor(scan[None, :], dtype=torch.float32, device=self.device)
        dummy_waypoints = self.build_waypoint_features(waypoints=None, robot_pose=None)
        goal_t_single = self.process_goal_info(goal_info[0], goal_info[1], goal_info[2], dummy_waypoints)
        goal_t = goal_t_single.unsqueeze(0)
        geom_t = torch.as_tensor(np.asarray(candidates, dtype=np.float32), dtype=torch.float32, device=self.device)

        laser_batch = laser_t.repeat(geom_t.shape[0], 1)
        goal_batch = goal_t.repeat(geom_t.shape[0], 1)

        with torch.no_grad():
            q_vals = self.value_net(laser_batch, goal_batch, geom_t).cpu().numpy()

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

        self.value_net.train()
        r_scalar = rewards_eff_t - self.scalar_lambda * safety_costs_t

        with torch.no_grad():
            self.target_value_net.eval()

            laser_next_np = (laser_next_t.cpu().numpy() * 9.0).astype(np.float32)
            goal_next_np = goal_next_t.cpu().numpy().astype(np.float32)
            norm_dist = goal_next_np[:, 0]
            cos_next = goal_next_np[:, 1]
            sin_next = goal_next_np[:, 2]
            goal_dist_next = norm_dist * 30.0

            q_next_list: List[float] = []
            for i in range(states_t.shape[0]):
                scan_next = laser_next_np[i]
                gd = float(goal_dist_next[i])
                gc = float(cos_next[i])
                gs = float(sin_next[i])

                candidates = self._generate_frontier_candidates(scan_next, gd, gc, gs)
                if not candidates:
                    q_next_list.append(0.0)
                    continue

                subgoals = torch.tensor(candidates, dtype=torch.float32, device=self.device)
                laser_i = torch.tensor(scan_next / 9.0, dtype=torch.float32, device=self.device).unsqueeze(0)
                laser_i = laser_i.repeat(subgoals.size(0), 1)
                goal_i = torch.tensor(goal_next_np[i], dtype=torch.float32, device=self.device).unsqueeze(0)
                goal_i = goal_i.repeat(subgoals.size(0), 1)

                q_vals = self.target_value_net(laser_i, goal_i, subgoals)
                q_next_list.append(float(torch.max(q_vals).item()))

            q_next = torch.tensor(q_next_list, device=self.device, dtype=torch.float32)
            target = r_scalar + self.gamma_high * not_done * q_next

        q_pred = self.value_net(laser_t, goal_t, actions_t)
        loss = self.value_loss_fn(q_pred, target.detach())

        self.value_optimizer.zero_grad()
        loss.backward()
        self.value_optimizer.step()
        self._soft_update_target(self.value_net, self.target_value_net)

        self.iter_count += 1
        return {
            "loss_total": float(loss.item()),
            "q_mean": float(q_pred.mean().item()),
            "r_eff_mean": float(rewards_eff_t.mean().item()),
            "c_safe_mean": float(safety_costs_t.mean().item()),
            "r_scalar_mean": float(r_scalar.mean().item()),
            "q_next_mean": float(q_next.mean().item()),
        }

    def save_model(self, filename, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        torch.save(self.value_net.state_dict(), f"{directory}/{filename}.pth")
        print(f"模型已保存到 {directory}/{filename}.pth")

    def load_model(self, filename, directory):
        path = Path(directory) / f"{filename}.pth"
        try:
            state_dict = torch.load(path, map_location=self.device)
            self.value_net.load_state_dict(state_dict)
            self.target_value_net.load_state_dict(self.value_net.state_dict())
            print(f"模型已从 {path} 加载")
        except FileNotFoundError as e:
            print(f"加载模型时出错: {e}")
        except RuntimeError as e:
            print(f"加载模型参数不匹配: {e}")

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
            return float(r), float(goal_dir)

        self.value_net.eval()

        scan = np.asarray(laser_scan, dtype=np.float32)
        scan = np.nan_to_num(scan, nan=self.frontier_max_distance, posinf=self.frontier_max_distance, neginf=0.0)
        scan = np.clip(scan, 0.0, self.frontier_max_distance)

        laser_t = torch.as_tensor(scan[None, :], dtype=torch.float32, device=self.device)
        dummy_waypoints = self.build_waypoint_features(waypoints=None, robot_pose=None)
        goal_t_single = self.process_goal_info(goal_info[0], goal_info[1], goal_info[2], dummy_waypoints)
        goal_t = goal_t_single.unsqueeze(0)
        geom_t = torch.as_tensor(np.asarray(candidates, dtype=np.float32), dtype=torch.float32, device=self.device)

        laser_batch = laser_t.repeat(geom_t.shape[0], 1)
        goal_batch = goal_t.repeat(geom_t.shape[0], 1)

        with torch.no_grad():
            q_vals = self.value_net(laser_batch, goal_batch, geom_t).cpu().numpy()

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

        self.value_net.train()

        # Scalar reward for TD target
        r_scalar = rewards_eff_t - self.scalar_lambda * safety_costs_t

        with torch.no_grad():
            self.target_value_net.eval()

            laser_next_np = (laser_next_t.cpu().numpy() * 9.0).astype(np.float32)
            goal_next_np = goal_next_t.cpu().numpy().astype(np.float32)

            norm_dist = goal_next_np[:, 0]
            cos_next = goal_next_np[:, 1]
            sin_next = goal_next_np[:, 2]
            goal_dist_next = norm_dist * 30.0

            q_next_list: List[float] = []

            for i in range(states_t.shape[0]):
                scan_next = laser_next_np[i]
                gd = float(goal_dist_next[i])
                gc = float(cos_next[i])
                gs = float(sin_next[i])

                candidates = self._generate_frontier_candidates(scan_next, gd, gc, gs)
                if not candidates:
                    q_next_list.append(0.0)
                    continue

                subgoals = torch.tensor(candidates, dtype=torch.float32, device=self.device)

                laser_i = torch.tensor(scan_next / 9.0, dtype=torch.float32, device=self.device).unsqueeze(0)
                laser_i = laser_i.repeat(subgoals.size(0), 1)

                goal_i = torch.tensor(goal_next_np[i], dtype=torch.float32, device=self.device).unsqueeze(0)
                goal_i = goal_i.repeat(subgoals.size(0), 1)

                q_vals = self.target_value_net(laser_i, goal_i, subgoals)
                q_next_list.append(float(torch.max(q_vals).item()))

            q_next = torch.tensor(q_next_list, device=self.device, dtype=torch.float32)
            target = r_scalar + self.gamma_high * not_done * q_next

        q_pred = self.value_net(laser_t, goal_t, actions_t)
        loss = self.value_loss_fn(q_pred, target.detach())

        self.value_optimizer.zero_grad()
        loss.backward()
        self.value_optimizer.step()

        self._soft_update_target(self.value_net, self.target_value_net)

        self.iter_count += 1

        metrics = {
            "loss_total": float(loss.item()),
            "q_mean": float(q_pred.mean().item()),
            "r_eff_mean": float(rewards_eff_t.mean().item()),
            "c_safe_mean": float(safety_costs_t.mean().item()),
            "r_scalar_mean": float(r_scalar.mean().item()),
            "q_next_mean": float(q_next.mean().item()),
        }
        return metrics

    def save_model(self, filename, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        torch.save(self.value_net.state_dict(), f"{directory}/{filename}.pth")
        print(f"模型已保存到 {directory}/{filename}.pth")

    def load_model(self, filename, directory):
        path = Path(directory) / f"{filename}.pth"
        try:
            state_dict = torch.load(path, map_location=self.device)
            self.value_net.load_state_dict(state_dict)
            self.target_value_net.load_state_dict(self.value_net.state_dict())
            print(f"模型已从 {path} 加载")
        except FileNotFoundError as e:
            print(f"加载模型时出错: {e}")
        except RuntimeError as e:
            print(f"加载模型参数不匹配: {e}")
