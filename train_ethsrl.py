"""
Training entry point for the ETHSRL+GP hierarchical navigation system.

The script follows the structure of the original ``robot_nav/rl_train.py`` while
integrating the newly implemented high-level planner and low-level controller.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch

from ethsrl.core.integration import HierarchicalNavigationSystem
from robot_nav.SIM_ENV.sim import SIM
from robot_nav.replay_buffer import ReplayBuffer


@dataclass
class TrainingConfig:
    """Container for configurable training hyper-parameters."""

    buffer_size: int = 100_000
    batch_size: int = 64
    max_epochs: int = 40
    episodes_per_epoch: int = 60
    max_steps: int = 300
    train_every_n_episodes: int = 2
    training_iterations: int = 80
    exploration_noise: float = 0.2
    min_buffer_size: int = 1_000
    max_lin_velocity: float = 0.5
    max_ang_velocity: float = 1.0
    eval_episodes: int = 6


class TD3ReplayAdapter:
    """Thin wrapper matching the replay-buffer API expected by the controller."""

    def __init__(self, buffer_size: int, random_seed: int = 666) -> None:
        self._buffer = ReplayBuffer(buffer_size=buffer_size, random_seed=random_seed)

    def add(self, state, action, reward, done, next_state) -> None:
        self._buffer.add(state, action, reward, done, next_state)

    def size(self) -> int:
        return self._buffer.size()

    def sample(self, batch_size: int):
        states, actions, rewards, dones, next_states = self._buffer.sample_batch(batch_size)
        return states, actions, rewards, next_states, dones

    def clear(self) -> None:
        self._buffer.clear()


def get_robot_pose(sim: SIM) -> Tuple[float, float, float]:
    """Extract the robot pose from the IR-Sim wrapper and return (x, y, theta)."""

    robot_state = sim.env.get_robot_state()
    return (
        float(robot_state[0].item()),
        float(robot_state[1].item()),
        float(robot_state[2].item()),
    )


def evaluate(system: HierarchicalNavigationSystem, sim: SIM, config: TrainingConfig, epoch: int) -> None:
    """Run evaluation rollouts without exploration noise and log summary statistics."""

    print("-" * 46)
    print(f"Epoch {epoch}: running evaluation trajectories")

    avg_reward = 0.0
    collisions = 0
    successes = 0

    for _ in range(config.eval_episodes):
        system.reset()
        latest_scan, distance, cos, sin, collision, goal, prev_action, reward = sim.reset()
        prev_action = [0.0, 0.0]
        done = False
        step = 0

        while not done and step < config.max_steps:
            robot_pose = get_robot_pose(sim)
            goal_info = [distance, cos, sin]

            # Trigger subgoal update if required
            if system.current_subgoal is None or system.high_level_planner.check_triggers(
                latest_scan, robot_pose, goal_info
            ):
                subgoal_distance, subgoal_angle = system.high_level_planner.generate_subgoal(
                    latest_scan, distance, cos, sin
                )
            else:
                subgoal_distance, subgoal_angle = system.current_subgoal

            state = system.low_level_controller.process_observation(
                latest_scan, subgoal_distance, subgoal_angle, prev_action
            )

            action = system.low_level_controller.predict_action(state, add_noise=False)
            lin_cmd = float(np.clip((action[0] + 1.0) / 4.0, 0.0, config.max_lin_velocity))
            ang_cmd = float(np.clip(action[1], -config.max_ang_velocity, config.max_ang_velocity))

            latest_scan, distance, cos, sin, collision, goal, _, reward = sim.step(
                lin_velocity=lin_cmd, ang_velocity=ang_cmd
            )

            prev_action = [lin_cmd, ang_cmd]
            avg_reward += reward
            step += 1

            if collision:
                collisions += 1
                done = True
            if goal:
                successes += 1
                done = True

    avg_reward /= max(config.eval_episodes, 1)
    avg_collision_rate = collisions / max(config.eval_episodes, 1)
    avg_success_rate = successes / max(config.eval_episodes, 1)

    print(f"Average reward   : {avg_reward:.2f}")
    print(f"Collision rate   : {avg_collision_rate:.2f}")
    print(f"Goal reach rate  : {avg_success_rate:.2f}")
    print("-" * 46)

    writer = system.low_level_controller.writer
    writer.add_scalar("eval/avg_reward", avg_reward, epoch)
    writer.add_scalar("eval/collision_rate", avg_collision_rate, epoch)
    writer.add_scalar("eval/goal_rate", avg_success_rate, epoch)


def main(args=None):
    """Main training loop for ETHSRL+GP."""

    # Device and system initialisation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = TrainingConfig()

    system = HierarchicalNavigationSystem(device=device)
    replay_buffer = TD3ReplayAdapter(buffer_size=config.buffer_size)

    sim = SIM(world_file="worlds/env_a.yaml", disable_plotting=False)

    episode = 0
    epoch = 0

    while epoch < config.max_epochs:
        system.reset()
        latest_scan, distance, cos, sin, collision, goal, prev_action, reward = sim.reset()
        prev_action = [0.0, 0.0]

        step = 0
        episode_reward = 0.0
        done = False

        while not done and step < config.max_steps:
            robot_pose = get_robot_pose(sim)
            goal_info = [distance, cos, sin]

            # Determine whether a new subgoal is required
            if system.current_subgoal is None or system.high_level_planner.check_triggers(
                latest_scan, robot_pose, goal_info
            ):
                subgoal_distance, subgoal_angle = system.high_level_planner.generate_subgoal(
                    latest_scan, distance, cos, sin
                )
            else:
                subgoal_distance, subgoal_angle = system.current_subgoal

            state = system.low_level_controller.process_observation(
                latest_scan, subgoal_distance, subgoal_angle, prev_action
            )

            action = system.low_level_controller.predict_action(
                state, add_noise=True, noise_scale=config.exploration_noise
            )
            action = np.clip(action, -1.0, 1.0)

            lin_cmd = float(np.clip((action[0] + 1.0) / 4.0, 0.0, config.max_lin_velocity))
            ang_cmd = float(np.clip(action[1], -config.max_ang_velocity, config.max_ang_velocity))

            latest_scan, distance, cos, sin, collision, goal, executed_action, reward = sim.step(
                lin_velocity=lin_cmd, ang_velocity=ang_cmd
            )

            episode_reward += reward

            next_prev_action = [executed_action[0], executed_action[1]]
            next_state = system.low_level_controller.process_observation(
                latest_scan,
                system.current_subgoal[0] if system.current_subgoal else subgoal_distance,
                system.current_subgoal[1] if system.current_subgoal else subgoal_angle,
                next_prev_action,
            )

            done = collision or goal or step == config.max_steps - 1
            replay_buffer.add(state, action, reward, float(done), next_state)

            prev_action = next_prev_action
            step += 1

        episode += 1

        writer = system.low_level_controller.writer
        writer.add_scalar("train/episode_reward", episode_reward, episode)

        if (
            replay_buffer.size() >= config.min_buffer_size
            and episode % config.train_every_n_episodes == 0
        ):
            for _ in range(config.training_iterations):
                system.low_level_controller.update(
                    replay_buffer,
                    batch_size=config.batch_size,
                    discount=0.99,
                    tau=0.005,
                    policy_noise=0.2,
                    noise_clip=0.5,
                    policy_freq=2,
                )

        if episode % config.episodes_per_epoch == 0:
            epoch += 1
            evaluate(system, sim, config, epoch)


if __name__ == "__main__":
    main()
