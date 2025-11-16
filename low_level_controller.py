from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


def _split_state(state: torch.Tensor, laser_dim: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split a batched state tensor into (laser, subgoal, prev_action)."""

    laser = state[..., :laser_dim]
    subgoal = state[..., laser_dim : laser_dim + 2]
    prev_action = state[..., laser_dim + 2 : laser_dim + 4]
    return laser, subgoal, prev_action


class LaserEncoder(nn.Module):
    """Light-weight 1-D CNN encoder for laser ranges."""

    def __init__(self, laser_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, laser_dim)
            conv_out = self.conv(dummy).view(1, -1)
        self.linear = nn.Sequential(
            nn.Linear(conv_out.shape[1], hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, laser: torch.Tensor) -> torch.Tensor:
        x = laser.unsqueeze(1)  # [B, 1, laser_dim]
        x = self.conv(x)
        x = x.flatten(start_dim=1)
        return self.linear(x)


class FeatureEmbedding(nn.Module):
    """Generic MLP embedding."""

    def __init__(self, in_dim: int, hidden_dim: int = 64, out_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RecurrentActor(nn.Module):
    """TD3 actor with CNN + embeddings feeding an LSTM."""

    def __init__(self, laser_dim: int, action_dim: int, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.laser_dim = laser_dim
        self.action_dim = action_dim
        self.laser_encoder = LaserEncoder(laser_dim, hidden_dim)
        self.subgoal_embed = FeatureEmbedding(2, hidden_dim, hidden_dim)
        self.prev_action_embed = FeatureEmbedding(2, hidden_dim, hidden_dim)
        feature_dim = hidden_dim * 3
        self.pre_fc = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def _encode(self, state: torch.Tensor) -> torch.Tensor:
        laser, subgoal, prev_action = _split_state(state, self.laser_dim)
        orig_shape = laser.shape
        flat = laser.reshape(-1, self.laser_dim)
        laser_feat = self.laser_encoder(flat)
        subgoal_feat = self.subgoal_embed(subgoal.reshape(-1, 2))
        prev_feat = self.prev_action_embed(prev_action.reshape(-1, 2))
        features = torch.cat((laser_feat, subgoal_feat, prev_feat), dim=-1)
        features = self.pre_fc(features)
        return features.view(orig_shape[0], orig_shape[1], -1)

    def forward(
        self, state_seq: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        encoded = self._encode(state_seq)
        outputs, new_hidden = self.lstm(encoded, hidden)
        outputs = self.dropout(outputs)
        actions = self.head(outputs)
        return actions, new_hidden

    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(1, batch_size, self.lstm.hidden_size, device=device)
        c = torch.zeros(1, batch_size, self.lstm.hidden_size, device=device)
        return h, c


class CriticNetwork(nn.Module):
    """Feed-forward critic that reuses the CNN embeddings."""

    def __init__(self, laser_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.laser_dim = laser_dim
        self.laser_encoder = LaserEncoder(laser_dim, hidden_dim)
        self.subgoal_embed = FeatureEmbedding(2, hidden_dim, hidden_dim)
        self.prev_action_embed = FeatureEmbedding(2, hidden_dim, hidden_dim)
        feat_dim = hidden_dim * 3 + action_dim
        self.q_net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        laser, subgoal, prev_action = _split_state(states, self.laser_dim)
        batch_shape = laser.shape
        flat = laser.reshape(-1, self.laser_dim)
        laser_feat = self.laser_encoder(flat)
        subgoal_feat = self.subgoal_embed(subgoal.reshape(-1, 2))
        prev_feat = self.prev_action_embed(prev_action.reshape(-1, 2))
        feats = torch.cat((laser_feat, subgoal_feat, prev_feat, actions.reshape(-1, actions.shape[-1])), dim=-1)
        q_val = self.q_net(feats)
        return q_val.view(batch_shape[0], batch_shape[1], 1)


class TwinCritic(nn.Module):
    def __init__(self, laser_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.q1 = CriticNetwork(laser_dim, action_dim, hidden_dim)
        self.q2 = CriticNetwork(laser_dim, action_dim, hidden_dim)

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.q1(states, actions), self.q2(states, actions)

    def q1_value(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.q1(states, actions)


class LowLevelController:
    """CNNTD3 controller with recurrent actor."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        device: Optional[torch.device] = None,
        save_directory: Optional[Path] = None,
        model_name: str = "low_level_controller",
        load_model: bool = False,
        max_lin_velocity: float = 0.5,
        max_ang_velocity: float = 1.0,
        sequence_length: int = 15,
        hidden_dim: int = 128,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-3,
        l2_lambda: float = 1e-5,
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.laser_dim = state_dim - 4
        self.max_action = max_action
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_directory = Path(save_directory or "models")
        self.save_directory.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.sequence_length = sequence_length
        self.l2_lambda = l2_lambda
        self.max_lin_velocity = max_lin_velocity
        self.max_ang_velocity = max_ang_velocity

        self.actor = RecurrentActor(self.laser_dim, action_dim, hidden_dim=hidden_dim).to(self.device)
        self.actor_target = RecurrentActor(self.laser_dim, action_dim, hidden_dim=hidden_dim).to(self.device)
        self.critic = TwinCritic(self.laser_dim, action_dim, hidden_dim=hidden_dim).to(self.device)
        self.critic_target = TwinCritic(self.laser_dim, action_dim, hidden_dim=hidden_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        run_name = f"runs/{model_name}_{int(time.time())}"
        self.writer = SummaryWriter(log_dir=self.save_directory / run_name)

        self.actor_hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self.total_it = 0

        if load_model:
            self.load_model(model_name, self.save_directory)

    # ------------------------------------------------------------------
    # Observation processing & inference
    # ------------------------------------------------------------------
    def process_observation(
        self,
        laser_scan: Sequence[float],
        subgoal_distance: float,
        subgoal_angle: float,
        prev_action: Optional[Sequence[float]],
    ) -> np.ndarray:
        laser = np.asarray(laser_scan, dtype=np.float32)
        if not np.all(np.isfinite(laser)):
            finite = laser[np.isfinite(laser)]
            if finite.size:
                laser = np.nan_to_num(laser, nan=float(finite.mean()), posinf=8.0, neginf=0.0)
            else:
                laser = np.zeros_like(laser, dtype=np.float32)
        subgoal = np.array([float(subgoal_distance or 0.0), float(subgoal_angle or 0.0)], dtype=np.float32)
        if prev_action is None:
            prev = np.zeros(2, dtype=np.float32)
        else:
            prev = np.asarray(prev_action, dtype=np.float32)
            if prev.shape != (2,):
                prev = np.zeros(2, dtype=np.float32)
        state = np.concatenate((laser, subgoal, prev), dtype=np.float32)
        return state

    def reset_hidden_state(self) -> None:
        self.actor_hidden = None

    def predict_action(self, state: np.ndarray, add_noise: bool = False, noise_scale: float = 0.1) -> np.ndarray:
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).view(1, 1, -1)
        if self.actor_hidden is None:
            self.actor_hidden = self.actor.init_hidden(batch_size=1, device=self.device)
        action_tensor, self.actor_hidden = self.actor(state_tensor, self.actor_hidden)
        action = action_tensor.squeeze(0).squeeze(0)
        if add_noise and noise_scale > 0:
            action = action + noise_scale * torch.randn_like(action)
        action = torch.clamp(action, -1.0, 1.0)
        return action.detach().cpu().numpy()

    # ------------------------------------------------------------------
    # Training logic
    # ------------------------------------------------------------------
    def update(
        self,
        replay_buffer,
        batch_size: int,
        discount: float,
        tau: float,
        policy_noise: float,
        noise_clip: float,
        policy_freq: int,
        sequence_length: Optional[int] = None,
    ) -> Optional[dict]:
        seq_len = sequence_length or self.sequence_length
        try:
            (
                states,
                actions,
                rewards,
                dones,
                next_states,
                mask,
            ) = replay_buffer.sample_sequences(batch_size, seq_len)
        except ValueError:
            return None

        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(-1)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(-1)
        next_states = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        mask_tensor = torch.as_tensor(mask, dtype=torch.float32, device=self.device).unsqueeze(-1)
        valid_steps = torch.clamp(mask_tensor.sum(), min=1.0)

        with torch.no_grad():
            target_actions, _ = self.actor_target(next_states, hidden=None)
            if policy_noise > 0:
                noise = torch.randn_like(target_actions) * policy_noise
                noise = torch.clamp(noise, -noise_clip, noise_clip)
                target_actions = torch.clamp(target_actions + noise, -1.0, 1.0)
            target_q1, target_q2 = self.critic_target(next_states, target_actions)
            target_q = torch.min(target_q1, target_q2)
            target = rewards + (1.0 - dones) * discount * target_q

        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = ((mask_tensor * (current_q1 - target) ** 2).sum() + (mask_tensor * (current_q2 - target) ** 2).sum()) / (
            2.0 * valid_steps
        )

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=5.0)
        self.critic_optimizer.step()

        actor_loss_value = torch.tensor(0.0, device=self.device)
        self.total_it += 1
        if self.total_it % policy_freq == 0:
            pi_actions, _ = self.actor(states, hidden=None)
            q1_pi = self.critic.q1_value(states, pi_actions)
            actor_loss_value = -(mask_tensor * q1_pi).sum() / valid_steps
            if self.l2_lambda > 0:
                reg = torch.tensor(0.0, device=self.device)
                for param in self.actor.lstm.parameters():
                    reg = reg + torch.sum(param ** 2)
                actor_loss_value = actor_loss_value + self.l2_lambda * reg
            self.actor_optimizer.zero_grad()
            actor_loss_value.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=5.0)
            self.actor_optimizer.step()

            self._soft_update(self.actor_target, self.actor, tau)
            self._soft_update(self.critic_target, self.critic, tau)

        metrics = {
            "critic_loss": float(critic_loss.detach().cpu().item()),
            "actor_loss": float(actor_loss_value.detach().cpu().item()),
        }
        self.writer.add_scalar("train/critic_loss", metrics["critic_loss"], self.total_it)
        self.writer.add_scalar("train/actor_loss", metrics["actor_loss"], self.total_it)
        return metrics

    @staticmethod
    def _soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
        for tgt_param, src_param in zip(target.parameters(), source.parameters()):
            tgt_param.data.copy_(tgt_param.data * (1.0 - tau) + src_param.data * tau)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def save_model(self, filename: str, directory: Path | str) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{filename}.pth"
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "actor_opt": self.actor_optimizer.state_dict(),
                "critic_opt": self.critic_optimizer.state_dict(),
            },
            path,
        )

    def load_model(self, filename: str, directory: Path | str) -> None:
        path = Path(directory) / f"{filename}.pth"
        if not path.exists():
            return
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint.get("actor", {}))
        self.critic.load_state_dict(checkpoint.get("critic", {}))
        if "actor_target" in checkpoint:
            self.actor_target.load_state_dict(checkpoint["actor_target"])
        if "critic_target" in checkpoint:
            self.critic_target.load_state_dict(checkpoint["critic_target"])
        if "actor_opt" in checkpoint:
            self.actor_optimizer.load_state_dict(checkpoint["actor_opt"])
        if "critic_opt" in checkpoint:
            self.critic_optimizer.load_state_dict(checkpoint["critic_opt"])

