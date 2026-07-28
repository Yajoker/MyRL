from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HighLevelValueNetScalar(nn.Module):
    """Scalar high-level value network Q(s, g) without value decomposition."""

    def __init__(
        self,
        *,
        belief_dim: int,
        goal_info_dim: int,
        geom_dim: int = 2,
        hidden_dim: int = 192,
    ) -> None:
        super().__init__()
        self.belief_dim = int(belief_dim)
        self.goal_info_dim = int(goal_info_dim)
        self.geom_dim = int(geom_dim)
        if self.belief_dim <= 0:
            raise ValueError("belief_dim (the number of beams) must be positive.")
        if self.goal_info_dim <= 0 or self.geom_dim <= 0:
            raise ValueError("goal_info_dim and geom_dim must be positive.")

        self.cnn1 = nn.Conv1d(2, 8, kernel_size=5, stride=2)
        self.cnn2 = nn.Conv1d(8, 16, kernel_size=3, stride=2)
        self.cnn3 = nn.Conv1d(16, 8, kernel_size=3, stride=1)

        self.goal_embed = nn.Linear(self.goal_info_dim, 64)
        self.subgoal_embed = nn.Linear(self.geom_dim, 16)

        cnn_out_dim = self._get_cnn_output_dim(self.belief_dim)
        concat_dim = int(cnn_out_dim + 64 + 16)

        self.fc1 = nn.Linear(concat_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q_head = nn.Linear(hidden_dim, 1)

    def _get_cnn_output_dim(self, belief_dim: int) -> int:
        try:
            dummy = torch.zeros(1, 2, belief_dim)
            x = self.cnn1(dummy)
            x = self.cnn2(x)
            x = self.cnn3(x)
        except RuntimeError as exc:
            raise ValueError(
                f"belief_dim={belief_dim} is too small for the scalar value CNN."
            ) from exc
        return int(x.view(1, -1).shape[1])

    def forward(self, laser: torch.Tensor, goal_info: torch.Tensor, subgoal_geom: torch.Tensor) -> torch.Tensor:
        if laser.dim() != 3 or tuple(laser.shape[1:]) != (2, self.belief_dim):
            raise ValueError(
                "laser must have shape [batch, 2, belief_dim], "
                f"expected [batch, 2, {self.belief_dim}], got {tuple(laser.shape)}."
            )
        if goal_info.dim() != 2 or goal_info.shape[1] != self.goal_info_dim:
            raise ValueError(
                f"goal_info must have shape [batch, {self.goal_info_dim}], "
                f"got {tuple(goal_info.shape)}."
            )
        if subgoal_geom.dim() != 2 or subgoal_geom.shape[1] != self.geom_dim:
            raise ValueError(
                f"subgoal_geom must have shape [batch, {self.geom_dim}], "
                f"got {tuple(subgoal_geom.shape)}."
            )
        if not (
            laser.shape[0] == goal_info.shape[0] == subgoal_geom.shape[0]
        ):
            raise ValueError(
                "laser, goal_info and subgoal_geom batch sizes must match."
            )

        x = laser
        x = F.relu(self.cnn1(x))
        x = F.relu(self.cnn2(x))
        x = F.relu(self.cnn3(x))
        x = x.view(x.size(0), -1)

        g = F.relu(self.goal_embed(goal_info))
        geom = F.relu(self.subgoal_embed(subgoal_geom))

        h = torch.cat([x, g, geom], dim=1)
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        q = self.q_head(h).squeeze(-1)
        return q
