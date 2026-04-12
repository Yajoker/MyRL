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

        self.cnn1 = nn.Conv1d(1, 8, kernel_size=5, stride=2)
        self.cnn2 = nn.Conv1d(8, 16, kernel_size=3, stride=2)
        self.cnn3 = nn.Conv1d(16, 8, kernel_size=3, stride=1)

        self.goal_embed = nn.Linear(goal_info_dim, 64)
        self.subgoal_embed = nn.Linear(geom_dim, 16)

        cnn_out_dim = self._get_cnn_output_dim(belief_dim)
        concat_dim = int(cnn_out_dim + 64 + 16)

        self.fc1 = nn.Linear(concat_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q_head = nn.Linear(hidden_dim, 1)

    def _get_cnn_output_dim(self, belief_dim: int) -> int:
        dummy = torch.zeros(1, 1, belief_dim)
        x = self.cnn1(dummy)
        x = self.cnn2(x)
        x = self.cnn3(x)
        return int(x.view(1, -1).shape[1])

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
        q = self.q_head(h).squeeze(-1)
        return q
