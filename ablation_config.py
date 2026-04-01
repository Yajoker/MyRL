from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AblationConfig:
    """Ablation switches for controlled experiments.

    Notes:
        Defaults correspond to the current baseline:
        - event-triggered replanning
        - dual-head value decomposition
    """

    # A1: replanning mode
    replan_mode: str = "event"  # "event" | "periodic"
    replan_k: int = 10  # only used when periodic
    allow_subgoal_immediate: bool = True

    # A2: value mode
    value_mode: str = "dual"  # "dual" | "scalar"
    scalar_lambda: float = 1.0  # Q-target uses r_eff - lambda * c_safe

    def exp_tag(self) -> str:
        parts: list[str] = []
        if self.replan_mode == "periodic":
            parts.append(f"replan-periodicK{int(self.replan_k)}")
            if self.allow_subgoal_immediate:
                parts.append("subgoal-immediate")
        else:
            parts.append("replan-event")
        parts.append(f"value-{self.value_mode}")
        return "_".join(parts)

    @staticmethod
    def _getenv(name: str, default: str) -> str:
        val = os.getenv(name)
        return default if val is None or val == "" else str(val)

    @classmethod
    def from_env(cls) -> "AblationConfig":
        """Create config from environment variables.

        Supported env vars:
            - MYRL_REPLAN_MODE: event|periodic
            - MYRL_REPLAN_K: integer
            - MYRL_ALLOW_SUBGOAL_IMMEDIATE: 0|1
            - MYRL_VALUE_MODE: dual|scalar
            - MYRL_SCALAR_LAMBDA: float

        This is intentionally optional: if env vars are unset, baseline behavior is preserved.
        """

        replan_mode = cls._getenv("MYRL_REPLAN_MODE", "event").strip().lower()
        value_mode = cls._getenv("MYRL_VALUE_MODE", "dual").strip().lower()

        try:
            replan_k = int(float(cls._getenv("MYRL_REPLAN_K", "10")))
        except ValueError:
            replan_k = 10

        allow_subgoal_immediate_raw = cls._getenv("MYRL_ALLOW_SUBGOAL_IMMEDIATE", "1").strip().lower()
        allow_subgoal_immediate = allow_subgoal_immediate_raw not in {"0", "false", "no", "off"}

        try:
            scalar_lambda = float(cls._getenv("MYRL_SCALAR_LAMBDA", "1.0"))
        except ValueError:
            scalar_lambda = 1.0

        if replan_mode not in {"event", "periodic"}:
            replan_mode = "event"
        if value_mode not in {"dual", "scalar"}:
            value_mode = "dual"

        replan_k = max(1, int(replan_k))

        return cls(
            replan_mode=replan_mode,
            replan_k=replan_k,
            allow_subgoal_immediate=allow_subgoal_immediate,
            value_mode=value_mode,
            scalar_lambda=scalar_lambda,
        )

