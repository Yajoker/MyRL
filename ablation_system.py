from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from high_level_planner import HighLevelPlanner, TriggerFlags
from temporal_lidar import TemporalLidarObservation


@dataclass
class PeriodicReplanState:
    last_replan_step: int = 0
    last_check_step: int = 0


class HighLevelPlannerPeriodic(HighLevelPlanner):
    """A1 ablation: replace event-triggered replanning with fixed K-step replanning.

    This class keeps the canonical trigger computation interface intact:
    - `check_triggers(...)` still returns TriggerFlags (useful for logging)
    - `should_replan(flags)` ignores risk/stagnation/time rules and uses a periodic schedule

    Important:
        The training loop in `myrl/train.py` calls `check_triggers(..., current_step=steps)`
        and then `should_replan(flags)`. We rely on that order to access `current_step`.
    """

    def __init__(
        self,
        *args,
        replan_k: int = 10,
        allow_subgoal_immediate: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.replan_k = max(1, int(replan_k))
        self.allow_subgoal_immediate = bool(allow_subgoal_immediate)
        self._periodic = PeriodicReplanState(last_replan_step=0, last_check_step=0)

    def check_triggers(
        self,
        observation: TemporalLidarObservation,
        robot_pose,
        current_step: int = 0,
    ) -> TriggerFlags:
        flags = super().check_triggers(
            observation,
            robot_pose,
            current_step=current_step,
        )
        self._periodic.last_check_step = int(current_step)
        return flags

    def should_replan(self, flags: TriggerFlags) -> bool:
        current_step = int(getattr(self._periodic, "last_check_step", 0))
        periodic_ready = (current_step - int(self._periodic.last_replan_step)) >= int(self.replan_k)

        if self.allow_subgoal_immediate and getattr(flags, "subgoal_reached", False):
            return True

        return bool(periodic_ready)

    def generate_subgoal(self, *args, current_step: Optional[int] = None, **kwargs):
        result = super().generate_subgoal(*args, current_step=current_step, **kwargs)
        if current_step is None:
            current_step = int(getattr(self._periodic, "last_check_step", 0))
        self._periodic.last_replan_step = int(current_step)
        return result

    def reset_runtime_state(self) -> None:
        """Reset base trigger state and the episode-local periodic schedule."""

        super().reset_runtime_state()
        self._periodic = PeriodicReplanState(
            last_replan_step=0,
            last_check_step=0,
        )
