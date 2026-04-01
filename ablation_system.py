from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from high_level_planner import HighLevelPlanner, TriggerFlags


@dataclass
class PeriodicReplanState:
    last_replan_step: int = 0
    last_check_step: int = 0


class HighLevelPlannerPeriodic(HighLevelPlanner):
    """A1 ablation: replace event-triggered replanning with fixed K-step replanning.

    This class keeps the existing trigger computation interface intact:
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
        laser_scan,
        robot_pose,
        goal_info,
        risk_index: float,
        current_step: int = 0,
        window_metrics: Optional[dict] = None,
    ) -> TriggerFlags:
        flags = super().check_triggers(
            laser_scan,
            robot_pose,
            goal_info,
            risk_index=risk_index,
            current_step=current_step,
            window_metrics=window_metrics,
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
