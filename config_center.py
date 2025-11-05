"""Centralised configuration dataclasses for the ETHSRL navigation stack.

This module collects all hyper-parameters that need to stay in sync across
integration, planning, control and training components.  By funnelling access
through this single module we avoid duplicated defaults and keep "single source
of truth" semantics for key limits and thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MotionConfig:
    """Motion primitive limits and time discretisation."""

    v_max: float = 0.5
    omega_max: float = 1.0
    dt: float = 0.3

    def __post_init__(self) -> None:  # type: ignore[override]
        if self.v_max <= 0:
            raise ValueError("v_max must be positive")
        if self.omega_max <= 0:
            raise ValueError("omega_max must be positive")
        if self.dt <= 0:
            raise ValueError("dt must be positive")


@dataclass(frozen=True)
class TriggerConfig:
    """High-level trigger thresholds and timing rules."""

    safety_trigger_distance: float = 0.7
    subgoal_reach_threshold: float = 0.5
    stagnation_steps: int = 30
    progress_epsilon: float = 0.1
    min_interval: float = 1.0

    def __post_init__(self) -> None:  # type: ignore[override]
        if self.safety_trigger_distance <= 0:
            raise ValueError("safety_trigger_distance must be positive")
        if self.subgoal_reach_threshold <= 0:
            raise ValueError("subgoal_reach_threshold must be positive")
        if self.stagnation_steps <= 0:
            raise ValueError("stagnation_steps must be positive")
        if self.progress_epsilon < 0:
            raise ValueError("progress_epsilon must be non-negative")
        if self.min_interval < 0:
            raise ValueError("min_interval must be non-negative")


@dataclass(frozen=True)
class PlannerConfig:
    """Static global planning configuration."""

    resolution: float = 0.25
    safety_margin: float = 0.35
    waypoint_lookahead: int = 3
    window_spacing: float = 2.0
    window_radius: float = 0.6

    def __post_init__(self) -> None:  # type: ignore[override]
        if self.resolution <= 0:
            raise ValueError("resolution must be positive")
        if self.safety_margin < 0:
            raise ValueError("safety_margin must be non-negative")
        if self.waypoint_lookahead <= 0:
            raise ValueError("waypoint_lookahead must be positive")
        if self.window_spacing <= 0:
            raise ValueError("window_spacing must be positive")
        if self.window_radius <= 0:
            raise ValueError("window_radius must be positive")


@dataclass(frozen=True)
class WindowRuntimeConfig:
    """Runtime rules for waypoint window tracking."""

    margin: float = 0.15
    step_limit: int = 80

    def __post_init__(self) -> None:  # type: ignore[override]
        if self.margin < 0:
            raise ValueError("margin must be non-negative")
        if self.step_limit <= 0:
            raise ValueError("step_limit must be positive")


@dataclass(frozen=True)
class IntegrationConfig:
    """Aggregate configuration passed into the integration layer."""

    motion: MotionConfig = field(default_factory=MotionConfig)
    trigger: TriggerConfig = field(default_factory=TriggerConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    window: WindowRuntimeConfig = field(default_factory=WindowRuntimeConfig)

    def with_updates(
        self,
        *,
        motion: MotionConfig | None = None,
        trigger: TriggerConfig | None = None,
        planner: PlannerConfig | None = None,
        window: WindowRuntimeConfig | None = None,
    ) -> "IntegrationConfig":
        """Return a copy with selected sub-configs replaced."""

        return IntegrationConfig(
            motion=motion or self.motion,
            trigger=trigger or self.trigger,
            planner=planner or self.planner,
            window=window or self.window,
        )
