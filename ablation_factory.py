from __future__ import annotations

from pathlib import Path
from typing import Optional

from ablation_config import AblationConfig
from ablation_system import HighLevelPlannerPeriodic
from high_level_planner import HighLevelPlanner
from high_level_planner_scalar import HighLevelPlannerScalar, HighLevelPlannerScalarPeriodic
from integration import HierarchicalNavigationSystem


def create_system(
    ab_cfg: Optional[AblationConfig] = None,
    *,
    load_models: bool = False,
    models_directory: Path = Path("myrl/models"),
    **kwargs,
) -> HierarchicalNavigationSystem:
    """Factory that returns a system configured for the desired ablation.

    This is optional sugar; you can also rely on env-vars + integration-level switching.
    """

    ab_cfg = ab_cfg or AblationConfig.from_env()

    # Create base system.
    system = HierarchicalNavigationSystem(
        load_models=load_models,
        models_directory=models_directory,
        **kwargs,
    )

    # Swap planner according to ablation config.
    old = system.high_level_planner

    PlannerCls = HighLevelPlanner
    planner_kwargs = {}

    if ab_cfg.replan_mode == "periodic" and ab_cfg.value_mode == "scalar":
        PlannerCls = HighLevelPlannerScalarPeriodic
        planner_kwargs.update(
            scalar_lambda=float(ab_cfg.scalar_lambda),
            replan_k=int(ab_cfg.replan_k),
            allow_subgoal_immediate=bool(ab_cfg.allow_subgoal_immediate),
        )
    elif ab_cfg.replan_mode == "periodic":
        PlannerCls = HighLevelPlannerPeriodic
        planner_kwargs.update(
            replan_k=int(ab_cfg.replan_k),
            allow_subgoal_immediate=bool(ab_cfg.allow_subgoal_immediate),
        )
    elif ab_cfg.value_mode == "scalar":
        PlannerCls = HighLevelPlannerScalar
        planner_kwargs.update(
            scalar_lambda=float(ab_cfg.scalar_lambda),
        )

    if PlannerCls is HighLevelPlanner:
        return system

    # Rebuild planner with matching parameters.
    if PlannerCls is HighLevelPlannerScalar:
        new_planner = PlannerCls(
            belief_dim=old.belief_dim,
            device=old.device,
            save_directory=old.save_directory,
            model_name=old.model_name,
            load_model=load_models,
            step_duration=system.step_duration,
            min_interval=None,
            subgoal_reach_threshold=system.subgoal_threshold,
            waypoint_lookahead=system.waypoint_lookahead,
            trigger_config=old.event_trigger._config,
            planner_config=old.planner_config,
            **planner_kwargs,
        )
    else:
        new_planner = PlannerCls(
            belief_dim=old.belief_dim,
            device=old.device,
            save_directory=old.save_directory,
            model_name=old.model_name,
            load_model=load_models,
            step_duration=system.step_duration,
            min_interval=None,
            subgoal_reach_threshold=system.subgoal_threshold,
            waypoint_lookahead=system.waypoint_lookahead,
            trigger_config=old.event_trigger._config,
            planner_config=old.planner_config,
            **planner_kwargs,
        )

    system.high_level_planner = new_planner
    return system
