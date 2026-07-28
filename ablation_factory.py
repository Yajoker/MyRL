from __future__ import annotations

from pathlib import Path
from typing import Optional

from ablation_config import AblationConfig
from integration import HierarchicalNavigationSystem
from robot_nav.SIM_ENV.sensor_metadata import LidarMetadata


def create_system(
    ab_cfg: Optional[AblationConfig] = None,
    *,
    lidar_metadata: LidarMetadata,
    load_models: bool = False,
    models_directory: Path = Path("myrl/models"),
    **kwargs,
) -> HierarchicalNavigationSystem:
    """Return a navigation system configured for one ablation.

    Planner selection belongs to :class:`HierarchicalNavigationSystem`, which
    also owns the shared temporal LiDAR processor and passes the simulator
    metadata and configured maximum angular velocity to the selected planner.
    Keeping construction in one place prevents the factory from rebuilding a
    second planner with stale legacy arguments.
    """

    ab_cfg = ab_cfg or AblationConfig.from_env()
    return HierarchicalNavigationSystem(
        lidar_metadata=lidar_metadata,
        ablation_config=ab_cfg,
        load_models=load_models,
        models_directory=models_directory,
        **kwargs,
    )
