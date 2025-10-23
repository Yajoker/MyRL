from pathlib import Path
import time
import numpy as np
import torch

from ethsrl.core.control.low_level_controller import LowLevelController
from ethsrl.core.planning.high_level_planner import HighLevelPlanner


class HierarchicalNavigationSystem:
    """
    Hierarchical navigation system integrating high-level planning and low-level control.
    
    This system combines event-triggered high-level planning with reactive 
    low-level control to achieve efficient and safe robot navigation.
    """
    def __init__(self, 
                 laser_dim=180,
                 action_dim=2,
                 max_action=1.0,
                 device=None,
                 load_models=False,
                 models_directory=Path("ethsrl/models")):
        """
        Initialize the hierarchical navigation system.
        
        Args:
            laser_dim: Dimension of laser scan data
            action_dim: Dimension of action space
            max_action: Maximum action magnitude
            device: Computation device (CPU/GPU)
            load_models: Whether to load pre-trained models
            models_directory: Directory containing model files
        """
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Calculate state dimensions
        high_level_state_dim = laser_dim + 3  # laser + [goal_distance, goal_cos, goal_sin]
        low_level_state_dim = laser_dim + 4   # laser + [subgoal_distance, subgoal_angle, prev_lin_vel, prev_ang_vel]
        
        # Initialize high-level planner
        self.high_level_planner = HighLevelPlanner(
            belief_dim=laser_dim,
            device=self.device,
            save_directory=models_directory / "high_level",
            model_name="high_level_planner",
            load_model=load_models
        )
        
        # Initialize low-level controller
        self.low_level_controller = LowLevelController(
            state_dim=low_level_state_dim,
            action_dim=action_dim,
            max_action=max_action,
            device=self.device,
            save_directory=models_directory / "low_level",
            model_name="low_level_controller",
            load_model=load_models
        )
        
        # System state
        self.current_subgoal = None
        self.prev_action = [0.0, 0.0]  # [lin_vel, ang_vel]
        self.step_count = 0
        self.last_replanning_step = 0
    
    def step(self, laser_scan, goal_distance, goal_cos, goal_sin, robot_pose):
        """
        Execute a step of the hierarchical navigation system.
        
        Args:
            laser_scan: Current laser scan readings
            goal_distance: Distance to global goal
            goal_cos: Cosine of angle to global goal
            goal_sin: Sine of angle to global goal
            robot_pose: Current robot pose [x, y, theta]
            
        Returns:
            Action [linear_velocity, angular_velocity]
        """
        self.step_count += 1
        
        # Determine if we need to generate a new subgoal
        should_replan = False
        
        # If no subgoal exists, definitely need to generate one
        if self.current_subgoal is None:
            should_replan = True
        else:
            # Check if event triggers indicate we should replan
            should_replan = self.high_level_planner.check_triggers(
                laser_scan, 
                robot_pose,
                [goal_distance, goal_cos, goal_sin]
            )
        
        # Generate new subgoal if needed
        if should_replan:
            subgoal_distance, subgoal_angle = self.high_level_planner.generate_subgoal(
                laser_scan, goal_distance, goal_cos, goal_sin
            )
            self.current_subgoal = (subgoal_distance, subgoal_angle)
            self.last_replanning_step = self.step_count
            print(f"New subgoal: distance={subgoal_distance:.2f}m, angle={subgoal_angle:.2f}rad")
        
        # Execute low-level control to track the subgoal
        low_level_state = self.low_level_controller.process_observation(
            laser_scan,
            self.current_subgoal[0],  # subgoal distance
            self.current_subgoal[1],  # subgoal angle
            self.prev_action          # previous action
        )
        
        # Get action from low-level controller
        action = self.low_level_controller.predict_action(low_level_state)
        
        # Convert network output to robot commands
        linear_velocity = (action[0] + 1) / 4  # Map [-1, 1] to [0, 0.5]
        angular_velocity = action[1]           # Keep [-1, 1] range
        
        # Store action for next step
        self.prev_action = [linear_velocity, angular_velocity]
        
        return [linear_velocity, angular_velocity]
    
    def reset(self):
        """Reset the navigation system state."""
        self.current_subgoal = None
        self.prev_action = [0.0, 0.0]
        self.step_count = 0
        self.last_replanning_step = 0


def create_navigation_system(load_models=False):
    """
    Factory function to create a hierarchical navigation system.
    
    Args:
        load_models: Whether to load pre-trained models
        
    Returns:
        Initialized HierarchicalNavigationSystem
    """
    return HierarchicalNavigationSystem(
        laser_dim=180,
        action_dim=2,
        max_action=1.0,
        load_models=load_models
    )