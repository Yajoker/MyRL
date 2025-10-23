import numpy as np
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


class SubgoalNetwork(nn.Module):
    """
    Neural network for generating navigation subgoals based on belief state.
    
    This network processes laser scan data along with goal information to
    produce subgoal distance and angle values, enabling intermediate waypoint
    generation for safer and more efficient navigation.
    """
    def __init__(self, belief_dim=90, hidden_dim=256):
        """
        Initialize the subgoal generation network.
        
        Args:
            belief_dim: Dimension of the belief state input
            hidden_dim: Dimension of hidden layers
        """
        super(SubgoalNetwork, self).__init__()
        
        # CNN for laser scan processing
        self.cnn1 = nn.Conv1d(1, 8, kernel_size=5, stride=2)
        self.cnn2 = nn.Conv1d(8, 16, kernel_size=3, stride=2)
        self.cnn3 = nn.Conv1d(16, 8, kernel_size=3, stride=1)
        
        # Process global goal information
        self.goal_embed = nn.Linear(3, 32)  # distance, cos, sin
        
        # Fully connected layers
        cnn_output_dim = self._get_cnn_output_dim(belief_dim)
        self.fc1 = nn.Linear(cnn_output_dim + 32, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        
        # Output layers for subgoal distance and angle
        self.distance_head = nn.Linear(hidden_dim // 2, 1)
        self.angle_head = nn.Linear(hidden_dim // 2, 1)
        
    def _get_cnn_output_dim(self, belief_dim):
        """Calculate the flattened output dimension of the CNN layers"""
        x = torch.zeros(1, 1, belief_dim)
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        return x.numel()
        
    def forward(self, belief_state, goal_info):
        """
        Forward pass through the subgoal network.
        
        Args:
            belief_state: Tensor containing the laser scan data
            goal_info: Tensor containing [distance, cos, sin] to global goal
            
        Returns:
            Tuple containing (subgoal_distance, subgoal_angle)
        """
        # Process laser scan
        laser = belief_state.unsqueeze(1)  # Add channel dimension
        x = F.relu(self.cnn1(laser))
        x = F.relu(self.cnn2(x))
        x = F.relu(self.cnn3(x))
        x = x.flatten(start_dim=1)
        
        # Process goal info
        g = F.relu(self.goal_embed(goal_info))
        
        # Combine features
        combined = torch.cat((x, g), dim=1)
        
        # Process through fully connected layers
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))
        
        # Generate subgoal parameters
        distance = 3 * torch.sigmoid(self.distance_head(x))  # Range [0, 3] meters
        angle = torch.tanh(self.angle_head(x)) * np.pi  # Range [-π, π]
        
        return distance, angle


class EventTrigger:
    """
    Implements multi-source event triggering mechanisms for high-level planning.
    
    This class contains various trigger conditions that determine when a new subgoal
    should be generated based on environmental factors, robot state, and timing constraints.
    """
    def __init__(self, 
                 base_threshold=0.7, 
                 complexity_factor=0.3,
                 safe_distance=0.5,
                 heading_threshold=0.3,
                 min_interval=1.0):
        """
        Initialize the event trigger mechanism.
        
        Args:
            base_threshold: Base risk threshold τ0
            complexity_factor: Environment complexity influence coefficient α
            safe_distance: Safe distance threshold in meters
            heading_threshold: Heading change threshold in radians
            min_interval: Minimum time interval between triggers in seconds
        """
        self.base_threshold = base_threshold
        self.complexity_factor = complexity_factor
        self.safe_distance = safe_distance
        self.heading_threshold = heading_threshold
        self.min_interval = min_interval
        
        # State variables
        self.last_trigger_time = 0.0
        self.last_heading = 0.0
        self.last_subgoal = None
        self.current_threshold = base_threshold
        
    def risk_assessment_trigger(self, risk_level, env_complexity):
        """
        Trigger based on environmental risk assessment.
        
        Args:
            risk_level: Current assessed risk level [0, 1]
            env_complexity: Computed environmental complexity [0, 1]
            
        Returns:
            Boolean indicating if trigger condition is met
        """
        # Dynamically adjust threshold based on environment complexity
        self.current_threshold = self.base_threshold * (1 - self.complexity_factor * env_complexity)
        return risk_level > self.current_threshold
    
    def obstacle_proximity_trigger(self, min_obstacle_dist):
        """
        Trigger based on proximity to obstacles.
        
        Args:
            min_obstacle_dist: Distance to the closest obstacle in meters
            
        Returns:
            Boolean indicating if trigger condition is met
        """
        return min_obstacle_dist < self.safe_distance * 1.5
    
    def heading_change_trigger(self, current_heading):
        """
        Trigger based on significant changes in heading.
        
        Args:
            current_heading: Current robot heading in radians
            
        Returns:
            Boolean indicating if trigger condition is met
        """
        heading_change = abs(self.last_heading - current_heading)
        # Handle wrapping around ±π
        if heading_change > np.pi:
            heading_change = 2 * np.pi - heading_change
            
        self.last_heading = current_heading
        return heading_change > self.heading_threshold
    
    def subgoal_reachability_trigger(self, current_pos, subgoal_pos, min_obstacle_dist):
        """
        Trigger based on subgoal reachability assessment.
        
        Args:
            current_pos: Current robot position [x, y]
            subgoal_pos: Current subgoal position [x, y]
            min_obstacle_dist: Distance to the closest obstacle
            
        Returns:
            Boolean indicating if trigger condition is met
        """
        if self.last_subgoal is None:
            return False
            
        # Calculate distance to subgoal
        subgoal_dist = np.linalg.norm(np.array(current_pos) - np.array(subgoal_pos))
        
        # If subgoal is close, check if we've reached it
        if subgoal_dist < 0.3:
            return True
            
        # If path to subgoal is blocked
        if min_obstacle_dist < self.safe_distance and subgoal_dist > self.safe_distance:
            # Calculate vector to obstacle and to subgoal
            if np.dot(current_pos, subgoal_pos) / (np.linalg.norm(current_pos) * np.linalg.norm(subgoal_pos)) > 0.7:
                return True
                
        return False
        
    def time_based_trigger(self):
        """
        Ensure minimum time interval between triggers.
        
        Returns:
            Boolean indicating if enough time has passed since last trigger
        """
        current_time = time.time()
        if current_time - self.last_trigger_time > self.min_interval:
            return True
        return False
    
    def reset_time(self):
        """Reset the last trigger time to the current time"""
        self.last_trigger_time = time.time()


class HighLevelPlanner:
    """
    High-level planner that generates subgoals for navigation based on event triggers.
    
    This class implements the subgoal generation strategy using a neural network
    and manages the event triggering mechanism to determine when to compute new subgoals.
    """
    def __init__(self, 
                 belief_dim=90,
                 device=None,
                 save_directory=Path("ethsrl/models/high_level"),
                 model_name="high_level_planner",
                 load_model=False,
                 load_directory=None):
        """
        Initialize the high-level planner.
        
        Args:
            belief_dim: Dimension of the belief state
            device: Computation device (CPU/GPU)
            save_directory: Directory to save model checkpoints
            model_name: Name for saved model files
            load_model: Whether to load a pre-trained model
            load_directory: Directory to load model from (if None, uses save_directory)
        """
        self.belief_dim = belief_dim
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize subgoal generation network
        self.subgoal_network = SubgoalNetwork(belief_dim=belief_dim).to(self.device)
        
        # Initialize event trigger
        self.event_trigger = EventTrigger()
        
        # Training setup
        self.optimizer = torch.optim.Adam(self.subgoal_network.parameters(), lr=1e-4)
        self.writer = SummaryWriter(comment=model_name)
        self.iter_count = 0
        self.model_name = model_name
        self.save_directory = save_directory
        
        # Current state tracking
        self.current_subgoal = None
        self.last_goal_distance = float('inf')
        self.last_goal_direction = 0.0
        
        # Load pre-trained model if requested
        if load_model:
            load_dir = load_directory if load_directory else save_directory
            self.load_model(filename=model_name, directory=load_dir)
    
    def process_laser_scan(self, laser_scan):
        """
        Process raw laser scan data into a belief state representation.
        
        Args:
            laser_scan: Raw laser scan readings
            
        Returns:
            Processed laser scan as a tensor
        """
        laser_scan = np.array(laser_scan)
        
        # Handle infinite values
        inf_mask = np.isinf(laser_scan)
        laser_scan[inf_mask] = 7.0  # Replace with max range
        
        # Normalize to [0, 1]
        laser_scan = laser_scan / 7.0
        
        return torch.FloatTensor(laser_scan).to(self.device)
    
    def process_goal_info(self, distance, cos_angle, sin_angle):
        """
        Process goal information into a tensor.
        
        Args:
            distance: Distance to global goal
            cos_angle: Cosine of angle to global goal
            sin_angle: Sine of angle to global goal
            
        Returns:
            Processed goal information as a tensor
        """
        # Normalize distance
        norm_distance = min(distance / 10.0, 1.0)
        
        # Combine into tensor
        goal_info = torch.FloatTensor([norm_distance, cos_angle, sin_angle]).to(self.device)
        
        return goal_info
    
    def check_triggers(self, laser_scan, robot_pose, goal_info, min_obstacle_dist=None):
        """
        Check if any event triggers are activated.
        
        Args:
            laser_scan: Current laser scan readings
            robot_pose: Current robot pose [x, y, theta]
            goal_info: Global goal information [distance, cos, sin]
            min_obstacle_dist: Distance to closest obstacle (if None, computed from laser_scan)
            
        Returns:
            Boolean indicating if a new subgoal should be generated
        """
        # If minimum time hasn't passed, don't trigger
        if not self.event_trigger.time_based_trigger():
            return False
            
        # Compute minimum obstacle distance if not provided
        if min_obstacle_dist is None:
            valid_scans = laser_scan[~np.isinf(laser_scan)]
            min_obstacle_dist = np.min(valid_scans) if valid_scans.size > 0 else float('inf')
        
        # Compute environment complexity
        env_complexity = self.compute_environment_complexity(laser_scan)
        
        # Compute risk level (simplified)
        risk_level = 1.0 - min(min_obstacle_dist / 3.0, 1.0)
        
        # Check individual triggers
        risk_trigger = self.event_trigger.risk_assessment_trigger(risk_level, env_complexity)
        obstacle_trigger = self.event_trigger.obstacle_proximity_trigger(min_obstacle_dist)
        heading_trigger = self.event_trigger.heading_change_trigger(robot_pose[2])
        
        # Create subgoal position from current subgoal (if exists)
        subgoal_pos = None
        if self.current_subgoal is not None:
            distance, angle = self.current_subgoal
            subgoal_x = robot_pose[0] + distance * np.cos(robot_pose[2] + angle)
            subgoal_y = robot_pose[1] + distance * np.sin(robot_pose[2] + angle)
            subgoal_pos = [subgoal_x, subgoal_y]
            
        reachability_trigger = self.event_trigger.subgoal_reachability_trigger(
            [robot_pose[0], robot_pose[1]], 
            subgoal_pos if subgoal_pos else [robot_pose[0], robot_pose[1]],
            min_obstacle_dist
        )
        
        # Combine triggers
        trigger_new_subgoal = risk_trigger or obstacle_trigger or heading_trigger or reachability_trigger
        
        # If triggered, reset time counter
        if trigger_new_subgoal:
            self.event_trigger.reset_time()
            
        return trigger_new_subgoal
    
    def generate_subgoal(self, laser_scan, goal_distance, goal_cos, goal_sin):
        """
        Generate a new subgoal based on current state.
        
        Args:
            laser_scan: Processed laser scan data
            goal_distance: Distance to global goal
            goal_cos: Cosine of angle to global goal
            goal_sin: Sine of angle to global goal
            
        Returns:
            Tuple containing (subgoal_distance, subgoal_angle)
        """
        # Process inputs
        laser_tensor = self.process_laser_scan(laser_scan)
        goal_tensor = self.process_goal_info(goal_distance, goal_cos, goal_sin)
        
        # Generate subgoal using network
        with torch.no_grad():
            distance, angle = self.subgoal_network(
                laser_tensor.unsqueeze(0),
                goal_tensor.unsqueeze(0)
            )
            
        # Convert to numpy arrays
        subgoal_distance = distance.cpu().numpy().item()
        subgoal_angle = angle.cpu().numpy().item()
        
        # Store for future reference
        self.current_subgoal = (subgoal_distance, subgoal_angle)
        self.last_goal_distance = goal_distance
        self.last_goal_direction = np.arctan2(goal_sin, goal_cos)
        
        return subgoal_distance, subgoal_angle
    
    def compute_environment_complexity(self, laser_scan):
        """
        Compute the complexity of the current environment based on laser scan.
        
        Args:
            laser_scan: Laser scan readings
            
        Returns:
            Environment complexity score in range [0, 1]
        """
        # Replace infinite values with max range
        scan = np.array(laser_scan)
        scan[np.isinf(scan)] = 7.0
        
        # Simple complexity metric based on:
        # 1. Variance in scan readings (more variance = more complex)
        # 2. Average distance (closer obstacles = more complex)
        variance = np.var(scan) / 10.0  # Normalized variance
        avg_distance = np.mean(scan) / 7.0  # Normalized mean
        
        # Compute complexity score (inverse of average distance, weighted by variance)
        complexity = (1.0 - avg_distance) * (0.5 + 0.5 * min(variance, 1.0))
        
        return min(complexity, 1.0)  # Ensure in [0, 1]
    
    def filter_unsafe_subgoals(self, laser_scan, candidate_subgoals):
        """
        Filter out unsafe subgoal options based on laser scan data.
        
        Args:
            laser_scan: Current laser scan data
            candidate_subgoals: List of (distance, angle) tuples
            
        Returns:
            List of safe (distance, angle) tuples
        """
        safe_subgoals = []
        
        # Convert scan to cartesian coordinates for easier processing
        angles = np.linspace(-np.pi, np.pi, len(laser_scan))
        
        for subgoal in candidate_subgoals:
            distance, angle = subgoal
            
            # Check if the path to the subgoal is clear
            index = int((angle + np.pi) / (2 * np.pi) * len(laser_scan))
            index = max(0, min(index, len(laser_scan) - 1))  # Ensure valid index
            
            # Check if subgoal is within safe distance
            if distance < laser_scan[index] - self.event_trigger.safe_distance:
                safe_subgoals.append(subgoal)
                
        # If all subgoals are unsafe, select the safest one
        if not safe_subgoals and candidate_subgoals:
            safest_distance = 0
            safest_subgoal = None
            
            for subgoal in candidate_subgoals:
                distance, angle = subgoal
                index = int((angle + np.pi) / (2 * np.pi) * len(laser_scan))
                index = max(0, min(index, len(laser_scan) - 1))
                
                if laser_scan[index] > safest_distance:
                    safest_distance = laser_scan[index]
                    safest_subgoal = subgoal
                    
            if safest_subgoal:
                safe_subgoals.append(safest_subgoal)
                
        return safe_subgoals
    
    def update_planner(self, states, actions, rewards, next_states, dones, batch_size=64):
        """
        Update the planner's neural network using collected experience.
        
        Args:
            states: Batch of environment states
            actions: Batch of actions taken
            rewards: Batch of rewards received
            next_states: Batch of resulting states
            dones: Batch of done flags
            batch_size: Training batch size
            
        Returns:
            Dictionary of training metrics
        """
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device).unsqueeze(1)
        
        # Split states into laser and goal components
        laser_scans = states[:, :-3]
        goal_info = states[:, -3:]
        
        # Generate subgoals
        subgoal_distances, subgoal_angles = self.subgoal_network(laser_scans, goal_info)
        subgoals = torch.cat([subgoal_distances, subgoal_angles], dim=1)
        
        # Compute loss (simplified example using MSE)
        # In a real implementation, you might use more sophisticated loss functions
        loss = F.mse_loss(subgoals, actions)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update training counters
        self.iter_count += 1
        
        # Log metrics
        self.writer.add_scalar('planner/loss', loss.item(), self.iter_count)
        
        return {
            'loss': loss.item(),
            'avg_distance': subgoal_distances.mean().item(),
            'avg_angle': subgoal_angles.mean().item()
        }
    
    def save_model(self, filename, directory):
        """
        Save the model parameters to files.
        
        Args:
            filename: Base name for saved files
            directory: Directory to save to
        """
        # Create directory if it doesn't exist
        Path(directory).mkdir(parents=True, exist_ok=True)
        
        # Save model
        torch.save(self.subgoal_network.state_dict(), f"{directory}/{filename}.pth")
        print(f"Model saved to {directory}/{filename}.pth")
    
    def load_model(self, filename, directory):
        """
        Load model parameters from files.
        
        Args:
            filename: Base name of files to load
            directory: Directory to load from
        """
        try:
            self.subgoal_network.load_state_dict(torch.load(f"{directory}/{filename}.pth"))
            print(f"Model loaded from {directory}/{filename}.pth")
        except FileNotFoundError as e:
            print(f"Error loading model: {e}")