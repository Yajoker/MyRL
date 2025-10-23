from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


class LowLevelActorNetwork(nn.Module):
    """
    Actor network for the low-level controller based on CNNTD3.
    
    Modified to accept subgoal information (distance and angle) instead of global goal.
    """
    def __init__(self, action_dim):
        super(LowLevelActorNetwork, self).__init__()
        
        # CNN for laser scan processing
        self.cnn1 = nn.Conv1d(1, 4, kernel_size=8, stride=4)
        self.cnn2 = nn.Conv1d(4, 8, kernel_size=8, stride=4)
        self.cnn3 = nn.Conv1d(8, 4, kernel_size=4, stride=2)
        
        # Subgoal embedding (distance and angle)
        self.subgoal_embed = nn.Linear(2, 10)
        
        # Previous action embedding
        self.action_embed = nn.Linear(2, 10)
        
        # Fully connected layers
        self.layer_1 = nn.Linear(36, 400)  # 16 (CNN output) + 10 (subgoal) + 10 (prev action)
        torch.nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="leaky_relu")
        
        self.layer_2 = nn.Linear(400, 300)
        torch.nn.init.kaiming_uniform_(self.layer_2.weight, nonlinearity="leaky_relu")
        
        self.layer_3 = nn.Linear(300, action_dim)
        self.tanh = nn.Tanh()

    def forward(self, s):
        """
        Forward pass through the Actor network.
        
        Args:
            s: Input state tensor with shape (batch_size, state_dim)
               Contains laser scan data and additional info:
               - Laser scan data
               - Subgoal distance and angle
               - Previous linear and angular velocity
               
        Returns:
            Action tensor with values in range [-1, 1]
        """
        if len(s.shape) == 1:
            s = s.unsqueeze(0)
            
        laser = s[:, :-4]  # Laser scan data
        subgoal = s[:, -4:-2]  # Subgoal (distance, angle)
        prev_act = s[:, -2:]  # Previous action (linear_vel, angular_vel)
        
        # Process laser scan
        laser = laser.unsqueeze(1)
        l = F.leaky_relu(self.cnn1(laser))
        l = F.leaky_relu(self.cnn2(l))
        l = F.leaky_relu(self.cnn3(l))
        l = l.flatten(start_dim=1)
        
        # Process subgoal
        g = F.leaky_relu(self.subgoal_embed(subgoal))
        
        # Process previous action
        a = F.leaky_relu(self.action_embed(prev_act))
        
        # Concatenate all features
        s = torch.concat((l, g, a), dim=-1)
        
        # Final layers
        s = F.leaky_relu(self.layer_1(s))
        s = F.leaky_relu(self.layer_2(s))
        a = self.tanh(self.layer_3(s))
        return a


class LowLevelCriticNetwork(nn.Module):
    """
    Critic network for the low-level controller based on CNNTD3.
    
    Uses the same structure as CNNTD3's critic but with modified input processing.
    """
    def __init__(self, action_dim):
        super(LowLevelCriticNetwork, self).__init__()
        
        # CNN for laser scan processing
        self.cnn1 = nn.Conv1d(1, 4, kernel_size=8, stride=4)
        self.cnn2 = nn.Conv1d(4, 8, kernel_size=8, stride=4)
        self.cnn3 = nn.Conv1d(8, 4, kernel_size=4, stride=2)
        
        # Subgoal embedding
        self.subgoal_embed = nn.Linear(2, 10)
        
        # Previous action embedding
        self.action_embed = nn.Linear(2, 10)
        
        # Q1 network
        self.layer_1 = nn.Linear(36, 400)
        torch.nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="leaky_relu")
        self.layer_2_s = nn.Linear(400, 300)
        torch.nn.init.kaiming_uniform_(self.layer_2_s.weight, nonlinearity="leaky_relu")
        self.layer_2_a = nn.Linear(action_dim, 300)
        torch.nn.init.kaiming_uniform_(self.layer_2_a.weight, nonlinearity="leaky_relu")
        self.layer_3 = nn.Linear(300, 1)
        torch.nn.init.kaiming_uniform_(self.layer_3.weight, nonlinearity="leaky_relu")
        
        # Q2 network
        self.layer_4 = nn.Linear(36, 400)
        torch.nn.init.kaiming_uniform_(self.layer_4.weight, nonlinearity="leaky_relu")
        self.layer_5_s = nn.Linear(400, 300)
        torch.nn.init.kaiming_uniform_(self.layer_5_s.weight, nonlinearity="leaky_relu")
        self.layer_5_a = nn.Linear(action_dim, 300)
        torch.nn.init.kaiming_uniform_(self.layer_5_a.weight, nonlinearity="leaky_relu")
        self.layer_6 = nn.Linear(300, 1)
        torch.nn.init.kaiming_uniform_(self.layer_6.weight, nonlinearity="leaky_relu")

    def forward(self, s, action):
        """
        Forward pass through both Q-networks of the Critic.
        
        Args:
            s: State tensor
            action: Action tensor
            
        Returns:
            Tuple of Q-values (Q1, Q2)
        """
        laser = s[:, :-4]  # Laser scan data
        subgoal = s[:, -4:-2]  # Subgoal (distance, angle)
        prev_act = s[:, -2:]  # Previous action
        
        laser = laser.unsqueeze(1)
        
        # Process laser scan
        l = F.leaky_relu(self.cnn1(laser))
        l = F.leaky_relu(self.cnn2(l))
        l = F.leaky_relu(self.cnn3(l))
        l = l.flatten(start_dim=1)
        
        # Process subgoal
        g = F.leaky_relu(self.subgoal_embed(subgoal))
        
        # Process previous action
        a = F.leaky_relu(self.action_embed(prev_act))
        
        # Concatenate features
        s = torch.concat((l, g, a), dim=-1)
        
        # Q1 computation
        s1 = F.leaky_relu(self.layer_1(s))
        self.layer_2_s(s1)
        self.layer_2_a(action)
        s11 = torch.mm(s1, self.layer_2_s.weight.data.t())
        s12 = torch.mm(action, self.layer_2_a.weight.data.t())
        s1 = F.leaky_relu(s11 + s12 + self.layer_2_a.bias.data)
        q1 = self.layer_3(s1)
        
        # Q2 computation
        s2 = F.leaky_relu(self.layer_4(s))
        self.layer_5_s(s2)
        self.layer_5_a(action)
        s21 = torch.mm(s2, self.layer_5_s.weight.data.t())
        s22 = torch.mm(action, self.layer_5_a.weight.data.t())
        s2 = F.leaky_relu(s21 + s22 + self.layer_5_a.bias.data)
        q2 = self.layer_6(s2)
        
        return q1, q2


class LowLevelController:
    """
    Low-level execution controller based on CNNTD3 algorithm.
    
    This controller processes laser scan data and subgoal information to generate
    low-level control commands (linear and angular velocity).
    """
    def __init__(
        self, 
        state_dim, 
        action_dim, 
        max_action,
        device,
        lr=1e-4,
        save_every=0,
        load_model=False,
        save_directory=Path("ethsrl/models/low_level"),
        model_name="low_level_controller",
        load_directory=None,
    ):
        """
        Initialize the low-level controller.
        
        Args:
            state_dim: Dimension of the state space (laser + subgoal + prev_action)
            action_dim: Dimension of the action space (2 for [lin_vel, ang_vel])
            max_action: Maximum action magnitude
            device: Torch device (CPU or GPU)
            lr: Learning rate
            save_every: Save model every N updates (0 to disable)
            load_model: Whether to load a pre-trained model
            save_directory: Directory to save models
            model_name: Name for saved model files
            load_directory: Directory to load models from (if None, uses save_directory)
        """
        self.device = device
        self.action_dim = action_dim
        self.max_action = max_action
        self.state_dim = state_dim
        
        # Initialize actor network and target
        self.actor = LowLevelActorNetwork(action_dim).to(device)
        self.actor_target = LowLevelActorNetwork(action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        
        # Initialize critic networks and targets
        self.critic = LowLevelCriticNetwork(action_dim).to(device)
        self.critic_target = LowLevelCriticNetwork(action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        
        # Training setup
        self.writer = SummaryWriter(comment=model_name)
        self.iter_count = 0
        self.save_every = save_every
        self.model_name = model_name
        self.save_directory = save_directory
        
        if load_model:
            load_dir = load_directory if load_directory else save_directory
            self.load_model(filename=model_name, directory=load_dir)
    
    def process_observation(self, laser_scan, subgoal_distance, subgoal_angle, prev_action):
        """
        Process raw observations into the state vector expected by the network.
        
        Args:
            laser_scan: Raw laser scan data
            subgoal_distance: Distance to subgoal
            subgoal_angle: Angle to subgoal
            prev_action: Previous action [lin_vel, ang_vel]
            
        Returns:
            Processed state vector
        """
        # Normalize laser scan data (handling infinities)
        laser_scan = np.array(laser_scan)
        inf_mask = np.isinf(laser_scan)
        laser_scan[inf_mask] = 7.0  # Replace infinities with max range
        laser_scan /= 7.0  # Normalize to [0, 1]
        
        # Normalize subgoal distance and angle
        norm_distance = min(subgoal_distance / 10.0, 1.0)  # Normalize to [0, 1]
        norm_angle = subgoal_angle / np.pi  # Normalize to [-1, 1]
        
        # Process previous action
        lin_vel = prev_action[0] * 2  # Scale to appropriate range
        ang_vel = (prev_action[1] + 1) / 2  # Scale to [0, 1]
        
        # Combine all components
        state = laser_scan.tolist() + [norm_distance, norm_angle] + [lin_vel, ang_vel]
        
        return np.array(state)
    
    def predict_action(self, state, add_noise=False, noise_scale=0.1):
        """
        Predict action based on current state.
        
        Args:
            state: Processed state vector
            add_noise: Whether to add exploration noise
            noise_scale: Scale of the exploration noise
            
        Returns:
            Action [lin_vel, ang_vel]
        """
        state_tensor = torch.FloatTensor(state).to(self.device)
        
        # Get action from actor network
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy().flatten()
        
        # Add exploration noise if required
        if add_noise:
            action += np.random.normal(0, noise_scale, size=self.action_dim)
            action = np.clip(action, -self.max_action, self.max_action)
        
        return action
    
    def update(self, replay_buffer, batch_size=64, discount=0.99, tau=0.005, 
               policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        """
        Update the controller parameters using samples from the replay buffer.
        
        Args:
            replay_buffer: Buffer containing experience samples
            batch_size: Size of the mini-batch
            discount: Discount factor for future rewards
            tau: Soft update parameter
            policy_noise: Noise added to target actions
            noise_clip: Maximum noise magnitude
            policy_freq: Frequency of actor updates relative to critic updates
        """
        # Sample a batch from the replay buffer
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        
        # Convert to tensors
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device).reshape(-1, 1)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).to(self.device).reshape(-1, 1)
        
        # Get next action with noise for regularization (TD3 technique)
        next_action = self.actor_target(next_state)
        noise = torch.FloatTensor(np.random.normal(0, policy_noise, size=(batch_size, self.action_dim))).to(self.device)
        noise = noise.clamp(-noise_clip, noise_clip)
        next_action = (next_action + noise).clamp(-self.max_action, self.max_action)
        
        # Compute target Q values
        target_q1, target_q2 = self.critic_target(next_state, next_action)
        target_q = torch.min(target_q1, target_q2)
        target_q = reward + (1 - done) * discount * target_q
        
        # Compute current Q values
        current_q1, current_q2 = self.critic(state, action)
        
        # Compute critic loss and update
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Delayed policy updates (TD3 technique)
        actor_loss = None
        if self.iter_count % policy_freq == 0:
            # Compute actor loss
            actor_loss = -self.critic.forward(state, self.actor(state))[0].mean()
            
            # Update actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Soft update target networks
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        
        # Increment iteration counter
        self.iter_count += 1
        
        # Logging
        self.writer.add_scalar('Loss/critic', critic_loss.item(), self.iter_count)
        if actor_loss is not None:
            self.writer.add_scalar('Loss/actor', actor_loss.item(), self.iter_count)
        
        # Save model if needed
        if self.save_every > 0 and self.iter_count % self.save_every == 0:
            self.save_model(filename=self.model_name, directory=self.save_directory)
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item() if actor_loss is not None else None,
            'q_value': current_q1.mean().item()
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
        
        # Save actor and critic models
        torch.save(self.actor.state_dict(), f"{directory}/{filename}_actor.pth")
        torch.save(self.actor_target.state_dict(), f"{directory}/{filename}_actor_target.pth")
        torch.save(self.critic.state_dict(), f"{directory}/{filename}_critic.pth")
        torch.save(self.critic_target.state_dict(), f"{directory}/{filename}_critic_target.pth")
        print(f"Model saved to {directory}/{filename}_*.pth")
    
    def load_model(self, filename, directory):
        """
        Load model parameters from files.
        
        Args:
            filename: Base name of files to load
            directory: Directory to load from
        """
        try:
            self.actor.load_state_dict(torch.load(f"{directory}/{filename}_actor.pth"))
            self.actor_target.load_state_dict(torch.load(f"{directory}/{filename}_actor_target.pth"))
            self.critic.load_state_dict(torch.load(f"{directory}/{filename}_critic.pth"))
            self.critic_target.load_state_dict(torch.load(f"{directory}/{filename}_critic_target.pth"))
            print(f"Model loaded from {directory}/{filename}_*.pth")
        except FileNotFoundError as e:
            print(f"Error loading model: {e}")