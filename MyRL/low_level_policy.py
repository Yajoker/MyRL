"""低层控制策略 - 实时生成控制命令"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class LowLevelActor(nn.Module):
    """
    低层策略网络（Actor）
    """
    def __init__(self, input_dim, hidden_dim=256, action_dim=2):
        super(LowLevelActor, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # 输出范围[-1, 1]
        )
    
    def forward(self, x):
        return self.net(x)

class LowLevelCritic(nn.Module):
    """
    低层价值网络（Critic）
    """
    def __init__(self, input_dim, hidden_dim=256, action_dim=2):
        super(LowLevelCritic, self).__init__()
        
        # Q1
        self.q1 = nn.Sequential(
            nn.Linear(input_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Q2 (双Q网络，用于减少过估计)
        self.q2 = nn.Sequential(
            nn.Linear(input_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        return self.q1(sa), self.q2(sa)
    
    def q1_value(self, state, action):
        sa = torch.cat([state, action], 1)
        return self.q1(sa)

class TD3Controller:
    """
    基于TD3算法的低层控制器
    """
    def __init__(self, 
                 state_dim, 
                 action_dim=2, 
                 hidden_dim=256, 
                 lr=3e-4, 
                 gamma=0.99,
                 tau=0.005,
                 policy_noise=0.2,
                 noise_clip=0.5,
                 policy_freq=2,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.action_dim = action_dim
        
        # 初始化Actor网络
        self.actor = LowLevelActor(state_dim, hidden_dim, action_dim).to(device)
        self.actor_target = LowLevelActor(state_dim, hidden_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        
        # 初始化Critic网络
        self.critic = LowLevelCritic(state_dim, hidden_dim, action_dim).to(device)
        self.critic_target = LowLevelCritic(state_dim, hidden_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        
        # 训练步数计数器
        self.total_it = 0
    
    def select_action(self, state, add_noise=True):
        """
        选择动作
        
        参数:
            state: 当前状态
            add_noise: 是否添加噪声（用于探索）
            
        返回:
            action: 选择的动作
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action = self.actor(state_tensor).cpu().numpy().flatten()
            
            if add_noise:
                noise = np.random.normal(0, 0.1, size=self.action_dim)
                action = np.clip(action + noise, -1.0, 1.0)
            
            return action
    
    def train(self, replay_buffer, batch_size=256):
        """
        训练TD3控制器
        
        参数:
            replay_buffer: 经验回放缓冲区
            batch_size: 批次大小
        """
        self.total_it += 1
        
        # 从缓冲区采样
        state, action, next_state, reward, done = replay_buffer.sample(batch_size)
        
        # 转换为tensor
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        done = torch.FloatTensor(done).to(self.device)
        
        # Critic更新
        with torch.no_grad():
            # 选择下一个动作并添加目标策略噪声
            noise = torch.FloatTensor(action.shape).data.normal_(0, self.policy_noise).to(self.device)
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
            next_action = torch.clamp(self.actor_target(next_state) + noise, -1, 1)
            
            # 目标Q值
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + (1.0 - done) * self.gamma * target_q
        
        # 当前Q值
        current_q1, current_q2 = self.critic(state, action)
        
        # Critic损失
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # 更新Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # 延迟策略更新
        if self.total_it % self.policy_freq == 0:
            # Actor损失
            actor_loss = -self.critic.q1_value(state, self.actor(state)).mean()
            
            # 更新Actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # 软更新目标网络
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic, self.critic_target)
    
    def _soft_update(self, local_model, target_model):
        """目标网络的软更新"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
    
    def save(self, path):
        """保存模型"""
        torch.save(self.actor.state_dict(), f"{path}_actor.pth")
        torch.save(self.critic.state_dict(), f"{path}_critic.pth")
    
    def load(self, path):
        """加载模型"""
        self.actor.load_state_dict(torch.load(f"{path}_actor.pth"))
        self.critic.load_state_dict(torch.load(f"{path}_critic.pth"))
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())