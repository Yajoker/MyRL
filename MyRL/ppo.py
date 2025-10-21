"""PPO (Proximal Policy Optimization) 算法实现"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

class PPOActor(nn.Module):
    """PPO Actor网络"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(PPOActor, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 均值和方差输出
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))
    
    def forward(self, x):
        x = self.network(x)
        mean = self.mean(x)
        std = torch.exp(self.log_std).expand_as(mean)
        
        return mean, std
    
    def get_action(self, state, deterministic=False):
        """选择动作"""
        mean, std = self.forward(state)
        
        if deterministic:
            return mean
        
        # 使用正态分布采样
        dist = Normal(mean, std)
        action = dist.sample()
        
        # 计算对数概率
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        
        return action, log_prob
    
    def evaluate(self, state, action):
        """评估动作的对数概率和熵"""
        mean, std = self.forward(state)
        dist = Normal(mean, std)
        
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        
        return log_prob, entropy

class PPOCritic(nn.Module):
    """PPO Critic网络"""
    def __init__(self, state_dim, hidden_dim=256):
        super(PPOCritic, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.network(x)

class PPOAgent:
    """
    PPO算法代理
    """
    def __init__(self, 
                 state_dim, 
                 action_dim, 
                 hidden_dim=256,
                 lr_actor=3e-4,
                 lr_critic=1e-3,
                 gamma=0.99,
                 gae_lambda=0.95,
                 clip_ratio=0.2,
                 value_coef=0.5,
                 entropy_coef=0.01,
                 max_grad_norm=0.5,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        初始化PPO代理
        
        参数:
            state_dim: 状态维度
            action_dim: 动作维度
            hidden_dim: 隐藏层维度
            lr_actor: actor学习率
            lr_critic: critic学习率
            gamma: 折扣因子
            gae_lambda: GAE lambda参数
            clip_ratio: PPO裁剪参数
            value_coef: 值函数损失系数
            entropy_coef: 熵正则化系数
            max_grad_norm: 梯度裁剪阈值
            device: 计算设备
        """
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # 初始化网络
        self.actor = PPOActor(state_dim, action_dim, hidden_dim).to(device)
        self.critic = PPOCritic(state_dim, hidden_dim).to(device)
        
        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # 经验缓存
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.masks = []
        self.next_state = None
        self.next_value = None
    
    def select_action(self, state, deterministic=False):
        """
        选择动作
        
        参数:
            state: 状态
            deterministic: 是否使用确定性策略
            
        返回:
            action: 选择的动作
            log_prob: 动作的对数概率
        """
        with torch.no_grad():
            if not isinstance(state, torch.Tensor):
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            else:
                state = state.to(self.device)
            
            # 计算值函数
            value = self.critic(state)
            
            # 选择动作
            action, log_prob = self.actor.get_action(state, deterministic)
            
            # 添加到经验缓存
            if not deterministic:
                self.states.append(state)
                self.actions.append(action)
                self.values.append(value)
                self.log_probs.append(log_prob)
            
            return action.cpu().numpy().flatten(), log_prob.cpu().numpy().flatten()
    
    def compute_gae(self, next_value, rewards, masks, values):
        """
        计算广义优势估计(GAE)
        """
        values = values + [next_value]
        gae = 0
        returns = []
        
        # 从后向前计算
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * masks[step] - values[step]
            gae = delta + self.gamma * self.gae_lambda * masks[step] * gae
            returns.insert(0, gae + values[step])
            
        return returns
    
    def update(self, next_state=None, next_value=None, n_epochs=10, batch_size=64):
        """
        更新PPO策略
        
        参数:
            next_state: 下一个状态
            next_value: 下一个状态的值函数
            n_epochs: 每次更新的训练轮数
            batch_size: 批次大小
            
        返回:
            policy_loss: 策略损失
            value_loss: 值函数损失
            entropy_loss: 熵损失
        """
        # 如果没有足够的经验，不进行更新
        if len(self.states) < batch_size:
            return 0, 0, 0
        
        # 如果没有提供next_value，则计算它
        if next_value is None and next_state is not None:
            with torch.no_grad():
                next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
                next_value = self.critic(next_state)
                
        # 计算优势函数
        returns = self.compute_gae(next_value, self.rewards, self.masks, self.values)
        
        # 准备训练数据
        states = torch.cat(self.states)
        actions = torch.cat(self.actions)
        returns = torch.cat(returns).detach()
        values = torch.cat(self.values).detach()
        log_probs = torch.cat(self.log_probs).detach()
        advantages = returns - values
        
        # 归一化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 清空缓存
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.masks = []
        
        # 多次训练
        policy_losses = []
        value_losses = []
        entropy_losses = []
        
        for _ in range(n_epochs):
            # 创建数据加载器
            indices = np.random.permutation(len(states))
            for start_idx in range(0, len(states), batch_size):
                # 采样批次
                idx = indices[start_idx:start_idx + batch_size]
                
                # 提取批次数据
                state_batch = states[idx]
                action_batch = actions[idx]
                advantage_batch = advantages[idx]
                return_batch = returns[idx]
                old_log_prob_batch = log_probs[idx]
                
                # 计算新的动作概率和熵
                new_log_prob, entropy = self.actor.evaluate(state_batch, action_batch)
                
                # 计算ratio
                ratio = torch.exp(new_log_prob - old_log_prob_batch)
                
                # PPO裁剪目标
                surr1 = ratio * advantage_batch
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantage_batch
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 值函数损失
                value_pred = self.critic(state_batch)
                value_loss = F.mse_loss(value_pred, return_batch)
                
                # 熵正则化
                entropy_loss = -entropy.mean()
                
                # 总损失
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # 更新网络
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
        
        # 返回平均损失
        return np.mean(policy_losses), np.mean(value_losses), np.mean(entropy_losses)
    
    def add_experience(self, reward, done):
        """
        添加奖励和终止标志到经验缓存
        
        参数:
            reward: 奖励
            done: 终止标志
        """
        self.rewards.append(torch.tensor([reward], device=self.device))
        self.masks.append(torch.tensor([1.0 - float(done)], device=self.device))
    
    def save(self, path):
        """
        保存模型
        
        参数:
            path: 保存路径
        """
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict()
        }, path)
        
    def load(self, path):
        """
        加载模型
        
        参数:
            path: 加载路径
        """
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])