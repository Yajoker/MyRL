"""分层经验回放缓冲区"""

import numpy as np
from collections import deque
import random

class PrioritizedReplayBuffer:
    """
    优先级经验回放缓冲区
    """
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        """
        初始化优先级经验回放缓冲区
        
        参数:
            capacity: 缓冲区容量
            alpha: 优先级指数，控制采样偏向程度
            beta: 重要性采样指数，用于修正偏向引入的bias
            beta_increment: beta增量，随时间增加beta
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0
    
    def add(self, state, action, next_state, reward, done):
        """
        添加经验到缓冲区
        
        参数:
            state: 当前状态
            action: 执行的动作
            next_state: 下一状态
            reward: 获得的奖励
            done: 是否结束
        """
        # 创建经验元组
        experience = (state, action, next_state, reward, done)
        
        # 如果缓冲区未满，扩展缓冲区
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            # 替换旧经验
            self.buffer[self.position] = experience
        
        # 分配最大优先级给新经验
        self.priorities[self.position] = self.max_priority
        
        # 更新位置指针
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        """
        按优先级采样经验批次
        
        参数:
            batch_size: 批次大小
            
        返回:
            batch: 经验批次 (state, action, next_state, reward, done)
            indices: 采样的索引
            weights: 重要性采样权重
        """
        # 确保缓冲区不为空
        n_samples = min(len(self.buffer), batch_size)
        if n_samples == 0:
            return None, None, None
        
        # 更新beta值
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # 计算采样概率
        priorities = self.priorities[:len(self.buffer)]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # 按概率采样
        indices = np.random.choice(len(self.buffer), n_samples, p=probabilities, replace=False)
        
        # 计算重要性采样权重
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()  # 归一化
        weights = np.array(weights, dtype=np.float32)
        
        # 提取经验批次
        batch = [self.buffer[idx] for idx in indices]
        
        # 解包批次
        states, actions, next_states, rewards, dones = zip(*batch)
        
        return (np.array(states), np.array(actions), np.array(next_states), 
                np.array(rewards).reshape(-1, 1), np.array(dones).reshape(-1, 1)), indices, weights
    
    def update_priorities(self, indices, priorities):
        """
        更新经验优先级
        
        参数:
            indices: 经验索引
            priorities: 新优先级
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
        
        # 更新最大优先级
        self.max_priority = max(self.max_priority, priorities.max())
    
    def __len__(self):
        return len(self.buffer)

class HierarchicalReplayBuffer:
    """
    分层经验回放缓冲区，支持课程学习
    将经验分为简单、中等和困难三个级别
    """
    def __init__(self, 
                 easy_capacity=10000, 
                 medium_capacity=20000, 
                 hard_capacity=30000,
                 alpha=0.6, 
                 beta=0.4, 
                 beta_increment=0.001,
                 safety_weight=0.5):
        """
        初始化分层经验回放缓冲区
        
        参数:
            easy_capacity: 简单经验的容量
            medium_capacity: 中等经验的容量
            hard_capacity: 困难经验的容量
            alpha: 优先级指数
            beta: 重要性采样指数
            beta_increment: beta增量
            safety_weight: 安全干预样本权重
        """
        self.easy_buffer = PrioritizedReplayBuffer(easy_capacity, alpha, beta, beta_increment)
        self.medium_buffer = PrioritizedReplayBuffer(medium_capacity, alpha, beta, beta_increment)
        self.hard_buffer = PrioritizedReplayBuffer(hard_capacity, alpha, beta, beta_increment)
        
        self.safety_weight = safety_weight
        self.training_step = 0
        
        # 初始采样比例（随训练进行调整）
        self.easy_ratio = 0.7
        self.medium_ratio = 0.2
        self.hard_ratio = 0.1
    
    def add(self, state, action, next_state, reward, done, difficulty=None, safety_triggered=False):
        """
        添加经验到适当的缓冲区
        
        参数:
            state: 当前状态
            action: 执行的动作
            next_state: 下一状态
            reward: 获得的奖励
            done: 是否结束
            difficulty: 经验难度 ('easy', 'medium', 'hard')
            safety_triggered: 安全层是否触发
        """
        # 如果未指定难度，根据奖励和安全触发判断
        if difficulty is None:
            if done and reward < 0:  # 失败
                difficulty = 'hard'
            elif safety_triggered:   # 安全干预
                difficulty = 'medium'
            else:                    # 正常状态
                difficulty = 'easy'
        
        # 添加到适当的缓冲区
        if difficulty == 'easy':
            self.easy_buffer.add(state, action, next_state, reward, done)
        elif difficulty == 'medium':
            self.medium_buffer.add(state, action, next_state, reward, done)
        elif difficulty == 'hard':
            self.hard_buffer.add(state, action, next_state, reward, done)
        else:
            # 默认为中等难度
            self.medium_buffer.add(state, action, next_state, reward, done)
    
    def sample(self, batch_size):
        """
        从三个缓冲区采样批次
        
        参数:
            batch_size: 总批次大小
            
        返回:
            batch: 经验批次
            indices: 样本索引字典
            weights: 重要性采样权重
        """
        self.training_step += 1
        
        # 更新采样比例（课程学习）
        self._update_sampling_ratio()
        
        # 计算各缓冲区的采样数量
        easy_size = int(batch_size * self.easy_ratio)
        medium_size = int(batch_size * self.medium_ratio)
        hard_size = batch_size - easy_size - medium_size
        
        # 初始化结果
        all_samples = []
        all_weights = []
        indices = {'easy': [], 'medium': [], 'hard': []}
        
        # 从简单缓冲区采样
        if easy_size > 0 and len(self.easy_buffer) > 0:
            easy_batch, easy_indices, easy_weights = self.easy_buffer.sample(easy_size)
            if easy_batch is not None:
                all_samples.append(easy_batch)
                all_weights.extend(easy_weights)
                indices['easy'] = easy_indices
        
        # 从中等缓冲区采样
        if medium_size > 0 and len(self.medium_buffer) > 0:
            medium_batch, medium_indices, medium_weights = self.medium_buffer.sample(medium_size)
            if medium_batch is not None:
                all_samples.append(medium_batch)
                all_weights.extend(medium_weights)
                indices['medium'] = medium_indices
        
        # 从困难缓冲区采样
        if hard_size > 0 and len(self.hard_buffer) > 0:
            hard_batch, hard_indices, hard_weights = self.hard_buffer.sample(hard_size)
            if hard_batch is not None:
                all_samples.append(hard_batch)
                all_weights.extend(hard_weights * 1.5)  # 给困难样本更高权重
                indices['hard'] = hard_indices
        
        # 合并样本
        if not all_samples:
            return None, None, None
        
        states = np.concatenate([s[0] for s in all_samples])
        actions = np.concatenate([s[1] for s in all_samples])
        next_states = np.concatenate([s[2] for s in all_samples])
        rewards = np.concatenate([s[3] for s in all_samples])
        dones = np.concatenate([s[4] for s in all_samples])
        
        batch = (states, actions, next_states, rewards, dones)
        weights = np.array(all_weights)
        
        return batch, indices, weights
    
    def update_priorities(self, indices, priorities):
        """
        更新各缓冲区的优先级
        
        参数:
            indices: 包含各缓冲区索引的字典
            priorities: 优先级数组
        """
        offset = 0
        
        # 更新简单缓冲区
        if 'easy' in indices and indices['easy']:
            n_easy = len(indices['easy'])
            self.easy_buffer.update_priorities(indices['easy'], priorities[offset:offset+n_easy])
            offset += n_easy
        
        # 更新中等缓冲区
        if 'medium' in indices and indices['medium']:
            n_medium = len(indices['medium'])
            self.medium_buffer.update_priorities(indices['medium'], priorities[offset:offset+n_medium])
            offset += n_medium
        
        # 更新困难缓冲区
        if 'hard' in indices and indices['hard']:
            n_hard = len(indices['hard'])
            self.hard_buffer.update_priorities(indices['hard'], priorities[offset:offset+n_hard])
    
    def _update_sampling_ratio(self):
        """更新采样比例，实现课程学习"""
        # 随训练进展，增加困难样本比例
        progress = min(1.0, self.training_step / 100000)  # 假设10万步为完整课程
        
        # 调整采样比例
        self.easy_ratio = max(0.1, 0.7 - 0.6 * progress)
        self.hard_ratio = min(0.7, 0.1 + 0.6 * progress)
        self.medium_ratio = 1.0 - self.easy_ratio - self.hard_ratio
    
    def __len__(self):
        """缓冲区总长度"""
        return len(self.easy_buffer) + len(self.medium_buffer) + len(self.hard_buffer)