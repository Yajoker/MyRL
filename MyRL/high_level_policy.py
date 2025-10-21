"""高层策略网络 - 生成子目标和导航模式"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class HighLevelPolicy(nn.Module):
    """
    高层策略网络 - 负责在触发时生成子目标和导航模式
    使用PPO算法实现策略学习
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        """
        初始化高层策略网络
        
        参数:
            state_dim: 状态向量维度
            action_dim: 动作向量维度（对于子目标生成，通常是2或3）
            hidden_dim: 隐藏层维度
        """
        super(HighLevelPolicy, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 策略网络：输出动作的均值和标准差
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
        # 值函数网络：估计状态价值
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 导航模式分类器（如果需要）：
        # 0: 正常导航，1: 避障模式，2: 紧急避险模式
        self.mode_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3)  # 假设有3种导航模式
        )
    
    def forward(self, state):
        """
        前向传播
        
        参数:
            state: 状态向量
            
        返回:
            action_mean: 动作均值
            action_std: 动作标准差
            value: 状态价值估计
            mode_probs: 导航模式概率
        """
        # 共享特征提取
        features = self.policy_net(state)
        
        # 策略输出
        action_mean = self.mean(features)
        action_log_std = self.log_std(features)
        action_log_std = torch.clamp(action_log_std, -20, 2)  # 防止数值不稳定
        action_std = torch.exp(action_log_std)
        
        # 值函数输出
        value = self.value_net(state)
        
        # 导航模式输出
        mode_logits = self.mode_net(features)
        mode_probs = F.softmax(mode_logits, dim=-1)
        
        return action_mean, action_std, value, mode_probs
    
    def get_action(self, state, deterministic=False):
        """
        生成子目标动作
        
        参数:
            state: 状态向量
            deterministic: 是否使用确定性策略
            
        返回:
            action: 采样的动作
            log_prob: 动作的对数概率
            mode: 选择的导航模式
        """
        with torch.no_grad():
            # 前向传播
            action_mean, action_std, _, mode_probs = self.forward(state)
            
            # 动作采样
            if deterministic:
                action = action_mean
            else:
                normal = torch.distributions.Normal(action_mean, action_std)
                action = normal.sample()
                
            # 计算对数概率
            log_prob = self._compute_log_prob(action, action_mean, action_std)
            
            # 选择导航模式
            mode = torch.argmax(mode_probs, dim=-1)
            
            return action, log_prob, mode
    
    def evaluate_actions(self, states, actions):
        """
        评估给定动作的值
        
        参数:
            states: 状态批次
            actions: 动作批次
            
        返回:
            values: 值函数估计
            action_log_probs: 动作的对数概率
            entropy: 策略熵
            modes: 导航模式
        """
        # 前向传播
        action_mean, action_std, values, mode_probs = self.forward(states)
        
        # 计算动作的对数概率
        action_log_probs = self._compute_log_prob(actions, action_mean, action_std)
        
        # 计算策略熵
        normal = torch.distributions.Normal(action_mean, action_std)
        entropy = normal.entropy().sum(1, keepdim=True)
        
        # 选择导航模式
        modes = torch.argmax(mode_probs, dim=-1)
        
        return values, action_log_probs, entropy, modes
    
    def _compute_log_prob(self, actions, mean, std):
        """
        计算动作的对数概率
        
        参数:
            actions: 动作批次
            mean: 动作均值
            std: 动作标准差
            
        返回:
            log_prob: 动作的对数概率
        """
        normal = torch.distributions.Normal(mean, std)
        return normal.log_prob(actions).sum(1, keepdim=True)

class HighLevelController:
    """
    高层控制器 - 管理子目标生成和导航模式选择
    """
    def __init__(self, state_dim, trigger, policy_network=None):
        """
        初始化高层控制器
        
        参数:
            state_dim: 状态向量维度
            trigger: 事件触发器实例
            policy_network: 高层策略网络（如果为None则创建新的）
        """
        self.state_dim = state_dim
        self.trigger = trigger
        
        # 创建或使用提供的策略网络
        if policy_network is None:
            self.policy = HighLevelPolicy(state_dim, action_dim=2)  # 子目标为(x, y)
        else:
            self.policy = policy_network
        
        # 当前子目标和导航模式
        self.current_subgoal = None
        self.current_mode = 0  # 默认为正常导航模式
        
        # 统计信息
        self.trigger_count = 0
        self.subgoal_history = []
    
    def select_subgoal(self, state, goal, obstacles=None, scan_data=None, local_path=None):
        """
        生成或选择子目标
        
        参数:
            state: 机器人当前状态
            goal: 全局目标
            obstacles: 障碍物列表
            scan_data: 激光雷达数据
            local_path: 局部路径
            
        返回:
            subgoal: 选择的子目标
            mode: 导航模式
            triggered: 是否触发了更新
        """
        # 首先检查是否触发高层决策
        triggered, reason = self.trigger.check_trigger(
            state, goal, self.current_subgoal, obstacles, scan_data, local_path
        )
        
        if triggered or self.current_subgoal is None:
            # 触发了高层决策，生成新的子目标
            self.trigger_count += 1
            
            # 将状态转换为tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            # 使用策略网络生成子目标偏移
            with torch.no_grad():
                action, _, mode = self.policy.get_action(state_tensor, deterministic=True)
                action = action.squeeze(0).numpy()
                mode = mode.item()
            
            # 根据环境和策略网络输出生成实际子目标
            new_subgoal = self._generate_subgoal(state, goal, action, local_path)
            
            # 更新当前子目标和导航模式
            self.current_subgoal = new_subgoal
            self.current_mode = mode
            
            # 记录子目标历史
            self.subgoal_history.append((self.current_subgoal, reason))
            if len(self.subgoal_history) > 100:
                self.subgoal_history.pop(0)
            
            return new_subgoal, mode, True
        
        # 未触发，继续使用当前子目标
        return self.current_subgoal, self.current_mode, False
    
    def _generate_subgoal(self, state, goal, action_offset, local_path=None):
        """
        基于策略网络输出和环境信息生成子目标
        
        参数:
            state: 机器人状态
            goal: 全局目标
            action_offset: 策略网络输出的偏移量
            local_path: 局部路径
            
        返回:
            subgoal: 生成的子目标坐标
        """
        robot_pos = np.array(state[:2])
        goal_pos = np.array(goal)
        
        # 计算到目标的距离
        dist_to_goal = np.linalg.norm(goal_pos - robot_pos)
        
        if dist_to_goal < 1.0:
            # 如果接近目标，直接使用目标作为子目标
            return goal_pos
        
        if local_path is not None and len(local_path) > 1:
            # 使用局部路径点作为参考
            path_idx = min(3, len(local_path) - 1)  # 使用第4个点或最后一个点
            base_subgoal = np.array(local_path[path_idx])
        else:
            # 如果没有局部路径，在机器人与目标之间创建子目标
            direction = (goal_pos - robot_pos) / dist_to_goal
            dist = min(2.0, dist_to_goal * 0.7)  # 子目标距离为到目标距离的70%，最大2米
            base_subgoal = robot_pos + direction * dist
        
        # 应用策略网络生成的偏移量，调整子目标
        # 将偏移量缩放到合理范围（例如±0.5米）
        offset_scale = 0.5
        offset = action_offset * offset_scale
        
        # 应用偏移量
        subgoal = base_subgoal + offset
        
        # 确保子目标在合理范围内
        # 防止子目标太远或太近
        vec_to_subgoal = subgoal - robot_pos
        dist_to_subgoal = np.linalg.norm(vec_to_subgoal)
        
        if dist_to_subgoal > 3.0:  # 限制最大距离为3米
            vec_to_subgoal = vec_to_subgoal / dist_to_subgoal * 3.0
            subgoal = robot_pos + vec_to_subgoal
        elif dist_to_subgoal < 0.5:  # 确保最小距离为0.5米
            if dist_to_subgoal > 0:
                vec_to_subgoal = vec_to_subgoal / dist_to_subgoal * 0.5
                subgoal = robot_pos + vec_to_subgoal
            else:
                # 继续前面的HighLevelController类实现
                subgoal = robot_pos + np.array([0.5, 0])  # 默认前进0.5米
        
        return subgoal
    
    def reset(self):
        """重置控制器状态"""
        self.current_subgoal = None
        self.current_mode = 0
        self.subgoal_history = []