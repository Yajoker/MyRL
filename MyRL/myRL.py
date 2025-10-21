"""主代理类 - 集成所有组件并提供统一接口"""

import numpy as np
import torch
import time
from collections import deque

from models.global_planner import GlobalPathPlanner
from models.high_level_trigger import MultiSourceTrigger
from models.high_level_policy import HighLevelController
from models.low_level_policy import TD3Controller
from models.safety_layer import LyapunovSafetyLayer
from models.perception import LidarProcessor, HistoryEncoder
from utils.buffer import HierarchicalReplayBuffer
from utils.curriculum import CurriculumScheduler

class ETHSRLAgent:
    """
    事件触发分层安全强化学习代理
    """
    def __init__(self, 
                 state_dim, 
                 action_dim=2, 
                 config=None,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        初始化ETHSRL代理
        
        参数:
            state_dim: 状态向量维度
            action_dim: 动作向量维度
            config: 配置参数
            device: 计算设备
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config if config else {}
        self.device = device
        
        # 初始化全局路径规划器（战略层）
        self.global_planner = GlobalPathPlanner(
            resolution=self.config.get('PLANNER_RESOLUTION', 0.1)
        )
        
        # 初始化高层触发器
        self.high_trigger = MultiSourceTrigger(
            risk_threshold=self.config.get('RISK_THRESHOLD', 0.7),
            proximity_threshold=self.config.get('PROXIMITY_THRESHOLD', 0.5),
            heading_threshold=self.config.get('HEADING_THRESHOLD', 0.3),
            subgoal_check_dist=self.config.get('SUBGOAL_CHECK_DIST', 2.0),
            min_trigger_interval=self.config.get('MIN_TRIGGER_INTERVAL', 1.0),
            env_complexity_factor=self.config.get('ENV_COMPLEXITY_FACTOR', 0.3)
        )
        
        # 初始化高层控制器（战术层）
        self.high_controller = HighLevelController(
            state_dim=state_dim,
            trigger=self.high_trigger
        )
        
        # 初始化低层控制器（执行层）
        self.low_controller = TD3Controller(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=self.config.get('LOW_LEVEL_HIDDEN_DIM', 256),
            lr=self.config.get('LR_ACTOR', 3e-4),
            gamma=self.config.get('GAMMA', 0.99),
            device=device
        )
        
        # 初始化安全层
        self.safety_layer = LyapunovSafetyLayer(
            k_d=self.config.get('K_D', 1.0),
            k_theta=self.config.get('K_THETA', 1.0),
            k_path=self.config.get('K_PATH', 0.5),
            v_threshold=self.config.get('V_THRESHOLD', 0.5),
            epsilon=self.config.get('EPSILON', 0.01),
            blend_factor=self.config.get('BLEND_FACTOR', 10),
            blend_threshold=self.config.get('BLEND_THRESHOLD', 0.2)
        )
        
        # 初始化感知处理器
        self.lidar_processor = LidarProcessor(
            grid_size=self.config.get('GRID_SIZE', 64),
            resolution=self.config.get('RESOLUTION', 0.1),
            max_range=self.config.get('MAX_RANGE', 10.0)
        )
        
        # 初始化历史编码器
        self.history_encoder = HistoryEncoder(
            observation_dim=state_dim,
            action_dim=action_dim,
            history_length=self.config.get('HISTORY_LENGTH', 8)
        )
        
        # 初始化经验回放缓冲区
        self.replay_buffer = HierarchicalReplayBuffer(
            easy_capacity=self.config.get('EASY_BUFFER_SIZE', 10000),
            medium_capacity=self.config.get('MEDIUM_BUFFER_SIZE', 20000),
            hard_capacity=self.config.get('HARD_BUFFER_SIZE', 30000),
            alpha=self.config.get('ALPHA', 0.6),
            beta=self.config.get('BETA', 0.4),
            beta_increment=self.config.get('BETA_INCREMENT', 0.001)
        )
        
        # 初始化课程调度器
        self.curriculum = CurriculumScheduler(
            total_epochs=self.config.get('TOTAL_EPOCHS', 3000),
            stage1_ratio=self.config.get('STAGE1_RATIO', 0.3),
            stage2_ratio=self.config.get('STAGE2_RATIO', 0.7)
        )
        
        # 运行时状态
        self.current_state = None
        self.current_goal = None
        self.current_subgoal = None
        self.local_path = None
        self.global_path = None
        self.obstacles = None
        self.last_action = None
        self.grid_map = None
        
        # 统计信息
        self.episode_reward = 0
        self.episode_steps = 0
        self.total_steps = 0
        self.high_trigger_count = 0
        self.safety_trigger_count = 0
        self.episode_history = []
        self.metrics = {
            'rewards': [],
            'success_rate': [],
            'collision_rate': [],
            'high_triggers': [],
            'safety_triggers': []
        }
    
    def select_action(self, state, goal, obstacles=None, scan_data=None, deterministic=False):
        """
        选择动作 - 主决策函数
        
        参数:
            state: 当前状态
            goal: 目标位置
            obstacles: 障碍物列表 (可选)
            scan_data: 激光扫描数据 (可选)
            deterministic: 是否使用确定性策略
            
        返回:
            action: 选择的动作
            info: 附加信息
        """
        # 更新当前状态和目标
        self.current_state = state
        self.current_goal = goal
        self.obstacles = obstacles
        
        # 处理感知输入
        if scan_data is not None:
            grid_map = self.lidar_processor.process_scan(scan_data, state)
            self.grid_map = grid_map
        
        # 更新历史编码器
        if self.last_action is not None:
            self.history_encoder.update(state, self.last_action)
        else:
            self.history_encoder.update(state)
        
        # 如果全局路径不存在或需要重规划，进行规划
        if self.global_path is None or self.global_planner.need_replan(
            self.config.get('REPLAN_THRESHOLD', 5)
        ):
            try:
                self.global_path = self.global_planner.plan(self.grid_map, [0, 0], goal)
            except ValueError:
                # 如果规划失败，使用直线路径
                self.global_path = [goal]
        
        # 获取局部路径窗口
        self.local_path = self.global_planner.get_local_path_window(
            state[:2],
            self.config.get('LOCAL_WINDOW_SIZE', 10)
        )
        
        # 高层触发检查和子目标选择
        subgoal, nav_mode, triggered = self.high_controller.select_subgoal(
            state, goal, obstacles, scan_data, self.local_path
        )
        
        if triggered:
            self.high_trigger_count += 1
            
        # 保存当前子目标
        self.current_subgoal = subgoal
        
        # 低层策略生成原始动作
        raw_action = self.low_controller.select_action(state, add_noise=not deterministic)
        
        # 安全层检查
        if self.curriculum.should_use_safety_layer():
            is_safe, safe_action, safety_info = self.safety_layer.check_safety(
                state, subgoal, raw_action, self.local_path
            )
            
            if not is_safe:
                self.safety_trigger_count += 1
                action = safe_action
                safety_triggered = True
            else:
                action = raw_action
                safety_triggered = False
        else:
            action = raw_action
            safety_triggered = False
        
        # 更新last_action
        self.last_action = action
        
        # 返回选择的动作和附加信息
        info = {
            'subgoal': subgoal,
            'nav_mode': nav_mode,
            'high_triggered': triggered,
            'safety_triggered': safety_triggered,
            'raw_action': raw_action
        }
        
        return action, info
    
    def train(self, batch_size=64):
        """
        训练代理
        
        参数:
            batch_size: 批次大小
            
        返回:
            训练信息
        """
        if len(self.replay_buffer) < batch_size:
            return {'status': 'buffer_too_small'}
        
        # 从经验回放缓冲区采样批次
        batch, indices, weights = self.replay_buffer.sample(batch_size)
        
        if batch is None:
            return {'status': 'no_samples'}
        
        # 根据课程阶段决定训练哪些组件
        train_info = {}
        
        # 训练低层控制器
        if self.curriculum.should_train_low_level():
            self.low_controller.train(self.replay_buffer, batch_size)
            train_info['low_level_trained'] = True
        
        # 更新经验优先级
        priorities = np.ones(len(indices['easy']) + len(indices['medium']) + len(indices['hard']))
        self.replay_buffer.update_priorities(indices, priorities)
        
        return {
            'status': 'trained',
            **train_info
        }
    
    def store_experience(self, state, action, reward, done, next_state, info=None):
        """
        存储经验到回放缓冲区
        
        参数:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            done: 是否结束
            next_state: 下一状态
            info: 附加信息
        """
        # 提取安全触发信息
        safety_triggered = False
        if info and 'safety_triggered' in info:
            safety_triggered = info['safety_triggered']
        
        # 存储经验
        self.replay_buffer.add(state, action, next_state, reward, done, None, safety_triggered)
        
        # 更新统计信息
        self.episode_reward += reward
        self.episode_steps += 1
        self.total_steps += 1
        
        if done:
            # 记录本轮结束信息
            episode_info = {
                'reward': self.episode_reward,
                'steps': self.episode_steps,
                'high_triggers': self.high_trigger_count,
                'safety_triggers': self.safety_trigger_count,
                'success': reward > 0  # 假设正奖励表示成功
            }
            
            self.episode_history.append(episode_info)
            
            # 更新指标
            self.metrics['rewards'].append(self.episode_reward)
            
            # 重置回合计数器
            self.episode_reward = 0
            self.episode_steps = 0
            self.high_trigger_count = 0
            self.safety_trigger_count = 0
    
    def update_curriculum(self, epoch):
        """
        更新课程学习阶段
        
        参数:
            epoch: 当前训练轮数
        """
        self.curriculum.update(epoch)
        
        # 根据课程调整参数
        self.high_trigger.risk_threshold = self.curriculum.get_risk_threshold()
    
    def save(self, path):
        """
        保存模型
        
        参数:
            path: 保存路径
        """
        torch.save({
            'high_policy': self.high_controller.policy.state_dict(),
            'low_policy': self.low_controller.actor.state_dict(),
            'low_critic': self.low_controller.critic.state_dict()
        }, path)
        
        print(f"Model saved to {path}")
    
    def load(self, path):
        """
        加载模型
        
        参数:
            path: 加载路径
        """
        checkpoint = torch.load(path)
        
        self.high_controller.policy.load_state_dict(checkpoint['high_policy'])
        self.low_controller.actor.load_state_dict(checkpoint['low_policy'])
        self.low_controller.critic.load_state_dict(checkpoint['low_critic'])
        
        # 更新目标网络
        self.low_controller.actor_target.load_state_dict(self.low_controller.actor.state_dict())
        self.low_controller.critic_target.load_state_dict(self.low_controller.critic.state_dict())
        
        print(f"Model loaded from {path}")
    
    def reset(self):
        """重置代理状态"""
        # 重置路径规划器
        self.global_planner.reset()
        
        # 重置高层控制器
        self.high_controller.reset()
        
        # 重置安全层
        self.safety_layer.reset()
        
        # 重置历史编码器
        self.history_encoder = HistoryEncoder(
            observation_dim=self.state_dim,
            action_dim=self.action_dim,
            history_length=self.config.get('HISTORY_LENGTH', 8)
        )
        
        # 重置运行时状态
        self.current_state = None
        self.current_goal = None
        self.current_subgoal = None
        self.local_path = None
        self.global_path = None
        self.obstacles = None
        self.last_action = None
        self.grid_map = None
        
        # 重置回合统计
        self.episode_reward = 0
        self.episode_steps = 0
        self.high_trigger_count = 0
        self.safety_trigger_count = 0
    
    def get_stats(self):
        """
        获取代理统计信息
        
        返回:
            stats: 统计信息字典
        """
        # 计算成功率
        success_rate = 0
        if self.episode_history:
            success_rate = sum(1 for ep in self.episode_history if ep['success']) / len(self.episode_history)
        
        # 计算触发频率
        avg_high_triggers = 0
        avg_safety_triggers = 0
        if self.episode_history:
            avg_high_triggers = sum(ep['high_triggers'] for ep in self.episode_history) / len(self.episode_history)
            avg_safety_triggers = sum(ep['safety_triggers'] for ep in self.episode_history) / len(self.episode_history)
        
        return {
            'success_rate': success_rate,
            'high_trigger_rate': avg_high_triggers,
            'safety_trigger_rate': avg_safety_triggers,
            'total_steps': self.total_steps,
            'buffer_size': len(self.replay_buffer),
            'current_stage': self.curriculum.get_stage()
        }