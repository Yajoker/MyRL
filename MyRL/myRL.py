"""主代理类 - 集成所有组件并提供统一接口"""

import numpy as np
import torch
import time
from collections import deque

# 修复导入路径问题
from global_planner import GlobalPathPlanner  
from high_level_trigger import MultiSourceTrigger
from high_level_policy import HighLevelController
from low_level_policy import TD3Controller
from safety_layer import LyapunovSafetyLayer
from perception import LidarProcessor, HistoryEncoder
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
        
        # 添加初始化代码: 初始化last_action数组
        self._last_action = np.zeros(action_dim, dtype=np.float32)
        
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

    def prepare_state(self, latest_scan, distance, cos, sin, collision, goal, a):
        """
        将当前观测拼成固定长度的一维状态向量。
        目标：输出长度与 train.py 的 state_dim 一致（默认为 95）。
        组成：scan(88) + [distance, cos, sin, collision, goal](5) + last_action(2) = 95
        """
        max_range = float(self.config.get("MAX_RANGE", 10.0))
        scan_len = int(self.config.get("SCAN_LEN", 88))  # 让总维度匹配 95
        scan = np.asarray(latest_scan, dtype=np.float32)
        scan = np.clip(scan, 0.0, max_range) / max_range

        # 下采样/裁剪/填充到固定 scan_len
        if scan.size >= scan_len:
            k = max(1, scan.size // scan_len)
            scan_ds = scan[::k][:scan_len]
            if scan_ds.size < scan_len:
                scan_ds = np.pad(scan_ds, (0, scan_len - scan_ds.size))
        else:
            pad = scan_len - scan.size
            scan_ds = np.pad(scan, (0, pad))

        # 上一步动作
        if a is not None:
            a_arr = np.asarray(a, dtype=np.float32).reshape(-1)
            self._last_action[:min(2, a_arr.size)] = a_arr[:min(2, a_arr.size)]

        feats = np.array(
            [float(distance), float(cos), float(sin), float(bool(collision)), float(bool(goal))],
            dtype=np.float32,
        )
        state = np.concatenate([scan_ds, feats, self._last_action], axis=0).astype(np.float32)

        # 终止标志
        terminal = bool(collision) or bool(goal)
        return state, terminal

    def select_action(self, state, goal=None, scan_data=None, deterministic=False):
        """
        根据当前状态选择动作
        
        参数:
            state: 当前环境状态
            goal: 目标位置
            scan_data: 激光雷达扫描数据
            deterministic: 是否使用确定性策略(默认为False)
            
        返回:
            action: 选定的动作
            info: 额外信息
        """
        # 更新当前状态
        self.current_state = state
        if goal is not None:
            self.current_goal = goal
        
        # 处理激光扫描数据
        if scan_data is not None:
            # 更新激光雷达处理器并获取障碍物
            self.obstacles = self.lidar_processor.update(scan_data)
            # 获取占据栅格图
            self.grid_map = self.lidar_processor.get_occupancy_grid()
        
        # 首先检查全局路径是否已规划
        if self.global_path is None and self.current_goal is not None:
            # 为当前位置和目标规划全局路径
            start_pos = (state[0], state[1])  # 假设state前两个元素是x,y位置
            self.global_path = self.global_planner.plan_path(
                start_pos, 
                self.current_goal, 
                self.obstacles
            )
            print(f"已规划全局路径，从 {start_pos} 到 {self.current_goal}，路径点数: {len(self.global_path) if self.global_path else 0}")
        
        # 检查是否触发高层控制器更新子目标
        trigger_update, reason = self.high_trigger.check_trigger(
            state, 
            self.current_goal, 
            self.current_subgoal,
            self.obstacles,
            scan_data=scan_data,
            local_path=self.local_path
        )
        
        if trigger_update:
            self.high_trigger_count += 1
            self.current_subgoal, mode, triggered = self.high_controller.select_subgoal(
                state, 
                self.current_goal,
                obstacles=self.obstacles,
                scan_data=scan_data,
                local_path=self.local_path
            )
            
            # 如果路径发生变化，重新规划全局路径
            self.global_path = self.global_planner.plan_path(
                (state[0], state[1]), 
                self.current_goal, 
                self.obstacles
            )
        
        try:
            # 获取局部路径窗口
            if self.global_path is not None:
                self.local_path = self.global_planner.get_local_path_window(
                    (state[0], state[1]),
                    window_size=self.config.get('LOCAL_WINDOW_SIZE', 10)
                )
            else:
                # 如果全局路径仍然为None，创建一个简单的直线路径
                self.local_path = [(state[0], state[1]), self.current_goal]
                print("警告: 使用简单直线路径替代全局规划")
        except ValueError as e:
            # 捕获"全局路径尚未规划"错误，进行处理
            print(f"错误: {e}")
            # 尝试重新规划路径
            if self.current_goal is not None:
                self.global_path = self.global_planner.plan_path(
                    (state[0], state[1]), 
                    self.current_goal, 
                    self.obstacles
                )
                print(f"重新规划全局路径，路径点数: {len(self.global_path) if self.global_path else 0}")
                # 再次尝试获取局部窗口
                if self.global_path is not None:
                    self.local_path = self.global_planner.get_local_path_window(
                        (state[0], state[1]),
                        window_size=self.config.get('LOCAL_WINDOW_SIZE', 10)
                    )
                else:
                    # 仍然无法规划，使用简单直线路径
                    self.local_path = [(state[0], state[1]), self.current_goal]
                    print("警告: 使用简单直线路径替代全局规划")
        
        # 从低层控制器获取动作
        raw_action = self.low_controller.select_action(
            state, 
            add_noise=not deterministic  # 根据deterministic参数决定是否添加噪声
        )
        
        # 通过安全层调整动作（如果配置启用）
        safety_triggered = False
        if self.config.get('SAFETY_LAYER_ENABLED', True):
            is_safe, safe_action, safety_info = self.safety_layer.check_safety(
                state, 
                self.current_subgoal or self.current_goal,
                raw_action,
                path=self.local_path
            )
            
            if not is_safe:
                self.safety_trigger_count += 1
                safety_triggered = True
                action = safe_action
            else:
                action = raw_action
        else:
            action = raw_action
        
        # 更新历史编码器
        self.history_encoder.update(state, action)
        
        # 记住上一个动作
        self.last_action = action
        self._last_action = np.asarray(action, dtype=np.float32)
        
        # 创建返回信息
        info = {
            'high_triggered': trigger_update,
            'subgoal': self.current_subgoal,
            'local_path': self.local_path,
            'global_path': self.global_path,
            'grid_map': self.grid_map,
            'safety_triggered': safety_triggered
        }
        
        return action, info
    
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
        
        # 重置_last_action数组
        self._last_action = np.zeros(self.action_dim, dtype=np.float32)
        
        # 重置回合统计
        self.episode_reward = 0
        self.episode_steps = 0
        self.high_trigger_count = 0
        self.safety_trigger_count = 0

    def update(self, batch_size=256, iterations=1):
        """
        使用经验回放更新代理的策略
        
        参数:
            batch_size: 每次更新的样本数
            iterations: 更新迭代次数
        """
        if len(self.replay_buffer) < batch_size:
            return
        
        update_info = {
            'actor_loss': 0,
            'critic_loss': 0,
            'td_error': 0
        }
        
        # 更新低层控制器
        for _ in range(iterations):
            self.low_controller.train(self.replay_buffer, batch_size)
            
        return update_info
    
    def update_curriculum(self, epoch):
        """
        更新课程学习参数
        
        参数:
            epoch: 当前训练轮数
        """
        self.curriculum.update(epoch)
        
        # 根据当前阶段更新高层触发阈值
        self.high_trigger.risk_threshold = self.curriculum.get_risk_threshold()
        
        # 根据当前阶段调整安全层参数
        self.safety_layer.v_threshold = self.config.get('V_THRESHOLD', 0.5) * (1.0 + 0.2 * (epoch / self.curriculum.total_epochs))
    
    def store_experience(self, state, action, reward, terminal, next_state, info=None):
        """
        存储经验到回放缓冲区
        
        参数:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            terminal: 是否终止
            info: 额外信息
        """
        # 计算难度（可根据奖励、是否碰撞等确定）
        difficulty = 'easy'
        safety_triggered = info.get('safety_triggered', False) if info else False
        
        if terminal and reward < 0:  # 碰撞
            difficulty = 'hard'
        elif terminal and reward > 0:  # 成功到达
            difficulty = 'medium'
        elif safety_triggered:  # 安全干预
            difficulty = 'medium'
        
        # 添加到缓冲区
        self.replay_buffer.add(
            state, action, next_state, reward, terminal,
            difficulty=difficulty, 
            safety_triggered=safety_triggered
        )
        
        # 更新统计信息
        self.episode_reward += reward
        self.episode_steps += 1
        self.total_steps += 1
    
    def train(self, batch_size=256):
        """
        训练模型
        
        参数:
            batch_size: 批次大小
        """
        return self.update(batch_size=batch_size, iterations=1)
        
    def on_episode_end(self, success=False, collision=False):
        """
        回合结束时的处理
        
        参数:
            success: 是否成功完成任务
            collision: 是否发生碰撞
        """
        # 记录本回合统计
        self.episode_history.append({
            'reward': self.episode_reward,
            'steps': self.episode_steps,
            'success': success,
            'collision': collision,
            'high_triggers': self.high_trigger_count,
            'safety_triggers': self.safety_trigger_count
        })
        
        # 更新全局统计
        self.metrics['rewards'].append(self.episode_reward)
        
        # 计算成功率和碰撞率（使用最近100回合数据）
        recent_history = self.episode_history[-100:]
        success_rate = sum(ep['success'] for ep in recent_history) / len(recent_history)
        collision_rate = sum(ep['collision'] for ep in recent_history) / len(recent_history)
        
        self.metrics['success_rate'].append(success_rate)
        self.metrics['collision_rate'].append(collision_rate)
        self.metrics['high_triggers'].append(self.high_trigger_count)
        self.metrics['safety_triggers'].append(self.safety_trigger_count)
        
        # 为下一回合重置统计
        self.episode_reward = 0
        self.episode_steps = 0
        self.high_trigger_count = 0
        self.safety_trigger_count = 0
        
    def save(self, path):
        """
        保存代理模型
        
        参数:
            path: 保存路径
        """
        save_dict = {
            'low_controller': self.low_controller.state_dict(),
            'high_controller': self.high_controller.state_dict(),
            'total_steps': self.total_steps,
            'metrics': self.metrics
        }
        torch.save(save_dict, path)
        
    def load(self, path):
        """
        加载代理模型
        
        参数:
            path: 加载路径
        """
        checkpoint = torch.load(path)
        self.low_controller.load_state_dict(checkpoint['low_controller'])
        self.high_controller.load_state_dict(checkpoint['high_controller'])
        self.total_steps = checkpoint.get('total_steps', 0)
        self.metrics = checkpoint.get('metrics', self.metrics)
