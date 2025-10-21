"""Lyapunov安全层 - 确保控制动作的安全性"""

import numpy as np

class LyapunovSafetyLayer:
    """
    几何Lyapunov安全层 - 提供安全控制动作修正
    """
    def __init__(
        self,
        k_d=1.0,
        k_theta=1.0,
        k_path=0.5,
        v_threshold=0.5,
        epsilon=0.01,
        blend_factor=10,
        blend_threshold=0.2
    ):
        """
        初始化安全层
        
        参数:
            k_d: 距离增益
            k_theta: 角度增益
            k_path: 路径跟踪增益
            v_threshold: Lyapunov值阈值，超过此值触发安全修正
            epsilon: Lyapunov导数的最小减小量
            blend_factor: 平滑混合因子（控制sigmoid陡度）
            blend_threshold: 混合阈值（V值的中点）
        """
        self.k_d = k_d
        self.k_theta = k_theta
        self.k_path = k_path
        self.v_threshold = v_threshold
        self.epsilon = epsilon
        self.blend_factor = blend_factor
        self.blend_threshold = blend_threshold
        
        # 用于追踪Lyapunov值的变化
        self.prev_v = None
        
        # 安全统计
        self.trigger_count = 0
        self.intervention_history = []
    
    def check_safety(self, state, subgoal, action, path=None, dt=0.1):
        """
        检查动作是否满足安全条件，如不满足则修正
        
        参数:
            state: 机器人状态 [x, y, theta, v, omega]
            subgoal: 子目标位置 [x, y]
            action: 原始控制动作 [v, omega]
            path: 局部路径
            dt: 时间步长，用于计算Lyapunov导数
            
        返回:
            is_safe: 原始动作是否安全
            safe_action: 安全动作（如果不安全则修正）
            info: 安全检查的附加信息
        """
        # 计算当前Lyapunov值
        v_current = self._compute_lyapunov(state, subgoal, path)
        
        # 如果是首次调用，初始化prev_v
        if self.prev_v is None:
            self.prev_v = v_current
            return True, action, {"V": v_current, "trigger": False}
        
        # 预测下一状态的Lyapunov值
        next_state = self._predict_next_state(state, action, dt)
        v_next = self._compute_lyapunov(next_state, subgoal, path)
        
        # 计算Lyapunov导数
        v_dot = (v_next - v_current) / dt
        
        # 安全条件：
        # 1. V值小于阈值，且
        # 2. V的导数为负（V在减小）或减小速度足够快
        is_safe = (v_current < self.v_threshold) and (v_dot < -self.epsilon)
        
        # 如果不安全，修正动作
        if not is_safe:
            self.trigger_count += 1
            safe_action = self._modify_action(state, subgoal, action, path)
            
            # 记录干预
            self.intervention_history.append({
                "V": v_current,
                "V_dot": v_dot,
                "original_action": action,
                "safe_action": safe_action
            })
            
            if len(self.intervention_history) > 100:
                self.intervention_history.pop(0)
            
            info = {
                "V": v_current,
                "V_dot": v_dot,
                "trigger": True,
                "blend": self._compute_blend_factor(v_current)
            }
            
            return False, safe_action, info
        
        # 更新prev_v
        self.prev_v = v_current
        
        return True, action, {"V": v_current, "V_dot": v_dot, "trigger": False}
    
    def _compute_lyapunov(self, state, subgoal, path=None):
        """
        计算Lyapunov函数值 V = 0.5 * (k_d * d^2 + k_theta * theta^2 + k_path * e_path^2)
        
        参数:
            state: 机器人状态 [x, y, theta, v, omega]
            subgoal: 子目标位置 [x, y]
            path: 局部路径
            
        返回:
            V: Lyapunov值
        """
        # 提取机器人位置和朝向
        robot_pos = np.array(state[:2])
        robot_theta = state[2]
        
        # 计算到子目标的距离
        subgoal_pos = np.array(subgoal)
        d = np.linalg.norm(subgoal_pos - robot_pos)
        
        # 计算到子目标的朝向误差
        goal_dir = subgoal_pos - robot_pos
        if np.linalg.norm(goal_dir) > 1e-6:
            goal_theta = np.arctan2(goal_dir[1], goal_dir[0])
        else:
            goal_theta = robot_theta  # 如果距离太近，使用当前朝向
        
        # 朝向误差，归一化到[-pi, pi]
        theta_err = goal_theta - robot_theta
        theta_err = (theta_err + np.pi) % (2 * np.pi) - np.pi
        
        # 计算路径跟踪误差
        e_path = 0
        if path is not None and len(path) > 1:
            # 找到路径上距离机器人最近的点
            min_dist = float('inf')
            for i in range(len(path) - 1):
                p1 = np.array(path[i])
                p2 = np.array(path[i + 1])
                
                # 计算点到线段的距离
                line_vec = p2 - p1
                point_vec = robot_pos - p1
                line_len = np.linalg.norm(line_vec)
                
                if line_len < 1e-6:
                    continue
                
                # 计算投影比例
                proj_ratio = np.dot(point_vec, line_vec) / (line_len * line_len)
                proj_ratio = max(0, min(1, proj_ratio))
                
                # 计算投影点
                proj_point = p1 + proj_ratio * line_vec
                
                # 计算距离
                dist = np.linalg.norm(robot_pos - proj_point)
                if dist < min_dist:
                    min_dist = dist
            
            e_path = min_dist
        
        # 计算Lyapunov值
        V = 0.5 * (self.k_d * d**2 + self.k_theta * theta_err**2 + self.k_path * e_path**2)
        
        return V
    
    def _predict_next_state(self, state, action, dt):
        """
        简单运动学模型，预测下一状态
        
        参数:
            state: 当前状态 [x, y, theta, v, omega]
            action: 控制动作 [v, omega]
            dt: 时间步长
            
        返回:
            next_state: 预测的下一状态
        """
        x, y, theta = state[:3]
        v, omega = action
        
        # 运动学模型
        next_x = x + v * np.cos(theta) * dt
        next_y = y + v * np.sin(theta) * dt
        next_theta = theta + omega * dt
        
        # 归一化角度到[-pi, pi]
        next_theta = (next_theta + np.pi) % (2 * np.pi) - np.pi
        
        # 构建下一状态
        next_state = state.copy()
        next_state[0] = next_x
        next_state[1] = next_y
        next_state[2] = next_theta
        
        return next_state
    
    def _modify_action(self, state, subgoal, action, path=None):
        """
        几何Lyapunov修正控制律
        
        参数:
            state: 机器人状态
            subgoal: 子目标位置
            action: 原始控制动作 [v, omega]
            path: 局部路径
            
        返回:
            safe_action: 修正后的安全动作
        """
        # 提取机器人状态和子目标
        robot_pos = np.array(state[:2])
        robot_theta = state[2]
        subgoal_pos = np.array(subgoal)
        
        # 提取原始动作
        v_raw, omega_raw = action
        
        # 计算到子目标的距离和方向
        goal_dir = subgoal_pos - robot_pos
        d = np.linalg.norm(goal_dir)
        
        if d > 1e-6:
            goal_theta = np.arctan2(goal_dir[1], goal_dir[0])
        else:
            goal_theta = robot_theta
        
        # 朝向误差
        theta_err = goal_theta - robot_theta
        theta_err = (theta_err + np.pi) % (2 * np.pi) - np.pi
        
        # 路径跟踪误差和方向
        path_theta = robot_theta  # 默认值
        e_path = 0
        
        if path is not None and len(path) > 1:
            # 寻找最近路径点和方向
            # (简化版，实际应用可能需要更复杂的路径跟踪逻辑)
            min_dist = float('inf')
            closest_idx = 0
            
            for i, point in enumerate(path):
                dist = np.linalg.norm(np.array(point) - robot_pos)
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = i
            
            # 获取路径方向
            if closest_idx < len(path) - 1:
                path_vec = np.array(path[closest_idx + 1]) - np.array(path[closest_idx])
                if np.linalg.norm(path_vec) > 1e-6:
                    path_theta = np.arctan2(path_vec[1], path_vec[0])
            
            e_path = min_dist
        
        # 路径朝向误差
        path_err = path_theta - robot_theta
        path_err = (path_err + np.pi) % (2 * np.pi) - np.pi
        
        # 几何Lyapunov修正控制律
        # 1. 速度修正：距离远则减速
        v_corr = v_raw - self.k_d * d * np.cos(theta_err)
        
        # 2. 角速度修正：转向目标
        omega_corr = omega_raw - self.k_theta * theta_err
        
        # 3. 路径跟踪修正
        omega_corr -= self.k_path * path_err
        
        # 限制动作范围
        v_safe = np.clip(v_corr, 0.0, 1.0)  # 假设速度范围[0, 1]
        omega_safe = np.clip(omega_corr, -1.0, 1.0)  # 假设角速度范围[-1, 1]
        
        # 计算Lyapunov值用于混合
        v_current = self._compute_lyapunov(state, subgoal, path)
        blend = self._compute_blend_factor(v_current)
        
        # 平滑混合原始动作和安全动作
        v_final = (1 - blend) * v_raw + blend * v_safe
        omega_final = (1 - blend) * omega_raw + blend * omega_safe
        
        return np.array([v_final, omega_final])
    
    def _compute_blend_factor(self, v):
        """
        计算混合因子lambda，控制原始动作和安全动作的混合比例
        使用sigmoid函数实现平滑过渡
        
        参数:
            v: 当前Lyapunov值
            
        返回:
            blend: 混合因子[0, 1]
        """
        x = self.blend_factor * (v - self.blend_threshold)
        return 1 / (1 + np.exp(-x))
    
    def reset(self):
        """重置安全层状态"""
        self.prev_v = None
        self.trigger_count = 0
        self.intervention_history = []