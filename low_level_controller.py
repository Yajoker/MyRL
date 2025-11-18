"""
低层控制器模块
基于CNNTD3算法实现的机器人低层运动控制器
处理激光雷达数据和子目标信息，生成底层控制指令
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


class LowLevelActorNetwork(nn.Module):
    """引入LSTM记忆的低层Actor网络。"""

    def __init__(self, action_dim, lstm_hidden_dim: int = 128, dropout_p: float = 0.1):
        super(LowLevelActorNetwork, self).__init__()

        self.cnn1 = nn.Conv1d(1, 4, kernel_size=8, stride=4)
        self.cnn2 = nn.Conv1d(4, 8, kernel_size=8, stride=4)
        self.cnn3 = nn.Conv1d(8, 4, kernel_size=4, stride=2)

        self.subgoal_embed = nn.Linear(2, 10)
        self.action_embed = nn.Linear(2, 10)

        self.feature_dim = 36
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm = nn.LSTM(self.feature_dim, lstm_hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(p=dropout_p) if dropout_p > 0 else nn.Identity()

        self.fc1 = nn.Linear(lstm_hidden_dim, 256)
        torch.nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity="leaky_relu")
        self.fc2 = nn.Linear(256, action_dim)
        self.tanh = nn.Tanh()

    def init_hidden(self, batch_size: int, device: torch.device):
        """创建零初始化的LSTM隐状态。"""

        h0 = torch.zeros(1, batch_size, self.lstm_hidden_dim, device=device)
        c0 = torch.zeros(1, batch_size, self.lstm_hidden_dim, device=device)
        return h0, c0

    def _encode_features(self, s: torch.Tensor) -> torch.Tensor:
        """将原始状态编码为压缩特征。"""

        laser = s[:, :-4]
        subgoal = s[:, -4:-2]
        prev_act = s[:, -2:]

        laser = laser.unsqueeze(1)
        l = F.leaky_relu(self.cnn1(laser))
        l = F.leaky_relu(self.cnn2(l))
        l = F.leaky_relu(self.cnn3(l))
        l = l.flatten(start_dim=1)

        g = F.leaky_relu(self.subgoal_embed(subgoal))
        a = F.leaky_relu(self.action_embed(prev_act))
        return torch.concat((l, g, a), dim=-1)

    def forward(self, s: torch.Tensor, hidden=None, return_hidden: bool = False):
        """按序列展开Actor网络。"""

        if s.dim() == 1:
            s = s.unsqueeze(0)

        was_sequence = s.dim() == 3
        if not was_sequence:
            s = s.unsqueeze(1)

        batch_size, seq_len, _ = s.shape
        flat = s.reshape(batch_size * seq_len, -1)
        features = self._encode_features(flat).reshape(batch_size, seq_len, -1)

        if hidden is None or hidden[0].shape[1] != batch_size:
            hidden = self.init_hidden(batch_size, s.device)

        lstm_out, next_hidden = self.lstm(features, hidden)
        lstm_out = self.dropout(lstm_out)
        out = F.leaky_relu(self.fc1(lstm_out))
        actions = self.tanh(self.fc2(out))

        if not was_sequence:
            actions = actions.squeeze(1)

        if return_hidden:
            return actions, next_hidden
        return actions

class LowLevelCriticNetwork(nn.Module):
    """
    低层控制器的Critic网络
    评估状态-动作对的Q值，使用双Q网络结构减少过估计
    """

    def __init__(self, action_dim):
        """初始化Critic网络

        Args:
            action_dim: 动作空间的维度
        """
        super(LowLevelCriticNetwork, self).__init__()

        # CNN层用于处理激光雷达数据（与Actor网络相同结构）
        self.cnn1 = nn.Conv1d(1, 4, kernel_size=8, stride=4)
        self.cnn2 = nn.Conv1d(4, 8, kernel_size=8, stride=4)
        self.cnn3 = nn.Conv1d(8, 4, kernel_size=4, stride=2)

        # 子目标信息嵌入层
        self.subgoal_embed = nn.Linear(2, 10)

        # 历史动作嵌入层
        self.action_embed = nn.Linear(2, 10)

        # Q1网络结构
        self.layer_1 = nn.Linear(36, 400)  # 第一层全连接
        torch.nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="leaky_relu")
        self.layer_2_s = nn.Linear(400, 300)  # 状态分支
        torch.nn.init.kaiming_uniform_(self.layer_2_s.weight, nonlinearity="leaky_relu")
        self.layer_2_a = nn.Linear(action_dim, 300)  # 动作分支
        torch.nn.init.kaiming_uniform_(self.layer_2_a.weight, nonlinearity="leaky_relu")
        self.layer_3 = nn.Linear(300, 1)  # Q值输出
        torch.nn.init.kaiming_uniform_(self.layer_3.weight, nonlinearity="leaky_relu")

        # Q2网络结构（与Q1结构相同但参数独立）
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
        Critic网络的前向传播，计算两个Q值

        Args:
            s: 状态张量
            action: 动作张量

        Returns:
            两个Q值的元组 (Q1, Q2)
        """
        if s.dim() == 1:
            s = s.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)

        if action.device != s.device:
            action = action.to(s.device)

        # 分割状态张量的各个部分
        laser = s[:, :-4]  # 激光雷达数据
        subgoal = s[:, -4:-2]  # 子目标信息
        prev_act = s[:, -2:]  # 历史动作

        laser = laser.unsqueeze(1)  # 增加通道维度

        # 处理激光雷达数据
        l = F.leaky_relu(self.cnn1(laser))
        l = F.leaky_relu(self.cnn2(l))
        l = F.leaky_relu(self.cnn3(l))
        l = l.flatten(start_dim=1)

        # 处理子目标信息
        g = F.leaky_relu(self.subgoal_embed(subgoal))

        # 处理历史动作
        a = F.leaky_relu(self.action_embed(prev_act))

        # 拼接特征
        s = torch.concat((l, g, a), dim=-1)

        # Q1值计算
        s1 = F.leaky_relu(self.layer_1(s))  # 状态特征提取
        s1_state = self.layer_2_s(s1)
        s1_action = self.layer_2_a(action)
        s1 = F.leaky_relu(s1_state + s1_action)  # 合并并激活
        q1 = self.layer_3(s1)  # 输出Q1值

        # Q2值计算（与Q1相同结构）
        s2 = F.leaky_relu(self.layer_4(s))
        s2_state = self.layer_5_s(s2)
        s2_action = self.layer_5_a(action)
        s2 = F.leaky_relu(s2_state + s2_action)
        q2 = self.layer_6(s2)

        return q1, q2


class LowLevelController:
    """
    低层执行控制器类
    基于CNNTD3算法，处理传感器数据并生成底层控制指令
    """

    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            device,
            lr=3e-4,
            save_every=0,
            load_model=False,
            save_directory=Path("ethsrl/models/low_level"),
            model_name="low_level_controller",
            load_directory=None,
            *,
            max_lin_velocity: float = 1.0,
            max_ang_velocity: float = 1.0,
            lstm_hidden_dim: int = 128,
            lstm_dropout: float = 0.1,
    ):
        """
        初始化低层控制器

        Args:
            state_dim: 状态空间维度（激光数据 + 子目标 + 历史动作）
            action_dim: 动作空间维度（通常为2，表示[线速度, 角速度]）
            max_action: 最大动作值
            device: PyTorch设备（CPU或GPU）
            lr: 学习率
            save_every: 每N次更新保存一次模型（0表示禁用）
            load_model: 是否加载预训练模型
            save_directory: 模型保存目录
            model_name: 模型文件名
            load_directory: 模型加载目录（如果为None则使用save_directory）
        """
        self.device = device
        self.action_dim = action_dim
        self.max_action = max_action
        self.state_dim = state_dim
        self.max_lin_velocity = float(max_lin_velocity)
        self.max_ang_velocity = float(max_ang_velocity)

        # 初始化Actor网络和目标网络
        self.actor = LowLevelActorNetwork(
            action_dim, lstm_hidden_dim=lstm_hidden_dim, dropout_p=lstm_dropout
        ).to(device)
        self.actor_target = LowLevelActorNetwork(
            action_dim, lstm_hidden_dim=lstm_hidden_dim, dropout_p=lstm_dropout
        ).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())  # 复制初始权重
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        # 初始化Critic网络和目标网络
        self.critic = LowLevelCriticNetwork(action_dim).to(device)
        self.critic_target = LowLevelCriticNetwork(action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # 训练设置
        self.writer = SummaryWriter(comment=model_name)  # TensorBoard记录器
        self.iter_count = 0  # 迭代计数器
        self.save_every = save_every  # 保存频率
        self.model_name = model_name  # 模型名称
        self.save_directory = save_directory  # 保存目录
        self._eval_hidden_state = None  # 推理阶段的LSTM隐状态

        # 如果指定加载模型，则从文件加载
        if load_model:
            load_dir = load_directory if load_directory else save_directory
            self.load_model(filename=model_name, directory=load_dir)

    def reset_hidden_state(self) -> None:
        """清空推理时缓存的LSTM隐状态。"""

        self._eval_hidden_state = None

    def _scale_actor_output(self, raw_action: torch.Tensor) -> torch.Tensor:
        """Convert the normalized Actor output to environment action space."""
        lin_cmd = ((raw_action[..., 0] + 1.0) / 4.0).clamp(0.0, self.max_lin_velocity)
        ang_cmd = raw_action[..., 1].clamp(-self.max_ang_velocity, self.max_ang_velocity)
        return torch.stack((lin_cmd, ang_cmd), dim=-1)

    def process_observation(self, laser_scan, subgoal_distance, subgoal_angle, prev_action):
        """
        处理原始观测数据，转换为网络期望的状态向量

        Args:
            laser_scan: 原始激光雷达扫描数据
            subgoal_distance: 到子目标的距离
            subgoal_angle: 到子目标的角度
            prev_action: 历史动作 [线速度, 角速度]

        Returns:
            处理后的状态向量
        """
        # 归一化激光雷达数据（处理无穷大值）
        laser_scan = np.array(laser_scan)
        inf_mask = np.isinf(laser_scan)  # 检测无穷大值
        laser_scan[inf_mask] = 7.0  # 将无穷大替换为最大范围值
        laser_scan /= 7.0  # 归一化到[0, 1]范围

        # 归一化子目标距离和角度
        norm_distance = min(subgoal_distance / 10.0, 1.0)  # 归一化到[0, 1]，最大10米
        norm_angle = subgoal_angle / np.pi  # 归一化到[-1, 1]范围

        # 处理历史动作
        lin_vel = prev_action[0] * 2  # 缩放到适当范围
        ang_vel = (prev_action[1] + 1) / 2  # 缩放到[0, 1]范围

        # 组合所有组件
        state = laser_scan.tolist() + [norm_distance, norm_angle] + [lin_vel, ang_vel]

        return np.array(state)

    def predict_action(self, state, add_noise=False, noise_scale=0.1):
        """
        基于当前状态预测动作

        Args:
            state: 处理后的状态向量
            add_noise: 是否添加探索噪声
            noise_scale: 探索噪声的尺度

        Returns:
            动作 [线速度, 角速度]
        """
        # 将状态转换为张量并移动到设备
        state_tensor = torch.FloatTensor(state).to(self.device)

        # 通过Actor网络获取动作（不计算梯度）
        with torch.no_grad():
            if self._eval_hidden_state is None:
                self._eval_hidden_state = self.actor.init_hidden(batch_size=1, device=self.device)
            action_tensor, hidden = self.actor(
                state_tensor,
                hidden=self._eval_hidden_state,
                return_hidden=True,
            )
            self._eval_hidden_state = tuple(h.detach() for h in hidden)
            action = action_tensor.cpu().numpy().flatten()

        # 如果需要，添加探索噪声
        if add_noise:
            action += np.random.normal(0, noise_scale, size=self.action_dim)
            # 裁剪动作到合法范围
            action = np.clip(action, -self.max_action, self.max_action)

        return action

    def update(self, replay_buffer, batch_size=64, discount=0.99, tau=0.005,
               policy_noise=0.2, noise_clip=0.5, policy_freq=2, sequence_length: int = 10):
        """使用序列经验回放样本更新网络参数。"""

        try:
            batch = replay_buffer.sample_sequences(batch_size, sequence_length)
        except ValueError:
            return None

        state = torch.FloatTensor(batch["states"]).to(self.device)
        action = torch.FloatTensor(batch["actions"]).to(self.device)
        reward = torch.FloatTensor(batch["rewards"]).to(self.device)
        next_state = torch.FloatTensor(batch["next_states"]).to(self.device)
        done = torch.FloatTensor(batch["dones"]).to(self.device)
        mask = torch.FloatTensor(batch["mask"]).to(self.device).unsqueeze(-1)

        batch_total, seq_len, _ = state.shape
        state_flat = state.reshape(batch_total * seq_len, -1)
        next_state_flat = next_state.reshape(batch_total * seq_len, -1)
        action_flat = action.reshape(batch_total * seq_len, -1)
        mask_flat = mask.reshape(batch_total * seq_len, 1)
        valid_steps = mask_flat.sum().clamp(min=1.0)

        with torch.no_grad():
            target_action = self.actor_target(next_state)
            target_action = self._scale_actor_output(target_action)
            noise = torch.normal(
                mean=0.0,
                std=policy_noise,
                size=target_action.shape,
                device=self.device,
            ).clamp(-noise_clip, noise_clip)
            target_action = target_action + noise
            target_action[..., 0] = target_action[..., 0].clamp(0.0, self.max_lin_velocity)
            target_action[..., 1] = target_action[..., 1].clamp(-self.max_ang_velocity, self.max_ang_velocity)
            target_action_flat = target_action.reshape(batch_total * seq_len, -1)

            target_q1, target_q2 = self.critic_target(next_state_flat, target_action_flat)
            target_q = torch.min(target_q1, target_q2).reshape(batch_total, seq_len, 1)
            target_q = reward + (1 - done) * discount * target_q
            target_q = target_q * mask

        current_q1, current_q2 = self.critic(state_flat, action_flat)
        current_q1 = current_q1.reshape(batch_total, seq_len, 1)
        current_q2 = current_q2.reshape(batch_total, seq_len, 1)

        q1_loss = (F.smooth_l1_loss(current_q1, target_q, reduction='none') * mask).sum() / valid_steps
        q2_loss = (F.smooth_l1_loss(current_q2, target_q, reduction='none') * mask).sum() / valid_steps
        critic_loss = q1_loss + q2_loss

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = None
        if self.iter_count % policy_freq == 0:
            actor_actions = self.actor(state)
            scaled_actor_actions = self._scale_actor_output(actor_actions)
            scaled_actor_flat = scaled_actor_actions.reshape(batch_total * seq_len, -1)
            policy_q1 = self.critic(state_flat, scaled_actor_flat)[0].reshape(batch_total, seq_len, 1)
            actor_loss = -(policy_q1 * mask).sum() / valid_steps

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        self.iter_count += 1

        self.writer.add_scalar('Loss/critic', critic_loss.item(), self.iter_count)
        if actor_loss is not None:
            self.writer.add_scalar('Loss/actor', actor_loss.item(), self.iter_count)

        if self.save_every > 0 and self.iter_count % self.save_every == 0:
            self.save_model(filename=self.model_name, directory=self.save_directory)

        q_value = ((current_q1 * mask).sum() / valid_steps).item()
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item() if actor_loss is not None else None,
            'q_value': q_value,
        }

    def save_model(self, filename, directory):
        """
        保存模型参数到文件

        Args:
            filename: 保存文件的基础名称
            directory: 保存目录
        """
        # 如果目录不存在则创建
        Path(directory).mkdir(parents=True, exist_ok=True)

        # 保存Actor和Critic模型
        torch.save(self.actor.state_dict(), f"{directory}/{filename}_actor.pth")
        torch.save(self.actor_target.state_dict(), f"{directory}/{filename}_actor_target.pth")
        torch.save(self.critic.state_dict(), f"{directory}/{filename}_critic.pth")
        torch.save(self.critic_target.state_dict(), f"{directory}/{filename}_critic_target.pth")
        print(f"模型已保存到 {directory}/{filename}_*.pth")

    def load_model(self, filename, directory):
        """
        从文件加载模型参数

        Args:
            filename: 要加载的文件的基础名称
            directory: 加载目录
        """
        try:
            # 加载所有网络参数
            self.actor.load_state_dict(torch.load(f"{directory}/{filename}_actor.pth"))
            self.actor_target.load_state_dict(torch.load(f"{directory}/{filename}_actor_target.pth"))
            self.critic.load_state_dict(torch.load(f"{directory}/{filename}_critic.pth"))
            self.critic_target.load_state_dict(torch.load(f"{directory}/{filename}_critic_target.pth"))
            print(f"模型已从 {directory}/{filename}_*.pth 加载")
            self.reset_hidden_state()
        except FileNotFoundError as e:
            print(f"加载模型时出错: {e}")
