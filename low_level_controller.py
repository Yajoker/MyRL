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
    """
    低层控制器的Actor网络
    基于CNNTD3架构，处理激光雷达扫描数据、子目标信息和历史动作
    输出机器人的线速度和角速度控制指令
    """

    def __init__(self, action_dim):
        """初始化Actor网络

        Args:
            action_dim: 动作空间的维度，通常为2（线速度和角速度）
        """
        super(LowLevelActorNetwork, self).__init__()

        # CNN层用于处理激光雷达扫描数据
        # 输入: 1通道的激光数据，输出: 4个特征图
        self.cnn1 = nn.Conv1d(1, 4, kernel_size=8, stride=4)
        # 第二层CNN，输入4个特征图，输出8个特征图
        self.cnn2 = nn.Conv1d(4, 8, kernel_size=8, stride=4)
        # 第三层CNN，输入8个特征图，输出4个特征图
        self.cnn3 = nn.Conv1d(8, 4, kernel_size=4, stride=2)

        # 子目标信息嵌入层（距离和角度）
        self.subgoal_embed = nn.Linear(2, 10)

        # 历史动作嵌入层
        self.action_embed = nn.Linear(2, 10)

        # 全连接层
        # 输入维度: 16(CNN输出) + 10(子目标) + 10(历史动作) = 36
        self.layer_1 = nn.Linear(36, 400)
        # 使用Kaiming初始化权重，适用于LeakyReLU激活函数
        torch.nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="leaky_relu")

        # 第二层全连接
        self.layer_2 = nn.Linear(400, 300)
        torch.nn.init.kaiming_uniform_(self.layer_2.weight, nonlinearity="leaky_relu")

        # 输出层，生成动作
        self.layer_3 = nn.Linear(300, action_dim)
        # Tanh激活函数将输出限制在[-1, 1]范围内
        self.tanh = nn.Tanh()

    def forward(self, s):
        """
        Actor网络的前向传播

        Args:
            s: 输入状态张量，形状为(batch_size, state_dim)
               包含激光雷达数据、子目标信息和历史动作

        Returns:
            动作张量，值在范围[-1, 1]内
        """
        # 如果输入是1维张量，增加batch维度
        if len(s.shape) == 1:
            s = s.unsqueeze(0)

        # 分割状态张量的各个部分
        laser = s[:, :-4]  # 激光雷达扫描数据
        subgoal = s[:, -4:-2]  # 子目标信息（距离，角度）
        prev_act = s[:, -2:]  # 历史动作（线速度，角速度）

        # 处理激光雷达数据
        laser = laser.unsqueeze(1)  # 增加通道维度
        l = F.leaky_relu(self.cnn1(laser))  # 第一层CNN + LeakyReLU激活
        l = F.leaky_relu(self.cnn2(l))  # 第二层CNN + LeakyReLU激活
        l = F.leaky_relu(self.cnn3(l))  # 第三层CNN + LeakyReLU激活
        l = l.flatten(start_dim=1)  # 展平特征图

        # 处理子目标信息
        g = F.leaky_relu(self.subgoal_embed(subgoal))

        # 处理历史动作
        a = F.leaky_relu(self.action_embed(prev_act))

        # 拼接所有特征
        s = torch.concat((l, g, a), dim=-1)

        # 全连接层处理
        s = F.leaky_relu(self.layer_1(s))  # 第一层全连接 + LeakyReLU
        s = F.leaky_relu(self.layer_2(s))  # 第二层全连接 + LeakyReLU
        a = self.tanh(self.layer_3(s))  # 输出层 + Tanh激活

        return a


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
            save_directory=Path("myrl/models/low_level"),
            model_name="low_level_controller",
            load_directory=None,
            *,
            max_lin_velocity: float = 1.0,
            max_ang_velocity: float = 1.0,
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
        self.max_action = float(max_action)
        if not np.isclose(self.max_action, 1.0):
            raise ValueError("max_action must be 1.0 to keep TD3 actions normalized in [-1, 1].")
        self.state_dim = state_dim
        self.max_lin_velocity = float(max_lin_velocity)
        self.max_ang_velocity = float(max_ang_velocity)

        # 初始化Actor网络和目标网络
        self.actor = LowLevelActorNetwork(action_dim).to(device)
        self.actor_target = LowLevelActorNetwork(action_dim).to(device)
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

        # 如果指定加载模型，则从文件加载
        if load_model:
            load_dir = load_directory if load_directory else save_directory
            self.load_model(filename=model_name, directory=load_dir)

    def scale_action_for_env(self, action):
        """Scale a normalized action ``a∈[-1,1]^2`` to ``(v, ω)`` commands."""

        lin_scale = self.max_lin_velocity / 2.0
        if isinstance(action, torch.Tensor):
            clipped = action.clamp(-self.max_action, self.max_action)
            lin_cmd = (clipped[..., 0] + 1.0) * lin_scale
            ang_cmd = clipped[..., 1] * self.max_ang_velocity
            return torch.stack((lin_cmd, ang_cmd), dim=-1)

        action_arr = np.asarray(action, dtype=np.float32)
        clipped = np.clip(action_arr, -self.max_action, self.max_action)
        lin_cmd = (clipped[..., 0] + 1.0) * lin_scale
        ang_cmd = clipped[..., 1] * self.max_ang_velocity
        scaled = np.stack((lin_cmd, ang_cmd), axis=-1)
        return scaled.astype(np.float32, copy=False)

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
        laser_scan[inf_mask] = 9.0  # 将无穷大替换为最大范围值
        laser_scan /= 9.0  # 归一化到[0, 1]范围

        # 归一化子目标距离和角度
        # norm_distance = min(subgoal_distance / 10.0, 1.0)  # 归一化到[0, 1]，最大10米
        norm_distance = float(np.tanh(subgoal_distance / 10.0))
        norm_angle = subgoal_angle / np.pi  # 归一化到[-1, 1]范围

        # 处理历史动作（已经是[-1,1]范围的归一化动作）
        prev_action = np.asarray(prev_action, dtype=np.float32)
        if prev_action.shape[-1] != 2:
            raise ValueError("prev_action must contain 2 elements: [linear, angular].")
        prev_action = np.clip(prev_action, -self.max_action, self.max_action)

        # 组合所有组件
        state = laser_scan.tolist() + [norm_distance, norm_angle] + prev_action.tolist()

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
            action = self.actor(state_tensor).cpu().numpy().flatten()

        # 如果需要，添加探索噪声
        if add_noise:
            action += np.random.normal(0, noise_scale, size=self.action_dim)
            action = np.clip(action, -self.max_action, self.max_action)
        else:
            action = np.clip(action, -self.max_action, self.max_action)

        return action

    def update(self, replay_buffer, batch_size=64, discount=0.99, tau=0.005,
               policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        """
        使用经验回放缓冲区中的样本更新控制器参数

        Args:
            replay_buffer: 包含经验样本的回放缓冲区
            batch_size: 小批量大小
            discount: 未来奖励的折扣因子
            tau: 软更新参数
            policy_noise: 添加到目标动作的噪声
            noise_clip: 最大噪声幅度
            policy_freq: Actor更新频率相对于Critic的频率

        Returns:
            包含损失信息的字典
        """
        # 从回放缓冲区采样一个小批量
        # ReplayBuffer 返回的顺序为 (states, actions, rewards, dones, next_states)
        state, action, reward, done, next_state = replay_buffer.sample(batch_size)

        # 转换为张量并移动到设备
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device).reshape(-1, 1)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).to(self.device).reshape(-1, 1)

        # 获取带噪声的下一个动作并计算目标Q值（TD3技术）
        with torch.no_grad():
            next_action = self.actor_target(next_state)
            noise = (torch.zeros_like(action)).data.normal_(0, policy_noise).to(self.device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)  # 取两个Q值的最小值（双Q学习）
            target_q = reward + (1 - done) * discount * target_q  # 贝尔曼方程

        # 计算当前Q值
        current_q1, current_q2 = self.critic(state, action)

        # 计算Critic损失并更新
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        self.critic_optimizer.zero_grad()  # 清零梯度
        critic_loss.backward()  # 反向传播
        self.critic_optimizer.step()  # 更新参数

        # 延迟策略更新（TD3技术）
        actor_loss = None
        if self.iter_count % policy_freq == 0:
            # 计算Actor损失（最大化Q值）
            actor_action = self.actor(state).clamp(-self.max_action, self.max_action)
            actor_loss = -self.critic.forward(state, actor_action)[0].mean()

            # 更新Actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # 软更新目标网络
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        # 增加迭代计数器
        self.iter_count += 1

        # 记录到TensorBoard
        self.writer.add_scalar('Loss/critic', critic_loss.item(), self.iter_count)
        if actor_loss is not None:
            self.writer.add_scalar('Loss/actor', actor_loss.item(), self.iter_count)

        # 如果需要，保存模型
        if self.save_every > 0 and self.iter_count % self.save_every == 0:
            self.save_model(filename=self.model_name, directory=self.save_directory)

        # 返回损失信息
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item() if actor_loss is not None else None,
            'q_value': current_q1.mean().item()
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
        except FileNotFoundError as e:
            print(f"加载模型时出错: {e}")
