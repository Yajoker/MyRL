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
        self.cnn1 = nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2)  # 第一层卷积
        self.cnn2 = nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2)  # 第二层卷积
        self.cnn3 = nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2)  # 第三层卷积

        # 计算展平后的激光特征维度
        # 假设激光输入维度为180，经过三次卷积+步幅后会被逐步减小
        # 这里预估一个较大的值，实际运行时若有尺寸不匹配可调整
        laser_feature_dim = 64 * 23  # 64个通道 * 约23个时间步

        # 子目标和历史动作嵌入层
        self.subgoal_embed = nn.Linear(2, 16)  # 子目标距离和角度嵌入层
        self.action_embed = nn.Linear(2, 10)  # 历史动作嵌入层

        # 全连接层用于融合特征
        self.layer_1 = nn.Linear(laser_feature_dim + 16 + 10, 256)
        self.layer_2 = nn.Linear(256, 128)
        self.layer_3 = nn.Linear(128, action_dim)  # 输出动作维度

        # 激活函数
        self.tanh = nn.Tanh()  # 用于将输出限制在[-1, 1]范围内

    def forward(self, s):
        """
        Actor网络的前向传播

        Args:
            s: 输入状态张量，包含激光雷达扫描、子目标和历史动作信息

        Returns:
            动作张量，包含线速度和角速度 [-1,1] 范围
        """
        # 假设输入s包含：激光数据(180维), 子目标(2维), 历史动作(2维)
        # 将其拆分
        laser = s[..., :180]  # 激光雷达数据
        subgoal = s[..., 180:182]  # 子目标距离和角度
        prev_act = s[..., 182:184]  # 历史动作（线速度和角速度）

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

        # CNN层用于处理激光雷达扫描数据
        self.cnn1 = nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2)
        self.cnn2 = nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2)
        self.cnn3 = nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2)

        # 与Actor相同的激光特征维度估计
        laser_feature_dim = 64 * 23

        # 子目标和历史动作嵌入层
        self.subgoal_embed = nn.Linear(2, 16)
        self.action_embed = nn.Linear(2, 10)

        # 第一条Q网络
        self.layer_1_s = nn.Linear(laser_feature_dim + 16 + 10, 256)
        self.layer_2_s = nn.Linear(256, 256)
        self.layer_3_s = nn.Linear(256, 300)
        self.layer_3_a = nn.Linear(action_dim, 300)
        self.layer_4 = nn.Linear(300, 1)

        # 第二条Q网络
        self.layer_4_s = nn.Linear(laser_feature_dim + 16 + 10, 256)
        self.layer_5_s = nn.Linear(256, 256)
        self.layer_5_a = nn.Linear(action_dim, 300)
        self.layer_6 = nn.Linear(300, 1)

        # 初始化权重（可选）
        torch.nn.init.kaiming_uniform_(self.cnn1.weight, nonlinearity="leaky_relu")
        torch.nn.init.kaiming_uniform_(self.cnn2.weight, nonlinearity="leaky_relu")
        torch.nn.init.kaiming_uniform_(self.cnn3.weight, nonlinearity="leaky_relu")

        torch.nn.init.kaiming_uniform_(self.layer_1_s.weight, nonlinearity="leaky_relu")
        torch.nn.init.kaiming_uniform_(self.layer_2_s.weight, nonlinearity="leaky_relu")
        torch.nn.init.kaiming_uniform_(self.layer_3_s.weight, nonlinearity="leaky_relu")
        torch.nn.init.kaiming_uniform_(self.layer_3_a.weight, nonlinearity="leaky_relu")
        torch.nn.init.kaiming_uniform_(self.layer_4.weight, nonlinearity="leaky_relu")

        torch.nn.init.kaiming_uniform_(self.layer_4_s.weight, nonlinearity="leaky_relu")
        torch.nn.init.kaiming_uniform_(self.layer_5_s.weight, nonlinearity="leaky_relu")
        torch.nn.init.kaiming_uniform_(self.layer_5_a.weight, nonlinearity="leaky_relu")
        torch.nn.init.kaiming_uniform_(self.layer_6.weight, nonlinearity="leaky_relu")

    def forward(self, s, action):
        """
        Critic网络的前向传播，计算两个Q值

        Args:
            s: 状态张量
            action: 动作张量

        Returns:
            两个Q值张量，分别对应两个Critic网络的输出
        """
        # 拆分状态信息
        laser = s[..., :180]
        subgoal = s[..., 180:182]
        prev_act = s[..., 182:184]

        # 处理激光雷达数据
        laser = laser.unsqueeze(1)
        l = F.leaky_relu(self.cnn1(laser))
        l = F.leaky_relu(self.cnn2(l))
        l = F.leaky_relu(self.cnn3(l))
        l = l.flatten(start_dim=1)

        # 处理子目标和历史动作嵌入
        g = F.leaky_relu(self.subgoal_embed(subgoal))
        a = F.leaky_relu(self.action_embed(prev_act))

        s2 = torch.concat((l, g, a), dim=-1)

        # 第一条Q网络
        s1 = F.leaky_relu(self.layer_1_s(s2))
        s1 = F.leaky_relu(self.layer_2_s(s1))
        s1 = F.leaky_relu(self.layer_3_s(s1) + self.layer_3_a(action))
        q1 = self.layer_4(s1)

        # 第二条Q网络
        s2 = F.leaky_relu(self.layer_4_s(s2))
        s2 = F.leaky_relu(self.layer_5_s(s2))
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
            save_directory: Path,
            model_name: str = "low_level_controller",
            load_model: bool = False,
            load_directory: Path | None = None,
            max_lin_velocity: float = 0.5,
            max_ang_velocity: float = 1.0,
            lr: float = 1e-4,
            save_every: int = 5,
    ):
        """
        初始化低层控制器

        Args:
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            max_action: 动作的最大值（归一化动作空间上限，一般为1.0）
            device: 计算设备（CPU或GPU）
            save_directory: 模型保存路径
            model_name: 模型名称
            load_model: 是否从文件加载已有模型
            load_directory: 模型加载路径
            max_lin_velocity: 最大线速度（环境物理空间）
            max_ang_velocity: 最大角速度（环境物理空间）
            lr: 学习率
            save_every: 每隔多少次迭代保存一次模型
        """
        self.device = device
        self.action_dim = action_dim
        self.max_action = max_action
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
            prev_action: 上一步的动作 [线速度, 角速度]

        Returns:
            处理后的状态向量（numpy数组）
        """
        # 提取激光雷达数据并归一化
        laser_scan = np.array(laser_scan, dtype=np.float32)
        # 将最大有效距离设置为10米，并归一化到[0, 1]
        laser_scan = np.clip(laser_scan, 0.0, 10.0) / 10.0

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
            动作 [线速度, 角速度]（归一化到 [-1, 1] 范围，环境执行前再映射到真实物理速度）
        """
        # 将状态转换为张量并移动到设备
        state_tensor = torch.FloatTensor(state).to(self.device)

        # 通过Actor网络获取动作（不计算梯度）
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy().flatten()

        # 如果需要，添加探索噪声（在归一化动作空间中）
        if add_noise:
            action += np.random.normal(0, noise_scale, size=self.action_dim)
            # 裁剪动作到合法范围 [-max_action, max_action]
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
        # 注意：这里所有操作都在归一化动作空间 [-1, 1] 内完成，
        # 环境的物理缩放仅在与仿真交互时进行。
        with torch.no_grad():
            # 目标 Actor 给出下一状态下的归一化动作
            next_action = self.actor_target(next_state)

            # 在归一化动作空间中添加平滑噪声（policy smoothing）
            noise = torch.normal(
                mean=0.0,
                std=policy_noise,
                size=next_action.shape,
                device=self.device,
            )
            noise = noise.clamp(-noise_clip, noise_clip)  # 裁剪噪声幅度

            # 得到加入噪声后的目标动作，并裁剪到合法范围 [-max_action, max_action]
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

            # 使用目标 Critic 网络计算目标Q值
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)  # 取两个Q值的最小值（双Q学习）
            target_q = reward + (1 - done) * discount * target_q  # 贝尔曼方程

        # 计算当前Q值
        current_q1, current_q2 = self.critic(state, action)

        # 计算Critic损失并更新
        critic_loss = F.smooth_l1_loss(current_q1, target_q) + F.smooth_l1_loss(current_q2, target_q)
        self.critic_optimizer.zero_grad()  # 清零梯度
        critic_loss.backward()  # 反向传播
        self.critic_optimizer.step()  # 更新参数

        # 延迟策略更新（TD3技术）
        actor_loss = None
        if self.iter_count % policy_freq == 0:
            # 计算Actor损失（最大化Q值）
            # 这里同样在归一化动作空间中工作，直接使用 Actor 的输出作为 Critic 的输入。
            actor_action = self.actor(state)
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
            self.save_model()

        # 返回损失信息
        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item() if actor_loss is not None else None,
        }

    def save_model(self):
        """
        保存当前模型参数到文件

        文件命名格式为：
        - {model_name}_actor.pth
        - {model_name}_actor_target.pth
        - {model_name}_critic.pth
        - {model_name}_critic_target.pth
        """
        directory = self.save_directory
        directory.mkdir(parents=True, exist_ok=True)  # 确保目录存在

        model_name = self.model_name

        # 保存Actor和Critic网络参数
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
