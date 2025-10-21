import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from numpy import inf


class RolloutBuffer:
    """
    用于存储PPO训练中的rollout数据（转换）的缓冲区。

    属性:
        actions (list): 智能体采取的动作。
        states (list): 智能体观察到的状态。
        logprobs (list): 动作的对数概率。
        rewards (list): 从环境中获得的奖励。
        state_values (list): 状态的价值估计。
        is_terminals (list): 指示情节终止的标志。
    """

    def __init__(self):
        """
        初始化空列表来存储缓冲区元素。
        """
        self.actions = []  # 存储动作
        self.states = []  # 存储状态
        self.logprobs = []  # 存储动作的对数概率
        self.rewards = []  # 存储奖励
        self.state_values = []  # 存储状态价值
        self.is_terminals = []  # 存储终止标志

    def clear(self):
        """
        清除缓冲区中所有存储的数据。
        """
        del self.actions[:]  # 清空动作列表
        del self.states[:]  # 清空状态列表
        del self.logprobs[:]  # 清空对数概率列表
        del self.rewards[:]  # 清空奖励列表
        del self.state_values[:]  # 清空状态价值列表
        del self.is_terminals[:]  # 清空终止标志列表

    def add(self, state, action, reward, terminal, next_state):
        """
        向缓冲区添加一个转换。（部分实现）

        参数:
            state (list or np.array): 当前观察到的状态。
            action (list or np.array): 采取的动作。
            reward (float): 采取动作后获得的奖励。
            terminal (bool): 情节是否终止。
            next_state (list or np.array): 采取动作后的结果状态。
        """
        self.states.append(state)  # 添加状态
        self.rewards.append(reward)  # 添加奖励
        self.is_terminals.append(terminal)  # 添加终止标志


class ActorCritic(nn.Module):
    """
    PPO的Actor-Critic神经网络模型。

    属性:
        actor (nn.Sequential): 策略网络（actor），输出动作均值。
        critic (nn.Sequential): 价值网络（critic），预测状态价值。
        action_var (Tensor): 动作分布的协方差矩阵对角线。
        device (str): 用于计算的设备（'cpu'或'cuda'）。
        max_action (float): 动作值的裁剪范围。
    """

    def __init__(self, state_dim, action_dim, action_std_init, max_action, device):
        """
        初始化Actor和Critic网络。

        参数:
            state_dim (int): 输入状态的维度。
            action_dim (int): 动作空间的维度。
            action_std_init (float): 动作分布的初始标准差。
            max_action (float): 动作允许的最大值（裁剪范围）。
            device (str): 运行模型的设备。
        """
        super(ActorCritic, self).__init__()

        self.device = device  # 设置设备
        self.max_action = max_action  # 最大动作值

        self.action_dim = action_dim  # 动作维度
        # 初始化动作方差
        self.action_var = torch.full(
            (action_dim,), action_std_init * action_std_init
        ).to(self.device)

        # actor网络定义
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 400),  # 第一层全连接
            nn.Tanh(),  # Tanh激活函数
            nn.Linear(400, 300),  # 第二层全连接
            nn.Tanh(),  # Tanh激活函数
            nn.Linear(300, action_dim),  # 输出层
            nn.Tanh(),  # Tanh激活函数，限制输出范围
        )

        # critic网络定义
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 400),  # 第一层全连接
            nn.Tanh(),  # Tanh激活函数
            nn.Linear(400, 300),  # 第二层全连接
            nn.Tanh(),  # Tanh激活函数
            nn.Linear(300, 1),  # 输出层，输出状态价值
        )

    def set_action_std(self, new_action_std):
        """
        为动作分布设置新的标准差。

        参数:
            new_action_std (float): 新的标准差。
        """
        # 更新动作方差
        self.action_var = torch.full(
            (self.action_dim,), new_action_std * new_action_std
        ).to(self.device)

    def forward(self):
        """
        前向方法未直接使用，因此未实现。

        抛出:
            NotImplementedError: 调用时总是抛出。
        """
        raise NotImplementedError

    def act(self, state, sample):
        """
        计算动作、其对数概率和状态价值。

        参数:
            state (Tensor): 输入状态张量。
            sample (bool): 是否从动作分布中采样或使用均值。

        返回:
            (Tuple[Tensor, Tensor, Tensor]): 采样（或均值）动作、对数概率和状态价值。
        """
        action_mean = self.actor(state)  # 通过actor网络获取动作均值
        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)  # 创建协方差矩阵
        dist = MultivariateNormal(action_mean, cov_mat)  # 创建多元正态分布

        if sample:
            # 从分布中采样并裁剪到允许范围内
            action = torch.clip(
                dist.sample(), min=-self.max_action, max=self.max_action
            )
        else:
            action = dist.mean  # 使用分布均值作为动作
        action_logprob = dist.log_prob(action)  # 计算动作的对数概率
        state_val = self.critic(state)  # 通过critic网络获取状态价值

        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, state, action):
        """
        评估给定状态和动作的动作对数概率、熵和状态价值。

        参数:
            state (Tensor): 状态批次。
            action (Tensor): 动作批次。

        返回:
            (Tuple[Tensor, Tensor, Tensor]): 动作对数概率、状态价值和分布熵。
        """
        action_mean = self.actor(state)  # 获取动作均值

        action_var = self.action_var.expand_as(action_mean)  # 扩展动作方差以匹配均值形状
        cov_mat = torch.diag_embed(action_var).to(self.device)  # 创建批次协方差矩阵
        dist = MultivariateNormal(action_mean, cov_mat)  # 创建多元正态分布

        # 对于单动作环境
        if self.action_dim == 1:
            action = action.reshape(-1, self.action_dim)  # 重塑动作张量

        action_logprobs = dist.log_prob(action)  # 计算动作对数概率
        dist_entropy = dist.entropy()  # 计算分布熵
        state_values = self.critic(state)  # 获取状态价值

        return action_logprobs, state_values, dist_entropy


class PPO:
    """
    近端策略优化（PPO）实现，用于连续控制任务。

    属性:
        max_action (float): 最大动作值。
        action_std (float): 动作分布的标准差。
        action_std_decay_rate (float): 动作标准差衰减率。
        min_action_std (float): 允许的最小动作标准差。
        state_dim (int): 状态空间的维度。
        gamma (float): 未来奖励的折扣因子。
        eps_clip (float): 策略更新的裁剪范围。
        device (str): 模型计算的设备（'cpu'或'cuda'）。
        save_every (int): 保存模型检查点的间隔（以迭代次数计）。
        model_name (str): 保存/加载模型时使用的名称。
        save_directory (Path): 保存模型检查点的目录。
        iter_count (int): 完成的训练迭代次数。
        buffer (RolloutBuffer): 存储轨迹的缓冲区。
        policy (ActorCritic): 当前的actor-critic网络。
        optimizer (torch.optim.Optimizer): actor和critic的优化器。
        policy_old (ActorCritic): 用于计算PPO更新的旧actor-critic网络。
        MseLoss (nn.Module): 均方误差损失函数。
        writer (SummaryWriter): TensorBoard摘要写入器。
    """

    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            lr_actor=0.0003,
            lr_critic=0.001,
            gamma=0.99,
            eps_clip=0.2,
            action_std_init=0.6,
            action_std_decay_rate=0.015,
            min_action_std=0.1,
            device="cpu",
            save_every=10,
            load_model=False,
            save_directory=Path("robot_nav/models/PPO/checkpoint"),
            model_name="PPO",
            load_directory=Path("robot_nav/models/PPO/checkpoint"),
    ):
        self.max_action = max_action  # 最大动作值
        self.action_std = action_std_init  # 动作标准差
        self.action_std_decay_rate = action_std_decay_rate  # 标准差衰减率
        self.min_action_std = min_action_std  # 最小标准差
        self.state_dim = state_dim  # 状态维度
        self.gamma = gamma  # 折扣因子
        self.eps_clip = eps_clip  # 裁剪参数
        self.device = device  # 设备
        self.save_every = save_every  # 保存间隔
        self.model_name = model_name  # 模型名称
        self.save_directory = save_directory  # 保存目录
        self.iter_count = 0  # 迭代计数器

        self.buffer = RolloutBuffer()  # 创建rollout缓冲区

        # 创建策略网络
        self.policy = ActorCritic(
            state_dim, action_dim, action_std_init, self.max_action, self.device
        ).to(device)
        # 创建优化器，为actor和critic设置不同的学习率
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.policy.actor.parameters(), "lr": lr_actor},
                {"params": self.policy.critic.parameters(), "lr": lr_critic},
            ]
        )

        # 创建旧策略网络
        self.policy_old = ActorCritic(
            state_dim, action_dim, action_std_init, self.max_action, self.device
        ).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())  # 复制权重
        if load_model:
            self.load(filename=model_name, directory=load_directory)  # 加载预训练模型

        self.MseLoss = nn.MSELoss()  # 均方误差损失
        self.writer = SummaryWriter(comment=model_name)  # TensorBoard写入器

    def set_action_std(self, new_action_std):
        """
        为动作分布设置新的标准差。

        参数:
            new_action_std (float): 新的标准差值。
        """
        self.action_std = new_action_std  # 更新标准差
        self.policy.set_action_std(new_action_std)  # 更新策略网络的标准差
        self.policy_old.set_action_std(new_action_std)  # 更新旧策略网络的标准差

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        """
        以固定速率衰减动作标准差，直到达到最小阈值。

        参数:
            action_std_decay_rate (float): 减少标准差的量。
            min_action_std (float): 标准差的最小值。
        """
        print(
            "--------------------------------------------------------------------------------------------"
        )
        self.action_std = self.action_std - action_std_decay_rate  # 衰减标准差
        self.action_std = round(self.action_std, 4)  # 四舍五入到4位小数
        if self.action_std <= min_action_std:
            self.action_std = min_action_std  # 确保不低于最小值
            print(
                "setting actor output action_std to min_action_std : ", self.action_std
            )
        else:
            print("setting actor output action_std to : ", self.action_std)
        self.set_action_std(self.action_std)  # 应用新的标准差
        print(
            "--------------------------------------------------------------------------------------------"
        )

    def get_action(self, state, add_noise):
        """
        使用当前策略采样动作（可选择添加噪声），如果添加噪声则存储在缓冲区中。

        参数:
            state (array_like): 策略的输入状态。
            add_noise (bool): 是否从分布中采样（True）或使用确定性均值（False）。

        返回:
            (np.ndarray): 采样的动作。
        """

        with torch.no_grad():  # 禁用梯度计算
            state = torch.FloatTensor(state).to(self.device)  # 转换为张量并移动到设备
            action, action_logprob, state_val = self.policy_old.act(state, add_noise)  # 获取动作

        if add_noise:
            # 将动作相关信息添加到缓冲区
            # self.buffer.states.append(state)  # 注释掉的状态添加
            self.buffer.actions.append(action)  # 添加动作
            self.buffer.logprobs.append(action_logprob)  # 添加对数概率
            self.buffer.state_values.append(state_val)  # 添加状态价值

        return action.detach().cpu().numpy().flatten()  # 返回numpy数组格式的动作

    def train(self, replay_buffer, iterations, batch_size):
        """
        使用基于存储的rollout缓冲区的PPO损失训练策略和价值函数。

        参数:
            replay_buffer (object): 用于兼容性的占位符（未使用）。
            iterations (int): 每次更新优化策略的周期数。
            batch_size (int): 批次大小（未使用；训练使用整个缓冲区）。
        """
        # 蒙特卡洛估计回报
        rewards = []  # 存储折扣回报
        discounted_reward = 0  # 初始化折扣回报
        for reward, is_terminal in zip(
                reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)
        ):
            if is_terminal:
                discounted_reward = 0  # 如果终止，重置折扣回报
            discounted_reward = reward + (self.gamma * discounted_reward)  # 计算折扣回报
            rewards.insert(0, discounted_reward)  # 在列表开头插入

        # 归一化回报
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)  # 转换为张量
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)  # 归一化

        # 将列表转换为张量
        assert len(self.buffer.actions) == len(self.buffer.states)  # 确保数据一致性

        # 处理状态数据
        states = [torch.tensor(st, dtype=torch.float32) for st in self.buffer.states]
        old_states = torch.squeeze(torch.stack(states, dim=0)).detach().to(self.device)
        # 处理动作数据
        old_actions = (
            torch.squeeze(torch.stack(self.buffer.actions, dim=0))
                .detach()
                .to(self.device)
        )
        # 处理对数概率数据
        old_logprobs = (
            torch.squeeze(torch.stack(self.buffer.logprobs, dim=0))
                .detach()
                .to(self.device)
        )
        # 处理状态价值数据
        old_state_values = (
            torch.squeeze(torch.stack(self.buffer.state_values, dim=0))
                .detach()
                .to(self.device)
        )

        # 计算优势函数
        advantages = rewards.detach() - old_state_values.detach()

        av_state_values = 0  # 平均状态价值
        max_state_value = -inf  # 最大状态价值
        av_loss = 0  # 平均损失

        # 优化策略K个周期
        for _ in range(iterations):
            # 评估旧动作和值
            logprobs, state_values, dist_entropy = self.policy.evaluate(
                old_states, old_actions
            )

            # 匹配state_values张量维度与rewards张量
            state_values = torch.squeeze(state_values)  # 压缩维度
            av_state_values += torch.mean(state_values)  # 累加平均状态价值
            max_state_value = max(max_state_value, max(state_values))  # 更新最大状态价值

            # 计算比率 (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # 计算替代损失
            surr1 = ratios * advantages  # 未裁剪的替代损失
            surr2 = (
                    torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages  # 裁剪的替代损失
            )

            # 裁剪目标PPO的最终损失
            loss = (
                    -torch.min(surr1, surr2)  # 策略损失（取负号因为要最大化）
                    + 0.5 * self.MseLoss(state_values, rewards)  # 价值函数损失
                    - 0.01 * dist_entropy  # 熵正则化项
            )

            # 执行梯度下降
            self.optimizer.zero_grad()  # 清零梯度
            loss.mean().backward()  # 反向传播
            self.optimizer.step()  # 更新参数
            av_loss += loss.mean()  # 累加损失

        # 将新权重复制到旧策略
        self.policy_old.load_state_dict(self.policy.state_dict())
        # 清空缓冲区
        self.buffer.clear()
        # 衰减动作标准差
        self.decay_action_std(self.action_std_decay_rate, self.min_action_std)
        self.iter_count += 1  # 增加迭代计数

        # 为TensorBoard写入新值
        self.writer.add_scalar("train/loss", av_loss / iterations, self.iter_count)
        self.writer.add_scalar(
            "train/avg_value", av_state_values / iterations, self.iter_count
        )
        self.writer.add_scalar("train/max_value", max_state_value, self.iter_count)

        # 如果达到保存点，保存模型
        if self.save_every > 0 and self.iter_count % self.save_every == 0:
            self.save(filename=self.model_name, directory=self.save_directory)

    def prepare_state(self, latest_scan, distance, cos, sin, collision, goal, action):
        """
        将原始传感器和导航数据转换为策略的归一化状态向量。

        参数:
            latest_scan (list[float]): LIDAR扫描数据。
            distance (float): 到目标的距离。
            cos (float): 到目标的角度余弦值。
            sin (float): 到目标的角度正弦值。
            collision (bool): 机器人是否发生碰撞。
            goal (bool): 机器人是否到达目标。
            action (tuple[float, float]): 最后采取的动作（线速度和角速度）。

        返回:
            (tuple[list[float], int]): 处理后的状态向量和终止标志（如果终止则为1，否则为0）。
        """
        latest_scan = np.array(latest_scan)  # 转换为numpy数组

        # 处理无限值（如超出传感器范围的读数）
        inf_mask = np.isinf(latest_scan)
        latest_scan[inf_mask] = 7.0  # 将无限值替换为最大值

        # 计算分箱参数以压缩激光数据
        max_bins = self.state_dim - 5  # 最大分箱数（减去其他状态分量）
        bin_size = int(np.ceil(len(latest_scan) / max_bins))  # 计算每个分箱的大小

        # 初始化存储每个分箱最小值的列表
        min_values = []

        # 循环遍历数据并创建分箱
        for i in range(0, len(latest_scan), bin_size):
            # 获取当前分箱
            bin = latest_scan[i: i + min(bin_size, len(latest_scan) - i)]
            # 找到当前分箱中的最小值并添加到min_values列表中
            min_values.append(min(bin) / 7)  # 归一化到[0, 1]范围

        # 归一化其他状态分量
        distance /= 10  # 距离归一化
        lin_vel = action[0] * 2  # 线速度归一化
        ang_vel = (action[1] + 1) / 2  # 角速度归一化到[0, 1]

        # 拼接所有状态分量：压缩的激光数据 + 目标信息 + 上一动作
        state = min_values + [distance, cos, sin] + [lin_vel, ang_vel]

        # 确保状态维度正确
        assert len(state) == self.state_dim
        # 确定是否终止（碰撞或达成目标）
        terminal = 1 if collision or goal else 0

        return state, terminal

    def save(self, filename, directory):
        """
        将当前策略模型保存到指定目录。

        参数:
            filename (str): 模型文件的基础名称。
            directory (Path): 保存模型的目录。
        """
        Path(directory).mkdir(parents=True, exist_ok=True)  # 创建目录（如果不存在）
        torch.save(
            self.policy_old.state_dict(), "%s/%s_policy.pth" % (directory, filename)  # 保存旧策略状态字典
        )

    def load(self, filename, directory):
        """
        从保存的检查点加载策略模型。

        参数:
            filename (str): 模型文件的基础名称。
            directory (Path): 加载模型的目录。
        """
        # 加载旧策略网络
        self.policy_old.load_state_dict(
            torch.load(
                "%s/%s_policy.pth" % (directory, filename),
                map_location=lambda storage, loc: storage,  # 确保加载到正确设备
            )
        )
        # 加载当前策略网络
        self.policy.load_state_dict(
            torch.load(
                "%s/%s_policy.pth" % (directory, filename),
                map_location=lambda storage, loc: storage,  # 确保加载到正确设备
            )
        )
        print(f"Loaded weights from: {directory}")  # 打印加载信息