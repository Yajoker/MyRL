"""
经验回放缓冲区模块
为离策略强化学习算法提供经验存储和采样功能
"""

import random
from collections import deque  # 双端队列，用于高效地添加和删除元素
import itertools  # 迭代工具，用于高效循环操作
from typing import Deque, Tuple

import numpy as np

class ReplayBuffer(object):
    """
    标准的经验回放缓冲区，用于离策略强化学习算法

    存储（状态，动作，奖励，完成标志，下一状态）的元组，最多达到固定容量，
    能够采样不相关的小批次数据进行训练
    """

    def __init__(self, buffer_size, random_seed=123):
        """
        初始化回放缓冲区

        参数:
            buffer_size: 缓冲区中最多存储的转移样本数量
            random_seed: 随机数生成器的种子
        """
        self.buffer_size = buffer_size  # 缓冲区的最大容量
        self.count = 0  # 当前缓冲区中的样本数量
        self.buffer = deque()  # 使用双端队列存储经验样本
        random.seed(random_seed)  # 设置随机种子以保证可重复性

    def add(self, s, a, r, t, s2):
        """
        向缓冲区添加一个转移样本

        参数:
            s: 当前状态
            a: 执行的动作
            r: 获得的奖励
            t: 完成标志（如果回合结束则为True）
            s2: 下一个状态
        """
        # 将经验打包成元组
        experience = (s, a, r, t, s2)
        
        # 如果缓冲区未满，直接添加到末尾
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            # 缓冲区已满，移除最旧的样本，添加新的样本（FIFO策略）
            self.buffer.popleft()  # 移除队列最前端的样本
            self.buffer.append(experience)  # 在队列末尾添加新样本

    def size(self):
        """
        获取缓冲区中当前元素的数量

        返回:
            当前缓冲区大小
        """
        return self.count

    def sample_batch(self, batch_size):
        """
        从缓冲区中采样一个批次的经验样本

        参数:
            batch_size: 要采样的经验样本数量

        返回:
            元组，包含批次的：状态、动作、奖励、完成标志、下一状态
        """
        # 如果请求的批次大小大于当前缓冲区大小，则采样所有可用样本
        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        # 将采样的批次数据分别提取到不同的数组中
        s_batch = np.array([_[0] for _ in batch])  # 状态批次
        a_batch = np.array([_[1] for _ in batch])  # 动作批次
        r_batch = np.array([_[2] for _ in batch])  # 奖励批次
        t_batch = np.array([_[3] for _ in batch])  # 完成标志批次
        s2_batch = np.array([_[4] for _ in batch])  # 下一状态批次

        return s_batch, a_batch, r_batch, t_batch, s2_batch

    def return_buffer(self):
        """
        将整个缓冲区内容作为单独的数组返回

        返回:
            元组，包含完整的：状态数组、动作数组、奖励数组、完成标志数组、下一状态数组
        """
        # 提取缓冲区中所有样本的各个组成部分
        s = np.array([_[0] for _ in self.buffer])  # 所有状态
        a = np.array([_[1] for _ in self.buffer])  # 所有动作
        r = np.array([_[2] for _ in self.buffer]).reshape(-1, 1)  # 所有奖励，重塑为列向量
        t = np.array([_[3] for _ in self.buffer]).reshape(-1, 1)  # 所有完成标志，重塑为列向量
        s2 = np.array([_[4] for _ in self.buffer])  # 所有下一状态

        return s, a, r, t, s2

    def clear(self):
        """
        清空缓冲区的所有内容
        """
        self.buffer.clear()  # 清空双端队列
        self.count = 0  # 重置计数器


class RolloutReplayBuffer(object):
    """
    存储完整回合轨迹的回放缓冲区，允许访问历史轨迹

    对于基于过去状态序列的算法（例如基于RNN的策略）非常有用
    """

    def __init__(self, buffer_size, random_seed=123, history_len=10):
        """
        初始化轨迹回放缓冲区

        参数:
            buffer_size: 最多存储的回合（轨迹）数量
            random_seed: 随机数生成器的种子
            history_len: 为每个采样状态返回的过去步数
        """
        self.buffer_size = buffer_size  # 缓冲区的最大容量（按回合数）
        self.count = 0  # 完整回合的数量
        self.buffer = deque(maxlen=buffer_size)  # 使用固定长度的双端队列
        random.seed(random_seed)  # 设置随机种子
        self.buffer.append([])  # 初始化第一个空回合
        self.history_len = history_len  # 历史序列的长度

    def add(self, s, a, r, t, s2):
        """
        向当前回合添加一个转移样本

        如果转移结束了回合（t=True），则开始一个新的回合

        参数:
            s: 当前状态
            a: 执行的动作
            r: 获得的奖励
            t: 完成标志
            s2: 下一个状态
        """
        # 将经验打包成元组
        experience = (s, a, r, t, s2)
        
        if t:
            # 如果回合结束
            self.count += 1  # 增加完整回合计数
            self.buffer[-1].append(experience)  # 将最终经验添加到当前回合
            self.buffer.append([])  # 开始一个新的空回合
        else:
            # 回合未结束，将经验添加到当前回合
            self.buffer[-1].append(experience)

    def size(self):
        """
        获取缓冲区中完整回合的数量

        返回:
            回合数量
        """
        return self.count

    def sample_batch(self, batch_size):
        """
        从完整回合中采样一批状态序列和相应的转移

        为每个采样的转移返回过去`history_len`步，必要时用最早的步进行填充

        参数:
            batch_size: 要采样的序列数量

        返回:
            元组，包含：过去状态序列、动作、奖励、完成标志、下一状态序列
        """
        # 排除最后一个可能未完成的回合
        available_episodes = list(itertools.islice(self.buffer, 0, len(self.buffer) - 1))
        
        # 采样回合批次
        if self.count < batch_size:
            # 如果可用回合数少于批次大小，采样所有可用回合
            batch = random.sample(available_episodes, self.count)
        else:
            # 否则采样指定数量的回合
            batch = random.sample(available_episodes, batch_size)

        # 为每个采样的回合随机选择一个时间步索引
        idx = [random.randint(0, len(b) - 1) for b in batch]

        # 初始化批次数组
        s_batch = []  # 状态序列批次
        s2_batch = []  # 下一状态序列批次
        
        # 为每个采样的回合构建状态序列
        for i in range(len(batch)):
            if idx[i] == len(batch[i]):
                # 如果选择了回合的最后一个时间步
                s = batch[i]  # 使用整个回合作为状态序列
                s2 = batch[i]  # 使用整个回合作为下一状态序列
            else:
                # 使用从开始到选定时间步的序列
                s = batch[i][: idx[i] + 1]
                s2 = batch[i][: idx[i] + 1]
            
            # 提取状态并反转顺序（最新的状态在最后）
            s = [v[0] for v in s]  # 提取所有状态
            s = s[::-1]  # 反转顺序，使最新的状态在序列末尾

            # 提取下一状态并反转顺序
            s2 = [v[4] for v in s2]  # 提取所有下一状态
            s2 = s2[::-1]  # 反转顺序

            # 如果序列长度小于历史长度，进行填充
            if len(s) < self.history_len:
                missing = self.history_len - len(s)  # 计算缺失的长度
                s += [s[-1]] * missing  # 用最后一个状态填充缺失部分
                s2 += [s2[-1]] * missing  # 用最后一个下一状态填充缺失部分
            else:
                # 如果序列过长，截取指定历史长度
                s = s[: self.history_len]
                s2 = s2[: self.history_len]
            
            # 再次反转，恢复原始时间顺序（最旧的状态在前）
            s = s[::-1]
            s_batch.append(s)  # 添加到状态批次
            
            s2 = s2[::-1]  # 恢复下一状态的原始时间顺序
            s2_batch.append(s2)  # 添加到下一状态批次

        # 提取选定时间步的动作、奖励和完成标志
        a_batch = np.array([batch[i][idx[i]][1] for i in range(len(batch))])  # 动作批次
        r_batch = np.array([batch[i][idx[i]][2] for i in range(len(batch))]).reshape(-1, 1)  # 奖励批次，重塑为列向量
        t_batch = np.array([batch[i][idx[i]][3] for i in range(len(batch))]).reshape(-1, 1)  # 完成标志批次，重塑为列向量

        # 返回所有批次数据
        return np.array(s_batch), a_batch, r_batch, t_batch, np.array(s2_batch)

    def clear(self):
        """
        清空缓冲区中存储的所有回合
        """
        self.buffer.clear()  # 清空双端队列
        self.count = 0  # 重置回合计数器


class HighLevelReplayBuffer:
    """Simple replay buffer for high-level (state, action, reward, done, next_state) tuples."""

    def __init__(self, buffer_size: int, random_seed: int = 666) -> None:
        self._buffer: Deque[Tuple[np.ndarray, np.ndarray, float, float, np.ndarray]] = deque(maxlen=buffer_size)
        random.seed(random_seed)

    def add(self, state, action, reward, done, next_state) -> None:
        state_arr = np.asarray(state, dtype=np.float32)
        action_arr = np.asarray(action, dtype=np.float32)
        next_state_arr = np.asarray(next_state, dtype=np.float32)
        reward_val = float(reward)
        done_val = float(done)
        self._buffer.append((state_arr, action_arr, reward_val, done_val, next_state_arr))

    def size(self) -> int:
        return len(self._buffer)

    def sample(self, batch_size: int):
        batch = random.sample(self._buffer, min(batch_size, len(self._buffer)))
        states = np.stack([entry[0] for entry in batch])
        actions = np.stack([entry[1] for entry in batch])
        rewards = np.array([entry[2] for entry in batch], dtype=np.float32)
        dones = np.array([entry[3] for entry in batch], dtype=np.float32)
        next_states = np.stack([entry[4] for entry in batch])
        return states, actions, rewards, dones, next_states

    def clear(self) -> None:
        self._buffer.clear()


class CostReplayBuffer:
    """Replay buffer for long-horizon cost critic supervision."""

    def __init__(self, max_size: int):
        self._buffer: Deque[Tuple[np.ndarray, np.ndarray, float]] = deque(maxlen=max_size)

    def add(self, state, geom, cost) -> None:
        state_arr = np.asarray(state, dtype=np.float32)
        geom_arr = np.asarray(geom, dtype=np.float32)
        self._buffer.append((state_arr, geom_arr, float(cost)))

    def sample(self, batch_size: int):
        batch = random.sample(self._buffer, min(batch_size, len(self._buffer)))
        states = np.stack([entry[0] for entry in batch])
        geoms = np.stack([entry[1] for entry in batch])
        costs = np.array([entry[2] for entry in batch], dtype=np.float32)
        return states, geoms, costs

    def __len__(self) -> int:
        return len(self._buffer)
