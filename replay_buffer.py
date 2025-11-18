"""
经验回放缓冲区模块
为离策略强化学习算法提供经验存储和采样功能
"""

import random
from collections import deque  # 双端队列，用于高效地添加和删除元素
import itertools  # 迭代工具，用于高效循环操作

import numpy as np

class ReplayBuffer(object):
    """经验回放缓冲区，支持单步和序列采样。"""

    def __init__(self, buffer_size, random_seed=123):
        """初始化回放缓冲区。

        参数:
            buffer_size: 最多存储的转移数量
            random_seed: 随机种子
        """

        self.buffer_size = int(buffer_size)
        self.count = 0  # 包含当前未结束回合在内的转移数量
        self._episodes = deque()  # 已完成的回合
        self._current_episode = []  # 当前未完成回合
        self.state_dim = None
        self.action_dim = None
        random.seed(random_seed)

    # ------------------------------------------------------------------
    # 基础操作
    # ------------------------------------------------------------------
    def _enforce_capacity(self):
        """确保已完成的回合满足容量限制。"""

        while self.count > self.buffer_size and self._episodes:
            removed = self._episodes.popleft()
            self.count -= len(removed)

    def _finalize_episode(self):
        """将当前回合放入缓冲区。"""

        if self._current_episode:
            self._episodes.append(list(self._current_episode))
            self._current_episode.clear()
            self._enforce_capacity()

    def add(self, s, a, r, t, s2):
        """添加一次转移。"""

        state = np.asarray(s, dtype=np.float32)
        action = np.asarray(a, dtype=np.float32)
        next_state = np.asarray(s2, dtype=np.float32)
        reward = float(r)
        done = float(t)

        if self.state_dim is None:
            self.state_dim = state.shape[-1]
        if self.action_dim is None:
            self.action_dim = action.shape[-1]

        experience = (state, action, reward, done, next_state)
        self._current_episode.append(experience)
        self.count += 1

        if done:
            self._finalize_episode()

    def size(self):
        """返回当前存储的转移数量。"""

        return self.count

    def _iter_all_transitions(self):
        """便利函数：迭代所有已完成和当前回合的转移。"""

        for episode in self._episodes:
            for exp in episode:
                yield exp
        for exp in self._current_episode:
            yield exp

    def sample_batch(self, batch_size):
        """兼容旧接口的单步随机采样。"""

        transitions = list(self._iter_all_transitions())
        if not transitions:
            raise ValueError("Replay buffer is empty")

        batch_size = min(batch_size, len(transitions))
        batch = random.sample(transitions, batch_size)

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        t_batch = np.array([_[3] for _ in batch])
        s2_batch = np.array([_[4] for _ in batch])

        return s_batch, a_batch, r_batch, t_batch, s2_batch

    def sample_sequences(self, batch_size, sequence_length):
        """随机采样定长序列。

        返回字典，包含状态、动作、奖励、终止标志、下一状态、掩码以及有效长度。
        """

        if self.state_dim is None or self.action_dim is None:
            raise ValueError("Replay buffer does not contain any samples yet")

        completed = [ep for ep in self._episodes if ep]
        if self._current_episode:
            completed.append(self._current_episode)

        if not completed:
            raise ValueError("Replay buffer does not have enough sequences")

        seq_len = int(sequence_length)
        batch_size = int(batch_size)

        states = np.zeros((batch_size, seq_len, self.state_dim), dtype=np.float32)
        next_states = np.zeros_like(states)
        actions = np.zeros((batch_size, seq_len, self.action_dim), dtype=np.float32)
        rewards = np.zeros((batch_size, seq_len, 1), dtype=np.float32)
        dones = np.zeros((batch_size, seq_len, 1), dtype=np.float32)
        mask = np.zeros((batch_size, seq_len), dtype=np.float32)
        lengths = np.zeros(batch_size, dtype=np.int64)

        for i in range(batch_size):
            episode = random.choice(completed)
            start = random.randint(0, len(episode) - 1)
            length = 0
            for t in range(seq_len):
                idx = start + t
                if idx >= len(episode):
                    break
                s, a, r, d, s2 = episode[idx]
                states[i, t] = s
                actions[i, t] = a
                rewards[i, t, 0] = r
                dones[i, t, 0] = d
                next_states[i, t] = s2
                mask[i, t] = 1.0
                length += 1
                if d >= 1.0:
                    break
            lengths[i] = length

        if not mask.any():
            raise ValueError("Failed to sample any valid transitions")

        return {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
            "next_states": next_states,
            "mask": mask,
            "lengths": lengths,
        }

    def return_buffer(self):
        """返回缓冲区全部内容。"""

        transitions = list(self._iter_all_transitions())
        if not transitions:
            return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

        s = np.array([_[0] for _ in transitions])
        a = np.array([_[1] for _ in transitions])
        r = np.array([_[2] for _ in transitions]).reshape(-1, 1)
        t = np.array([_[3] for _ in transitions]).reshape(-1, 1)
        s2 = np.array([_[4] for _ in transitions])

        return s, a, r, t, s2

    def clear(self):
        """清空缓冲区。"""

        self._episodes.clear()
        self._current_episode.clear()
        self.count = 0
        self.state_dim = None
        self.action_dim = None


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
