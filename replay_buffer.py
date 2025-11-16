"""
经验回放缓冲区模块
为离策略强化学习算法提供经验存储和采样功能
"""

import random
from collections import deque
from typing import Deque, List, Optional, Tuple

import numpy as np


Transition = Tuple[np.ndarray, np.ndarray, float, float, np.ndarray]


class ReplayBuffer(object):
    """
    支持序列采样的经验回放缓冲区。

    数据按 episode 组织，既可以随机采样单步，也可以采样连续子序列，
    方便 TD3 等 RNN 版算法执行 BPTT。
    """

    def __init__(self, buffer_size, random_seed=123):
        self.buffer_size = buffer_size
        self.episodes: Deque[List[Transition]] = deque()
        self.current_episode: List[Transition] = []
        self.total_steps = 0
        self.count = 0  # 与旧版接口保持一致，方便外部模块直接查询缓冲区大小
        self.state_dim: Optional[int] = None
        self.action_dim: Optional[int] = None
        self._flat_cache: Optional[List[Transition]] = None  # 聚合视图缓存
        random.seed(random_seed)

    # ------------------------------------------------------------------
    # 基础操作
    # ------------------------------------------------------------------
    def _invalidate_cache(self) -> None:
        self._flat_cache = None

    def _finalize_episode(self) -> None:
        if not self.current_episode:
            return
        self.episodes.append(list(self.current_episode))
        self.current_episode.clear()
        self._invalidate_cache()
        self._trim_to_capacity()

    def _trim_to_capacity(self) -> None:
        trimmed = False
        while self.total_steps > self.buffer_size and self.episodes:
            removed = self.episodes.popleft()
            self.total_steps -= len(removed)
            trimmed = True
        if trimmed:
            self._invalidate_cache()
        self.count = self.total_steps

    def add(self, s, a, r, t, s2):
        state = np.asarray(s, dtype=np.float32)
        action = np.asarray(a, dtype=np.float32)
        reward = float(r)
        done = float(t)
        next_state = np.asarray(s2, dtype=np.float32)
        transition: Transition = (state, action, reward, done, next_state)
        self.current_episode.append(transition)
        self.total_steps += 1
        self.count = self.total_steps
        self._invalidate_cache()
        if self.state_dim is None:
            self.state_dim = int(state.shape[-1])
        if self.action_dim is None:
            self.action_dim = int(action.shape[-1])
        if done >= 1.0:
            self._finalize_episode()
        else:
            self._trim_to_capacity()

    def size(self):
        return self.total_steps

    def clear(self):
        self.episodes.clear()
        self.current_episode.clear()
        self.total_steps = 0
        self.count = 0
        self.state_dim = None
        self.action_dim = None
        self._invalidate_cache()

    def _all_transitions(self) -> List[Transition]:
        if self._flat_cache is None:
            data: List[Transition] = []
            for episode in self.episodes:
                data.extend(episode)
            data.extend(self.current_episode)
            self._flat_cache = data
        return list(self._flat_cache)

    @property
    def buffer(self) -> List[Transition]:
        """与旧版接口兼容，返回展平后的 transition 列表副本。"""
        return self._all_transitions()

    # ------------------------------------------------------------------
    # 采样 API
    # ------------------------------------------------------------------
    def sample_batch(self, batch_size):
        data = self._all_transitions()
        if not data:
            raise ValueError("Replay buffer is empty")
        sample_size = min(batch_size, len(data))
        batch = random.sample(data, sample_size)
        s_batch = np.array([item[0] for item in batch], dtype=np.float32)
        a_batch = np.array([item[1] for item in batch], dtype=np.float32)
        r_batch = np.array([item[2] for item in batch], dtype=np.float32)
        d_batch = np.array([item[3] for item in batch], dtype=np.float32)
        s2_batch = np.array([item[4] for item in batch], dtype=np.float32)
        return s_batch, a_batch, r_batch, d_batch, s2_batch

    def can_sample_sequence(self, batch_size: int, seq_len: int) -> bool:
        try:
            _ = self.sample_sequences(batch_size, seq_len)
            return True
        except ValueError:
            return False

    def _available_episodes(self) -> List[List[Transition]]:
        episodes: List[List[Transition]] = [ep for ep in self.episodes if ep]
        if self.current_episode:
            episodes.append(self.current_episode)
        return episodes

    def sample_sequences(self, batch_size: int, seq_len: int):
        if self.state_dim is None or self.action_dim is None:
            raise ValueError("Replay buffer has no data")
        episodes = self._available_episodes()
        if not episodes:
            raise ValueError("No episodes available for sequence sampling")

        states = np.zeros((batch_size, seq_len, self.state_dim), dtype=np.float32)
        next_states = np.zeros_like(states)
        actions = np.zeros((batch_size, seq_len, self.action_dim), dtype=np.float32)
        rewards = np.zeros((batch_size, seq_len), dtype=np.float32)
        dones = np.zeros((batch_size, seq_len), dtype=np.float32)
        mask = np.zeros((batch_size, seq_len), dtype=np.float32)

        for idx in range(batch_size):
            episode = random.choice(episodes)
            if not episode:
                continue
            start = random.randint(0, len(episode) - 1)
            step = 0
            while step < seq_len and (start + step) < len(episode):
                s, a, r, d, s2 = episode[start + step]
                states[idx, step] = s
                actions[idx, step] = a
                rewards[idx, step] = r
                dones[idx, step] = d
                next_states[idx, step] = s2
                mask[idx, step] = 1.0
                step += 1
                if d >= 1.0:
                    break

        if mask.sum() == 0:
            raise ValueError("Unable to sample any valid sequences")

        return states, actions, rewards, dones, next_states, mask

    # ------------------------------------------------------------------
    # 兼容旧接口
    # ------------------------------------------------------------------
    def return_buffer(self):
        data = self._all_transitions()
        if not data:
            raise ValueError("Replay buffer is empty")
        s = np.array([item[0] for item in data], dtype=np.float32)
        a = np.array([item[1] for item in data], dtype=np.float32)
        r = np.array([item[2] for item in data], dtype=np.float32).reshape(-1, 1)
        d = np.array([item[3] for item in data], dtype=np.float32).reshape(-1, 1)
        s2 = np.array([item[4] for item in data], dtype=np.float32)
        return s, a, r, d, s2


class RolloutReplayBuffer(object):
    """
    存储完整回合轨迹的回放缓冲区，允许访问历史轨迹

    对于基于过去状态序列的算法（例如基于RNN的策略）非常有用
    """

    def __init__(self, buffer_size, random_seed=123, history_len=10):
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque(maxlen=buffer_size)
        random.seed(random_seed)
        self.buffer.append([])
        self.history_len = history_len

    def add(self, s, a, r, t, s2):
        experience = (s, a, r, t, s2)
        if t:
            self.count += 1
            self.buffer[-1].append(experience)
            self.buffer.append([])
        else:
            self.buffer[-1].append(experience)

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        available_episodes = list(self.buffer)[:-1]
        if self.count < batch_size:
            batch = random.sample(available_episodes, self.count)
        else:
            batch = random.sample(available_episodes, batch_size)

        idx = [random.randint(0, len(b) - 1) for b in batch]

        s_batch = []
        s2_batch = []
        for episode, pointer in zip(batch, idx):
            start = max(0, pointer - self.history_len + 1)
            history = episode[start : pointer + 1]
            history_states = [item[0] for item in history]
            history_next_states = [item[4] for item in history]
            s_batch.append(history_states)
            s2_batch.append(history_next_states)

        a_batch = [episode[pointer][1] for episode, pointer in zip(batch, idx)]
        r_batch = [episode[pointer][2] for episode, pointer in zip(batch, idx)]
        t_batch = [episode[pointer][3] for episode, pointer in zip(batch, idx)]

        return np.array(s_batch), np.array(a_batch), np.array(r_batch), np.array(t_batch), np.array(s2_batch)

    def clear(self):
        self.buffer.clear()
        self.buffer.append([])
        self.count = 0

