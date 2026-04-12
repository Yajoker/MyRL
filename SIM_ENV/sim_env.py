"""
仿真环境抽象基类模块
定义机器人导航仿真环境的统一接口和基础功能
"""

from abc import ABC, abstractmethod
import numpy as np


class SIM_ENV(ABC):
    """
    仿真环境抽象基类

    所有具体的仿真环境类都应该继承这个基类，
    并实现其中定义的抽象方法
    """

    @abstractmethod
    def step(self):
        """
        抽象方法：执行仿真环境的一个时间步

        子类必须实现这个方法，定义如何根据动作更新环境状态

        抛出:
            NotImplementedError: 如果子类没有实现此方法
        """
        raise NotImplementedError("step方法必须由子类实现")

    @abstractmethod
    def reset(self):
        """
        抽象方法：重置仿真环境到初始状态

        子类必须实现这个方法，定义如何初始化或重新开始仿真

        抛出:
            NotImplementedError: 如果子类没有实现此方法
        """
        raise NotImplementedError("reset方法必须由子类实现")

    @staticmethod
    def cossin(vec1, vec2):
        """
        计算两个二维向量之间夹角的余弦和正弦值

        参数:
            vec1: 第一个二维向量
            vec2: 第二个二维向量

        返回:
            tuple: 向量间夹角的(余弦值, 正弦值)元组
        """
        # 将向量归一化（单位化）
        vec1 = vec1 / np.linalg.norm(vec1)
        vec2 = vec2 / np.linalg.norm(vec2)
        # 计算点积得到余弦值
        cos = np.dot(vec1, vec2)
        # 通过二维叉积计算正弦值（带方向）
        sin = vec1[0] * vec2[1] - vec1[1] * vec2[0]
        return cos, sin

    @staticmethod
    @abstractmethod
    def get_reward():
        """
        抽象静态方法：计算当前状态的奖励值

        子类必须实现这个静态方法，定义奖励函数的计算逻辑

        抛出:
            NotImplementedError: 如果子类没有实现此方法
        """
        raise NotImplementedError("get_reward方法必须由子类实现")