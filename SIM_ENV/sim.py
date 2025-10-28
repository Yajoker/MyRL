"""
机器人导航仿真环境实现模块
基于IRSim库的机器人导航仿真环境封装
"""

import irsim
import numpy as np
import random

from robot_nav.SIM_ENV.sim_env import SIM_ENV


class SIM(SIM_ENV):
    """
    机器人导航仿真环境类

    封装IRSim环境，提供步进、重置和与移动机器人交互的方法，包括奖励计算。

    属性:
        env: IRSim仿真环境实例
        robot_goal: 机器人的目标位置
    """

    def __init__(self, world_file="env.yaml", disable_plotting=False):
        """
        初始化仿真环境

        参数:
            world_file: 世界配置文件路径(YAML格式)
            disable_plotting: 如果为True，禁用渲染和绘图
        """
        # 根据参数决定是否显示图形界面
        display = False if disable_plotting else True
        # 创建IRSim仿真环境实例
        self.env = irsim.make(
            world_file, disable_all_plot=disable_plotting, display=display
        )
        # 获取机器人信息
        robot_info = self.env.get_robot_info(0)
        # 记录机器人的目标位置
        self.robot_goal = robot_info.goal

    def step(self, lin_velocity=0.0, ang_velocity=0.1):
        """
        使用给定的控制命令在仿真中执行一个步进

        参数:
            lin_velocity: 应用于机器人的线速度
            ang_velocity: 应用于机器人的角速度

        返回:
            tuple: 包含最新的LIDAR扫描数据、到目标的距离、到目标的角度余弦和正弦、
                   碰撞标志、到达目标标志、应用的动作和计算的奖励值
        """
        # 在环境中执行动作
        self.env.step(action_id=0, action=np.array([[lin_velocity], [ang_velocity]]))
        # 渲染当前环境状态
        self.env.render()

        # 获取激光雷达扫描数据
        scan = self.env.get_lidar_scan()
        latest_scan = scan["ranges"]

        # 获取机器人当前状态
        robot_state = self.env.get_robot_state()
        # 计算到目标的向量
        goal_vector = [
            self.robot_goal[0].item() - robot_state[0].item(),
            self.robot_goal[1].item() - robot_state[1].item(),
        ]
        # 计算到目标的欧几里得距离
        distance = np.linalg.norm(goal_vector)
        # 检查是否到达目标
        goal = self.env.robot.arrive
        # 计算机器人朝向向量
        pose_vector = [np.cos(robot_state[2]).item(), np.sin(robot_state[2]).item()]
        # 计算机器人朝向与目标方向之间的夹角余弦和正弦
        cos, sin = self.cossin(pose_vector, goal_vector)
        # 获取碰撞状态
        collision = self.env.robot.collision
        # 记录执行的动作
        action = [lin_velocity, ang_velocity]
        # 计算当前步的奖励值
        reward = self.get_reward(goal, collision, action, latest_scan)

        # 返回所有观测值和状态信息
        return latest_scan, distance, cos, sin, collision, goal, action, reward

    def reset(self):
        """
        重置仿真环境，但不改变机器人位置、障碍物位置和目标位置

        Returns:
            (tuple): 初始观测值
        """
        # 只重置环境状态，保持原有位置配置
        self.env.reset()

        # 获取当前的目标位置
        self.robot_goal = self.env.robot.goal

        # 执行一步空动作来获取初始观测
        action = [0.0, 0.0]
        latest_scan, distance, cos, sin, _, _, action, reward = self.step(
            lin_velocity=action[0], ang_velocity=action[1]
        )
        return latest_scan, distance, cos, sin, False, False, action, reward

    @staticmethod
    def get_reward(goal, collision, action, laser_scan):
        """
        计算当前步的奖励值

        参数:
            goal: 是否到达目标
            collision: 是否发生碰撞
            action: 执行的动作[线速度, 角速度]
            laser_scan: 激光雷达扫描读数

        返回:
            float: 当前状态的计算奖励值
        """
        # 如果到达目标，给予高奖励
        if goal:
            return 100.0
        # 如果发生碰撞，给予高惩罚
        elif collision:
            return -100.0
        else:
            # 定义距离惩罚函数：当最小距离小于1.35时给予惩罚
            r3 = lambda x: 1.35 - x if x < 1.35 else 0.0
            # 奖励计算：鼓励前进，惩罚转弯和靠近障碍物
            return action[0] - abs(action[1]) / 2 - r3(min(laser_scan)) / 2
