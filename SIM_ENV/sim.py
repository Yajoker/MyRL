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

    def __init__(self, world_file="env_a.yaml", disable_plotting=False):
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

    def reset(
        self,
        robot_state=None,
        robot_goal=None,
        random_obstacles=True,
        random_obstacle_ids=None,
    ):
        """
        重置仿真环境，可选择设置机器人和障碍物状态

        参数:
            robot_state: 机器人的初始状态，格式为[x, y, theta, speed]的列表
            robot_goal: 机器人的目标状态
            random_obstacles: 是否随机重新定位障碍物
            random_obstacle_ids: 要随机化的特定障碍物ID列表

        返回:
            tuple: 重置后的初始观测值，包括LIDAR扫描、距离、余弦/正弦、
                   以及奖励相关的标志和值
        """
        # 如果未提供机器人初始状态，在1-9范围内随机生成位置
        if robot_state is None:
            robot_state = [[random.uniform(1, 9)], [random.uniform(1, 9)], [0]]

        # 设置机器人初始状态
        self.env.robot.set_state(
            state=np.array(robot_state),
            init=True,
        )

        # 如果需要随机重置障碍物位置
        if random_obstacles:
            # 如果未指定障碍物ID，默认重置所有7个障碍物
            if random_obstacle_ids is None:
                random_obstacle_ids = [i + 1 for i in range(7)]
            # 在指定范围内随机设置障碍物位置，确保不重叠
            self.env.random_obstacle_position(
                range_low=[0, 0, -3.14],
                range_high=[10, 10, 3.14],
                ids=random_obstacle_ids,
                non_overlapping=True,
            )

        # 设置机器人目标位置
        if robot_goal is None:
            # 随机设置目标位置，避开障碍物
            self.env.robot.set_random_goal(
                obstacle_list=self.env.obstacle_list,
                init=True,
                range_limits=[[1, 1, -3.141592653589793], [9, 9, 3.141592653589793]],
            )
        else:
            # 使用指定的目标位置
            self.env.robot.set_goal(np.array(robot_goal), init=True)

        # 重置环境
        self.env.reset()
        # 更新记录的目标位置
        self.robot_goal = self.env.robot.goal

        # 执行初始动作并获取初始状态
        action = [0.0, 0.0]
        latest_scan, distance, cos, sin, _, _, action, reward = self.step(
            lin_velocity=action[0], ang_velocity=action[1]
        )
        # 返回初始观测值，碰撞和目标标志设为False
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