from pathlib import Path
import time
from typing import Optional
import numpy as np
import torch

# 导入系统内部模块：低层控制器和高层规划器
from ethsrl.core.control.low_level_controller import LowLevelController
from ethsrl.core.planning.high_level_planner import HighLevelPlanner


class HierarchicalNavigationSystem:
    """
    分层导航系统类（Hierarchical Navigation System）

    该系统将事件触发的高层规划与反应式低层控制相结合，
    以实现高效、安全的自主机器人导航。
    """

    def __init__(
        self,
        laser_dim: int = 180,
        action_dim: int = 2,
        max_action: float = 1.0,
        device=None,
        load_models: bool = False,
        models_directory: Path = Path("ethsrl/models"),
        step_duration: float = 0.1,
        trigger_min_interval: float = 1.0,
    ) -> None:
        """
        初始化分层导航系统。

        参数:
            laser_dim: 激光雷达输入维度（例如180个角度的测距）
            action_dim: 动作维度（一般为 [线速度, 角速度]）
            max_action: 动作的最大幅值
            device: 计算设备（CPU 或 GPU）
            load_models: 是否加载已训练好的模型
            models_directory: 模型文件所在目录
        """
        # 设置计算设备：若未指定则自动检测 GPU，否则使用 CPU
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 计算状态维度
        low_level_state_dim = laser_dim + 4

        # 初始化高层规划器（负责生成子目标）
        self.high_level_planner = HighLevelPlanner(
            belief_dim=laser_dim,  # 输入维度（激光雷达特征）
            device=self.device,  # 使用的计算设备
            save_directory=models_directory / "high_level",  # 模型保存路径
            model_name="high_level_planner",  # 模型名称
            load_model=load_models,  # 是否加载已有模型
            step_duration=step_duration,
            min_interval=trigger_min_interval,
        )

        # 初始化低层控制器（负责执行动作）
        self.low_level_controller = LowLevelController(
            state_dim=low_level_state_dim,  # 低层输入维度
            action_dim=action_dim,  # 输出动作维度
            max_action=max_action,  # 最大动作值
            device=self.device,  # 计算设备
            save_directory=models_directory / "low_level",  # 模型保存路径
            model_name="low_level_controller",  # 模型名称
            load_model=load_models  # 是否加载已有模型
        )

        # 系统运行状态变量
        self.current_subgoal = None  # 当前子目标（距离、角度）
        self.current_subgoal_world: Optional[np.ndarray] = None  # 当前子目标在世界坐标系中的位置
        self.prev_action = [0.0, 0.0]  # 上一步执行的动作 [线速度, 角速度]
        self.step_count = 0  # 总步数计数
        self.last_replanning_step = 0  # 上次重新规划的步数（用于事件触发判断）
        self.step_duration = step_duration

    def step(self, laser_scan, goal_distance, goal_cos, goal_sin, robot_pose):
        """
        执行导航系统的单步操作（Step）。

        参数:
            laser_scan: 当前激光雷达读数（numpy 数组）
            goal_distance: 机器人到全局目标的距离
            goal_cos: 到全局目标方向的余弦值
            goal_sin: 到全局目标方向的正弦值
            robot_pose: 当前机器人位姿 [x, y, θ]

        返回:
            动作 [线速度, 角速度]
        """
        # 步数加一
        self.step_count += 1

        # 标志位：是否需要重新生成子目标
        if self.current_subgoal_world is None:
            should_replan = True
        else:
            should_replan = self.high_level_planner.check_triggers(
                laser_scan,
                robot_pose,
                [goal_distance, goal_cos, goal_sin],
                prev_action=self.prev_action,
                current_step=self.step_count,
            )

        subgoal_distance = None
        subgoal_angle = None

        if should_replan:
            subgoal_distance, subgoal_angle = self.high_level_planner.generate_subgoal(
                laser_scan,
                goal_distance,
                goal_cos,
                goal_sin,
                prev_action=self.prev_action,
                robot_pose=robot_pose,
                current_step=self.step_count,
            )
            planner_world = self.high_level_planner.current_subgoal_world
            self.current_subgoal_world = None if planner_world is None else np.asarray(planner_world, dtype=np.float32)
            self.last_replanning_step = self.step_count
            self.high_level_planner.event_trigger.reset_time(self.step_count)

        # 使用最新的机器人姿态计算子目标在机器人坐标系下的表示
        relative_geometry = self.high_level_planner.get_relative_subgoal(robot_pose)
        if relative_geometry[0] is None:
            if subgoal_distance is not None and subgoal_angle is not None:
                relative_geometry = (float(subgoal_distance), float(subgoal_angle))
            elif self.current_subgoal is not None:
                relative_geometry = self.current_subgoal
            else:
                relative_geometry = (0.0, 0.0)

        self.current_subgoal = (float(relative_geometry[0]), float(relative_geometry[1]))

        if should_replan:
            print(
                "New subgoal: distance={:.2f}m, angle={:.2f}rad".format(
                    self.current_subgoal[0], self.current_subgoal[1]
                )
            )

        # 生成低层输入状态，用于控制器决策
        low_level_state = self.low_level_controller.process_observation(
            laser_scan,
            self.current_subgoal[0],  # 子目标距离
            self.current_subgoal[1],  # 子目标角度
            self.prev_action  # 上一步的动作（用于平滑控制）
        )

        # 通过低层控制器预测下一步动作（网络输出）
        action = self.low_level_controller.predict_action(low_level_state)

        # 将网络输出映射为实际机器人可执行的速度命令
        # 将线速度从 [-1,1] 映射为 [0,0.5]
        linear_velocity = (action[0] + 1) / 4
        # 保持角速度在 [-1,1] 范围内
        angular_velocity = action[1]

        # 记录当前动作（用于下一次输入）
        self.prev_action = [linear_velocity, angular_velocity]

        # 返回控制命令
        return [linear_velocity, angular_velocity]

    def reset(self):
        """
        重置整个导航系统的内部状态。
        （例如在新仿真回合开始时调用）
        """
        self.current_subgoal = None  # 清空子目标
        self.current_subgoal_world = None
        self.prev_action = [0.0, 0.0]  # 重置上一步动作
        self.step_count = 0  # 步数归零
        self.last_replanning_step = 0  # 清除上次规划记录
        self.high_level_planner.current_subgoal = None
        self.high_level_planner.current_subgoal_world = None
        self.high_level_planner.prev_action = [0.0, 0.0]
        self.high_level_planner.event_trigger.last_subgoal = None
        self.high_level_planner.event_trigger.reset_state()


def create_navigation_system(load_models=False):
    """
    工厂函数：创建一个完整的分层导航系统实例。

    参数:
        load_models: 是否加载预训练模型

    返回:
        初始化好的 HierarchicalNavigationSystem 对象
    """
    return HierarchicalNavigationSystem(
        laser_dim=180,  # 激光维度
        action_dim=2,  # 动作维度
        max_action=1.0,  # 最大动作幅值
        load_models=load_models  # 是否加载模型
    )
