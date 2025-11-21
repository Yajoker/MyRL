from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class LowLevelRewardConfig:
    """Reward shaping coefficients for the low-level controller."""

    # 1. 子目标进展相关
    progress_weight: float = 5.0          # 略小一点，配合后面的缩放
    efficiency_penalty: float = 0.03      # 每步轻微时间成本

    # 2. 安全相关
    safety_weight: float = 1.0            # 提高安全项权重
    safety_sensitivity: float = 0.5       # 暂时保留但不使用（或直接删掉）
    safety_clearance: float = 0.6
    collision_distance: float = 0.3

    # 3. 终局项：在低层只保留很小的局部效果
    goal_bonus: float = 0.0               # 低层不再给终点奖励
    subgoal_bonus: float = 0.0            # 子目标奖励交给高层
    collision_penalty: float = -20.0       # 轻微局部惩罚
    timeout_penalty: float = 0.0          # 不在低层惩罚超时

    def __post_init__(self) -> None:  # type: ignore[override]
        """数据类初始化后验证方法"""
        if self.efficiency_penalty < 0:
            raise ValueError("efficiency_penalty must be non-negative")
        if self.safety_clearance <= 0:
            raise ValueError("safety_clearance must be positive")


@dataclass(frozen=True)
class HighLevelRewardConfig:
    """Reward shaping coefficients for the high-level planner."""

    # 1. 全局几何进展
    path_progress_weight: float = 1.0       # window index 的权重
    global_progress_weight: float = 4.0     # 到最终目标距离变化的权重（主导）

    # 2. 低层执行质量
    low_level_return_scale: float = 1.0     # 但注意我们会先对 low_level_return 做归一化
    subgoal_completion_bonus: float = 2.0   # 子目标成功的小奖励
    low_level_failure_penalty: float = -2.0 # “子目标期间失败”的轻微惩罚

    # 3. 终局事件（只在高层定义一次）
    goal_bonus: float = 80.0                # 大正奖励
    collision_penalty: float = -80.0        # 大负奖励
    timeout_penalty: float = -50.0          # 负但比碰撞轻


@dataclass(frozen=True)
class ShieldingConfig:
    """Velocity shielding parameters applied before executing commands."""

    enabled: bool = False                             # 是否启用速度缩放屏蔽
    safe_distance: float = 0.7                       # 安全距离阈值 d_safe
    gain: float = 8.0                                # Logistic 缩放的斜率系数 k
    angular_gain: float = 1.5                        # 安全距离内角速度放大系数 γ

    def __post_init__(self) -> None:  # type: ignore[override]
        if self.safe_distance <= 0:
            raise ValueError("safe_distance must be positive")
        if self.gain <= 0:
            raise ValueError("gain must be positive")
        if self.angular_gain < 1.0:
            raise ValueError("angular_gain must be at least 1.0")


@dataclass(frozen=True)
class MotionConfig:
    """Motion primitive limits and time discretisation."""

    v_max: float = 0.5                               # 最大线速度 (m/s)
    omega_max: float = 1.0                           # 最大角速度 (rad/s)
    dt: float = 0.3                                  # 控制时间步长 (s)
    shielding: ShieldingConfig = field(default_factory=ShieldingConfig)  # 速度缩放屏蔽配置

    def __post_init__(self) -> None:  # type: ignore[override]
        """数据类初始化后验证方法"""
        if self.v_max <= 0:
            raise ValueError("v_max must be positive")
        if self.omega_max <= 0:
            raise ValueError("omega_max must be positive")
        if self.dt <= 0:
            raise ValueError("dt must be positive")


@dataclass(frozen=True)
class TriggerConfig:
    """High-level trigger thresholds and timing rules."""

    safety_trigger_distance: float = 0.8             # 安全触发距离阈值
    subgoal_reach_threshold: float = 0.4             # 子目标到达判定阈值
    stagnation_steps: int = 30                       # 停滞步数阈值（检测是否卡住）
    stagnation_turn_threshold: float = 3.5           # 累计转向阈值（弧度）
    progress_epsilon: float = 0.1                    # 进度变化最小阈值下限（采用窗口比例时的兜底值）
    progress_epsilon_ratio: float = 0.02             # 进度阈值相对于窗口半径的比例
    min_interval: float = 1.2                        # 最小触发间隔时间（秒，优先使用步数配置）
    min_step_interval: int = 10                      # 最小触发间隔步数
    window_inside_hold: int = 3                      # 进入窗口后至少驻留的步数
    subgoal_smoothing_alpha: float = 0.7             # 子目标EMA平滑系数

    def __post_init__(self) -> None:  # type: ignore[override]
        """数据类初始化后验证方法"""
        if self.safety_trigger_distance <= 0:
            raise ValueError("safety_trigger_distance must be positive")
        if self.subgoal_reach_threshold <= 0:
            raise ValueError("subgoal_reach_threshold must be positive")
        if self.stagnation_steps <= 0:
            raise ValueError("stagnation_steps must be positive")
        if self.stagnation_turn_threshold < 0:
            raise ValueError("stagnation_turn_threshold must be non-negative")
        if self.progress_epsilon < 0:
            raise ValueError("progress_epsilon must be non-negative")
        if self.progress_epsilon_ratio < 0:
            raise ValueError("progress_epsilon_ratio must be non-negative")
        if self.min_interval < 0:
            raise ValueError("min_interval must be non-negative")
        if self.min_step_interval <= 0:
            raise ValueError("min_step_interval must be positive")
        if self.window_inside_hold < 0:
            raise ValueError("window_inside_hold must be non-negative")
        if not 0.0 <= self.subgoal_smoothing_alpha < 1.0:
            raise ValueError("subgoal_smoothing_alpha must be in [0, 1)")


@dataclass(frozen=True)
class PlannerConfig:
    """Static global planning configuration."""

    resolution: float = 0.25                         # 路径规划分辨率
    safety_margin: float = 0.35                      # 安全边界距离
    waypoint_lookahead: int = 3                      # 前瞻路径点的数量
    window_spacing: float = 2.0                      # 路径点窗口间距
    window_radius: float = 0.6                       # 路径点窗口半径
    subgoal_distance_normalizer: float = 1.5         # 低层观测中的子目标距离归一化尺度
    use_path_tangent: bool = True                    # 是否使用窗口切线作为子目标基准方向

    def __post_init__(self) -> None:  # type: ignore[override]
        """数据类初始化后验证方法"""
        if self.resolution <= 0:
            raise ValueError("resolution must be positive")
        if self.safety_margin < 0:
            raise ValueError("safety_margin must be non-negative")
        if self.waypoint_lookahead <= 0:
            raise ValueError("waypoint_lookahead must be positive")
        if self.window_spacing <= 0:
            raise ValueError("window_spacing must be positive")
        if self.window_radius <= 0:
            raise ValueError("window_radius must be positive")
        if self.subgoal_distance_normalizer <= 0:
            raise ValueError("subgoal_distance_normalizer must be positive")


@dataclass(frozen=True)
class SafetyCriticConfig:
    """Configuration for the tactical-layer safety critic."""

    progress_weight: float = 1.0                     # 进度得分的权重α
    risk_weight: float = 1.0                         # 风险惩罚的权重β
    distance_weight: float = 1.0                     # 进度得分中的距离项权重
    angle_weight: float = 0.35                       # 进度得分中的角度项权重
    update_batch_size: int = 64                      # Safety-Critic训练批次大小
    min_buffer_size: int = 128                       # 开始训练前所需的最少样本数
    max_buffer_size: int = 4096                      # Safety-Critic样本缓冲区最大容量
    target_clip_min: float = 0.0                     # 风险监督目标的最小裁剪值
    target_clip_max: float = 6.0                     # 风险监督目标的最大裁剪值

    def __post_init__(self) -> None:  # type: ignore[override]
        if self.progress_weight < 0:
            raise ValueError("progress_weight must be non-negative")
        if self.risk_weight < 0:
            raise ValueError("risk_weight must be non-negative")
        if self.distance_weight < 0:
            raise ValueError("distance_weight must be non-negative")
        if self.angle_weight < 0:
            raise ValueError("angle_weight must be non-negative")
        if self.update_batch_size <= 0:
            raise ValueError("update_batch_size must be positive")
        if self.min_buffer_size < 0:
            raise ValueError("min_buffer_size must be non-negative")
        if self.max_buffer_size <= 0:
            raise ValueError("max_buffer_size must be positive")
        if self.target_clip_min < 0:
            raise ValueError("target_clip_min must be non-negative")
        if self.target_clip_max <= self.target_clip_min:
            raise ValueError("target_clip_max must be greater than target_clip_min")


@dataclass(frozen=True)
class WindowRuntimeConfig:
    """Runtime rules for waypoint window tracking."""

    margin: float = 0.15                             # 窗口边界裕度
    step_limit: int = 80                             # 步数限制（在窗口内的最大步数）
    clip_to_window: bool = False                     # 是否强制将子目标裁剪到窗口半径内

    def __post_init__(self) -> None:  # type: ignore[override]
        """数据类初始化后验证方法"""
        if self.margin < 0:
            raise ValueError("margin must be non-negative")
        if self.step_limit <= 0:
            raise ValueError("step_limit must be positive")


@dataclass(frozen=True)
class TrainingConfig:
    """End-to-end training and evaluation hyper-parameters."""

    buffer_size: int = 50_000                        # 经验回放缓冲区大小
    batch_size: int = 64                             # 训练批次大小
    max_epochs: int = 60                             # 最大训练周期数
    episodes_per_epoch: int = 70                     # 每个周期的回合数
    max_steps: int = 350                             # 每个回合的最大步数
    train_every_n_episodes: int = 1                  # 每N个回合训练一次
    training_iterations: int = 20                   # 每次训练的迭代次数
    exploration_noise: float = 0.15                  # 探索噪声系数
    min_buffer_size: int = 0                     # 开始训练的最小缓冲区大小
    max_lin_velocity: float = 1.0                    # 最大线速度
    max_ang_velocity: float = 1.0                    # 最大角速度
    eval_episodes: int = 10                          # 评估回合数
    subgoal_radius: float = 0.4                      # 子目标判定阈值
    save_every: int = 5                              # 保存模型的频率（每N个周期）
    world_file: str = "env_b.yaml"                  # 环境配置文件
    waypoint_lookahead: int = 3                      # 高层使用的前瞻航点数
    global_plan_resolution: float = 0.25             # 全局规划分辨率
    global_plan_margin: float = 0.35                 # 全局规划安全裕度
    discount: float = 0.99                           # 折扣因子
    tau: float = 0.005                               # 目标网络软更新系数
    policy_noise: float = 0.2                        # 策略噪声
    noise_clip: float = 0.5                          # 噪声裁剪范围
    policy_freq: int = 2                             # 策略更新频率
    random_seed: Optional[int] = 666                 # 随机种子

    def __post_init__(self) -> None:  # type: ignore[override]
        """数据类初始化后验证方法"""
        if self.buffer_size <= 0:
            raise ValueError("buffer_size must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.max_epochs <= 0:
            raise ValueError("max_epochs must be positive")
        if self.episodes_per_epoch <= 0:
            raise ValueError("episodes_per_epoch must be positive")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.train_every_n_episodes <= 0:
            raise ValueError("train_every_n_episodes must be positive")
        if self.training_iterations <= 0:
            raise ValueError("training_iterations must be positive")
        if self.exploration_noise < 0:
            raise ValueError("exploration_noise must be non-negative")
        if self.min_buffer_size < 0:
            raise ValueError("min_buffer_size must be non-negative")
        if self.eval_episodes <= 0:
            raise ValueError("eval_episodes must be positive")
        if self.save_every < 0:
            raise ValueError("save_every must be non-negative")
        if self.discount <= 0 or self.discount > 1:
            raise ValueError("discount must be in (0, 1]")
        if self.tau <= 0 or self.tau > 1:
            raise ValueError("tau must be in (0, 1]")
        if self.policy_noise < 0:
            raise ValueError("policy_noise must be non-negative")
        if self.noise_clip < 0:
            raise ValueError("noise_clip must be non-negative")
        if self.policy_freq <= 0:
            raise ValueError("policy_freq must be positive")


@dataclass(frozen=True)
class IntegrationConfig:
    """Aggregate configuration passed into the integration layer."""

    motion: MotionConfig = field(default_factory=MotionConfig)                    # 运动配置
    trigger: TriggerConfig = field(default_factory=TriggerConfig)                 # 触发配置
    planner: PlannerConfig = field(default_factory=PlannerConfig)                 # 规划器配置
    safety_critic: SafetyCriticConfig = field(default_factory=SafetyCriticConfig) # Safety-Critic配置
    window: WindowRuntimeConfig = field(default_factory=WindowRuntimeConfig)      # 窗口运行时配置
    low_level_reward: LowLevelRewardConfig = field(default_factory=LowLevelRewardConfig)  # 低层奖励配置
    high_level_reward: HighLevelRewardConfig = field(default_factory=HighLevelRewardConfig)  # 高层奖励配置
    training: TrainingConfig = field(default_factory=TrainingConfig)              # 训练配置

    def with_updates(
        self,
        *,
        motion: MotionConfig | None = None,
        trigger: TriggerConfig | None = None,
        planner: PlannerConfig | None = None,
        safety_critic: SafetyCriticConfig | None = None,
        window: WindowRuntimeConfig | None = None,
        low_level_reward: LowLevelRewardConfig | None = None,
        high_level_reward: HighLevelRewardConfig | None = None,
        training: TrainingConfig | None = None,
    ) -> "IntegrationConfig":
        """Return a copy with selected sub-configs replaced."""
        # 返回更新了指定子配置的新IntegrationConfig实例
        return IntegrationConfig(
            motion=motion or self.motion,
            trigger=trigger or self.trigger,
            planner=planner or self.planner,
            safety_critic=safety_critic or self.safety_critic,
            window=window or self.window,
            low_level_reward=low_level_reward or self.low_level_reward,
            high_level_reward=high_level_reward or self.high_level_reward,
            training=training or self.training,
        )


@dataclass(frozen=True)
class ConfigBundle:
    """Single source of truth for all ETHSRL configuration domains."""

    integration: IntegrationConfig = field(default_factory=IntegrationConfig)     # 集成配置（包含所有子配置）

    def with_updates(
        self,
        *,
        integration: IntegrationConfig | None = None,
    ) -> "ConfigBundle":
        """返回更新了指定配置的新ConfigBundle实例"""
        return ConfigBundle(integration=integration or self.integration)

    @property
    def training(self) -> TrainingConfig:
        """便捷属性：直接访问训练配置"""
        return self.integration.training

    @property
    def low_level_reward(self) -> LowLevelRewardConfig:
        """便捷属性：直接访问低层奖励配置"""
        return self.integration.low_level_reward

    @property
    def high_level_reward(self) -> HighLevelRewardConfig:
        """便捷属性：直接访问高层奖励配置"""
        return self.integration.high_level_reward

    @property
    def safety_critic(self) -> SafetyCriticConfig:
        """便捷属性：直接访问安全评估模型配置"""
        return self.integration.safety_critic


# 模块导出列表
__all__ = [
    "ShieldingConfig",
    "MotionConfig",
    "TriggerConfig",
    "PlannerConfig",
    "SafetyCriticConfig",
    "WindowRuntimeConfig",
    "LowLevelRewardConfig",
    "HighLevelRewardConfig",
    "TrainingConfig",
    "IntegrationConfig",
    "ConfigBundle",
]
