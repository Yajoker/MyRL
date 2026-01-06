from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class LowLevelRewardConfig:
    """Reward shaping coefficients for the low-level controller."""

    # 1. 子目标进展相关
    progress_weight: float = 5.0          
    efficiency_penalty: float = 0.5      # 时间步惩罚

    # 2. 安全相关
    safety_weight: float = 1.0            # 提高安全项权重
    safety_sensitivity: float = 0.0       # 暂时保留但不使用（或直接删掉）
    safety_clearance: float = 0.6
    collision_distance: float = 0.3

    # 3. 终局项：在低层只保留很小的局部效果
    goal_bonus: float = 30.0              
    subgoal_bonus: float = 0.0            
    collision_penalty: float = -20.0       
    timeout_penalty: float = -20.0          

    def __post_init__(self) -> None:  # type: ignore[override]
        """数据类初始化后验证方法"""
        if self.efficiency_penalty < 0:
            raise ValueError("efficiency_penalty must be non-negative")
        if self.safety_clearance <= 0:
            raise ValueError("safety_clearance must be positive")


@dataclass(frozen=True)
class HighLevelRewardConfig:
    """Reward shaping coefficients for the high-level planner."""

    # 高层回报 Gi = alpha * Δd_global - beta_col * I[collision] - beta_time * T
    alpha_global_progress: float = 4.0      # 到最终目标距离减少的权重
    beta_collision: float = 80.0            # 碰撞惩罚权重
    beta_time: float = 0.5                  # 时间步惩罚权重
    gamma_short_cost: float = 1.0           # 子目标执行期间的累计安全成本权重
    gamma_near_steps: float = 0.5           # 近障碍步数权重
    lambda_col: float = 1.0                 # 每步安全成本中的碰撞权重
    lambda_near: float = 0.2                # 每步安全成本中的近障碍权重


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
    # 进度阈值：Δd_min(d) = eps_abs + eps_rel * d
    progress_epsilon_abs: float = 0.2                # 绝对改善下限（米）
    progress_epsilon_rel: float = 0.05               # 相对改善比例

    # 触发节奏：下限 + 上限
    min_interval: float = 1.2                        # 最小触发间隔时间（秒，优先使用步数配置）
    min_step_interval: int = 10                      # 最小触发间隔步数
    max_interval: float = 3.6                        # 最大触发间隔时间（秒，优先使用步数配置）
    max_step_interval: int = 30                      # 最大触发间隔步数

    # 风险判定：统一 risk_index
    risk_alpha: float = 0.6                          # min 与分位数加权
    risk_trigger_threshold: float = 0.8              # 事件触发阈值
    risk_near_threshold: float = 0.3                 # 近障计数阈值（奖励用）
    risk_percentile: float = 10.0                    # 计算分位数使用的百分位
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
        if self.progress_epsilon_abs < 0:
            raise ValueError("progress_epsilon_abs must be non-negative")
        if self.progress_epsilon_rel < 0:
            raise ValueError("progress_epsilon_rel must be non-negative")
        if self.min_interval < 0:
            raise ValueError("min_interval must be non-negative")
        if self.min_step_interval <= 0:
            raise ValueError("min_step_interval must be positive")
        if self.max_interval < 0:
            raise ValueError("max_interval must be non-negative")
        if self.max_step_interval <= 0:
            raise ValueError("max_step_interval must be positive")
        if self.max_step_interval < self.min_step_interval:
            raise ValueError("max_step_interval must be >= min_step_interval")
        if self.risk_alpha < 0 or self.risk_alpha > 1:
            raise ValueError("risk_alpha must be in [0, 1]")
        if self.risk_trigger_threshold < 0:
            raise ValueError("risk_trigger_threshold must be non-negative")
        if self.risk_near_threshold < 0:
            raise ValueError("risk_near_threshold must be non-negative")
        if self.risk_percentile <= 0 or self.risk_percentile >= 100:
            raise ValueError("risk_percentile must be in (0, 100)")
        if self.window_inside_hold < 0:
            raise ValueError("window_inside_hold must be non-negative")
        if not 0.0 <= self.subgoal_smoothing_alpha < 1.0:
            raise ValueError("subgoal_smoothing_alpha must be in [0, 1)")


@dataclass(frozen=True)
class PlannerConfig:
    """High-level subgoal configuration for mapless navigation."""

    waypoint_lookahead: int = 3                      # 高层输入的前瞻占位维度
    anchor_radius: float = 0.6                       # 子目标基准半径（用于距离/角度裁剪）

    # 前沿引导的候选子目标生成
    frontier_num_candidates: int = 7                 # 每次生成的候选子目标总数
    frontier_min_dist: float = 0.8                   # 子目标距离下限（米）
    frontier_max_dist: float = 3.5                   # 子目标距离上限（米）
    frontier_gap_min_width: float = 0.2              # 最小前沿角宽（弧度）
    diverse_frontier_enabled: bool = True           # 是否启用多桶候选保留策略
    frontier_bucket_k_align: int = 3                 # 与目标方向对齐的保留数量
    frontier_bucket_k_clear: int = 2                 # 空旷度优先保留数量
    frontier_bucket_k_diverse: int = 2               # 多样性采样保留数量
    frontier_clear_window: int = 3                   # 计算空旷度时的窗口半径
    frontier_diverse_method: str = "farthest_angle"  # 多样性采样方式
    frontier_keep_goal_candidate: bool = True        # 是否强制保留目标方向候选

    # 连续性约束参数
    consistency_lambda: float = 0.5
    consistency_sigma_r: float = 1.0
    consistency_sigma_theta: float = 0.5

    # === 新增：多目标 Q 组合相关超参 ===
    safety_q_weight: float = 1.0       # λ_q: 决策时 Q_safe 的权重
    safety_loss_weight: float = 1.0    # λ_safe_loss: 训练时安全头 loss 的权重
    high_level_gamma: float = 0.99     # 高层 TD 折扣因子
    high_level_tau: float = 0.005      # 高层目标网络软更新系数
    high_level_double_q_enabled: bool = True        # 是否启用双价值网络
    high_level_double_q_update_mode: str = "alternate"  # 双Q更新模式
    high_level_double_q_fuse_mode: str = "mean"         # 推理融合方式
    high_level_double_q_target_eval: bool = True         # 目标网络是否用于评估
    high_level_double_q_log_net_id: bool = True          # 是否记录本轮更新的网络编号


    def __post_init__(self) -> None:  # type: ignore[override]
        """数据类初始化后验证方法"""
        if self.waypoint_lookahead <= 0:
            raise ValueError("waypoint_lookahead must be positive")
        if self.anchor_radius <= 0:
            raise ValueError("anchor_radius must be positive")
        if self.frontier_num_candidates <= 0:
            raise ValueError("frontier_num_candidates must be positive")
        if self.frontier_min_dist <= 0:
            raise ValueError("frontier_min_dist must be positive")
        if self.frontier_max_dist <= 0:
            raise ValueError("frontier_max_dist must be positive")
        if self.frontier_gap_min_width <= 0:
            raise ValueError("frontier_gap_min_width must be positive")
        if self.frontier_bucket_k_align < 0 or self.frontier_bucket_k_clear < 0 or self.frontier_bucket_k_diverse < 0:
            raise ValueError("frontier bucket sizes must be non-negative")
        total_bucket = self.frontier_bucket_k_align + self.frontier_bucket_k_clear + self.frontier_bucket_k_diverse
        if total_bucket > self.frontier_num_candidates:
            raise ValueError("sum of frontier buckets must not exceed frontier_num_candidates")
        if self.frontier_clear_window < 0:
            raise ValueError("frontier_clear_window must be non-negative")
        if self.consistency_lambda < 0:
            raise ValueError("consistency_lambda must be non-negative")
        if self.consistency_sigma_r <= 0:
            raise ValueError("consistency_sigma_r must be positive")
        if self.consistency_sigma_theta <= 0:
            raise ValueError("consistency_sigma_theta must be positive")
        if self.safety_q_weight < 0:
            raise ValueError("safety_q_weight must be non-negative")
        if self.safety_loss_weight < 0:
            raise ValueError("safety_loss_weight must be non-negative")
        if not 0 < self.high_level_gamma <= 1:
            raise ValueError("high_level_gamma must be in (0, 1]")
        if not 0 < self.high_level_tau <= 1:
            raise ValueError("high_level_tau must be in (0, 1]")
        if self.high_level_double_q_update_mode not in {"alternate"}:
            raise ValueError("high_level_double_q_update_mode must be 'alternate'")
        if self.high_level_double_q_fuse_mode not in {"mean", "min"}:
            raise ValueError("high_level_double_q_fuse_mode must be 'mean' or 'min'")


@dataclass(frozen=True)
class TrainingConfig:
    """End-to-end training and evaluation hyper-parameters."""

    buffer_size: int = 100_000                        # 经验回放缓冲区大小
    batch_size: int = 128                            # 训练批次大小
    max_epochs: int = 60                             # 最大训练周期数
    episodes_per_epoch: int = 70                     # 每个周期的回合数
    max_steps: int = 550                             # 每个回合的最大步数
    train_every_n_episodes: int = 1                  # 每N个回合训练一次
    training_iterations: int = 10                    # 每次训练的迭代次数
    exploration_noise: float = 0.17                  # 探索噪声系数
    min_buffer_size: int = 0                         # 开始训练的最小缓冲区大小
    max_lin_velocity: float = 1.0                    # 最大线速度
    max_ang_velocity: float = 1.0                    # 最大角速度
    eval_episodes: int = 10                          # 评估回合数
    subgoal_radius: float = 0.4                      # 子目标判定阈值
    save_every: int = 5                              # 保存模型的频率（每N个周期）
    world_file: str = "env1.yaml"                  # 环境配置文件
    waypoint_lookahead: int = 3                      # 高层使用的前瞻航点数
    discount: float = 0.99                           # 折扣因子
    tau: float = 0.005                               # 目标网络软更新系数
    policy_noise: float = 0.15                        # 策略噪声
    noise_clip: float = 0.3                          # 噪声裁剪范围
    policy_freq: int = 2                            # 策略更新频率
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
    low_level_reward: LowLevelRewardConfig = field(default_factory=LowLevelRewardConfig)  # 低层奖励配置
    high_level_reward: HighLevelRewardConfig = field(default_factory=HighLevelRewardConfig)  # 高层奖励配置
    training: TrainingConfig = field(default_factory=TrainingConfig)              # 训练配置

    def with_updates(
        self,
        *,
        motion: MotionConfig | None = None,
        trigger: TriggerConfig | None = None,
        planner: PlannerConfig | None = None,
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


# 模块导出列表
__all__ = [
    "ShieldingConfig",
    "MotionConfig",
    "TriggerConfig",
    "PlannerConfig",
    "LowLevelRewardConfig",
    "HighLevelRewardConfig",
    "TrainingConfig",
    "IntegrationConfig",
    "ConfigBundle",
]
