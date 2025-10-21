"""默认配置参数"""

class DefaultConfig:
    # 全局路径规划器参数
    PLANNER_RESOLUTION = 0.1
    LOCAL_WINDOW_SIZE = 10
    REPLAN_THRESHOLD = 5  # 当障碍物接近路径多少次时重新规划
    
    # 高层触发器参数
    RISK_THRESHOLD = 0.7
    PROXIMITY_THRESHOLD = 0.5  # 米
    HEADING_THRESHOLD = 0.3    # 弧度
    SUBGOAL_CHECK_DIST = 2.0   # 子目标可达性检查距离
    MIN_TRIGGER_INTERVAL = 1.0  # 秒
    ENV_COMPLEXITY_FACTOR = 0.3
    
    # 感知模块参数
    GRID_SIZE = 64
    RESOLUTION = 0.1
    MAX_RANGE = 10.0
    HISTORY_LENGTH = 8
    
    # 安全层参数
    K_D = 1.0
    K_THETA = 1.0
    K_PATH = 0.5
    V_THRESHOLD = 0.5
    EPSILON = 0.01
    BLEND_FACTOR = 10
    BLEND_THRESHOLD = 0.2
    
    # PPO策略参数
    HIGH_LEVEL_HIDDEN_DIM = 256
    LOW_LEVEL_HIDDEN_DIM = 256
    PPO_CLIP_RATIO = 0.2
    PPO_VALUE_COEF = 0.5
    PPO_ENTROPY_COEF = 0.01
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    LR_ACTOR = 3e-4
    LR_CRITIC = 1e-3
    MAX_GRAD_NORM = 0.5
    
    # 经验回放参数
    EASY_BUFFER_SIZE = 10000
    MEDIUM_BUFFER_SIZE = 20000
    HARD_BUFFER_SIZE = 30000
    ALPHA = 0.6
    BETA = 0.4
    BETA_INCREMENT = 0.001
    
    # 课程学习参数
    TOTAL_EPOCHS = 3000
    STAGE1_RATIO = 0.3
    STAGE2_RATIO = 0.7
    
    # 训练参数
    BATCH_SIZE = 64
    TRAIN_EPOCHS = 60
    EPISODES_PER_EPOCH = 70
    MAX_STEPS = 300
    TRAIN_EVERY_N = 2
    TRAINING_ITERATIONS = 80
    EVAL_EPISODES = 10
    SAVE_EVERY = 5