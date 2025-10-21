"""课程学习调度器 - 控制训练过程中的参数变化"""

import numpy as np

class CurriculumScheduler:
    """
    课程学习调度器，根据训练阶段调整参数
    """
    def __init__(self, total_epochs=3000, stage1_ratio=0.3, stage2_ratio=0.7):
        """
        初始化课程学习调度器
        
        参数:
            total_epochs: 总训练轮数
            stage1_ratio: 阶段1结束时的进度比例
            stage2_ratio: 阶段2结束时的进度比例
        """
        self.total_epochs = total_epochs
        self.stage1_end = int(total_epochs * stage1_ratio)
        self.stage2_end = int(total_epochs * stage2_ratio)
        self.current_epoch = 0
        
        # 初始参数
        self.initial_risk_threshold = 0.5  # 风险阈值(初始更敏感)
        self.final_risk_threshold = 0.8    # 风险阈值(最终更稳健)
        
        self.initial_learning_rate = 3e-4  # 初始学习率
        self.final_learning_rate = 1e-5    # 最终学习率
        
        self.initial_noise_scale = 0.3     # 初始探索噪声(更大)
        self.final_noise_scale = 0.05      # 最终探索噪声(更小)
    
    def update(self, epoch):
        """更新当前轮数"""
        self.current_epoch = epoch
    
    def get_stage(self):
        """
        获取当前训练阶段
        
        返回:
            stage: 训练阶段(1, 2, 3)
        """
        if self.current_epoch < self.stage1_end:
            return 1  # 阶段1：基础技能学习
        elif self.current_epoch < self.stage2_end:
            return 2  # 阶段2：安全校准
        else:
            return 3  # 阶段3：端到端微调
    
    def get_risk_threshold(self):
        """
        获取当前风险阈值
        
        返回:
            risk_threshold: 风险阈值
        """
        if self.current_epoch < self.stage2_end:
            # 阶段1和2：线性增加风险阈值
            progress = self.current_epoch / self.stage2_end
            return self.initial_risk_threshold + progress * (self.final_risk_threshold - self.initial_risk_threshold)
        else:
            # 阶段3：使用最终风险阈值
            return self.final_risk_threshold
    
    def get_learning_rate(self):
        """
        获取当前学习率
        
        返回:
            lr: 学习率
        """
        if self.current_epoch < self.stage2_end:
            # 阶段1和2：缓慢减小学习率
            progress = self.current_epoch / self.stage2_end
            return self.initial_learning_rate * (1 - 0.5 * progress)
        else:
            # 阶段3：使用较小的学习率进行微调
            return self.final_learning_rate
    
    def get_exploration_noise(self):
        """
        获取当前探索噪声尺度
        
        返回:
            noise_scale: 探索噪声尺度
        """
        # 随着训练进行，降低探索噪声
        progress = min(1.0, self.current_epoch / self.total_epochs)
        return self.initial_noise_scale + progress * (self.final_noise_scale - self.initial_noise_scale)
    
    def should_train_high_level(self):
        """是否应该训练高层策略"""
        stage = self.get_stage()
        return stage in [1, 3]  # 阶段1和3训练高层
    
    def should_train_low_level(self):
        """是否应该训练低层策略"""
        stage = self.get_stage()
        return stage in [1, 3]  # 阶段1和3训练低层
    
    def should_train_safety_layer(self):
        """是否应该训练安全层"""
        stage = self.get_stage()
        return stage in [2, 3]  # 阶段2和3训练安全层
    
    def should_use_safety_layer(self):
        """是否应该使用安全层"""
        stage = self.get_stage()
        return stage in [2, 3]  # 阶段2和3使用安全层