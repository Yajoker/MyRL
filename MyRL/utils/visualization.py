"""可视化工具 - 用于训练监控和结果分析"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arrow
import seaborn as sns

def plot_trajectory(trajectory, goal, obstacles=None, path=None, subgoals=None, safety_interventions=None, figsize=(10, 10)):
    """
    可视化机器人轨迹和环境
    
    参数:
        trajectory: 机器人轨迹点列表 [(x1, y1), (x2, y2), ...]
        goal: 目标位置 [x, y]
        obstacles: 障碍物列表 [[x1, y1, r1], [x2, y2, r2], ...] (可选)
        path: 全局路径 [(x1, y1), (x2, y2), ...] (可选)
        subgoals: 子目标列表 [(x1, y1), (x2, y2), ...] (可选)
        safety_interventions: 安全干预点列表 [(x1, y1), (x2, y2), ...] (可选)
        figsize: 图像大小
    """
    plt.figure(figsize=figsize)
    
    # 绘制轨迹
    if trajectory:
        x = [p[0] for p in trajectory]
        y = [p[1] for p in trajectory]
        plt.plot(x, y, 'b-', linewidth=2, label='Robot Trajectory')
        
        # 绘制起点
        plt.plot(x[0], y[0], 'go', markersize=10, label='Start')
    
    # 绘制目标
    plt.plot(goal[0], goal[1], 'r*', markersize=15, label='Goal')
    
    # 绘制全局路径
    if path:
        path_x = [p[0] for p in path]
        path_y = [p[1] for p in path]
        plt.plot(path_x, path_y, 'g--', linewidth=1, label='Global Path')
    
    # 绘制子目标
    if subgoals:
        sg_x = [sg[0] for sg in subgoals]
        sg_y = [sg[1] for sg in subgoals]
        plt.scatter(sg_x, sg_y, c='cyan', s=80, marker='x', label='Subgoals')
    
    # 绘制安全干预点
    if safety_interventions:
        si_x = [si[0] for si in safety_interventions]
        si_y = [si[1] for si in safety_interventions]
        plt.scatter(si_x, si_y, c='red', s=50, marker='+', label='Safety Interventions')
    
    # 绘制障碍物
    if obstacles:
        ax = plt.gca()
        for obs in obstacles:
            x, y = obs[0], obs[1]
            r = obs[2] if len(obs) > 2 else 0.5  # 默认半径0.5
            circle = Circle((x, y), r, fill=True, color='gray', alpha=0.5)
            ax.add_patch(circle)
    
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.title('Robot Navigation Trajectory')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.tight_layout()
    
    return plt.gcf()

def plot_training_metrics(metrics_dict, figsize=(15, 10)):
    """
    绘制训练指标
    
    参数:
        metrics_dict: 指标字典 {'metric_name': [values...], ...}
        figsize: 图像大小
    """
    n_metrics = len(metrics_dict)
    fig, axes = plt.subplots(n_metrics, 1, figsize=figsize, sharex=True)
    
    if n_metrics == 1:
        axes = [axes]
    
    for i, (metric_name, values) in enumerate(metrics_dict.items()):
        axes[i].plot(values)
        axes[i].set_title(metric_name)
        axes[i].grid(True)
    
    plt.tight_layout()
    return fig

def plot_trigger_analysis(high_triggers, safety_triggers, figsize=(12, 6)):
    """
    分析事件触发情况
    
    参数:
        high_triggers: 高层触发记录列表 [(time, reason), ...]
        safety_triggers: 安全层触发记录列表 [(time, V_value), ...]
        figsize: 图像大小
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # 分析高层触发原因
    if high_triggers:
        reasons = [t[1] for t in high_triggers]
        reason_counts = {}
        for r in reasons:
            reason_counts[r] = reason_counts.get(r, 0) + 1
        
        ax1.bar(reason_counts.keys(), reason_counts.values())
        ax1.set_title('High-Level Trigger Reasons')
        ax1.set_ylabel('Count')
        ax1.grid(True)
    
    # 分析安全层触发值
    if safety_triggers:
        v_values = [t[1] for t in safety_triggers]
        ax2.hist(v_values, bins=20)
        ax2.set_title('Safety Trigger Lyapunov Values')
        ax2.set_xlabel('V Value')
        ax2.set_ylabel('Frequency')
        ax2.grid(True)
    
    plt.tight_layout()
    return fig

def plot_attention_heatmap(attention_weights, scan_data=None, figsize=(8, 8)):
    """
    可视化注意力权重
    
    参数:
        attention_weights: 注意力权重矩阵
        scan_data: 对应的激光扫描数据 (可选)
        figsize: 图像大小
    """
    plt.figure(figsize=figsize)
    
    # 绘制注意力热图
    sns.heatmap(attention_weights, cmap='viridis')
    plt.title('Attention Weights Heatmap')
    
    # 如果提供了扫描数据，绘制极坐标图
    if scan_data is not None:
        plt.figure(figsize=figsize)
        
        # 计算角度
        angles = np.linspace(0, 2*np.pi, len(scan_data), endpoint=False)
        
        # 创建极坐标图
        ax = plt.subplot(111, polar=True)
        ax.plot(angles, scan_data)
        ax.set_title('Lidar Scan Data (Polar)')
        ax.grid(True)
    
    plt.tight_layout()
    return plt.gcf()