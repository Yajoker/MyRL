"""
基于事件触发分层安全强化学习的机器人导航训练主程序
使用ETHSRL(Event-Triggered Hierarchical Safe Reinforcement Learning)框架进行机器人导航
"""

from myRL import ETHSRLAgent  # 修改了导入路径
import torch
import numpy as np
import time
import os
from datetime import datetime

from robot_nav.SIM_ENV.sim import SIM  # 导入仿真环境
from utils.buffer import HierarchicalReplayBuffer  # 修改了导入路径

# 训练配置输出
print("=== ETHSRL Training Configuration ===")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
else:
    print("Training on CPU")
print("==================================")

def main(args=None):
    """主训练函数"""
    # ========== 训练参数配置 ==========
    action_dim = 2  # 模型输出的动作维度 [线速度, 角速度]
    max_action = 1  # 输出动作的最大绝对值
    state_dim = 95  # 神经网络输入状态的维度（状态向量的长度）
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # 设备选择：如果可用则使用CUDA，否则使用CPU
    nr_eval_episodes = 10  # 评估时使用的回合数
    max_epochs = 60  # 最大训练轮数
    epoch = 0  # 起始轮数
    episodes_per_epoch = 70  # 每个训练轮中运行的回合数
    episode = 0  # 起始回合数
    train_every_n = 2  # 每n个回合训练和更新一次网络参数
    training_iterations = 80  # 单次训练周期中使用的批次数
    batch_size = 64  # 每次训练迭代的批次大小
    max_steps = 300  # 单个回合中的最大步数
    steps = 0  # 起始步数
    load_saved_buffer = False  # 是否从assets/data.yml加载经验数据
    pretrain = False  # 是否使用加载的经验数据预训练模型（load_saved_buffer必须为True）
    pretraining_iterations = 10  # 预训练期间运行的训练迭代次数
    save_every = 5  # 每n个训练周期保存一次模型
    history_length = 8  # 历史观测长度，用于POMDP信念状态
    world_file = "worlds/env_a.yaml"  # 仿真环境世界文件
    save_dir = "saved_models/ETHSRL"  # 模型保存目录
    
    # ETHSRL特定参数
    risk_threshold = 0.7  # 风险触发阈值
    proximity_threshold = 0.5  # 障碍物接近阈值(米)
    heading_threshold = 0.3  # 航向变化阈值(弧度)
    min_trigger_interval = 1.0  # 最小触发间隔(秒)
    k_d = 1.0  # Lyapunov距离增益
    k_theta = 1.0  # Lyapunov角度增益
    v_threshold = 0.5  # Lyapunov阈值

    # 创建模型保存目录
    os.makedirs(save_dir, exist_ok=True)

    # ========== 训练初始化日志 ==========
    print("\n" + "="*60)
    print("🚀 Starting ETHSRL Navigation Training")
    print("="*60)
    print(f"📋 Training Configuration:")
    print(f"   • Device: {device}")
    print(f"   • State dim: {state_dim}, Action dim: {action_dim}")
    print(f"   • Max epochs: {max_epochs}, Episodes per epoch: {episodes_per_epoch}")
    print(f"   • Training iterations: {training_iterations}, Batch size: {batch_size}")
    print(f"   • History length: {history_length}")
    print(f"   • World file: {world_file}")
    print(f"   • Risk threshold: {risk_threshold}")
    print(f"   • Min trigger interval: {min_trigger_interval}s")
    print(f"   • Lyapunov parameters: k_d={k_d}, k_theta={k_theta}, V_threshold={v_threshold}")
    print("="*60)

    # ========== 模型初始化 ==========
    # 创建配置字典
    config = {
        'RISK_THRESHOLD': risk_threshold,
        'PROXIMITY_THRESHOLD': proximity_threshold,
        'HEADING_THRESHOLD': heading_threshold,
        'MIN_TRIGGER_INTERVAL': min_trigger_interval,
        'K_D': k_d,
        'K_THETA': k_theta,
        'V_THRESHOLD': v_threshold,
        'HISTORY_LENGTH': history_length,
        'TOTAL_EPOCHS': max_epochs,
        'BATCH_SIZE': batch_size,
        'TRAIN_EVERY_N': train_every_n
    }
    
    model = ETHSRLAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        config=config,
        device=device
    )

    # ========== 环境初始化 ==========
    sim = SIM(
        world_file=world_file, disable_plotting=False
    )  # 实例化仿真环境，启用图形显示

    # ========== 经验回放缓冲区初始化 ==========
    # 修改为直接使用HierarchicalReplayBuffer
    replay_buffer = HierarchicalReplayBuffer(
        easy_capacity=10000,
        medium_capacity=20000, 
        hard_capacity=30000,
        alpha=0.6,
        beta=0.4
    )

    # ========== 训练统计变量初始化 ==========
    episode_reward = 0.0  # 当前回合累计奖励
    epoch_total_reward = 0.0  # 当前epoch累计奖励
    epoch_total_steps = 0  # 当前epoch累计步数
    epoch_goal_count = 0  # 当前epoch成功到达目标次数
    epoch_collision_count = 0  # 当前epoch碰撞次数
    epoch_high_triggers = 0  # 当前epoch高层触发次数
    epoch_safety_triggers = 0  # 当前epoch安全层触发次数
    
    # ========== 初始状态获取 ==========
    # 执行初始步进，获取环境初始状态（零速度）
    latest_scan, distance, cos, sin, collision, goal, a, reward = sim.step(
        lin_velocity=0.0, ang_velocity=0.0
    )

    # ========== 主训练循环 ==========
    training_start_time = time.time()  # 记录训练开始时间
    print("\n📊 Training Progress:")
    
    while epoch < max_epochs:  # 训练直到达到最大轮数
        epoch_start_time = time.time()  # 记录当前epoch开始时间
        
        # 更新课程学习阶段
        model.update_curriculum(epoch)
        
        # 重置当前epoch的统计数据
        epoch_total_reward = 0.0
        epoch_total_steps = 0
        epoch_goal_count = 0
        epoch_collision_count = 0
        epoch_high_triggers = 0
        epoch_safety_triggers = 0
        
        # 每轮训练指定回合数
        while episode < episodes_per_epoch:
            # 准备当前状态表示
            state, terminal = model.prepare_state(
                latest_scan,  # 最新激光雷达扫描数据
                distance,     # 到目标的距离
                cos,          # 目标方向余弦值
                sin,          # 目标方向正弦值
                collision,    # 碰撞标志
                goal,         # 到达目标标志
                a             # 上一个动作
            )
            
            # 从模型获取动作（探索模式）
            goal_pos = [distance * cos, distance * sin]  # 相对目标位置
            action, info = model.select_action(state, goal_pos, scan_data=latest_scan)
            
            # 动作后处理：将线性速度从[-1,1]映射到[0,0.5]范围
            a_in = [
                (action[0] + 1) / 4,  # 线性速度：映射到[0, 0.5] m/s范围
                action[1],            # 角速度：保持原值
            ]

            # 在环境中执行动作，获取新状态和奖励
            latest_scan, distance, cos, sin, collision, goal, a, reward = sim.step(
                lin_velocity=a_in[0],   # 应用处理后的线速度
                ang_velocity=a_in[1]    # 应用角速度
            )
            
            # 准备下一个状态表示
            next_state, terminal = model.prepare_state(
                latest_scan, distance, cos, sin, collision, goal, a
            )
            
            # 存储经验
            model.store_experience(
                state, action, reward, terminal, next_state, info
            )
            
            # 同时添加到标准缓冲区（用于兼容）
            replay_buffer.add(
                state, action, next_state, reward, terminal
            )
            
            # 更新统计
            episode_reward += reward
            epoch_total_reward += reward
            steps += 1
            epoch_total_steps += 1
            
            # 记录触发事件
            if info['high_triggered']:
                epoch_high_triggers += 1
            if info['safety_triggered']:
                epoch_safety_triggers += 1

            # 检查回合是否结束（到达终止状态或达到最大步数）
            if terminal or steps == max_steps:
                # 记录成功/失败
                if goal:
                    epoch_goal_count += 1
                if collision:
                    epoch_collision_count += 1
                
                # 打印回合信息
                print(f"Episode {episode+1}/{episodes_per_epoch}: " + 
                      f"Reward={episode_reward:.2f}, Steps={steps}, " +
                      f"Result={'Success' if goal else 'Collision' if collision else 'Timeout'}, " +
                      f"High Triggers={model.high_trigger_count}, Safety Triggers={model.safety_trigger_count}")
                
                # 重置环境，获取新的初始状态
                latest_scan, distance, cos, sin, collision, goal, a, reward = sim.reset()
                episode += 1  # 回合计数增加
                
                # 定期训练模型
                if episode % train_every_n == 0:
                    model.train(batch_size=batch_size)  # 训练模型并更新参数

                # 重置回合统计
                episode_reward = 0.0
                steps = 0  # 重置步数计数器
                model.reset()  # 重置模型状态
            
        # 计算epoch统计
        success_rate = epoch_goal_count / episodes_per_epoch
        collision_rate = epoch_collision_count / episodes_per_epoch
        avg_reward = epoch_total_reward / episodes_per_epoch
        avg_steps = epoch_total_steps / episodes_per_epoch
        avg_high_triggers = epoch_high_triggers / episodes_per_epoch
        avg_safety_triggers = epoch_safety_triggers / episodes_per_epoch
        epoch_time = time.time() - epoch_start_time
        
        # 打印epoch结果
        print(f"\n== Epoch {epoch+1}/{max_epochs} 统计 ==")
        print(f"🎯 成功率: {success_rate:.2f}, 碰撞率: {collision_rate:.2f}")
        print(f"🏆 平均奖励: {avg_reward:.2f}, 平均步数: {avg_steps:.2f}")
        print(f"⚡ 平均高层触发: {avg_high_triggers:.2f}, 平均安全触发: {avg_safety_triggers:.2f}")
        print(f"⏱️ 用时: {epoch_time:.2f}秒")
        print("="*40)
        
        # 重置回合计数
        episode = 0
        epoch += 1  # 训练轮数增加
        
        # 保存模型
        if epoch % save_every == 0:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M")
            save_path = f"{save_dir}/ethsrl_epoch_{epoch}_{timestamp}.pt"
            model.save(save_path)
            print(f"📁 模型已保存: {save_path}")
        
        # 评估当前模型
        if epoch % save_every == 0:
            evaluate(model, sim, nr_eval_episodes, epoch)

    # 训练结束，保存最终模型
    final_path = f"{save_dir}/ethsrl_final.pt"
    model.save(final_path)
    print(f"📁 最终模型已保存: {final_path}")
    
    # 总训练时间
    total_training_time = time.time() - training_start_time
    hours = int(total_training_time // 3600)
    minutes = int((total_training_time % 3600) // 60)
    seconds = int(total_training_time % 60)
    print(f"\n🎉 训练完成! 总用时: {hours}h {minutes}m {seconds}s")
    
    # 最终评估
    print("\n🧪 开始最终评估...")
    evaluate(model, sim, nr_eval_episodes * 2, epoch)


def evaluate(model, sim, eval_episodes, epoch=None):
    """
    评估模型性能
    
    参数:
        model: 要评估的模型
        sim: 仿真环境实例
        eval_episodes: 评估回合数
        epoch: 当前训练轮数
    """
    print("\n" + "="*50)
    print(f"🔍 开始评估 (回合数: {eval_episodes})")
    print("="*50)
    
    # 初始化评估指标
    rewards = []
    steps_list = []
    success_count = 0
    collision_count = 0
    timeout_count = 0
    high_triggers_list = []
    safety_triggers_list = []
    
    for ep in range(eval_episodes):
        # 重置环境与模型
        latest_scan, distance, cos, sin, collision, goal, a, _ = sim.reset()
        model.reset()
        
        # 初始化回合统计
        episode_reward = 0
        steps = 0
        high_triggers_before = model.high_trigger_count
        safety_triggers_before = model.safety_trigger_count
        
        # 单回合循环
        done = False
        while not done and steps < 500:  # 评估用更长的步数限制
            # 准备当前状态
            state, _ = model.prepare_state(latest_scan, distance, cos, sin, collision, goal, a)
            
            # 选择动作（确定性模式）
            goal_pos = [distance * cos, distance * sin]
            action, _ = model.select_action(state, goal_pos, scan_data=latest_scan, deterministic=True)
            
            # 动作映射
            a_in = [(action[0] + 1) / 4, action[1]]
            
            # 执行动作
            latest_scan, distance, cos, sin, collision, goal, a, reward = sim.step(
                lin_velocity=a_in[0], ang_velocity=a_in[1]
            )
            
            # 更新统计
            episode_reward += reward
            steps += 1
            
            # 检查终止条件
            done = collision or goal
        
        # 计算此回合触发次数
        high_triggers = model.high_trigger_count - high_triggers_before
        safety_triggers = model.safety_trigger_count - safety_triggers_before
        
        # 更新统计
        rewards.append(episode_reward)
        steps_list.append(steps)
        high_triggers_list.append(high_triggers)
        safety_triggers_list.append(safety_triggers)
        
        if goal:
            success_count += 1
            result = "成功"
        elif collision:
            collision_count += 1
            result = "碰撞"
        else:
            timeout_count += 1
            result = "超时"
        
        # 打印回合结果
        print(f"回合 {ep+1}/{eval_episodes}: Reward={episode_reward:.2f}, " +
              f"步数={steps}, 结果={result}, " +
              f"高层触发={high_triggers}, 安全触发={safety_triggers}")
    
    # 计算评估指标
    success_rate = success_count / eval_episodes
    collision_rate = collision_count / eval_episodes
    timeout_rate = timeout_count / eval_episodes
    avg_reward = np.mean(rewards) if rewards else 0
    avg_steps = np.mean(steps_list) if steps_list else 0
    avg_high_triggers = np.mean(high_triggers_list) if high_triggers_list else 0
    avg_safety_triggers = np.mean(safety_triggers_list) if safety_triggers_list else 0
    
    # 打印评估结果
    print("\n📊 评估结果:")
    print(f"   • 成功率: {success_rate:.2f} ({success_count}/{eval_episodes})")
    print(f"   • 碰撞率: {collision_rate:.2f} ({collision_count}/{eval_episodes})")
    print(f"   • 超时率: {timeout_rate:.2f} ({timeout_count}/{eval_episodes})")
    print(f"   • 平均奖励: {avg_reward:.2f}")
    print(f"   • 平均步数: {avg_steps:.2f}")
    print(f"   • 平均高层触发: {avg_high_triggers:.2f}")
    print(f"   • 平均安全触发: {avg_safety_triggers:.2f}")
    
    # 记录评估指标到模型的TensorBoard
    if hasattr(model, 'writer'):
        model.writer.add_scalar("eval/success_rate", success_rate, epoch)
        model.writer.add_scalar("eval/collision_rate", collision_rate, epoch)
        model.writer.add_scalar("eval/avg_reward", avg_reward, epoch)
        model.writer.add_scalar("eval/avg_steps", avg_steps, epoch)
        model.writer.add_scalar("eval/avg_high_triggers", avg_high_triggers, epoch)
        model.writer.add_scalar("eval/avg_safety_triggers", avg_safety_triggers, epoch)
    
    print("="*50)
    return {
        'success_rate': success_rate,
        'collision_rate': collision_rate,
        'avg_reward': avg_reward,
        'avg_steps': avg_steps,
        'avg_high_triggers': avg_high_triggers,
        'avg_safety_triggers': avg_safety_triggers
    }


if __name__ == "__main__":
    main()  # 程序入口点，启动主训练函数
