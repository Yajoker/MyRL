"""
全局路径规划器，集成A*等经典规划算法
"""

import numpy as np
import heapq
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt

class GlobalPathPlanner:
    """
    全局路径规划器，用于生成从起点到目标的路径
    """
    def __init__(self, resolution=0.1, obstacle_inflation=0.2):
        """
        初始化路径规划器
        
        参数:
            resolution: 栅格分辨率（米/像素）
            obstacle_inflation: 障碍物膨胀距离
        """
        self.resolution = resolution
        self.obstacle_inflation = obstacle_inflation
        
        # 内部状态
        self._global_path = None
        self._local_window = None
        self._grid = None
        self._grid_resolution = resolution
        self._grid_offset = (0, 0)
        
        # A*搜索的方向（8个方向）
        self._directions = [
            (1, 0), (0, 1), (-1, 0), (0, -1),  # 上下左右
            (1, 1), (1, -1), (-1, 1), (-1, -1)  # 对角线
        ]
        
        # 平滑参数
        self._smooth_weight_data = 0.5
        self._smooth_weight_smooth = 0.3
        self._smooth_tolerance = 0.000001
        
    def reset(self):
        """
        重置规划器状态
        """
        self._global_path = None
        self._local_window = None
        self._grid = None
        self._grid_resolution = self.resolution
        self._grid_offset = (0, 0)
        
    def plan_path(self, start, goal, obstacles=None):
        """
        规划从起点到终点的路径
        
        参数:
            start: 起点坐标 (x, y)
            goal: 终点坐标 (x, y)
            obstacles: 障碍物列表，每个元素为 (x, y, radius)
            
        返回:
            path: 路径点列表，每个元素为 (x, y)
        """
        # 将坐标转换为网格索引
        start_node = self._coord_to_node(start)
        goal_node = self._coord_to_node(goal)
        
        # 使用A*算法寻找路径
        path_nodes = self._astar(start_node, goal_node, obstacles)
        
        if not path_nodes:
            print("警告: 无法找到有效路径，使用直线路径")
            # 如果找不到路径，返回直线路径
            self._global_path = [start, goal]
            return self._global_path
        
        # 将网格索引转换回坐标
        self._global_path = [self._node_to_coord(node) for node in path_nodes]
        
        # 路径平滑
        if len(self._global_path) > 2:
            self._global_path = self._smooth_path(self._global_path)
        
        return self._global_path
    
    def get_local_path_window(self, current_pos, window_size=10):
        """
        获取当前位置周围的局部路径窗口
        
        参数:
            current_pos: 当前位置 (x, y)
            window_size: 窗口大小，默认10个路径点
            
        返回:
            local_path: 局部路径点列表
        """
        if self._global_path is None:
            raise ValueError("全局路径尚未规划")
        
        if not self._global_path:  # 检查路径是否为空列表
            return [current_pos]   # 返回仅包含当前位置的路径
        
        # 找到全局路径上离当前位置最近的点
        min_dist = float('inf')
        min_idx = 0
        
        for i, point in enumerate(self._global_path):
            dist = ((point[0] - current_pos[0]) ** 2 + 
                    (point[1] - current_pos[1]) ** 2) ** 0.5
            if dist < min_dist:
                min_dist = dist
                min_idx = i
        
        # 获取从最近点开始的局部窗口
        end_idx = min(min_idx + window_size, len(self._global_path))
        self._local_window = self._global_path[min_idx:end_idx]
        
        # 如果局部窗口为空（可能是因为已经到达路径末尾），返回最后一个点
        if not self._local_window and self._global_path:
            self._local_window = [self._global_path[-1]]
        
        return self._local_window
    
    def get_path_distance(self, current_pos):
        """
        计算当前位置到目标的路径距离
        
        参数:
            current_pos: 当前位置 (x, y)
            
        返回:
            path_distance: 路径距离
        """
        if not self._global_path:
            return float('inf')
        
        # 找到全局路径上离当前位置最近的点
        min_dist = float('inf')
        min_idx = 0
        
        for i, point in enumerate(self._global_path):
            dist = ((point[0] - current_pos[0]) ** 2 + 
                    (point[1] - current_pos[1]) ** 2) ** 0.5
            if dist < min_dist:
                min_dist = dist
                min_idx = i
        
        # 计算从最近点到终点的路径距离
        path_distance = 0.0
        for i in range(min_idx, len(self._global_path) - 1):
            p1 = self._global_path[i]
            p2 = self._global_path[i + 1]
            path_distance += ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
            
        return path_distance
    
    def get_next_waypoint(self, current_pos, lookahead_distance=1.0):
        """
        获取前方指定距离的路径点
        
        参数:
            current_pos: 当前位置 (x, y)
            lookahead_distance: 前视距离
            
        返回:
            waypoint: 路径点坐标 (x, y)
        """
        if not self._global_path:
            return current_pos
        
        # 找到全局路径上离当前位置最近的点
        min_dist = float('inf')
        min_idx = 0
        
        for i, point in enumerate(self._global_path):
            dist = ((point[0] - current_pos[0]) ** 2 + 
                    (point[1] - current_pos[1]) ** 2) ** 0.5
            if dist < min_dist:
                min_dist = dist
                min_idx = i
        
        # 从最近点开始，找到前方指定距离的路径点
        accumulated_distance = 0.0
        for i in range(min_idx, len(self._global_path) - 1):
            p1 = self._global_path[i]
            p2 = self._global_path[i + 1]
            segment_distance = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
            
            if accumulated_distance + segment_distance >= lookahead_distance:
                # 计算在此线段上的插值点
                ratio = (lookahead_distance - accumulated_distance) / segment_distance
                x = p1[0] + ratio * (p2[0] - p1[0])
                y = p1[1] + ratio * (p2[1] - p1[1])
                return (x, y)
            
            accumulated_distance += segment_distance
        
        # 如果没有找到前方点，返回路径最后一点
        return self._global_path[-1]
    
    def visualize_path(self, obstacles=None, show=True):
        """
        可视化全局路径和障碍物
        
        参数:
            obstacles: 障碍物列表
            show: 是否显示图像
        """
        if self._global_path is None:
            print("没有可视化的路径")
            return
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # 绘制全局路径
        path_x = [p[0] for p in self._global_path]
        path_y = [p[1] for p in self._global_path]
        ax.plot(path_x, path_y, 'b-', linewidth=2, label='Global Path')
        
        # 绘制起点和终点
        if self._global_path:
            ax.plot(self._global_path[0][0], self._global_path[0][1], 'go', markersize=10, label='Start')
            ax.plot(self._global_path[-1][0], self._global_path[-1][1], 'ro', markersize=10, label='Goal')
            
        # 绘制局部窗口
        if self._local_window:
            local_x = [p[0] for p in self._local_window]
            local_y = [p[1] for p in self._local_window]
            ax.plot(local_x, local_y, 'g-', linewidth=3, alpha=0.7, label='Local Window')
            
        # 绘制障碍物
        if obstacles:
            for obs in obstacles:
                if len(obs) == 3:  # (x, y, radius) 格式
                    x, y, r = obs
                    circle = plt.Circle((x, y), r, color='r', alpha=0.3)
                    ax.add_artist(circle)
                else:  # (x, y) 格式
                    x, y = obs
                    ax.plot(x, y, 'rx', markersize=5)
        
        ax.set_aspect('equal')
        ax.legend()
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Global Path Planning')
        ax.grid(True)
        
        if show:
            plt.show()
        
        return fig, ax
    
    def _create_grid(self, start, goal, obstacles=None):
        """
        创建路径规划的栅格地图
        
        参数:
            start: 起点坐标
            goal: 终点坐标
            obstacles: 障碍物列表
            
        返回:
            grid: 栅格地图（0表示自由空间，1表示障碍物）
        """
        # 确定地图边界
        min_x = min(start[0], goal[0]) - 5.0
        max_x = max(start[0], goal[0]) + 5.0
        min_y = min(start[1], goal[1]) - 5.0
        max_y = max(start[1], goal[1]) + 5.0
        
        # 确保包含所有障碍物
        if obstacles:
            for obs in obstacles:
                if len(obs) == 3:  # (x, y, radius) 格式
                    x, y, r = obs
                    min_x = min(min_x, x - r - self.obstacle_inflation)
                    max_x = max(max_x, x + r + self.obstacle_inflation)
                    min_y = min(min_y, y - r - self.obstacle_inflation)
                    max_y = max(max_y, y + r + self.obstacle_inflation)
                else:  # (x, y) 格式
                    x, y = obs
                    min_x = min(min_x, x - self.obstacle_inflation)
                    max_x = max(max_x, x + self.obstacle_inflation)
                    min_y = min(min_y, y - self.obstacle_inflation)
                    max_y = max(max_y, y + self.obstacle_inflation)
        
        # 确定栅格尺寸
        width = int((max_x - min_x) / self.resolution) + 1
        height = int((max_y - min_y) / self.resolution) + 1
        
        # 创建栅格地图
        grid = np.zeros((height, width), dtype=np.uint8)
        
        # 记录网格偏移量
        self._grid_offset = (min_x, min_y)
        self._grid_resolution = self.resolution
        
        # 标记障碍物
        if obstacles:
            for obs in obstacles:
                if len(obs) == 3:  # (x, y, radius) 格式
                    x, y, r = obs
                    # 膨胀障碍物半径
                    inflated_r = r + self.obstacle_inflation
                    # 转换为栅格坐标
                    grid_x = int((x - min_x) / self.resolution)
                    grid_y = int((y - min_y) / self.resolution)
                    grid_r = int(inflated_r / self.resolution)
                    
                    # 标记圆形障碍物
                    y_indices, x_indices = np.ogrid[-grid_y:height-grid_y, -grid_x:width-grid_x]
                    mask = x_indices*x_indices + y_indices*y_indices <= grid_r*grid_r
                    grid[mask] = 1
                else:  # (x, y) 格式
                    x, y = obs
                    # 转换为栅格坐标
                    grid_x = int((x - min_x) / self.resolution)
                    grid_y = int((y - min_y) / self.resolution)
                    
                    # 膨胀点障碍物
                    inflation_cells = int(self.obstacle_inflation / self.resolution)
                    for i in range(-inflation_cells, inflation_cells + 1):
                        for j in range(-inflation_cells, inflation_cells + 1):
                            if (i*i + j*j <= inflation_cells*inflation_cells and
                                0 <= grid_y + i < height and
                                0 <= grid_x + j < width):
                                grid[grid_y + i, grid_x + j] = 1
        
        self._grid = grid
        return grid
    
    def _coord_to_node(self, coord):
        """
        将物理坐标转换为栅格节点索引
        
        参数:
            coord: 物理坐标 (x, y)
            
        返回:
            node: 栅格索引 (row, col)
        """
        if self._grid is None:
            return (int(coord[1] / self.resolution), int(coord[0] / self.resolution))
        
        col = int((coord[0] - self._grid_offset[0]) / self._grid_resolution)
        row = int((coord[1] - self._grid_offset[1]) / self._grid_resolution)
        return (row, col)
    
    def _node_to_coord(self, node):
        """
        将栅格节点索引转换为物理坐标
        
        参数:
            node: 栅格索引 (row, col)
            
        返回:
            coord: 物理坐标 (x, y)
        """
        if self._grid is None:
            return (node[1] * self.resolution, node[0] * self.resolution)
        
        x = node[1] * self._grid_resolution + self._grid_offset[0]
        y = node[0] * self._grid_resolution + self._grid_offset[1]
        return (x, y)
    
    def _heuristic(self, node, goal):
        """
        A*算法的启发式函数（欧几里得距离）
        
        参数:
            node: 当前节点 (row, col)
            goal: 目标节点 (row, col)
            
        返回:
            distance: 估计距离
        """
        return ((node[0] - goal[0]) ** 2 + (node[1] - goal[1]) ** 2) ** 0.5
    
    def _astar(self, start, goal, obstacles=None):
        """
        A*路径规划算法
        
        参数:
            start: 起点节点 (row, col)
            goal: 终点节点 (row, col)
            obstacles: 障碍物列表
            
        返回:
            path: 路径节点列表
        """
        # 创建栅格地图
        grid = self._create_grid(self._node_to_coord(start), self._node_to_coord(goal), obstacles)
        height, width = grid.shape
        
        # 检查起点和终点是否在栅格范围内且不是障碍物
        if (start[0] < 0 or start[0] >= height or
            start[1] < 0 or start[1] >= width or
            goal[0] < 0 or goal[0] >= height or
            goal[1] < 0 or goal[1] >= width):
            print("起点或终点超出栅格范围")
            return []
        
        if grid[start[0], start[1]] == 1 or grid[goal[0], goal[1]] == 1:
            print("起点或终点位于障碍物内")
            return []
        
        # 初始化开放列表和关闭列表
        open_list = []
        closed_set = set()
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self._heuristic(start, goal)}
        
        # 将起点加入开放列表
        heapq.heappush(open_list, (f_score[start], start))
        
        while open_list:
            # 获取f值最小的节点
            _, current = heapq.heappop(open_list)
            
            # 如果到达目标，构建路径并返回
            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return path[::-1]  # 反转路径，从起点到终点
            
            # 将当前节点加入关闭列表
            closed_set.add(current)
            
            # 检查所有相邻节点
            for dx, dy in self._directions:
                neighbor = (current[0] + dx, current[1] + dy)
                
                # 检查相邻节点是否有效
                if (neighbor[0] < 0 or neighbor[0] >= height or
                    neighbor[1] < 0 or neighbor[1] >= width):
                    continue
                
                # 检查相邻节点是否是障碍物或已在关闭列表中
                if grid[neighbor[0], neighbor[1]] == 1 or neighbor in closed_set:
                    continue
                
                # 计算通过当前节点到达相邻节点的代价
                tentative_g_score = g_score[current]
                if dx == 0 or dy == 0:  # 水平/垂直移动
                    tentative_g_score += 1.0
                else:  # 对角线移动
                    tentative_g_score += 1.414
                
                # 如果相邻节点不在开放列表中，或找到了更好的路径
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    # 更新相邻节点的信息
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + self._heuristic(neighbor, goal)
                    
                    # 将相邻节点加入开放列表
                    heapq.heappush(open_list, (f_score[neighbor], neighbor))
        
        # 如果无法找到路径，返回空列表
        print("找不到从起点到终点的路径")
        return []
    
    def _smooth_path(self, path, iterations=100):
        """
        平滑路径
        
        参数:
            path: 原始路径点列表
            iterations: 平滑迭代次数
            
        返回:
            smooth_path: 平滑后的路径点列表
        """
        if len(path) <= 2:
            return path
        
        # 创建路径的副本
        smooth_path = [[x, y] for x, y in path]
        
        # 梯度下降法平滑
        for _ in range(iterations):
            change = 0.0
            # 不修改起点和终点
            for i in range(1, len(path) - 1):
                for j in range(2):  # x和y坐标
                    # 数据拉力（保持接近原始路径）
                    data_pull = self._smooth_weight_data * (path[i][j] - smooth_path[i][j])
                    
                    # 平滑拉力（保持路径平滑）
                    smooth_pull = self._smooth_weight_smooth * (
                        smooth_path[i-1][j] + smooth_path[i+1][j] - 2.0 * smooth_path[i][j]
                    )
                    
                    # 更新路径点
                    change = data_pull + smooth_pull
                    smooth_path[i][j] += change
            
            # 如果变化很小，提前结束
            if abs(change) < self._smooth_tolerance:
                break
        
        # 转换回元组格式
        return [(p[0], p[1]) for p in smooth_path]
