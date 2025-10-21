"""全局路径规划器实现"""

import numpy as np
from heapq import heappush, heappop
import matplotlib.pyplot as plt

class GlobalPathPlanner:
    """
    全局路径规划器 - 使用A*算法生成全局路径，并提供局部路径窗口
    """
    def __init__(self, resolution=0.1, allow_diagonal=True):
        """
        初始化全局路径规划器
        
        参数:
            resolution: 地图分辨率 (米/像素)
            allow_diagonal: 是否允许对角线移动
        """
        self.resolution = resolution
        self.allow_diagonal = allow_diagonal
        self.global_path = None
        self.obstacle_count = 0  # 记录障碍物接近路径的次数，用于触发重规划
        
    def plan(self, grid_map, start, goal):
        """
        使用A*算法规划全局路径
        
        参数:
            grid_map: 二维栅格地图 (0=可通行, 1=障碍物)
            start: 起点坐标 (x, y) [米]
            goal: 终点坐标 (x, y) [米]
            
        返回:
            path: 世界坐标系中的路径点列表 [(x, y), ...] [米]
        """
        # 将实际坐标转换为栅格坐标
        start_grid = self._to_grid(start)
        goal_grid = self._to_grid(goal)
        
        # 检查起点和终点是否有效
        if not self._is_valid(grid_map, start_grid) or not self._is_valid(grid_map, goal_grid):
            raise ValueError("起点或终点不可通行")
        
        # 初始化A*算法所需数据结构
        open_set = []  # 优先队列，按f值排序
        closed_set = set()  # 已访问节点集合
        
        # g_score[node] = 从起点到node的最低成本
        g_score = {start_grid: 0}
        
        # f_score[node] = g_score[node] + h(node)
        f_score = {start_grid: self._heuristic(start_grid, goal_grid)}
        
        # 记录节点的父节点，用于重建路径
        came_from = {}
        
        # 将起点加入开放集
        heappush(open_set, (f_score[start_grid], start_grid))
        
        # A*主循环
        while open_set:
            # 获取f值最小的节点
            _, current = heappop(open_set)
            
            # 如果达到目标，重建并返回路径
            if current == goal_grid:
                path = self._reconstruct_path(came_from, current)
                world_path = [self._to_world(p) for p in path]
                self.global_path = world_path
                self.obstacle_count = 0  # 重置障碍物计数
                return world_path
            
            # 将当前节点加入已访问集合
            closed_set.add(current)
            
            # 检查所有相邻节点
            for neighbor in self._get_neighbors(grid_map, current):
                # 如果已经访问过，跳过
                if neighbor in closed_set:
                    continue
                
                # 计算从起点经过当前节点到邻居节点的成本
                # 对角线移动成本为√2，直线移动成本为1
                dx = abs(neighbor[0] - current[0])
                dy = abs(neighbor[1] - current[1])
                if dx == 1 and dy == 1:  # 对角线移动
                    tentative_g_score = g_score[current] + 1.414  # √2
                else:  # 直线移动
                    tentative_g_score = g_score[current] + 1.0
                
                # 如果找到更优路径或者首次访问此节点
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    # 更新路径和得分
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self._heuristic(neighbor, goal_grid)
                    
                    # 将邻居节点加入开放集
                    if neighbor not in [i[1] for i in open_set]:
                        heappush(open_set, (f_score[neighbor], neighbor))
        
        # 如果无法找到路径
        raise ValueError("无法找到从起点到终点的路径")
    
    def get_local_path_window(self, current_pos, window_size=10):
        """
        从全局路径中提取局部路径窗口
        
        参数:
            current_pos: 当前位置 (x, y) [米]
            window_size: 窗口大小 (点数)
            
        返回:
            local_path: 局部路径窗口 [(x, y), ...] [米]
        """
        if self.global_path is None:
            raise ValueError("全局路径尚未规划")
        
        # 找到全局路径上离当前位置最近的点
        distances = [np.linalg.norm(np.array(current_pos) - np.array(p)) for p in self.global_path]
        closest_idx = np.argmin(distances)
        
        # 提取局部窗口
        end_idx = min(len(self.global_path), closest_idx + window_size)
        return self.global_path[closest_idx:end_idx]
    
    def check_path_validity(self, grid_map, obstacle_threshold=0.5):
        """
        检查全局路径是否仍然有效（是否被障碍物阻塞）
        
        参数:
            grid_map: 当前栅格地图
            obstacle_threshold: 障碍物接近阈值 (米)
            
        返回:
            is_valid: 路径是否有效
            blocked_segment: 被阻塞的路径段 [(x1,y1), (x2,y2)]，如无阻塞则为None
        """
        if self.global_path is None or len(self.global_path) < 2:
            return True, None
        
        # 检查路径上每个点和相邻点之间的线段是否与障碍物相交
        for i in range(len(self.global_path) - 1):
            p1 = self.global_path[i]
            p2 = self.global_path[i + 1]
            
            # 如果线段与障碍物距离小于阈值，认为路径被阻塞
            if not self._is_segment_clear(grid_map, p1, p2, obstacle_threshold):
                self.obstacle_count += 1
                return False, (p1, p2)
        
        return True, None
    
    def need_replan(self, replan_threshold=5):
        """
        判断是否需要重新规划全局路径
        
        参数:
            replan_threshold: 重规划阈值，障碍物接近次数
            
        返回:
            need_replan: 是否需要重新规划
        """
        if self.obstacle_count >= replan_threshold:
            return True
        return False
    
    def reset(self):
        """重置规划器状态"""
        self.global_path = None
        self.obstacle_count = 0
    
    def visualize_path(self, grid_map=None, start=None, goal=None, figsize=(10, 10)):
        """
        可视化全局路径和局部路径窗口
        
        参数:
            grid_map: 栅格地图 (可选)
            start: 起点坐标 (可选)
            goal: 终点坐标 (可选)
            figsize: 图像大小
        """
        plt.figure(figsize=figsize)
        
        # 如果提供了地图，绘制地图
        if grid_map is not None:
            plt.imshow(grid_map.T, cmap='binary', origin='lower')
        
        # 如果有全局路径，绘制全局路径
        if self.global_path is not None:
            path_x = [p[0] for p in self.global_path]
            path_y = [p[1] for p in self.global_path]
            plt.plot(path_x, path_y, 'b-', linewidth=2, label='Global Path')
        
        # 如果提供了起点和终点，绘制起点和终点
        if start is not None:
            plt.plot(start[0], start[1], 'go', markersize=10, label='Start')
        if goal is not None:
            plt.plot(goal[0], goal[1], 'ro', markersize=10, label='Goal')
        
        plt.legend()
        plt.grid(True)
        plt.title('Global Path Planning')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.axis('equal')
        plt.show()
    
    def _to_grid(self, world_point):
        """
        将世界坐标转换为栅格坐标
        
        参数:
            world_point: 世界坐标 (x, y) [米]
            
        返回:
            grid_point: 栅格坐标 (row, col)
        """
        return (int(round(world_point[0] / self.resolution)), 
                int(round(world_point[1] / self.resolution)))
    
    def _to_world(self, grid_point):
        """
        将栅格坐标转换为世界坐标
        
        参数:
            grid_point: 栅格坐标 (row, col)
            
        返回:
            world_point: 世界坐标 (x, y) [米]
        """
        return (grid_point[0] * self.resolution, 
                grid_point[1] * self.resolution)
    
    def _heuristic(self, a, b):
        """
        启发式函数 - 曼哈顿距离
        
        参数:
            a, b: 两个栅格坐标
            
        返回:
            距离值
        """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def _is_valid(self, grid_map, grid_point):
        """
        检查栅格点是否有效（在地图内且不是障碍物）
        
        参数:
            grid_map: 栅格地图
            grid_point: 栅格坐标
            
        返回:
            is_valid: 是否有效
        """
        x, y = grid_point
        # 检查是否在地图边界内
        if x < 0 or x >= grid_map.shape[0] or y < 0 or y >= grid_map.shape[1]:
            return False
        # 检查是否是障碍物
        if grid_map[x, y] == 1:
            return False
        return True
    
    def _get_neighbors(self, grid_map, grid_point):
        """
        获取栅格点的相邻点
        
        参数:
            grid_map: 栅格地图
            grid_point: 栅格坐标
            
        返回:
            neighbors: 有效的相邻点列表
        """
        x, y = grid_point
        neighbors = []
        
        # 直线移动
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if self._is_valid(grid_map, (nx, ny)):
                neighbors.append((nx, ny))
        
        # 对角线移动（如果允许）
        if self.allow_diagonal:
            for dx, dy in [(1, 1), (1, -1), (-1, -1), (-1, 1)]:
                nx, ny = x + dx, y + dy
                if self._is_valid(grid_map, (nx, ny)):
                    # 确保对角线移动时两个相邻格子都不是障碍物
                    if self._is_valid(grid_map, (x, y + dy)) and self._is_valid(grid_map, (x + dx, y)):
                        neighbors.append((nx, ny))
        
        return neighbors
    
    def _reconstruct_path(self, came_from, current):
        """
        从came_from字典重建路径
        
        参数:
            came_from: 记录每个节点的父节点的字典
            current: 当前节点（终点）
            
        返回:
            path: 从起点到终点的路径
        """
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        
        # 返回从起点到终点的路径（逆序）
        return path[::-1]
    
    def _is_segment_clear(self, grid_map, p1, p2, threshold):
        """
        检查线段是否与障碍物相交
        
        参数:
            grid_map: 栅格地图
            p1, p2: 线段的两个端点
            threshold: 障碍物接近阈值
            
        返回:
            is_clear: 线段是否安全
        """
        # 将世界坐标转换为栅格坐标
        p1_grid = self._to_grid(p1)
        p2_grid = self._to_grid(p2)
        
        # 使用Bresenham算法计算线段上的所有格子
        cells = self._bresenham_line(p1_grid[0], p1_grid[1], p2_grid[0], p2_grid[1])
        
        # 检查线段上的所有格子是否有障碍物
        threshold_grid = int(threshold / self.resolution)
        for x, y in cells:
            if not self._is_valid(grid_map, (x, y)):
                return False
            
            # 检查周围的格子
            for dx in range(-threshold_grid, threshold_grid + 1):
                for dy in range(-threshold_grid, threshold_grid + 1):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < grid_map.shape[0] and 0 <= ny < grid_map.shape[1]:
                        if grid_map[nx, ny] == 1:
                            return False
        
        return True
    
    def _bresenham_line(self, x0, y0, x1, y1):
        """
        Bresenham直线算法，计算两点之间的所有栅格点
        
        参数:
            x0, y0: 起点栅格坐标
            x1, y1: 终点栅格坐标
            
        返回:
            cells: 线段上的所有栅格点
        """
        cells = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        while True:
            cells.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        
        return cells