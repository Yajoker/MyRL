"""Global path planning utilities for ETHSRL navigation.

全局路径规划工具模块，用于ETHSRL导航系统。

This module provides a lightweight grid-based A* planner that converts a
static world description (in IR-Sim YAML format) into a set of traversable
waypoints. The planner discretises the workspace, inflates obstacle segments
by a configurable safety margin, and exposes helpers for incremental replans
when dynamic obstacles are introduced.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from itertools import count
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import yaml


@dataclass(frozen=True)
class GridCell:
    """离散网格索引，用于路径规划器"""

    row: int  # 行索引
    col: int  # 列索引


@dataclass(frozen=True)
class WaypointWindow:
    """均匀分布在全局路径上的目标窗口."""

    center: np.ndarray
    radius: float

    def clone(self) -> "WaypointWindow":
        return WaypointWindow(center=self.center.copy(), radius=float(self.radius))


class GlobalPlanner:
    """基于网格的A*路径规划器，操作IR-Sim世界描述"""

    def __init__(
        self,
        world_file: Path,
        *,
        resolution: float = 0.25,
        safety_margin: float = 0.35,
        allow_diagonal: bool = True,
        window_spacing: float = 2.0,
        window_radius: float = 0.6,
    ) -> None:
        # 初始化全局路径规划器
        self.world_file = Path(world_file)
        if not self.world_file.exists():
            raise FileNotFoundError(f"World file not found: {self.world_file}")

        # 网格分辨率
        self.resolution = float(resolution)
        if self.resolution <= 0:
            raise ValueError("resolution must be positive")

        # 安全边距
        self.safety_margin = float(safety_margin)
        if self.safety_margin < 0:
            raise ValueError("safety_margin must be non-negative")

        # 是否允许对角线移动
        self.allow_diagonal = bool(allow_diagonal)

        # 均匀目标窗口的间距与半径
        self.window_spacing = max(0.1, float(window_spacing))
        self.window_radius = max(0.05, float(window_radius))

        # 动态障碍物存储: 列表格式为 (中心点, 半径)
        self.dynamic_obstacles: List[Tuple[np.ndarray, float]] = []

        # 加载世界配置和栅格化静态障碍物
        self._load_world()
        self._rasterise_static_obstacles()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def plan(self, start: Sequence[float], goal: Sequence[float]) -> List[WaypointWindow]:

        """计算从起点到目标点的目标窗口序列
        
        Args:
            start: 机器人起始位置，世界坐标系中的``(x, y)``坐标
            goal: 目标位置，世界坐标系中的``(x, y)``坐标

        Returns:
           
            从起点到目标点排序的目标窗口列表

        Raises:
            RuntimeError: 如果找不到可行路
        """

        # 转换起点和目标点为numpy数组
        start_xy = np.asarray(start, dtype=np.float32)[:2]
        goal_xy = np.asarray(goal, dtype=np.float32)[:2]

        # 打印一次世界坐标的起点与终点，便于调试验证
        print(
            f"[GlobalPlanner] world start={start_xy.tolist()} -> goal={goal_xy.tolist()}"
        )

        # 将世界坐标转换为网格坐标
        start_cell = self._world_to_cell(start_xy)
        goal_cell = self._world_to_cell(goal_xy)

        # 寻找最近的可行网格单元
        start_cell = self._nearest_traversable(start_cell)
        goal_cell = self._nearest_traversable(goal_cell)
        if start_cell is None or goal_cell is None:
            raise RuntimeError("Unable to locate traversable start/goal cells")

        # 优先队列条目包含 (f_cost, 计数器, g_cost, 网格单元)
        # 显式计数器防止heapq比较GridCell实例
        open_set: List[Tuple[float, int, float, GridCell]] = []
        counter = count()
        heapq.heappush(open_set, (0.0, next(counter), 0.0, start_cell))

        # A*算法数据结构
        came_from: dict[GridCell, GridCell] = {}  # 记录路径来源
        g_score: dict[GridCell, float] = {start_cell: 0.0}  # 从起点到当前点的实际代价
        f_score: dict[GridCell, float] = {
            start_cell: self._heuristic(start_cell, goal_cell)  # 估计总代价
        }

        closed: set[GridCell] = set()  # 已探索的节点集合

        # A*算法主循环
        while open_set:
            _, _, current_cost, current = heapq.heappop(open_set)
            if current in closed:
                continue
            if current == goal_cell:
                # 找到目标，重建路径
                path_cells = self._reconstruct_path(came_from, current)
                waypoints = [self._cell_center(cell) for cell in path_cells]
                simplified = self._post_process_path(waypoints)
                return self._discretise_into_windows(start_xy, goal_xy, simplified)

            closed.add(current)

            # 探索邻居节点
            for neighbour, step_cost in self._neighbours(current):
                if neighbour in closed:
                    continue
                if not self._is_traversable(neighbour):
                    continue

                # 计算新的代价
                tentative_cost = current_cost + step_cost
                if tentative_cost < g_score.get(neighbour, math.inf):
                    came_from[neighbour] = current
                    g_score[neighbour] = tentative_cost
                    priority = tentative_cost + self._heuristic(neighbour, goal_cell)
                    f_score[neighbour] = priority
                    heapq.heappush(
                        open_set,
                        (priority, next(counter), tentative_cost, neighbour),
                    )

        raise RuntimeError("A* search failed to find a valid path")

    def update_dynamic_obstacles(
        self, obstacles: Iterable[Sequence[float]], radius: float = 0.3
    ) -> None:
        """Register dynamic obstacle centres for subsequent plans."""
        """注册动态障碍物中心点，用于后续路径规划"""

        processed: List[Tuple[np.ndarray, float]] = []
        for centre in obstacles:
            centre_xy = np.asarray(centre, dtype=np.float32)[:2]
            processed.append((centre_xy, float(radius)))
        self.dynamic_obstacles = processed

    def clear_dynamic_obstacles(self) -> None:
        """Remove all dynamic obstacle annotations."""
        """清除所有动态障碍物标注"""
        self.dynamic_obstacles.clear()

    # ------------------------------------------------------------------
    # World parsing and discretisation helpers
    # 世界解析和离散化辅助函数
    # ------------------------------------------------------------------
    def _load_world(self) -> None:
        """Load world configuration from YAML file"""
        """从YAML文件加载世界配置"""
        with self.world_file.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)

        # 解析世界配置
        world_cfg = data.get("world", {})
        self.world_width = float(world_cfg.get("width", 20.0))  # 世界宽度
        self.world_height = float(world_cfg.get("height", 20.0))  # 世界高度
        offset = world_cfg.get("offset", [0.0, 0.0])
        self.origin = np.asarray(offset, dtype=np.float32)[:2]  # 世界原点

        # 计算世界边界
        self.x_min = float(self.origin[0])
        self.y_min = float(self.origin[1])
        self.x_max = self.x_min + self.world_width
        self.y_max = self.y_min + self.world_height

        # 计算网格行列数
        self.rows = int(math.ceil(self.world_height / self.resolution))
        self.cols = int(math.ceil(self.world_width / self.resolution))

        # 解析静态障碍物线段
        self.static_segments: List[Tuple[np.ndarray, np.ndarray]] = []
        for obstacle in data.get("obstacle", []):
            shape = obstacle.get("shape", {})
            vertices = shape.get("vertices", [])
            if not vertices:
                continue
            points = [np.asarray(v, dtype=np.float32)[:2] for v in vertices]
            if len(points) == 2:
                # 线段障碍物
                self.static_segments.append((points[0], points[1]))
            else:
                # 多边形障碍物，分解为线段
                for idx in range(len(points)):
                    start = points[idx]
                    end = points[(idx + 1) % len(points)]
                    self.static_segments.append((start, end))

        # 添加边界墙，确保路径在地图内
        corners = [
            np.array([self.x_min, self.y_min], dtype=np.float32),
            np.array([self.x_max, self.y_min], dtype=np.float32),
            np.array([self.x_max, self.y_max], dtype=np.float32),
            np.array([self.x_min, self.y_max], dtype=np.float32),
        ]
        for idx in range(4):
            self.static_segments.append((corners[idx], corners[(idx + 1) % 4]))

    def _rasterise_static_obstacles(self) -> None:
        """Rasterize static obstacles into occupancy grid"""
        """将静态障碍物栅格化到占据网格中"""
        occupancy = np.zeros((self.rows, self.cols), dtype=bool)
        for row in range(self.rows):
            for col in range(self.cols):
                centre = self._cell_center(GridCell(row, col))
                if self._point_blocked(centre):
                    occupancy[row, col] = True  # 标记被占据的网格
        self.occupancy = occupancy

    # ------------------------------------------------------------------
    # Geometry / numeric helpers
    # 几何/数值辅助函数
    # ------------------------------------------------------------------
    def _world_to_cell(self, point: np.ndarray) -> GridCell:
        """Convert world coordinates to grid cell coordinates"""
        """将世界坐标转换为网格坐标"""
        col = int((point[0] - self.x_min) / self.resolution)
        row = int((point[1] - self.y_min) / self.resolution)
        return GridCell(row=row, col=col)

    def _cell_center(self, cell: GridCell) -> np.ndarray:
        """Convert grid cell coordinates to world coordinates (center point)"""
        """将网格坐标转换为世界坐标（中心点）"""
        x = self.x_min + (cell.col + 0.5) * self.resolution
        y = self.y_min + (cell.row + 0.5) * self.resolution
        return np.array([x, y], dtype=np.float32)

    def _in_bounds(self, cell: GridCell) -> bool:
        """Check if grid cell is within world bounds"""
        """检查网格单元是否在世界边界内"""
        return 0 <= cell.row < self.rows and 0 <= cell.col < self.cols

    def _heuristic(self, a: GridCell, b: GridCell) -> float:
        """Calculate heuristic (Euclidean distance) between two grid cells"""
        """计算两个网格单元之间的启发式距离（欧几里得距离）"""
        dx = (a.col - b.col) * self.resolution
        dy = (a.row - b.row) * self.resolution
        return math.hypot(dx, dy)

    def _neighbours(self, cell: GridCell) -> Iterable[Tuple[GridCell, float]]:
        """Get traversable neighbours of a grid cell"""
        """获取网格单元的可通行邻居"""
        directions = [
            (-1, 0),  # 上
            (1, 0),   # 下
            (0, -1),  # 左
            (0, 1),   # 右
        ]
        if self.allow_diagonal:
            # 添加对角线方向
            directions.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])

        for dr, dc in directions:
            neighbour = GridCell(cell.row + dr, cell.col + dc)
            if not self._in_bounds(neighbour):
                continue
            # 计算移动代价：直线为resolution，对角线为resolution * √2
            cost = self.resolution if dr == 0 or dc == 0 else self.resolution * math.sqrt(2)
            yield neighbour, cost

    def _is_traversable(self, cell: GridCell) -> bool:
        """Check if a grid cell is traversable (not blocked by obstacles)"""
        """检查网格单元是否可通行（未被障碍物阻挡）"""
        if not self._in_bounds(cell):
            return False
        if self.occupancy[cell.row, cell.col]:
            return False
        # 检查动态障碍物
        centre = self._cell_center(cell)
        for dynamic_center, radius in self.dynamic_obstacles:
            if np.linalg.norm(centre - dynamic_center) <= radius + self.safety_margin:
                return False
        return True

    def _point_blocked(self, point: np.ndarray) -> bool:
        """Check if a world point is blocked by any obstacle"""
        """检查世界坐标点是否被任何障碍物阻挡"""
        x, y = float(point[0]), float(point[1])
        # 检查边界
        if x < self.x_min or x > self.x_max or y < self.y_min or y > self.y_max:
            return True

        # 计算膨胀距离
        inflation = self.safety_margin + 0.5 * self.resolution
        # 检查静态障碍物
        for start, end in self.static_segments:
            if self._segment_distance(point, start, end) <= inflation:
                return True

        # 检查动态障碍物
        for dynamic_center, radius in self.dynamic_obstacles:
            if np.linalg.norm(point - dynamic_center) <= radius + self.safety_margin:
                return True

        return False

    @staticmethod
    def _segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
        """Calculate shortest distance from point to line segment"""
        """计算点到线段的最短距离"""
        ax, ay = float(start[0]), float(start[1])
        bx, by = float(end[0]), float(end[1])
        px, py = float(point[0]), float(point[1])

        # 向量计算
        abx = bx - ax
        aby = by - ay
        apx = px - ax
        apy = py - ay
        ab_len_sq = abx * abx + aby * aby
        
        # 如果线段长度为0，直接返回点到起点的距离
        if ab_len_sq == 0:
            return math.hypot(apx, apy)
            
        # 计算投影参数t
        t = max(0.0, min(1.0, (apx * abx + apy * aby) / ab_len_sq))
        # 计算最近点坐标
        closest_x = ax + t * abx
        closest_y = ay + t * aby
        # 返回点到最近点的距离
        return math.hypot(px - closest_x, py - closest_y)

    def _nearest_traversable(self, cell: GridCell) -> Optional[GridCell]:
        """Find the nearest traversable grid cell to the given cell"""
        """查找给定网格单元最近的可通行网格单元"""
        if self._is_traversable(cell):
            return cell
        # 在逐渐增大的半径范围内搜索
        max_radius = max(self.rows, self.cols)
        for radius in range(1, max_radius + 1):
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    candidate = GridCell(cell.row + dr, cell.col + dc)
                    if not self._in_bounds(candidate):
                        continue
                    if self._is_traversable(candidate):
                        return candidate
        return None

    def _reconstruct_path(
        self, came_from: dict[GridCell, GridCell], current: GridCell
    ) -> List[GridCell]:
        """Reconstruct path from start to goal using came_from dictionary"""
        """使用came_from字典重建从起点到目标的路径"""
        path: List[GridCell] = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()  # 反转路径，从起点到目标
        return path

    def _post_process_path(self, raw_waypoints: List[np.ndarray]) -> List[np.ndarray]:
        """Simplify path by removing unnecessary waypoints"""
        """通过移除不必要的路径点来简化路径"""
        if not raw_waypoints:
            return raw_waypoints

        # 路径简化：使用射线投射检查直线是否畅通
        simplified: List[np.ndarray] = [raw_waypoints[0]]
        anchor = raw_waypoints[0]
        for waypoint in raw_waypoints[1:]:
            if self._line_is_free(anchor, waypoint):
                continue
            simplified.append(waypoint)
            anchor = waypoint
        # 确保包含终点
        if not np.allclose(simplified[-1], raw_waypoints[-1]):
            simplified.append(raw_waypoints[-1])

        # 移除可能由简化逻辑引入的重复点
        deduped: List[np.ndarray] = []
        for waypoint in simplified:
            if not deduped or np.linalg.norm(waypoint - deduped[-1]) > 1e-6:
                deduped.append(waypoint)
        return [np.asarray(w, dtype=np.float32) for w in deduped]

    def _discretise_into_windows(
        self,
        start_xy: np.ndarray,
        goal_xy: np.ndarray,
        path_points: List[np.ndarray],
    ) -> List[WaypointWindow]:
        """将路径离散化为等间距的目标窗口序列。"""

        if not path_points:
            return [WaypointWindow(center=goal_xy.copy(), radius=self.window_radius)]

        points: List[np.ndarray] = [np.asarray(start_xy, dtype=np.float32)]
        points.extend(np.asarray(p, dtype=np.float32) for p in path_points)
        if not np.allclose(points[-1], goal_xy):
            points.append(np.asarray(goal_xy, dtype=np.float32))

        windows: List[WaypointWindow] = []
        spacing = self.window_spacing
        radius = self.window_radius

        cursor = points[0].copy()
        distance_acc = 0.0
        point_idx = 1

        while point_idx < len(points):
            segment_end = points[point_idx]
            segment_vec = segment_end - cursor
            segment_len = float(np.linalg.norm(segment_vec))

            if segment_len < 1e-6:
                cursor = segment_end.copy()
                point_idx += 1
                continue

            direction = segment_vec / segment_len
            remaining = spacing - distance_acc

            if segment_len + distance_acc < spacing - 1e-6:
                distance_acc += segment_len
                cursor = segment_end.copy()
                point_idx += 1
                continue

            sample_point = cursor + direction * remaining
            windows.append(WaypointWindow(center=sample_point.astype(np.float32), radius=radius))
            cursor = sample_point
            distance_acc = 0.0

        if not windows:
            windows.append(WaypointWindow(center=goal_xy.copy(), radius=radius))
        elif np.linalg.norm(goal_xy - windows[-1].center) > 1e-3:
            windows.append(WaypointWindow(center=goal_xy.copy(), radius=radius))

        return windows

    def _line_is_free(self, start: np.ndarray, end: np.ndarray) -> bool:
        """Check if straight line between two points is obstacle-free"""
        """检查两点之间的直线是否无障碍物"""
        distance = float(np.linalg.norm(end - start))
        if distance == 0:
            return not self._point_blocked(start)
        # 计算采样步数
        steps = max(2, int(distance / (0.5 * self.resolution)))
        direction = (end - start) / distance
        # 沿直线采样检查
        for step in range(1, steps + 1):
            point = start + direction * (distance * step / steps)
            if self._point_blocked(point):
                return False
        return True
