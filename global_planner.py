"""Global path planning utilities for ETHSRL navigation.

全局路径规划工具模块，用于ETHSRL导航系统。

This module provides a lightweight grid-based A* planner that converts a
static world description (in IR-Sim YAML format) into a set of traversable
waypoints. The planner discretises the workspace, inflates obstacle segments
by a configurable safety margin, and exposes helpers for incremental replans
when dynamic obstacles are introduced.
"""

from __future__ import annotations

import heapq  # 优先队列（堆）实现
import math  # 数学函数
from dataclasses import dataclass  # 数据类装饰器
from itertools import count  # 计数器生成器
from pathlib import Path  # 路径处理
from typing import Iterable, List, Optional, Sequence, Tuple  # 类型注解

import numpy as np  # 数值计算库
import yaml  # YAML文件解析


@dataclass(frozen=True)
class GridCell:
    """离散网格索引，用于路径规划器"""
    row: int  # 行索引
    col: int  # 列索引


@dataclass(frozen=True)
class WaypointWindow:
    """均匀分布在全局路径上的目标窗口."""
    center: np.ndarray  # 窗口中心坐标
    radius: float  # 窗口半径

    def clone(self) -> "WaypointWindow":
        """创建窗口的深拷贝"""
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
        enable_smoothing: bool = True,
        bezier_step: float = 0.2,
    ) -> None:
        # 初始化全局路径规划器
        self.world_file = Path(world_file)
        # 检查世界文件是否存在
        if not self.world_file.exists():
            raise FileNotFoundError(f"World file not found: {self.world_file}")

        # 网格分辨率（米/单元格）
        self.resolution = float(resolution)
        if self.resolution <= 0:
            raise ValueError("resolution must be positive")

        # 安全边距（障碍物膨胀距离）
        self.safety_margin = float(safety_margin)
        if self.safety_margin < 0:
            raise ValueError("safety_margin must be non-negative")

        # 是否允许对角线移动
        self.allow_diagonal = bool(allow_diagonal)

        # 均匀目标窗口的间距与半径
        self.window_spacing = max(0.1, float(window_spacing))
        self.window_radius = max(0.05, float(window_radius))

        # 二次近似平滑设置
        self.enable_smoothing = bool(enable_smoothing)
        self.bezier_step = max(0.05, float(bezier_step))

        # 动态障碍物存储: 列表格式为 (中心点, 半径)
        self.dynamic_obstacles: List[Tuple[np.ndarray, float]] = []
        # 标记是否已打印路径摘要
        self._printed_route_summary = False

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
            RuntimeError: 如果找不到可行路径
        """
        # 转换起点和目标点为numpy数组
        start_xy = np.asarray(start, dtype=np.float32)[:2]
        goal_xy = np.asarray(goal, dtype=np.float32)[:2]

        # 打印一次世界坐标的起点与终点，便于调试验证
        if not self._printed_route_summary:
            print(
                f"[GlobalPlanner] world start={start_xy.tolist()} -> goal={goal_xy.tolist()}"
            )
            self._printed_route_summary = True

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
        counter = count()  # 无限计数器
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
            # 弹出代价最小的节点
            _, _, current_cost, current = heapq.heappop(open_set)
            if current in closed:
                continue
            if current == goal_cell:
                # 找到目标，重建路径
                path_cells = self._reconstruct_path(came_from, current)
                waypoints = [self._cell_center(cell) for cell in path_cells]
                simplified = self._post_process_path(waypoints)  # 路径简化
                smoothed = self._smooth_path_quadratic(simplified)  # 路径平滑
                return self._discretise_into_windows(start_xy, goal_xy, smoothed)  # 离散化为窗口

            closed.add(current)  # 标记为已探索

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
        """注册动态障碍物中心点，用于后续路径规划"""
        processed: List[Tuple[np.ndarray, float]] = []
        for centre in obstacles:
            centre_xy = np.asarray(centre, dtype=np.float32)[:2]  # 提取前两个坐标
            processed.append((centre_xy, float(radius)))
        self.dynamic_obstacles = processed  # 更新动态障碍物列表

    def clear_dynamic_obstacles(self) -> None:
        """清除所有动态障碍物标注"""
        self.dynamic_obstacles.clear()

    # ------------------------------------------------------------------
    # World parsing and discretisation helpers
    # 世界解析和离散化辅助函数
    # ------------------------------------------------------------------
    def _load_world(self) -> None:
        """从YAML文件加载世界配置"""
        with self.world_file.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)  # 安全加载YAML数据

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
            points = [np.asarray(v, dtype=np.float32)[:2] for v in vertices]  # 提取顶点坐标
            if len(points) == 2:
                # 线段障碍物
                self.static_segments.append((points[0], points[1]))
            else:
                # 多边形障碍物，分解为线段
                for idx in range(len(points)):
                    start = points[idx]
                    end = points[(idx + 1) % len(points)]  # 循环连接
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
        """将静态障碍物栅格化到占据网格中"""
        # 初始化占据网格（False表示可通行）
        occupancy = np.zeros((self.rows, self.cols), dtype=bool)
        for row in range(self.rows):
            for col in range(self.cols):
                centre = self._cell_center(GridCell(row, col))  # 获取网格中心坐标
                if self._point_blocked(centre):
                    occupancy[row, col] = True  # 标记被占据的网格
        self.occupancy = occupancy  # 保存占据网格

    # ------------------------------------------------------------------
    # Geometry / numeric helpers
    # 几何/数值辅助函数
    # ------------------------------------------------------------------
    def _world_to_cell(self, point: np.ndarray) -> GridCell:
        """将世界坐标转换为网格坐标"""
        col = int((point[0] - self.x_min) / self.resolution)
        row = int((point[1] - self.y_min) / self.resolution)
        return GridCell(row=row, col=col)

    def _cell_center(self, cell: GridCell) -> np.ndarray:
        """将网格坐标转换为世界坐标（中心点）"""
        x = self.x_min + (cell.col + 0.5) * self.resolution
        y = self.y_min + (cell.row + 0.5) * self.resolution
        return np.array([x, y], dtype=np.float32)

    def _in_bounds(self, cell: GridCell) -> bool:
        """检查网格单元是否在世界边界内"""
        return 0 <= cell.row < self.rows and 0 <= cell.col < self.cols

    def _heuristic(self, a: GridCell, b: GridCell) -> float:
        """计算两个网格单元之间的启发式距离（欧几里得距离）"""
        dx = (a.col - b.col) * self.resolution  # x方向距离
        dy = (a.row - b.row) * self.resolution  # y方向距离
        return math.hypot(dx, dy)  # 欧几里得距离

    def _neighbours(self, cell: GridCell) -> Iterable[Tuple[GridCell, float]]:
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
            yield neighbour, cost  # 返回邻居和移动代价

    def _is_traversable(self, cell: GridCell) -> bool:
        """检查网格单元是否可通行（未被障碍物阻挡）"""
        if not self._in_bounds(cell):
            return False
        if self.occupancy[cell.row, cell.col]:  # 检查静态障碍物
            return False
        # 检查动态障碍物
        centre = self._cell_center(cell)
        for dynamic_center, radius in self.dynamic_obstacles:
            if np.linalg.norm(centre - dynamic_center) <= radius + self.safety_margin:
                return False
        return True

    def _point_blocked(self, point: np.ndarray) -> bool:
        """检查世界坐标点是否被任何障碍物阻挡"""
        x, y = float(point[0]), float(point[1])
        # 检查边界
        if x < self.x_min or x > self.x_max or y < self.y_min or y > self.y_max:
            return True

        # 计算膨胀距离（安全边距 + 半个网格分辨率）
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
        """计算点到线段的最短距离"""
        ax, ay = float(start[0]), float(start[1])
        bx, by = float(end[0]), float(end[1])
        px, py = float(point[0]), float(point[1])

        # 向量计算
        abx = bx - ax
        aby = by - ay
        apx = px - ax
        apy = py - ay
        ab_len_sq = abx * abx + aby * aby  # 线段长度的平方
        
        # 如果线段长度为0，直接返回点到起点的距离
        if ab_len_sq == 0:
            return math.hypot(apx, apy)
            
        # 计算投影参数t（限制在[0,1]范围内）
        t = max(0.0, min(1.0, (apx * abx + apy * aby) / ab_len_sq))
        # 计算最近点坐标
        closest_x = ax + t * abx
        closest_y = ay + t * aby
        # 返回点到最近点的距离
        return math.hypot(px - closest_x, py - closest_y)

    def _nearest_traversable(self, cell: GridCell) -> Optional[GridCell]:
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
        return None  # 未找到可通行网格

    def _reconstruct_path(
        self, came_from: dict[GridCell, GridCell], current: GridCell
    ) -> List[GridCell]:
        """使用came_from字典重建从起点到目标的路径"""
        path: List[GridCell] = [current]
        while current in came_from:
            current = came_from[current]  # 回溯前驱节点
            path.append(current)
        path.reverse()  # 反转路径，从起点到目标
        return path

    def _post_process_path(self, raw_waypoints: List[np.ndarray]) -> List[np.ndarray]:
        """通过移除不必要的路径点来简化路径"""
        if not raw_waypoints:
            return raw_waypoints

        # 路径简化：使用射线投射检查直线是否畅通
        simplified: List[np.ndarray] = [raw_waypoints[0]]  # 起点必须保留
        anchor = raw_waypoints[0]  # 当前锚点
        last_visible = raw_waypoints[0]  # 最后一个可见点

        for waypoint in raw_waypoints[1:]:
            if self._line_is_free(anchor, waypoint):
                 # 暂不追加，延长"可见"范围
                last_visible = waypoint
                continue
            # 一旦不可直连，把"上一个仍可直连的点"纳入简化结果
            if not np.allclose(simplified[-1], last_visible):
                simplified.append(waypoint)
            anchor = last_visible
            last_visible = waypoint  # 继续尝试从新的 anchor 延伸

        # 收尾：确保包含终点
        if not np.allclose(simplified[-1], raw_waypoints[-1]):
            simplified.append(raw_waypoints[-1])

        # 移除可能由简化逻辑引入的重复点
        deduped: List[np.ndarray] = []
        for w in simplified:
            w_np = np.asarray(w, dtype=np.float32)
            if not deduped or np.linalg.norm(w_np - deduped[-1]) > 1e-6:
                deduped.append(w_np)
        return deduped
        
    def _smooth_path_quadratic(self, path_points: List[np.ndarray]) -> List[np.ndarray]:
        """使用二次贝塞尔近似对路径进行平滑，同时保持避障约束"""
        if not self.enable_smoothing or len(path_points) < 3:
            return [np.asarray(p, dtype=np.float32) for p in path_points]

        processed: List[np.ndarray] = [np.asarray(path_points[0], dtype=np.float32)]  # 起点
        idx = 0
        last_index = len(path_points) - 1

        while idx < last_index - 1:
            p0 = path_points[idx]  # 当前点
            p1 = path_points[idx + 1]  # 下一个点
            p2 = path_points[idx + 2]  # 下下个点

            # 采样二次贝塞尔曲线
            bezier_samples = self._sample_quadratic_bezier(p0, p1, p2)
            if self._curve_is_free(bezier_samples):  # 检查曲线是否无障碍
                # 跳过首个样本防止重复
                for sample in bezier_samples[1:]:
                    if np.linalg.norm(sample - processed[-1]) > 1e-4:
                        processed.append(sample.astype(np.float32))
                idx += 2  # 跳过两个点
            else:
                midpoint = np.asarray(p1, dtype=np.float32)
                if np.linalg.norm(midpoint - processed[-1]) > 1e-4:
                    processed.append(midpoint)
                idx += 1  # 只前进一个点

        # 追加剩余未处理的点
        for tail in path_points[idx + 1 :]:
            tail_np = np.asarray(tail, dtype=np.float32)
            if np.linalg.norm(tail_np - processed[-1]) > 1e-4:
                processed.append(tail_np)

        return self._deduplicate_points(processed)  # 去重

    def _sample_quadratic_bezier(
        self, p0: np.ndarray, p1: np.ndarray, p2: np.ndarray
    ) -> List[np.ndarray]:
        """采样二次贝塞尔曲线"""
        chord = float(np.linalg.norm(np.asarray(p2) - np.asarray(p0)))  # 弦长
        steps = 3 if chord < 1e-6 else max(3, int(math.ceil(chord / self.bezier_step)))
        ts = np.linspace(0.0, 1.0, steps, endpoint=True)  # 参数t
        samples: List[np.ndarray] = []
        for t in ts:
            inv = 1.0 - t
            # 二次贝塞尔曲线公式
            point = (inv * inv) * p0 + 2.0 * inv * t * p1 + (t * t) * p2
            samples.append(np.asarray(point, dtype=np.float32))
        return samples

    def _curve_is_free(self, samples: Sequence[np.ndarray]) -> bool:
        """检查曲线采样点是否全部无障碍"""
        for pt in samples:
            if self._point_blocked(pt):
                return False
        # 加密采样以减少穿障风险
        if len(samples) >= 2:
            dense: List[np.ndarray] = []
            for start, end in zip(samples[:-1], samples[1:]):
                seg_len = float(np.linalg.norm(end - start))
                if seg_len <= self.bezier_step:
                    continue
                extra_count = max(1, int(math.ceil(seg_len / self.bezier_step)) - 1)
                direction = (end - start) / seg_len
                for step in range(1, extra_count + 1):
                    point = start + direction * (seg_len * step / (extra_count + 1))
                    dense.append(point.astype(np.float32))
            for pt in dense:
                if self._point_blocked(pt):
                    return False
        return True

    @staticmethod
    def _deduplicate_points(points: Sequence[np.ndarray], tol: float = 1e-4) -> List[np.ndarray]:
        """去除重复点"""
        deduped: List[np.ndarray] = []
        for point in points:
            point_np = np.asarray(point, dtype=np.float32)
            if not deduped or np.linalg.norm(point_np - deduped[-1]) > tol:
                deduped.append(point_np)
        return deduped

    def _discretise_into_windows(
        self,
        start_xy: np.ndarray,
        goal_xy: np.ndarray,
        path_points: List[np.ndarray],
    ) -> List[WaypointWindow]:
        """将路径离散化为等间距的目标窗口序列"""
        if not path_points:
            return [WaypointWindow(center=goal_xy.copy(), radius=self.window_radius)]

        # 构建完整路径点列表（包含起点和终点）
        points: List[np.ndarray] = [np.asarray(start_xy, dtype=np.float32)]
        points.extend(np.asarray(p, dtype=np.float32) for p in path_points)
        if not np.allclose(points[-1], goal_xy):
            points.append(np.asarray(goal_xy, dtype=np.float32))

        windows: List[WaypointWindow] = []
        spacing = self.window_spacing  # 窗口间距
        radius = self.window_radius  # 窗口半径

        cursor = points[0].copy()  # 当前位置光标
        distance_acc = 0.0  # 累计距离
        point_idx = 1  # 当前路径点索引

        while point_idx < len(points):
            segment_end = points[point_idx]
            segment_vec = segment_end - cursor
            segment_len = float(np.linalg.norm(segment_vec))  # 段长度

            if segment_len < 1e-6:  # 忽略零长度段
                cursor = segment_end.copy()
                point_idx += 1
                continue

            direction = segment_vec / segment_len  # 单位方向向量
            remaining = spacing - distance_acc  # 剩余需要前进的距离

            if segment_len + distance_acc < spacing - 1e-6:
                # 整段长度不足一个间距，继续累计
                distance_acc += segment_len
                cursor = segment_end.copy()
                point_idx += 1
                continue

            # 生成新的目标窗口
            sample_point = cursor + direction * remaining
            windows.append(WaypointWindow(center=sample_point.astype(np.float32), radius=radius))
            cursor = sample_point  # 移动光标
            distance_acc = 0.0  # 重置累计距离

        # 确保包含终点窗口
        if not windows:
            windows.append(WaypointWindow(center=goal_xy.copy(), radius=radius))
        elif np.linalg.norm(goal_xy - windows[-1].center) > 1e-3:
            windows.append(WaypointWindow(center=goal_xy.copy(), radius=radius))

        return windows

    def _line_is_free(self, start: np.ndarray, end: np.ndarray) -> bool:
        """检查两点之间的直线是否无障碍物"""
        distance = float(np.linalg.norm(end - start))
        if distance == 0:
            return not self._point_blocked(start)
        # 计算采样步数
        steps = max(2, int(distance / (0.5 * self.resolution)))
        direction = (end - start) / distance  # 单位方向向量
        # 沿直线采样检查
        for step in range(1, steps + 1):
            point = start + direction * (distance * step / steps)
            if self._point_blocked(point):
                return False
        return True
