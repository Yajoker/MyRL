"""Two-frame LiDAR preprocessing and risk estimation.

This module is deliberately independent from the planner and controller.  A
single :class:`TemporalLidarProcessor` is expected to own the temporal state
for an episode and to publish the resulting immutable observation to all
consumers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

from robot_nav.SIM_ENV.sensor_metadata import LidarMetadata, RawLidarObservation

if TYPE_CHECKING:
    from myrl.config import TriggerConfig


@dataclass(frozen=True)
class RiskEstimate:
    """Clearance and time-to-collision risk for one LiDAR observation."""

    clearance: float
    ttc: float
    combined: float
    d_min_m: float
    distance_percentile_m: float
    ttc_percentile_s: float


@dataclass(frozen=True, eq=False)
class TemporalLidarObservation:
    """Immutable, network-ready representation of one simulator observation."""

    observation_id: int
    pose_xytheta: np.ndarray
    raw_scan_m: np.ndarray
    current_scan_m: np.ndarray
    lidar_channels: np.ndarray
    radial_closing_speed_mps: np.ndarray
    ttc_s: np.ndarray
    current_valid_mask: np.ndarray
    match_valid_mask: np.ndarray
    risk: RiskEstimate
    is_warmup: bool


def _immutable_array(
    values: np.ndarray,
    *,
    dtype: np.dtype,
    shape: Optional[Tuple[int, ...]] = None,
) -> np.ndarray:
    """Return a contiguous array backed by immutable bytes."""

    array = np.asarray(values, dtype=dtype)
    if shape is not None and array.shape != shape:
        raise ValueError(f"Expected array shape {shape}, got {array.shape}")
    contiguous = np.ascontiguousarray(array)
    immutable = np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype)
    return immutable.reshape(contiguous.shape)


def _same_array(left: np.ndarray, right: np.ndarray) -> bool:
    """Compare observation payloads while treating NaNs at equal positions alike."""

    if left.shape != right.shape:
        return False
    try:
        return bool(np.array_equal(left, right, equal_nan=True))
    except TypeError:  # NumPy versions before equal_nan support.
        equal = (left == right) | (np.isnan(left) & np.isnan(right))
        return bool(np.all(equal))


class TemporalLidarProcessor:
    """Build signed radial closing speed and TTC from consecutive scans.

    The processor enforces a strict observation sequence.  Re-reading the same
    simulator observation is idempotent, while skipped, reordered, or mutated
    observation payloads fail immediately.
    """

    _OFFSET_TOLERANCE_M = 1.0e-9
    _FOV_TOLERANCE_RAD = 1.0e-10

    def __init__(
        self,
        metadata: LidarMetadata,
        trigger_config: "TriggerConfig",
    ) -> None:
        if not isinstance(metadata, LidarMetadata):
            raise TypeError("metadata must be a LidarMetadata instance")

        beam_angles = np.asarray(metadata.beam_angles_rad, dtype=np.float64)
        if beam_angles.ndim != 1 or beam_angles.size == 0:
            raise ValueError("metadata.beam_angles_rad must be a non-empty 1-D array")
        if not np.all(np.isfinite(beam_angles)):
            raise ValueError("metadata.beam_angles_rad must contain only finite values")
        if beam_angles.size > 1 and not np.all(np.diff(beam_angles) > 0.0):
            raise ValueError("metadata.beam_angles_rad must be strictly increasing")

        offset = np.asarray(metadata.sensor_offset_xytheta, dtype=np.float64)
        if offset.shape != (3,) or not np.all(np.isfinite(offset)):
            raise ValueError("metadata.sensor_offset_xytheta must contain three finite values")
        if np.linalg.norm(offset[:2]) > self._OFFSET_TOLERANCE_M:
            raise ValueError(
                "Non-zero translational LiDAR offsets are not supported by the "
                "current point-robot TTC clearance model"
            )

        range_min = float(metadata.range_min_m)
        range_max = float(metadata.range_max_m)
        sample_period = float(metadata.sample_period_s)
        robot_radius = float(metadata.robot_radius_m)
        if not math.isfinite(range_min) or range_min < 0.0:
            raise ValueError("metadata.range_min_m must be finite and non-negative")
        if not math.isfinite(range_max) or range_max <= range_min:
            raise ValueError("metadata.range_max_m must be greater than range_min_m")
        if not math.isfinite(sample_period) or sample_period <= 0.0:
            raise ValueError("metadata.sample_period_s must be finite and positive")
        if not math.isfinite(robot_radius) or robot_radius < 0.0:
            raise ValueError("metadata.robot_radius_m must be finite and non-negative")

        safe_distance = float(trigger_config.safety_trigger_distance)
        risk_percentile = float(trigger_config.risk_percentile)
        risk_alpha = float(trigger_config.risk_alpha)
        max_interval = float(trigger_config.max_interval)
        if not math.isfinite(safe_distance) or safe_distance <= 0.0:
            raise ValueError("safety_trigger_distance must be finite and positive")
        if not math.isfinite(risk_percentile) or not 0.0 < risk_percentile < 100.0:
            raise ValueError("risk_percentile must be in (0, 100)")
        if not math.isfinite(risk_alpha) or not 0.0 <= risk_alpha <= 1.0:
            raise ValueError("risk_alpha must be in [0, 1]")
        if not math.isfinite(max_interval) or max_interval <= 0.0:
            raise ValueError("max_interval must be finite and positive")

        # Copy all method-defining inputs.  No later caller mutation can alter
        # the temporal geometry or the normalization rules.
        self._beam_angles = beam_angles.copy()
        self._beam_unit_vectors = np.column_stack(
            (np.cos(self._beam_angles), np.sin(self._beam_angles))
        )
        self._sensor_offset = offset.copy()
        self._beam_count = int(beam_angles.size)
        self._range_min_m = range_min
        self._range_max_m = range_max
        self._sample_period_s = sample_period
        self._robot_radius_m = robot_radius
        self._safe_distance_m = safe_distance
        self._risk_percentile = risk_percentile
        self._risk_alpha = risk_alpha
        self._ttc_horizon_s = max_interval

        self.reset()

    @property
    def beam_count(self) -> int:
        return self._beam_count

    def reset(self) -> None:
        """Clear all episode-local temporal and idempotency state."""

        self._last_observation_id: Optional[int] = None
        self._last_timestamp_s: Optional[float] = None
        self._last_payload_scan: Optional[np.ndarray] = None
        self._last_payload_pose: Optional[np.ndarray] = None
        self._last_output: Optional[TemporalLidarObservation] = None

        self._previous_scan_m: Optional[np.ndarray] = None
        self._previous_valid_mask: Optional[np.ndarray] = None
        self._previous_pose_xytheta: Optional[np.ndarray] = None

    def process(
        self,
        observation: RawLidarObservation,
    ) -> TemporalLidarObservation:
        """Process one raw observation, advancing history at most once."""

        if not isinstance(observation, RawLidarObservation):
            raise TypeError("observation must be a RawLidarObservation instance")

        observation_id = int(observation.observation_id)
        timestamp_s = float(observation.timestamp_s)
        raw_scan = np.asarray(observation.ranges_m, dtype=np.float64)
        pose = np.asarray(observation.pose_xytheta, dtype=np.float64)

        if raw_scan.shape != (self._beam_count,):
            raise ValueError(
                f"Expected {self._beam_count} LiDAR ranges, got shape {raw_scan.shape}"
            )
        if pose.shape != (3,) or not np.all(np.isfinite(pose)):
            raise ValueError("pose_xytheta must have shape (3,) and be finite")
        if not math.isfinite(timestamp_s):
            raise ValueError("timestamp_s must be finite")

        if self._last_observation_id is not None and observation_id == self._last_observation_id:
            if not self._same_cached_payload(timestamp_s, raw_scan, pose):
                raise ValueError(
                    f"Observation id {observation_id} was reused with a different payload"
                )
            assert self._last_output is not None
            return self._last_output

        expected_id = 0 if self._last_observation_id is None else self._last_observation_id + 1
        if observation_id != expected_id:
            raise ValueError(
                f"Expected observation id {expected_id}, got {observation_id}"
            )
        if self._last_timestamp_s is not None and timestamp_s <= self._last_timestamp_s:
            raise ValueError("Observation timestamps must be strictly increasing")

        raw_scan_f32 = np.asarray(raw_scan, dtype=np.float32)
        current_valid = (
            np.isfinite(raw_scan)
            & (raw_scan >= self._range_min_m)
            & (raw_scan <= self._range_max_m)
        )
        current_scan = np.nan_to_num(
            raw_scan,
            nan=self._range_max_m,
            posinf=self._range_max_m,
            neginf=self._range_min_m,
        )
        current_scan = np.clip(
            current_scan,
            self._range_min_m,
            self._range_max_m,
        )

        is_warmup = self._previous_scan_m is None
        if is_warmup:
            closing_speed = np.zeros(self._beam_count, dtype=np.float64)
            ttc_s = np.full(self._beam_count, np.inf, dtype=np.float64)
            match_valid = np.zeros(self._beam_count, dtype=bool)
            ttc_risk = 0.0
            ttc_percentile = float("inf")
        else:
            assert self._previous_valid_mask is not None
            assert self._previous_pose_xytheta is not None
            projected_scan, projected_valid = self._project_previous_scan(
                self._previous_scan_m,
                self._previous_valid_mask,
                self._previous_pose_xytheta,
                pose,
            )
            match_valid = current_valid & projected_valid
            closing_speed = self._compute_radial_closing_speed(
                projected_scan,
                match_valid,
                current_scan,
                self._previous_pose_xytheta,
                pose,
            )
            ttc_s, ttc_risk, ttc_percentile = self._compute_ttc(
                current_scan,
                current_valid,
                match_valid,
                closing_speed,
            )

        clearance_risk, d_min, distance_percentile = self._compute_clearance_risk(
            raw_scan_f32
        )
        combined_risk = max(clearance_risk, ttc_risk)

        range_channel = np.clip(
            current_scan / self._range_max_m,
            0.0,
            1.0,
        )
        closure_channel = np.clip(
            self._sample_period_s * closing_speed / self._safe_distance_m,
            -1.0,
            1.0,
        )
        lidar_channels = np.stack((range_channel, closure_channel), axis=0)

        output = TemporalLidarObservation(
            observation_id=observation_id,
            pose_xytheta=_immutable_array(pose, dtype=np.dtype(np.float32), shape=(3,)),
            raw_scan_m=_immutable_array(
                raw_scan_f32,
                dtype=np.dtype(np.float32),
                shape=(self._beam_count,),
            ),
            current_scan_m=_immutable_array(
                current_scan,
                dtype=np.dtype(np.float32),
                shape=(self._beam_count,),
            ),
            lidar_channels=_immutable_array(
                lidar_channels,
                dtype=np.dtype(np.float32),
                shape=(2, self._beam_count),
            ),
            radial_closing_speed_mps=_immutable_array(
                closing_speed,
                dtype=np.dtype(np.float32),
                shape=(self._beam_count,),
            ),
            ttc_s=_immutable_array(
                ttc_s,
                dtype=np.dtype(np.float32),
                shape=(self._beam_count,),
            ),
            current_valid_mask=_immutable_array(
                current_valid,
                dtype=np.dtype(np.bool_),
                shape=(self._beam_count,),
            ),
            match_valid_mask=_immutable_array(
                match_valid,
                dtype=np.dtype(np.bool_),
                shape=(self._beam_count,),
            ),
            risk=RiskEstimate(
                clearance=float(clearance_risk),
                ttc=float(ttc_risk),
                combined=float(combined_risk),
                d_min_m=float(d_min),
                distance_percentile_m=float(distance_percentile),
                ttc_percentile_s=float(ttc_percentile),
            ),
            is_warmup=is_warmup,
        )

        # Commit history only after all calculations and output construction
        # succeed, so a rejected observation cannot partially advance state.
        self._previous_scan_m = current_scan.copy()
        self._previous_valid_mask = current_valid.copy()
        self._previous_pose_xytheta = pose.copy()
        self._last_observation_id = observation_id
        self._last_timestamp_s = timestamp_s
        self._last_payload_scan = raw_scan.copy()
        self._last_payload_pose = pose.copy()
        self._last_output = output
        return output

    def _same_cached_payload(
        self,
        timestamp_s: float,
        scan: np.ndarray,
        pose: np.ndarray,
    ) -> bool:
        assert self._last_timestamp_s is not None
        assert self._last_payload_scan is not None
        assert self._last_payload_pose is not None
        return (
            timestamp_s == self._last_timestamp_s
            and _same_array(scan, self._last_payload_scan)
            and _same_array(pose, self._last_payload_pose)
        )

    def _sensor_world_pose(self, robot_pose: np.ndarray) -> Tuple[np.ndarray, float]:
        robot_yaw = float(robot_pose[2])
        cos_yaw = math.cos(robot_yaw)
        sin_yaw = math.sin(robot_yaw)
        rotation = np.array(
            ((cos_yaw, -sin_yaw), (sin_yaw, cos_yaw)),
            dtype=np.float64,
        )
        sensor_position = robot_pose[:2] + rotation @ self._sensor_offset[:2]
        sensor_yaw = robot_yaw + float(self._sensor_offset[2])
        return sensor_position, sensor_yaw

    def _project_previous_scan(
        self,
        previous_scan: np.ndarray,
        previous_valid: np.ndarray,
        previous_pose: np.ndarray,
        current_pose: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Project previous endpoints into the current sensor frame."""

        projected_scan = np.full(self._beam_count, np.inf, dtype=np.float64)
        if not np.any(previous_valid):
            return projected_scan, np.zeros(self._beam_count, dtype=bool)

        previous_sensor_position, previous_sensor_yaw = self._sensor_world_pose(
            previous_pose
        )
        current_sensor_position, current_sensor_yaw = self._sensor_world_pose(
            current_pose
        )

        previous_points_sensor = (
            self._beam_unit_vectors[previous_valid]
            * previous_scan[previous_valid, np.newaxis]
        )
        previous_cos = math.cos(previous_sensor_yaw)
        previous_sin = math.sin(previous_sensor_yaw)
        previous_rotation = np.array(
            ((previous_cos, -previous_sin), (previous_sin, previous_cos)),
            dtype=np.float64,
        )
        points_world = (
            previous_points_sensor @ previous_rotation.T
            + previous_sensor_position[np.newaxis, :]
        )

        current_cos = math.cos(current_sensor_yaw)
        current_sin = math.sin(current_sensor_yaw)
        current_rotation = np.array(
            ((current_cos, -current_sin), (current_sin, current_cos)),
            dtype=np.float64,
        )
        points_current = (
            points_world - current_sensor_position[np.newaxis, :]
        ) @ current_rotation

        projected_ranges = np.linalg.norm(points_current, axis=1)
        projected_angles = np.arctan2(points_current[:, 1], points_current[:, 0])
        inside_fov = (
            np.isfinite(projected_ranges)
            & np.isfinite(projected_angles)
            & (projected_ranges >= 0.0)
            & (
                projected_angles
                >= self._beam_angles[0] - self._FOV_TOLERANCE_RAD
            )
            & (
                projected_angles
                <= self._beam_angles[-1] + self._FOV_TOLERANCE_RAD
            )
        )
        if not np.any(inside_fov):
            return projected_scan, np.zeros(self._beam_count, dtype=bool)

        projected_ranges = projected_ranges[inside_fov]
        projected_angles = projected_angles[inside_fov]
        beam_indices = self._nearest_beam_indices(projected_angles)

        # Nearest-depth z-buffer for collisions in the angular projection.
        np.minimum.at(projected_scan, beam_indices, projected_ranges)
        projected_valid = np.isfinite(projected_scan)
        return projected_scan, projected_valid

    def _nearest_beam_indices(self, angles: np.ndarray) -> np.ndarray:
        if self._beam_count == 1:
            return np.zeros(angles.shape, dtype=np.int64)

        right = np.searchsorted(self._beam_angles, angles, side="left")
        right = np.clip(right, 0, self._beam_count - 1)
        left = np.clip(right - 1, 0, self._beam_count - 1)
        choose_right = (
            np.abs(self._beam_angles[right] - angles)
            < np.abs(angles - self._beam_angles[left])
        )
        return np.where(choose_right, right, left).astype(np.int64, copy=False)

    def _compute_radial_closing_speed(
        self,
        projected_scan: np.ndarray,
        match_valid: np.ndarray,
        current_scan: np.ndarray,
        previous_pose: np.ndarray,
        current_pose: np.ndarray,
    ) -> np.ndarray:
        closing_speed = np.zeros(self._beam_count, dtype=np.float64)
        if not np.any(match_valid):
            return closing_speed

        previous_sensor_position, _ = self._sensor_world_pose(previous_pose)
        current_sensor_position, current_sensor_yaw = self._sensor_world_pose(
            current_pose
        )
        sensor_displacement_world = (
            current_sensor_position - previous_sensor_position
        )
        current_cos = math.cos(current_sensor_yaw)
        current_sin = math.sin(current_sensor_yaw)
        current_rotation = np.array(
            ((current_cos, -current_sin), (current_sin, current_cos)),
            dtype=np.float64,
        )
        sensor_displacement_current = sensor_displacement_world @ current_rotation
        ego_radial_displacement = (
            self._beam_unit_vectors @ sensor_displacement_current
        )

        residual_displacement = projected_scan[match_valid] - current_scan[match_valid]
        total_closing_displacement = (
            ego_radial_displacement[match_valid] + residual_displacement
        )
        closing_speed[match_valid] = total_closing_displacement / self._sample_period_s
        return closing_speed

    def _compute_ttc(
        self,
        current_scan: np.ndarray,
        current_valid: np.ndarray,
        match_valid: np.ndarray,
        closing_speed: np.ndarray,
    ) -> Tuple[np.ndarray, float, float]:
        ttc_s = np.full(self._beam_count, np.inf, dtype=np.float64)
        positive_closing = current_valid & match_valid & (closing_speed > 0.0)
        if not np.any(positive_closing):
            return ttc_s, 0.0, float("inf")

        clearance = np.maximum(
            current_scan[positive_closing] - self._robot_radius_m,
            0.0,
        )
        denominator = np.maximum(
            closing_speed[positive_closing],
            np.finfo(np.float32).eps,
        )
        ttc_s[positive_closing] = clearance / denominator
        ttc_percentile = float(
            np.percentile(ttc_s[positive_closing], self._risk_percentile)
        )
        ttc_risk = float(
            np.clip(1.0 - ttc_percentile / self._ttc_horizon_s, 0.0, 1.0)
        )
        return ttc_s, ttc_risk, ttc_percentile

    def _compute_clearance_risk(
        self,
        raw_scan_f32: np.ndarray,
    ) -> Tuple[float, float, float]:
        """Replicate the planner's existing finite-scan clearance formula."""

        finite_scan = raw_scan_f32[np.isfinite(raw_scan_f32)]
        if finite_scan.size == 0:
            return 0.0, float("inf"), float("inf")

        d_min = float(np.min(finite_scan))
        percentile = float(np.percentile(finite_scan, self._risk_percentile))
        safe_distance = max(self._safe_distance_m, 1.0e-6)
        min_risk = max(0.0, (safe_distance - d_min) / safe_distance)
        percentile_risk = max(
            0.0,
            (safe_distance - percentile) / safe_distance,
        )
        clearance_risk = min(
            1.0,
            self._risk_alpha * min_risk
            + (1.0 - self._risk_alpha) * percentile_risk,
        )
        return float(clearance_risk), d_min, percentile
