"""
Drawer opening demo using RGB-D handle detection.

Pipeline:
1. Search with the head camera for an elongated drawer/cabinet handle.
2. Approach until the handle is front-facing and within arm reach.
3. Refine alignment with the wrist RGB-D camera.
4. Reach, close the gripper, and pull by retracting the arm.

This intentionally uses only OpenCV and depth, not MuJoCo object poses. For a
good drawer scene in simulation, set stretch_toolkit/sim_config.json to a
RoboCasa drawer task such as "OpenDrawer".
"""

from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# In passive-viewer runs the visible MuJoCo window uses GLFW. If the shell has
# MUJOCO_GL=egl, dynamic RGB-D renderers can fail with "Failed to make the EGL
# context current." Default this demo to GLFW, while leaving an override hook.
os.environ["MUJOCO_GL"] = os.environ.get("STRETCH_MUJOCO_GL", "glfw")
os.environ.setdefault("STRETCH_KEEP_CAMERAS", "1")
os.environ.setdefault("STRETCH_CAMERA_HZ", "10")
os.environ.setdefault(
    "STRETCH_PRELOAD_CAMERAS",
    "cam_d435i_rgb,cam_d435i_depth",
)

import cv2
import numpy as np

from stretch_toolkit import (
    BACKEND_NAME,
    HEAD_CAMERA,
    HEAD_RGB_CAMERA,
    WRIST_CAMERA,
    WRIST_RGB_CAMERA,
    RobotTransforms,
    StateController,
    controller,
    locate_object,
    merge_proportional,
    teleop,
)


Orientation = Literal["horizontal", "vertical", "any"]
HandleTarget = Literal["best", "top", "second", "third", "bottom"]


@dataclass
class HandleDetection:
    centroid: tuple[int, int]
    bbox: tuple[int, int, int, int]
    score: float
    angle_deg: float
    contour: np.ndarray
    mask: np.ndarray


class HandleDetector:
    """Classical RGB detector for likely drawer/cabinet handles.

    The detector looks for elongated high-contrast components. It combines
    Canny edges, low-saturation metal highlights, and dark hardware masks, then
    scores contours by elongation, area, centrality, and depth availability.
    """

    def __init__(
        self,
        orientation: Orientation = "horizontal",
        target: HandleTarget = "best",
        min_area: float = 45.0,
        min_aspect: float = 2.4,
        min_score: float = 2.4,
        min_x_fraction: float = 0.04,
        max_x_fraction: float = 0.96,
        min_y_fraction: float = 0.02,
        max_y_fraction: float = 0.92,
        edge_margin_fraction: float = 0.025,
        metal_v_min: int = 115,
        metal_s_max: int = 115,
        bright_ratio_min: float = 0.08,
        context_dark_ratio_min: float = 0.16,
        context_brick_ratio_max: float = 0.24,
        require_depth: bool = False,
    ) -> None:
        self.orientation = orientation
        self.target = target
        self.min_area = min_area
        self.min_aspect = min_aspect
        self.min_score = min_score
        self.min_x_fraction = min_x_fraction
        self.max_x_fraction = max_x_fraction
        self.min_y_fraction = min_y_fraction
        self.max_y_fraction = max_y_fraction
        self.edge_margin_fraction = edge_margin_fraction
        self.metal_v_min = metal_v_min
        self.metal_s_max = metal_s_max
        self.bright_ratio_min = bright_ratio_min
        self.context_dark_ratio_min = context_dark_ratio_min
        self.context_brick_ratio_max = context_brick_ratio_max
        self.require_depth = require_depth
        self.last_mask: np.ndarray | None = None

    def detect(
        self,
        rgb_frame: np.ndarray | None,
        depth_cam=None,
        sample_radius: int = 4,
        prefer_centroid: tuple[float, float] | None = None,
        lock_radius_fraction: float = 0.28,
        relax_locked_detection: bool = False,
    ) -> HandleDetection | None:
        if rgb_frame is None or rgb_frame.size == 0:
            self.last_mask = None
            return None

        frame = rgb_frame.copy()
        height, width = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = self._candidate_mask(frame)
        self.last_mask = mask

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections: list[HandleDetection] = []
        locked_tracking = prefer_centroid is not None and relax_locked_detection
        min_area = self.min_area * (0.45 if locked_tracking else 1.0)
        min_aspect = self.min_aspect * (0.65 if locked_tracking else 1.0)
        min_score = self.min_score * (0.55 if locked_tracking else 1.0)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            if w <= 0 or h <= 0:
                continue

            aspect = max(w, h) / max(1.0, min(w, h))
            if aspect < min_aspect:
                continue

            if self.orientation == "horizontal" and w < h:
                continue
            if self.orientation == "vertical" and h < w:
                continue

            if w > width * 0.85 or h > height * 0.85:
                continue

            if self.orientation == "horizontal":
                edge_margin = int(width * self.edge_margin_fraction)
                if not locked_tracking and (x <= edge_margin or x + w >= width - edge_margin):
                    continue
                min_w_fraction = 0.020 if locked_tracking else 0.035
                max_w_fraction = 0.55 if locked_tracking else 0.45
                max_h_fraction = 0.25 if locked_tracking else 0.18
                if w < width * min_w_fraction or w > width * max_w_fraction:
                    continue
                if h > height * max_h_fraction:
                    continue

            moments = cv2.moments(contour)
            if moments["m00"] <= 0:
                cx = x + w // 2
                cy = y + h // 2
            else:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])

            x_fraction = cx / max(1.0, width)
            if x_fraction < self.min_x_fraction or x_fraction > self.max_x_fraction:
                continue

            y_fraction = cy / max(1.0, height)
            if y_fraction < self.min_y_fraction or y_fraction > self.max_y_fraction:
                continue

            if self.orientation == "horizontal":
                roi_hsv = hsv[y : y + h, x : x + w]
                bright_handle = cv2.inRange(
                    roi_hsv,
                    np.array([0, 0, self.metal_v_min]),
                    np.array([180, self.metal_s_max, 255]),
                )
                bright_ratio = cv2.countNonZero(bright_handle) / max(1.0, float(w * h))
                bright_ratio_min = self.bright_ratio_min * (0.45 if locked_tracking else 1.0)
                if bright_ratio < bright_ratio_min:
                    continue

                dark_context_ratio, brick_context_ratio = self._mount_context_ratios(hsv, x, y, w, h)
                dark_context_min = self.context_dark_ratio_min
                brick_context_max = self.context_brick_ratio_max
                if prefer_centroid is not None and relax_locked_detection:
                    dark_context_min *= 0.55
                    brick_context_max = min(1.0, brick_context_max + 0.20)

                if dark_context_ratio < dark_context_min:
                    continue
                if brick_context_ratio > brick_context_max:
                    continue
            else:
                dark_context_ratio = 0.0
                brick_context_ratio = 0.0

            depth_score = 0.0
            if depth_cam is not None:
                depth = depth_cam.get_depth((cx, cy), sample_radius=sample_radius)
                if depth is None or not (0.10 <= depth <= 3.50):
                    if self.require_depth:
                        continue
                else:
                    depth_score = max(0.0, 1.0 - abs(depth - 1.0) / 2.5)

            roi = cv2.cvtColor(frame[y : y + h, x : x + w], cv2.COLOR_BGR2GRAY)
            contrast = float(np.std(roi)) / 64.0 if roi.size else 0.0

            center_x = (cx - width * 0.5) / max(1.0, width * 0.5)
            center_y = (cy - height * 0.55) / max(1.0, height * 0.55)
            centrality = 1.0 - min(1.0, 0.65 * abs(center_x) + 0.35 * abs(center_y))

            area_norm = min(1.0, area / (width * height * 0.015))
            orientation_bonus = 0.7 if (self.orientation == "horizontal" and w >= h) else 0.0
            if self.orientation == "vertical" and h >= w:
                orientation_bonus = 0.7
            elif self.orientation == "any":
                orientation_bonus = 0.3

            score = (
                min(3.5, aspect / 2.0)
                + area_norm
                + contrast
                + centrality
                + depth_score
                + orientation_bonus
                + min(0.8, dark_context_ratio * 1.6)
                - min(0.8, brick_context_ratio * 2.0)
            )
            if score < min_score:
                continue

            rect = cv2.minAreaRect(contour)
            detections.append(
                HandleDetection(
                    centroid=(cx, cy),
                    bbox=(x, y, w, h),
                    score=score,
                    angle_deg=float(rect[2]),
                    contour=contour,
                    mask=mask,
                )
            )

        if not detections:
            return None
        if prefer_centroid is not None:
            return self._select_locked_handle(
                detections,
                prefer_centroid,
                image_shape=(height, width),
                radius_fraction=lock_radius_fraction,
            )
        if self.target != "best":
            return self._select_target_handle(detections)
        return max(detections, key=lambda detection: detection.score)

    def _candidate_mask(self, bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if self.orientation == "horizontal":
            bright_metal = cv2.inRange(
                hsv,
                np.array([0, 0, self.metal_v_min]),
                np.array([180, self.metal_s_max, 255]),
            )
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
            bright_metal = cv2.morphologyEx(
                bright_metal,
                cv2.MORPH_OPEN,
                np.ones((3, 3), np.uint8),
            )
            bright_metal = cv2.morphologyEx(
                bright_metal,
                cv2.MORPH_CLOSE,
                horizontal_kernel,
                iterations=1,
            )
            return bright_metal

        # Bright metal handles tend to be low saturation. Dark/black handles
        # are captured by the dark mask. Canny catches geometry in either case.
        low_saturation = cv2.inRange(hsv, np.array([0, 0, 105]), np.array([180, 85, 255]))
        bright_metal = cv2.inRange(hsv, np.array([0, 0, 145]), np.array([180, 70, 255]))
        dark_hardware = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 85]))
        edges = cv2.Canny(gray, 45, 120)

        # Top-hat isolates small bright structures against dark cabinet faces,
        # which matches the silver cylindrical handles in the RoboCasa drawers.
        horizontal_hat_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 9))
        top_hat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, horizontal_hat_kernel)
        _, top_hat_mask = cv2.threshold(top_hat, 18, 255, cv2.THRESH_BINARY)

        combined = cv2.bitwise_or(edges, low_saturation)
        combined = cv2.bitwise_or(combined, bright_metal)
        combined = cv2.bitwise_or(combined, top_hat_mask)
        combined = cv2.bitwise_or(combined, dark_hardware)

        if self.orientation == "horizontal":
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (19, 5))
        elif self.orientation == "vertical":
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 19))
        else:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))

        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=1)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        return combined

    def _mount_context_ratios(
        self,
        hsv: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
    ) -> tuple[float, float]:
        height, width = hsv.shape[:2]
        pad_x = int(max(8, w * 0.35))
        pad_y = int(max(12, h * 3.0))
        x0 = max(0, x - pad_x)
        y0 = max(0, y - pad_y)
        x1 = min(width, x + w + pad_x)
        y1 = min(height, y + h + pad_y)
        context = hsv[y0:y1, x0:x1]
        if context.size == 0:
            return 0.0, 1.0

        dark_cabinet = cv2.inRange(
            context,
            np.array([0, 0, 30]),
            np.array([180, 95, 155]),
        )
        brick_or_wood = cv2.inRange(
            context,
            np.array([0, 45, 45]),
            np.array([25, 255, 235]),
        )
        brick_red_wrap = cv2.inRange(
            context,
            np.array([165, 45, 45]),
            np.array([180, 255, 235]),
        )
        brick_or_wood = cv2.bitwise_or(brick_or_wood, brick_red_wrap)

        total = float(context.shape[0] * context.shape[1])
        return (
            cv2.countNonZero(dark_cabinet) / max(1.0, total),
            cv2.countNonZero(brick_or_wood) / max(1.0, total),
        )

    def _select_target_handle(self, detections: list[HandleDetection]) -> HandleDetection:
        ordered = sorted(detections, key=lambda detection: detection.centroid[1])
        if self.target == "top":
            return ordered[0]
        if self.target == "second":
            return ordered[min(1, len(ordered) - 1)]
        if self.target == "third":
            return ordered[min(2, len(ordered) - 1)]
        if self.target == "bottom":
            return ordered[-1]
        return max(detections, key=lambda detection: detection.score)

    def _select_locked_handle(
        self,
        detections: list[HandleDetection],
        prefer_centroid: tuple[float, float],
        image_shape: tuple[int, int],
        radius_fraction: float,
    ) -> HandleDetection | None:
        height, width = image_shape
        max_distance = max(8.0, radius_fraction * math.hypot(width, height))
        px, py = prefer_centroid
        ranked: list[tuple[float, HandleDetection]] = []
        for detection in detections:
            cx, cy = detection.centroid
            distance = math.hypot(cx - px, cy - py)
            if distance > max_distance:
                continue
            tracking_score = detection.score - 2.0 * (distance / max_distance)
            ranked.append((tracking_score, detection))

        if not ranked:
            return None
        return max(ranked, key=lambda item: item[0])[1]


def clamp(value: float, lower: float = -1.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def close_window(name: str, enabled: bool) -> None:
    if not enabled:
        return
    try:
        cv2.destroyWindow(name)
    except cv2.error:
        pass


def smooth_centroid(
    previous: tuple[float, float] | None,
    current: tuple[int, int],
    smoothing: float,
) -> tuple[float, float]:
    if previous is None:
        return float(current[0]), float(current[1])
    return (
        smoothing * previous[0] + (1.0 - smoothing) * current[0],
        smoothing * previous[1] + (1.0 - smoothing) * current[1],
    )


def smooth_command(previous: float, current: float, smoothing: float) -> float:
    smoothed = smoothing * previous + (1.0 - smoothing) * current
    if abs(smoothed) < 0.005:
        return 0.0
    return smoothed


def tracking_depth_is_consistent(
    distance: float,
    previous_distance: float | None,
    max_jump: float,
) -> bool:
    return previous_distance is None or abs(distance - previous_distance) <= max_jump


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def tapered_pixel_command(
    error: float,
    *,
    sign: float,
    gain: float,
    deadband: float,
    max_command: float,
    taper_error: float = 0.24,
    min_limit_fraction: float = 0.35,
) -> float:
    if abs(error) <= deadband:
        return 0.0
    taper = min(1.0, max(min_limit_fraction, abs(error) / max(deadband, taper_error)))
    command_limit = max_command * taper
    return clamp(sign * gain * error, -command_limit, command_limit)


def estimate_surface_yaw_from_depth(
    rgb_frame: np.ndarray,
    detection: HandleDetection,
    depth_cam,
    robot_transforms: RobotTransforms,
    sample_step: int = 6,
    min_points: int = 24,
    depth_band: float = 0.38,
) -> tuple[float, int] | None:
    """Estimate drawer-face yaw from nearby dark cabinet depth points.

    Returns the yaw, in robot-base coordinates, of the direction the robot
    should face to be square to the local cabinet surface.
    """
    depth_image = depth_cam.depth_cam.get_frame()
    if depth_image is None or depth_image.size == 0:
        return None

    x, y, w, h = detection.bbox
    height, width = rgb_frame.shape[:2]
    pad_x = int(max(24, w * 2.2))
    pad_y = int(max(28, h * 4.5))
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(width, x + w + pad_x)
    y1 = min(height, y + h + pad_y)
    if x1 <= x0 or y1 <= y0:
        return None

    handle_depth = depth_cam.get_depth(detection.centroid, depth_image=depth_image, sample_radius=4)
    if handle_depth is None:
        return None

    hsv = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2HSV)
    cam_t = robot_transforms.get_cam_T(depth_cam)
    points: list[np.ndarray] = []

    for py in range(y0, y1, max(1, sample_step)):
        for px in range(x0, x1, max(1, sample_step)):
            hue, sat, val = hsv[py, px]
            is_dark_cabinet = sat <= 120 and 28 <= val <= 170
            if not is_dark_cabinet:
                continue

            distance = depth_cam.get_depth((px, py), depth_image=depth_image, sample_radius=1)
            if distance is None or not (0.15 <= distance <= 3.5):
                continue
            if abs(distance - handle_depth) > depth_band:
                continue

            x_norm = (px - depth_cam.cx) / depth_cam.fx
            y_norm = (py - depth_cam.cy) / depth_cam.fy
            point_cam = np.array([distance * x_norm, distance * y_norm, distance, 1.0])
            point_base = cam_t @ point_cam
            points.append(point_base[:3])

    if len(points) < min_points:
        return None

    points_np = np.asarray(points, dtype=float)
    center = points_np.mean(axis=0)
    centered = points_np - center
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None

    normal = vh[-1]
    normal_xy = normal[:2]
    norm_xy = float(np.linalg.norm(normal_xy))
    if norm_xy < 1e-4:
        return None
    normal_xy = normal_xy / norm_xy

    center_xy = center[:2]
    if np.dot(normal_xy, -center_xy) < 0:
        normal_xy = -normal_xy

    face_direction = -normal_xy
    yaw = math.atan2(face_direction[1], face_direction[0])
    return yaw, len(points)


def estimate_surface_yaw_from_lidar(
    ranges: np.ndarray | None,
    target_bearing: float,
    half_angle: float,
    max_range: float,
    min_points: int,
) -> tuple[float, int, float] | None:
    """Fit a local LiDAR line and return the yaw from robot to that surface."""
    if ranges is None or len(ranges) < 8:
        return None

    ranges = np.asarray(ranges, dtype=float).reshape(-1)
    angles = np.linspace(0.0, 2.0 * math.pi, len(ranges), endpoint=False)
    signed_delta = np.arctan2(np.sin(angles - target_bearing), np.cos(angles - target_bearing))
    valid = (
        np.isfinite(ranges)
        & (ranges >= 0.08)
        & (ranges <= max_range)
        & (np.abs(signed_delta) <= half_angle)
    )
    if int(np.count_nonzero(valid)) < min_points:
        return None

    hit_angles = angles[valid]
    hit_ranges = ranges[valid]
    points = np.column_stack(
        (
            hit_ranges * np.cos(hit_angles),
            hit_ranges * np.sin(hit_angles),
        )
    )

    center = points.mean(axis=0)
    centered = points - center
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None

    tangent = vh[0]
    normal = np.array([-tangent[1], tangent[0]], dtype=float)
    norm = float(np.linalg.norm(normal))
    if norm < 1e-4:
        return None
    normal /= norm

    if np.dot(normal, -center) < 0:
        normal = -normal

    face_direction = -normal
    yaw = math.atan2(face_direction[1], face_direction[0])
    return yaw, int(points.shape[0]), float(np.linalg.norm(center))


def draw_detection(frame: np.ndarray, detection: HandleDetection | None, label: str) -> np.ndarray:
    canvas = frame.copy()
    height, width = canvas.shape[:2]
    cv2.drawMarker(
        canvas,
        (width // 2, height // 2),
        (0, 0, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=16,
        thickness=2,
    )
    if detection is None:
        cv2.putText(
            canvas,
            f"{label}: no handle",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
        return canvas

    x, y, w, h = detection.bbox
    cx, cy = detection.centroid
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.circle(canvas, (cx, cy), 7, (0, 255, 255), -1)
    cv2.putText(
        canvas,
        f"{label}: score {detection.score:.1f}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )
    return canvas


def maybe_save_debug_frame(
    debug_dir: Path | None,
    last_save_time: float,
    frame: np.ndarray | None,
    detection: HandleDetection | None,
    mask: np.ndarray | None,
    label: str,
    now: float,
    period_s: float = 1.0,
) -> float:
    if debug_dir is None or frame is None or now - last_save_time < period_s:
        return last_save_time

    debug_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(debug_dir / f"{label}_rgb.png"), draw_detection(frame, detection, label))
    if mask is not None:
        cv2.imwrite(str(debug_dir / f"{label}_mask.png"), mask)
    return now


def perception_only_command(command: dict[str, float]) -> dict[str, float]:
    return {
        joint: velocity
        for joint, velocity in command.items()
        if joint in {"head_pan_counterclockwise", "head_tilt_up"}
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find, approach, grasp, and pull a drawer handle.")
    parser.add_argument("--dry-run", action="store_true", help="Perception test: allow head scan, freeze base/arm/gripper.")
    parser.add_argument("--no-display", action="store_true", help="Disable OpenCV visualization windows.")
    parser.add_argument("--orientation", choices=["horizontal", "vertical", "any"], default="horizontal")
    parser.add_argument("--target-handle", choices=["best", "top", "second", "third", "bottom"], default="best")
    parser.add_argument("--target-distance", type=float, default=0.62, help="Head-camera approach distance in meters.")
    parser.add_argument("--grasp-depth", type=float, default=0.18, help="Wrist-camera standoff depth before rolling/closing the gripper.")
    parser.add_argument("--grasp-depth-tolerance", type=float, default=0.035, help="Allowed wrist depth margin before rolling/closing.")
    parser.add_argument("--pull-seconds", type=float, default=3.5, help="How long to retract/pull after grasping.")
    parser.add_argument("--min-score", type=float, default=2.4, help="Minimum detector score for a valid handle.")
    parser.add_argument("--min-x", type=float, default=0.04, help="Ignore detections left of this image-width fraction.")
    parser.add_argument("--max-x", type=float, default=0.96, help="Ignore detections right of this image-width fraction.")
    parser.add_argument("--min-y", type=float, default=0.02, help="Ignore detections above this image-height fraction.")
    parser.add_argument("--max-y", type=float, default=0.92, help="Ignore detections below this image-height fraction.")
    parser.add_argument("--edge-margin", type=float, default=0.025, help="Reject handle boxes touching this image-width border fraction.")
    parser.add_argument("--metal-v-min", type=int, default=115, help="HSV V minimum for bright metal mask.")
    parser.add_argument("--metal-s-max", type=int, default=115, help="HSV S maximum for bright metal mask.")
    parser.add_argument("--bright-ratio-min", type=float, default=0.08, help="Minimum bright-metal ratio inside candidate box.")
    parser.add_argument("--context-dark-min", type=float, default=0.16, help="Minimum dark cabinet ratio around a handle.")
    parser.add_argument("--context-brick-max", type=float, default=0.24, help="Maximum brick/wall ratio around a handle.")
    parser.add_argument("--require-detection-depth", action="store_true", help="Only draw/select handles with valid depth.")
    parser.add_argument("--search-tilt", type=float, default=-0.10, help="Head tilt target while searching.")
    parser.add_argument("--scan-speed", type=float, default=0.12, help="Head pan speed while searching.")
    parser.add_argument("--head-kp", type=float, default=0.55, help="Head visual-servo proportional gain.")
    parser.add_argument("--head-max", type=float, default=0.22, help="Max head command during visual servo.")
    parser.add_argument("--approach-angle-deg", type=float, default=60.0, help="Angle window that still allows forward approach.")
    parser.add_argument("--approach-min-auth", type=float, default=0.25, help="Minimum forward authority while the handle is visible.")
    parser.add_argument("--base-forward-max", type=float, default=0.50, help="Max normalized base-forward command during approach.")
    parser.add_argument("--base-turn-max", type=float, default=0.55, help="Max normalized base-turn command during approach.")
    parser.add_argument("--approach-creep-max", type=float, default=0.22, help="Max normalized forward command while visually tracking.")
    parser.add_argument("--approach-pulse-seconds", type=float, default=0.45, help="Drive duration for each slow approach pulse.")
    parser.add_argument("--approach-pause-seconds", type=float, default=0.20, help="Settling duration between approach pulses.")
    parser.add_argument("--approach-lost-seconds", type=float, default=1.8, help="Local reacquire time before returning to global search.")
    parser.add_argument("--drive-center-x", type=float, default=0.18, help="Required horizontal image error before forward creep is allowed.")
    parser.add_argument("--drive-center-y", type=float, default=0.28, help="Required vertical image error before forward creep is allowed.")
    parser.add_argument("--local-reacquire-pan", type=float, default=0.06, help="Small head-pan sweep while recovering a locked handle.")
    parser.add_argument("--wrist-handoff-distance", type=float, default=0.95, help="Switch to wrist camera if head loses a close, aligned handle.")
    parser.add_argument("--wrist-handoff-angle-deg", type=float, default=10.0, help="Max recent base angle for lost-head wrist handoff.")
    parser.add_argument("--base-align-distance", type=float, default=0.95, help="Stop creeping and rotate the base for gripper-side alignment when this close.")
    parser.add_argument("--base-align-angle-deg", type=float, default=4.0, help="Base angle tolerance before wrist handoff.")
    parser.add_argument("--gripper-target-bearing-deg", type=float, default=-90.0, help="Target handle bearing in base frame when the gripper/arm points at it.")
    parser.add_argument("--gripper-roll-deg", type=float, default=90.0, help="Roll the gripper only after reaching the handle; use -90 for the opposite direction or 0 to leave it flat.")
    parser.add_argument("--gripper-roll-timeout", type=float, default=5.0, help="Max seconds to wait for delayed gripper roll before closing.")
    parser.add_argument("--base-align-turn-max", type=float, default=0.16, help="Max normalized base turn during close square-up.")
    parser.add_argument("--base-align-min-turn", type=float, default=0.035, help="Minimum normalized turn when square-up angle is still nonzero.")
    parser.add_argument("--base-align-stable-seconds", type=float, default=0.55, help="How long the base angle must stay aligned.")
    parser.add_argument("--base-align-forward-max", type=float, default=0.10, help="Max normalized forward creep while square to the drawer face.")
    parser.add_argument("--base-align-distance-margin", type=float, default=0.08, help="Allowed head distance margin before wrist handoff.")
    parser.add_argument("--disable-base-align-wrist-check", action="store_true", help="Do not use the wrist camera to confirm/refine base alignment before handoff.")
    parser.add_argument("--base-align-wrist-distance", type=float, default=1.05, help="Head distance below which base_align may trust wrist-camera confirmation.")
    parser.add_argument("--base-align-wrist-center-x", type=float, default=0.18, help="Required wrist horizontal pixel error before base_align handoff.")
    parser.add_argument("--base-align-wrist-center-y", type=float, default=0.18, help="Required wrist vertical pixel error before base_align handoff.")
    parser.add_argument("--base-settle-seconds", type=float, default=0.8, help="Hold base still before wrist visual servo after base motion.")
    parser.add_argument("--wrist-slide-kp", type=float, default=0.18, help="Base-forward gain for centering the locked wrist handle.")
    parser.add_argument("--wrist-slide-max", type=float, default=0.05, help="Max normalized base-forward command during wrist centering.")
    parser.add_argument("--wrist-slide-deadband", type=float, default=0.09, help="Ignore tiny wrist horizontal pixel errors below this fraction.")
    parser.add_argument("--wrist-slide-command-smoothing", type=float, default=0.72, help="Low-pass smoothing for wrist side-slide commands.")
    parser.add_argument("--wrist-slide-pulse-seconds", type=float, default=0.16, help="Drive duration for each wrist side-slide correction pulse.")
    parser.add_argument("--wrist-slide-pause-seconds", type=float, default=0.34, help="Settling duration between wrist side-slide correction pulses.")
    parser.add_argument("--wrist-slide-continuous", action="store_true", help="Drive wrist side-slide continuously instead of pulse/settle correction.")
    parser.add_argument("--wrist-slide-sign", type=float, default=1.0, help="Set to -1 if wrist horizontal centering moves the wrong way.")
    parser.add_argument("--enable-wrist-image-joints", action="store_true", help="Allow wrist yaw/pitch to center the wrist image; default keeps wrist fixed.")
    parser.add_argument("--wrist-lift-pixel-kp", type=float, default=1.2, help="Lift gain for centering the locked wrist handle vertically in the image.")
    parser.add_argument("--wrist-lift-max", type=float, default=0.38, help="Max normalized lift command during wrist image centering.")
    parser.add_argument("--wrist-lift-deadband", type=float, default=0.04, help="Ignore tiny wrist vertical pixel errors below this fraction.")
    parser.add_argument("--wrist-lift-sign", type=float, default=-1.0, help="Set to 1 if wrist vertical centering moves the wrong way.")
    parser.add_argument("--wrist-depth-seek-speed", type=float, default=0.18, help="Slow arm-out command while wrist sees a handle but has no valid depth.")
    parser.add_argument("--wrist-depth-seek-max-arm-out", type=float, default=0.55, help="Max arm extension while seeking valid wrist depth.")
    parser.add_argument("--wrist-depth-seek-center-x", type=float, default=0.22, help="Require wrist horizontal pixel error below this before seeking depth.")
    parser.add_argument("--wrist-depth-seek-center-y", type=float, default=0.22, help="Require wrist vertical pixel error below this before seeking depth.")
    parser.add_argument("--reach-arm-kp", type=float, default=3.0, help="Arm extension gain during final wrist reach.")
    parser.add_argument("--reach-arm-max-out", type=float, default=0.22, help="Max normalized arm-out command during final wrist reach.")
    parser.add_argument("--reach-arm-max-in", type=float, default=0.25, help="Max normalized arm-in command if final wrist reach gets too close.")
    parser.add_argument("--reach-min-depth", type=float, default=0.14, help="Emergency wrist depth floor; retract instead of pushing closer below this.")
    parser.add_argument("--reach-center-x", type=float, default=0.08, help="Required wrist horizontal pixel error before extending in reach.")
    parser.add_argument("--reach-center-y", type=float, default=0.10, help="Required wrist vertical pixel error before extending in reach.")
    parser.add_argument("--lock-depth-jump", type=float, default=0.35, help="Reject locked detections whose depth jumps by more than this many meters.")
    parser.add_argument("--disable-plane-align", action="store_true", help="Use handle bearing instead of depth plane normal during base_align.")
    parser.add_argument("--plane-align-min-points", type=int, default=24, help="Minimum cabinet depth points for plane-based base alignment.")
    parser.add_argument("--plane-align-depth-band", type=float, default=0.38, help="Depth band around the handle for cabinet plane samples.")
    parser.add_argument("--plane-align-sample-step", type=int, default=6, help="Pixel stride for cabinet plane sampling.")
    parser.add_argument("--disable-lidar-align", action="store_true", help="Disable LiDAR fallback for squaring to the drawer/cabinet face.")
    parser.add_argument("--prefer-lidar-align", action="store_true", help="Use LiDAR surface fit before the head depth-plane fit during base square-up.")
    parser.add_argument("--lidar-align-half-angle-deg", type=float, default=32.0, help="LiDAR angle window around gripper target bearing for local surface fit.")
    parser.add_argument("--lidar-align-max-range", type=float, default=1.8, help="Max LiDAR range used for close drawer/cabinet surface fitting.")
    parser.add_argument("--lidar-align-min-points", type=int, default=8, help="Minimum LiDAR points needed for surface alignment fallback.")
    parser.add_argument("--lock-radius", type=float, default=0.28, help="Image-diagonal fraction allowed around the locked handle.")
    parser.add_argument("--lock-smoothing", type=float, default=0.55, help="Centroid smoothing for the locked handle, 0 to 0.95.")
    parser.add_argument("--base-scan", action="store_true", help="Rotate the base during search if the handle is not found.")
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=Path("/tmp/stretch_drawer_debug"),
        help="Directory for latest camera/detector debug PNGs. Use '' to disable.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if str(args.debug_dir) == "":
        args.debug_dir = None
    args.grasp_depth = max(0.05, args.grasp_depth)
    args.grasp_depth_tolerance = max(0.005, args.grasp_depth_tolerance)
    args.approach_angle_deg = max(5.0, args.approach_angle_deg)
    args.approach_min_auth = clamp(args.approach_min_auth, 0.0, 1.0)
    args.base_forward_max = max(0.05, args.base_forward_max)
    args.base_turn_max = max(0.05, args.base_turn_max)
    args.approach_creep_max = max(0.03, args.approach_creep_max)
    args.approach_pulse_seconds = max(0.05, args.approach_pulse_seconds)
    args.approach_pause_seconds = max(0.0, args.approach_pause_seconds)
    args.approach_lost_seconds = max(0.4, args.approach_lost_seconds)
    args.drive_center_x = clamp(args.drive_center_x, 0.02, 0.5)
    args.drive_center_y = clamp(args.drive_center_y, 0.02, 0.5)
    args.local_reacquire_pan = clamp(args.local_reacquire_pan, 0.0, args.head_max)
    args.wrist_handoff_distance = max(args.target_distance, args.wrist_handoff_distance)
    args.wrist_handoff_angle_deg = max(1.0, args.wrist_handoff_angle_deg)
    args.base_align_distance = max(args.target_distance, args.base_align_distance)
    args.base_align_angle_deg = max(0.5, args.base_align_angle_deg)
    args.gripper_target_bearing_deg = clamp(args.gripper_target_bearing_deg, -179.0, 179.0)
    args.gripper_roll_deg = clamp(args.gripper_roll_deg, -180.0, 180.0)
    args.gripper_roll_timeout = max(0.1, args.gripper_roll_timeout)
    args.base_align_turn_max = clamp(args.base_align_turn_max, 0.02, args.base_turn_max)
    args.base_align_min_turn = clamp(args.base_align_min_turn, 0.0, args.base_align_turn_max)
    args.base_align_stable_seconds = max(0.05, args.base_align_stable_seconds)
    args.base_align_forward_max = clamp(args.base_align_forward_max, 0.0, args.base_forward_max)
    args.base_align_distance_margin = max(0.01, args.base_align_distance_margin)
    args.base_align_wrist_distance = max(args.target_distance, args.base_align_wrist_distance)
    args.base_align_wrist_center_x = clamp(args.base_align_wrist_center_x, 0.02, 0.5)
    args.base_align_wrist_center_y = clamp(args.base_align_wrist_center_y, 0.02, 0.5)
    args.base_settle_seconds = max(0.0, args.base_settle_seconds)
    args.wrist_slide_kp = max(0.0, args.wrist_slide_kp)
    args.wrist_slide_max = clamp(args.wrist_slide_max, 0.0, args.base_forward_max)
    args.wrist_slide_deadband = clamp(args.wrist_slide_deadband, 0.0, 0.25)
    args.wrist_slide_command_smoothing = clamp(args.wrist_slide_command_smoothing, 0.0, 0.95)
    args.wrist_slide_pulse_seconds = max(0.02, args.wrist_slide_pulse_seconds)
    args.wrist_slide_pause_seconds = max(0.0, args.wrist_slide_pause_seconds)
    args.wrist_slide_sign = -1.0 if args.wrist_slide_sign < 0.0 else 1.0
    args.wrist_lift_pixel_kp = max(0.0, args.wrist_lift_pixel_kp)
    args.wrist_lift_max = clamp(args.wrist_lift_max, 0.0, 0.8)
    args.wrist_lift_deadband = clamp(args.wrist_lift_deadband, 0.0, 0.25)
    args.wrist_lift_sign = -1.0 if args.wrist_lift_sign < 0.0 else 1.0
    args.wrist_depth_seek_speed = clamp(args.wrist_depth_seek_speed, 0.0, 0.6)
    args.wrist_depth_seek_max_arm_out = max(0.0, args.wrist_depth_seek_max_arm_out)
    args.wrist_depth_seek_center_x = clamp(args.wrist_depth_seek_center_x, 0.02, 0.5)
    args.wrist_depth_seek_center_y = clamp(args.wrist_depth_seek_center_y, 0.02, 0.5)
    args.reach_arm_kp = max(0.0, args.reach_arm_kp)
    args.reach_arm_max_out = clamp(args.reach_arm_max_out, 0.0, 0.6)
    args.reach_arm_max_in = clamp(args.reach_arm_max_in, 0.0, 0.6)
    args.reach_min_depth = clamp(args.reach_min_depth, 0.03, args.grasp_depth)
    args.reach_center_x = clamp(args.reach_center_x, 0.02, 0.5)
    args.reach_center_y = clamp(args.reach_center_y, 0.02, 0.5)
    args.lock_depth_jump = max(0.05, args.lock_depth_jump)
    args.plane_align_min_points = max(6, args.plane_align_min_points)
    args.plane_align_depth_band = max(0.05, args.plane_align_depth_band)
    args.plane_align_sample_step = max(2, args.plane_align_sample_step)
    args.lidar_align_half_angle_deg = clamp(args.lidar_align_half_angle_deg, 5.0, 90.0)
    args.lidar_align_max_range = max(0.25, args.lidar_align_max_range)
    args.lidar_align_min_points = max(3, args.lidar_align_min_points)
    args.lock_radius = max(0.05, args.lock_radius)
    args.lock_smoothing = clamp(args.lock_smoothing, 0.0, 0.95)
    args.min_x = clamp(args.min_x, 0.0, 0.95)
    args.max_x = clamp(args.max_x, args.min_x + 0.01, 1.0)
    args.min_y = clamp(args.min_y, 0.0, 0.95)
    args.max_y = clamp(args.max_y, args.min_y + 0.01, 1.0)
    args.edge_margin = clamp(args.edge_margin, 0.0, 0.25)
    args.context_dark_min = clamp(args.context_dark_min, 0.0, 1.0)
    args.context_brick_max = clamp(args.context_brick_max, 0.0, 1.0)
    print(f"\n=== Running on {BACKEND_NAME} backend ===")
    print("Press Ctrl+C to stop. Toggle manual mode with the configured teleop key if needed.\n")
    print(
        f"Delayed gripper roll target: {args.gripper_roll_deg:+.0f}deg. "
        f"Wrist slide mode: {'continuous' if args.wrist_slide_continuous else 'pulse/settle'}.\n"
    )
    if args.dry_run:
        print("Dry run: head pan/tilt may move; base, arm, and gripper stay frozen.\n")
    if args.debug_dir is not None:
        print(f"Saving latest detector snapshots to {args.debug_dir}\n")

    transforms = RobotTransforms(controller)
    detector = HandleDetector(
        orientation=args.orientation,
        target=args.target_handle,
        min_score=args.min_score,
        min_x_fraction=args.min_x,
        max_x_fraction=args.max_x,
        min_y_fraction=args.min_y,
        max_y_fraction=args.max_y,
        edge_margin_fraction=args.edge_margin,
        metal_v_min=args.metal_v_min,
        metal_s_max=args.metal_s_max,
        bright_ratio_min=args.bright_ratio_min,
        context_dark_ratio_min=args.context_dark_min,
        context_brick_ratio_max=args.context_brick_max,
        require_depth=args.require_detection_depth,
    )

    gripper_roll = math.radians(args.gripper_roll_deg)

    stow_pose = StateController(
        controller,
        {
            "wrist_roll_counterclockwise": 0.0,
            "wrist_yaw_counterclockwise": 0.0,
            "wrist_pitch_up": 0.0,
            "gripper_open": 0.45,
            "arm_out": 0.0,
        },
    )
    search_head_pose = StateController(
        controller,
        {
            "head_tilt_up": args.search_tilt,
        },
    )
    pre_grasp_pose = StateController(
        controller,
        {
            "wrist_roll_counterclockwise": 0.0,
            "gripper_open": 0.55,
        },
    )
    orient_gripper_pose = StateController(
        controller,
        {
            "wrist_roll_counterclockwise": gripper_roll,
            "gripper_open": 0.55,
        },
    )
    gripper_roll_pose = StateController(
        controller,
        {
            "wrist_roll_counterclockwise": gripper_roll,
        },
    )

    kp_head = args.head_kp
    kp_base_angle = 5.0 / math.pi
    kp_forward = 2.0
    kp_lift = 4.5
    kp_wrist_yaw = 0.7
    kp_wrist_pitch = 0.45
    gripper_target_bearing = math.radians(args.gripper_target_bearing_deg)

    phase = "search"
    search_dir = 1.0
    last_seen = time.perf_counter()
    phase_started = time.perf_counter()
    stable_since: float | None = None
    last_camera_warning = 0.0
    last_debug_save = 0.0
    last_depth_warning = 0.0
    locked_head_centroid: tuple[float, float] | None = None
    locked_wrist_centroid: tuple[float, float] | None = None
    last_wrist_slide_cmd = 0.0
    wrist_slide_cycle_started = phase_started
    last_approach_seen = 0.0
    last_handle_distance: float | None = None
    last_handle_angle: float | None = None
    base_aligned_for_lock = False

    print(f"Phase: {phase}")

    try:
        while True:
            auto_cmd: dict[str, float] = {}
            now = time.perf_counter()

            if phase == "search":
                rgb = HEAD_RGB_CAMERA.get_frame()
                if rgb is None:
                    detection = None
                    if now - last_camera_warning > 1.0:
                        print("\rWaiting for head RGB camera frame; robot is holding still.   ", end="", flush=True)
                        last_camera_warning = now
                    auto_cmd = {}
                else:
                    detection = detector.detect(rgb, HEAD_CAMERA)

                if rgb is not None and detection is not None:
                    last_seen = now
                    locked_head_centroid = smooth_centroid(None, detection.centroid, args.lock_smoothing)
                    phase = "approach"
                    phase_started = now
                    last_approach_seen = now
                    last_handle_distance = None
                    last_handle_angle = None
                    locked_wrist_centroid = None
                    last_wrist_slide_cmd = 0.0
                    base_aligned_for_lock = False
                    stable_since = None
                    print(f"\nPhase: {phase} locked={tuple(round(v, 1) for v in locked_head_centroid)}")
                elif rgb is not None:
                    state = controller.get_state()
                    pan = state.get("head_pan_counterclockwise", 0.0)
                    if pan > 0.9:
                        search_dir = -1.0
                    elif pan < -0.9:
                        search_dir = 1.0

                    auto_cmd["head_pan_counterclockwise"] = args.scan_speed * search_dir
                    if args.base_scan and now - last_seen > 3.0:
                        auto_cmd["base_counterclockwise"] = 0.12 * search_dir

                auto_cmd = merge_proportional(auto_cmd, stow_pose.get_command())
                auto_cmd = merge_proportional(auto_cmd, search_head_pose.get_command())
                if not args.no_display and rgb is not None:
                    cv2.imshow("Head RGB", draw_detection(rgb, detection, "search"))
                    if detector.last_mask is not None:
                        cv2.imshow("Handle mask", detector.last_mask)
                last_debug_save = maybe_save_debug_frame(
                    args.debug_dir, last_debug_save, rgb, detection, detector.last_mask, "head_search", now
                )

            elif phase == "approach":
                rgb = HEAD_RGB_CAMERA.get_frame()
                detection = detector.detect(
                    rgb,
                    HEAD_CAMERA,
                    prefer_centroid=locked_head_centroid,
                    lock_radius_fraction=args.lock_radius,
                    relax_locked_detection=True,
                )
                if detection is None:
                    lost_elapsed = now - last_approach_seen
                    if rgb is not None and locked_head_centroid is not None:
                        frame_cx = rgb.shape[1] / 2
                        frame_cy = rgb.shape[0] / 2
                        error_x = (locked_head_centroid[0] - frame_cx) / rgb.shape[1]
                        error_y = (locked_head_centroid[1] - frame_cy) / rgb.shape[0]
                        lost_elapsed = max(0.0, lost_elapsed)
                        local_sweep = args.local_reacquire_pan * math.sin(lost_elapsed * 7.0)
                        auto_cmd["head_pan_counterclockwise"] = clamp(
                            -kp_head * error_x + local_sweep,
                            -args.head_max,
                            args.head_max,
                        )
                        auto_cmd["head_tilt_up"] = clamp(
                            -kp_head * error_y,
                            -args.head_max,
                            args.head_max,
                        )

                    ready_for_wrist_handoff = (
                        last_handle_distance is not None
                        and last_handle_distance <= args.wrist_handoff_distance
                        and last_handle_angle is not None
                        and abs(wrap_angle(last_handle_angle - gripper_target_bearing))
                        <= math.radians(args.wrist_handoff_angle_deg)
                    )
                    if ready_for_wrist_handoff and lost_elapsed > 0.35:
                        phase = "base_settle"
                        phase_started = now
                        stable_since = None
                        locked_wrist_centroid = None
                        last_wrist_slide_cmd = 0.0
                        close_window("Head RGB", not args.no_display)
                        print(f"\nPhase: {phase} (head lost close handle; trying wrist)")
                    elif lost_elapsed > args.approach_lost_seconds:
                        phase = "search"
                        locked_head_centroid = None
                        locked_wrist_centroid = None
                        last_wrist_slide_cmd = 0.0
                        last_handle_distance = None
                        last_handle_angle = None
                        print(f"\nPhase: {phase}")
                    else:
                        print(
                            f"\rTracking lost; holding base for local reacquire "
                            f"{lost_elapsed:.1f}/{args.approach_lost_seconds:.1f}s   ",
                            end="",
                            flush=True,
                        )

                    auto_cmd = merge_proportional(auto_cmd, stow_pose.get_command())
                else:
                    cx, cy = detection.centroid
                    frame_cx = rgb.shape[1] / 2
                    frame_cy = rgb.shape[0] / 2
                    error_x = (cx - frame_cx) / rgb.shape[1]
                    error_y = (cy - frame_cy) / rgb.shape[0]
                    auto_cmd["head_pan_counterclockwise"] = clamp(
                        -kp_head * error_x, -args.head_max, args.head_max
                    )
                    auto_cmd["head_tilt_up"] = clamp(
                        -kp_head * error_y, -args.head_max, args.head_max
                    )

                    _, handle_base_t = locate_object(detection.centroid, HEAD_CAMERA, transforms, sample_radius=5)
                    if handle_base_t is not None:
                        x, y, z = handle_base_t[0:3, 3]
                        angle_z = math.atan2(y, x)
                        distance = math.sqrt(x * x + y * y)
                        distance_error = distance - args.target_distance
                        if not tracking_depth_is_consistent(
                            distance,
                            last_handle_distance,
                            args.lock_depth_jump,
                        ):
                            detection = None
                            print(
                                f"\rRejected locked target depth jump to {distance:.2f}m; holding base.   ",
                                end="",
                                flush=True,
                            )
                            auto_cmd = merge_proportional(auto_cmd, stow_pose.get_command())
                        else:
                            last_approach_seen = now
                            locked_head_centroid = smooth_centroid(
                                locked_head_centroid,
                                detection.centroid,
                                args.lock_smoothing,
                            )
                            last_handle_distance = distance
                            last_handle_angle = angle_z

                            turn_cmd = clamp(
                                -kp_base_angle * angle_z,
                                -args.base_turn_max,
                                args.base_turn_max,
                            )
                            approach_window = math.radians(args.approach_angle_deg)
                            alignment = 1.0 - min(1.0, abs(angle_z) / approach_window)
                            if distance_error > 0.0:
                                travel_auth = max(args.approach_min_auth, alignment)
                            else:
                                travel_auth = 1.0
                            forward_cmd = clamp(
                                kp_forward * distance_error * travel_auth,
                                -0.25,
                                min(args.base_forward_max, args.approach_creep_max),
                            )

                            centered_for_drive = (
                                abs(error_x) <= args.drive_center_x
                                and abs(error_y) <= args.drive_center_y
                            )
                            pulse_period = args.approach_pulse_seconds + args.approach_pause_seconds
                            in_drive_pulse = (
                                pulse_period <= 0.0
                                or ((now - phase_started) % pulse_period) < args.approach_pulse_seconds
                            )
                            if not centered_for_drive or not in_drive_pulse:
                                forward_cmd = 0.0

                            if (
                                not base_aligned_for_lock
                                and distance <= args.base_align_distance
                            ):
                                phase = "base_align"
                                phase_started = now
                                stable_since = None
                                forward_cmd = 0.0
                                print(f"\nPhase: {phase}")

                            auto_cmd["base_counterclockwise"] = turn_cmd
                            auto_cmd["base_forward"] = forward_cmd

                            wrist_z = transforms.get_wrist_cam_T()[2, 3]
                            auto_cmd["lift_up"] = clamp(kp_lift * (z - wrist_z), -0.6, 0.6)

                            at_distance = abs(distance_error) < 0.045
                            at_angle = abs(angle_z) < math.radians(5)
                            if at_distance and at_angle and stow_pose.is_at_goal():
                                stable_since = stable_since or now
                                if now - stable_since > 0.5:
                                    phase = "base_settle"
                                    phase_started = now
                                    stable_since = None
                                    locked_wrist_centroid = None
                                    last_wrist_slide_cmd = 0.0
                                    close_window("Head RGB", not args.no_display)
                                    print(f"\nPhase: {phase}")
                            else:
                                stable_since = None

                            print(
                                f"\rDist {distance:.2f}m  angle {math.degrees(angle_z):+5.1f}deg  "
                                f"z {z:.2f}m  fwd {forward_cmd:+.2f}  turn {turn_cmd:+.2f}  "
                                f"center {'ok' if centered_for_drive else 'hold'}  "
                                f"{'pulse' if in_drive_pulse else 'settle'}  score {detection.score:.1f}"
                                f"{'  DRY-RUN freezes base' if args.dry_run else ''}   ",
                                end="",
                                flush=True,
                            )
                    else:
                        last_approach_seen = now
                        locked_head_centroid = smooth_centroid(
                            locked_head_centroid,
                            detection.centroid,
                            args.lock_smoothing,
                        )
                        if now - last_depth_warning > 1.0:
                            print(
                                f"\rDetected handle score {detection.score:.1f}, waiting for valid head depth.   ",
                                end="",
                                flush=True,
                            )
                            last_depth_warning = now

                    auto_cmd = merge_proportional(auto_cmd, stow_pose.get_command())

                if not args.no_display and rgb is not None:
                    cv2.imshow("Head RGB", draw_detection(rgb, detection, "approach"))
                    if detector.last_mask is not None:
                        cv2.imshow("Handle mask", detector.last_mask)
                last_debug_save = maybe_save_debug_frame(
                    args.debug_dir, last_debug_save, rgb, detection, detector.last_mask, "head_approach", now
                )

            elif phase == "base_align":
                rgb = HEAD_RGB_CAMERA.get_frame()
                detection = detector.detect(
                    rgb,
                    HEAD_CAMERA,
                    prefer_centroid=locked_head_centroid,
                    lock_radius_fraction=args.lock_radius,
                    relax_locked_detection=True,
                )
                head_mask = detector.last_mask
                wrist_rgb_base = None
                wrist_detection_base = None
                wrist_mask_base = None
                if detection is None:
                    lost_elapsed = now - last_approach_seen
                    if rgb is not None and locked_head_centroid is not None:
                        frame_cx = rgb.shape[1] / 2
                        frame_cy = rgb.shape[0] / 2
                        error_x = (locked_head_centroid[0] - frame_cx) / rgb.shape[1]
                        error_y = (locked_head_centroid[1] - frame_cy) / rgb.shape[0]
                        lost_elapsed = max(0.0, lost_elapsed)
                        local_sweep = args.local_reacquire_pan * math.sin(lost_elapsed * 7.0)
                        auto_cmd["head_pan_counterclockwise"] = clamp(
                            -kp_head * error_x + local_sweep,
                            -args.head_max,
                            args.head_max,
                        )
                        auto_cmd["head_tilt_up"] = clamp(
                            -kp_head * error_y,
                            -args.head_max,
                            args.head_max,
                        )

                    ready_for_wrist_handoff = (
                        last_handle_distance is not None
                        and last_handle_distance <= args.wrist_handoff_distance
                        and last_handle_angle is not None
                        and abs(wrap_angle(last_handle_angle - gripper_target_bearing))
                        <= math.radians(args.wrist_handoff_angle_deg)
                    )
                    if ready_for_wrist_handoff and lost_elapsed > 0.35:
                        phase = "base_settle"
                        phase_started = now
                        stable_since = None
                        locked_wrist_centroid = None
                        last_wrist_slide_cmd = 0.0
                        close_window("Head RGB", not args.no_display)
                        print(f"\nPhase: {phase} (base align lost close handle; trying wrist)")
                    elif lost_elapsed > args.approach_lost_seconds:
                        phase = "search"
                        locked_head_centroid = None
                        locked_wrist_centroid = None
                        last_wrist_slide_cmd = 0.0
                        last_handle_distance = None
                        last_handle_angle = None
                        base_aligned_for_lock = False
                        print(f"\nPhase: {phase}")
                    else:
                        print(
                            f"\rBase align lost lock; holding base "
                            f"{lost_elapsed:.1f}/{args.approach_lost_seconds:.1f}s   ",
                            end="",
                            flush=True,
                        )

                    auto_cmd = merge_proportional(auto_cmd, stow_pose.get_command())
                else:
                    cx, cy = detection.centroid
                    frame_cx = rgb.shape[1] / 2
                    frame_cy = rgb.shape[0] / 2
                    error_x = (cx - frame_cx) / rgb.shape[1]
                    error_y = (cy - frame_cy) / rgb.shape[0]
                    auto_cmd["head_pan_counterclockwise"] = clamp(
                        -kp_head * error_x,
                        -args.head_max,
                        args.head_max,
                    )
                    auto_cmd["head_tilt_up"] = clamp(
                        -kp_head * error_y,
                        -args.head_max,
                        args.head_max,
                    )

                    _, handle_base_t = locate_object(detection.centroid, HEAD_CAMERA, transforms, sample_radius=5)
                    if handle_base_t is not None:
                        x, y, z = handle_base_t[0:3, 3]
                        angle_z = math.atan2(y, x)
                        distance = math.sqrt(x * x + y * y)
                        distance_error = distance - args.target_distance

                        if not tracking_depth_is_consistent(
                            distance,
                            last_handle_distance,
                            args.lock_depth_jump,
                        ):
                            detection = None
                            stable_since = None
                            print(
                                f"\rBase align rejected depth jump to {distance:.2f}m; holding base.   ",
                                end="",
                                flush=True,
                            )
                        else:
                            last_approach_seen = now
                            locked_head_centroid = smooth_centroid(
                                locked_head_centroid,
                                detection.centroid,
                                args.lock_smoothing,
                            )
                            last_handle_distance = distance
                            last_handle_angle = angle_z

                            align_yaw = angle_z
                            align_source = "handle"
                            align_points = 0
                            align_range: float | None = None
                            lidar_estimate: tuple[float, int, float] | None = None
                            lidar_checked = False

                            if args.prefer_lidar_align and not args.disable_lidar_align:
                                lidar_estimate = estimate_surface_yaw_from_lidar(
                                    controller.get_lidar_ranges(),
                                    target_bearing=gripper_target_bearing,
                                    half_angle=math.radians(args.lidar_align_half_angle_deg),
                                    max_range=args.lidar_align_max_range,
                                    min_points=args.lidar_align_min_points,
                                )
                                lidar_checked = True
                                if lidar_estimate is not None:
                                    align_yaw, align_points, align_range = lidar_estimate
                                    align_source = "lidar-plane"

                            if align_source == "handle" and not args.disable_plane_align:
                                plane_estimate = estimate_surface_yaw_from_depth(
                                    rgb,
                                    detection,
                                    HEAD_CAMERA,
                                    transforms,
                                    sample_step=args.plane_align_sample_step,
                                    min_points=args.plane_align_min_points,
                                    depth_band=args.plane_align_depth_band,
                                )
                                if plane_estimate is not None:
                                    align_yaw, align_points = plane_estimate
                                    align_source = "depth-plane"

                            if align_source == "handle" and not args.disable_lidar_align:
                                if not lidar_checked:
                                    lidar_estimate = estimate_surface_yaw_from_lidar(
                                        controller.get_lidar_ranges(),
                                        target_bearing=gripper_target_bearing,
                                        half_angle=math.radians(args.lidar_align_half_angle_deg),
                                        max_range=args.lidar_align_max_range,
                                        min_points=args.lidar_align_min_points,
                                    )
                                if lidar_estimate is not None:
                                    align_yaw, align_points, align_range = lidar_estimate
                                    align_source = "lidar-plane"

                            wrist_error_x: float | None = None
                            wrist_error_y: float | None = None
                            wrist_gripper_error: float | None = None
                            wrist_angle_z: float | None = None
                            wrist_centered = False
                            wrist_target_slide_cmd = 0.0
                            wrist_slide_cmd = 0.0
                            if (
                                not args.disable_base_align_wrist_check
                                and distance <= args.base_align_wrist_distance
                            ):
                                wrist_rgb_base = WRIST_RGB_CAMERA.get_frame()
                                if wrist_rgb_base is not None:
                                    wrist_detection_base = detector.detect(
                                        wrist_rgb_base,
                                        WRIST_CAMERA,
                                        prefer_centroid=locked_wrist_centroid,
                                        lock_radius_fraction=args.lock_radius,
                                        relax_locked_detection=True,
                                    )
                                    wrist_mask_base = detector.last_mask
                                    if wrist_detection_base is not None:
                                        locked_wrist_centroid = smooth_centroid(
                                            locked_wrist_centroid,
                                            wrist_detection_base.centroid,
                                            args.lock_smoothing,
                                        )
                                        wrist_frame_cx = wrist_rgb_base.shape[1] / 2
                                        wrist_frame_cy = wrist_rgb_base.shape[0] / 2
                                        wrist_error_x = (locked_wrist_centroid[0] - wrist_frame_cx) / wrist_rgb_base.shape[1]
                                        wrist_error_y = (locked_wrist_centroid[1] - wrist_frame_cy) / wrist_rgb_base.shape[0]
                                        wrist_centered = (
                                            abs(wrist_error_x) <= args.base_align_wrist_center_x
                                            and abs(wrist_error_y) <= args.base_align_wrist_center_y
                                        )

                                        _, wrist_handle_base_t = locate_object(
                                            wrist_detection_base.centroid,
                                            WRIST_CAMERA,
                                            transforms,
                                            sample_radius=5,
                                        )
                                        if wrist_handle_base_t is not None:
                                            wx, wy = wrist_handle_base_t[0:2, 3]
                                            wrist_angle_z = math.atan2(wy, wx)
                                            wrist_gripper_error = wrap_angle(wrist_angle_z - gripper_target_bearing)
                                            align_yaw = wrist_angle_z
                                            align_source = "wrist-depth"
                                            align_points = 1
                                            align_range = math.sqrt(wx * wx + wy * wy)

                            align_error = wrap_angle(align_yaw - gripper_target_bearing)

                            align_tol = math.radians(args.base_align_angle_deg)
                            turn_cmd = clamp(
                                -kp_base_angle * align_error,
                                -args.base_align_turn_max,
                                args.base_align_turn_max,
                            )
                            if abs(align_error) > align_tol and abs(turn_cmd) < args.base_align_min_turn:
                                turn_cmd = math.copysign(args.base_align_min_turn, turn_cmd or -align_error)
                            if abs(align_error) <= align_tol:
                                turn_cmd = 0.0

                            head_centered = abs(error_x) <= args.drive_center_x and abs(error_y) <= args.drive_center_y
                            using_wrist_center = wrist_error_x is not None and wrist_error_y is not None
                            centered = wrist_centered if using_wrist_center else head_centered
                            distance_error = distance - args.target_distance
                            close_enough = distance_error <= args.base_align_distance_margin
                            wrist_handoff_ok = (
                                using_wrist_center
                                and wrist_centered
                                and distance <= args.base_align_wrist_distance
                                and abs(align_error) <= align_tol * 1.5
                            )
                            close_for_handoff = close_enough or wrist_handoff_ok
                            square_enough_for_creep = abs(align_error) <= align_tol * 1.5
                            forward_cmd = 0.0
                            if square_enough_for_creep and using_wrist_center and not wrist_centered:
                                wrist_target_slide_cmd = tapered_pixel_command(
                                    wrist_error_x,
                                    sign=args.wrist_slide_sign,
                                    gain=args.wrist_slide_kp,
                                    deadband=args.wrist_slide_deadband,
                                    max_command=min(args.wrist_slide_max, args.base_align_forward_max),
                                )
                                if wrist_target_slide_cmd != 0.0:
                                    wrist_slide_cmd = smooth_command(
                                        last_wrist_slide_cmd,
                                        wrist_target_slide_cmd,
                                        args.wrist_slide_command_smoothing,
                                    )
                                    last_wrist_slide_cmd = wrist_slide_cmd
                                    forward_cmd = wrist_slide_cmd
                                else:
                                    last_wrist_slide_cmd = 0.0
                            elif square_enough_for_creep and centered and not wrist_handoff_ok and distance_error > args.base_align_distance_margin:
                                forward_cmd = clamp(
                                    kp_forward * distance_error,
                                    0.0,
                                    min(args.base_forward_max, args.base_align_forward_max),
                                )
                                last_wrist_slide_cmd = 0.0
                            else:
                                last_wrist_slide_cmd = 0.0

                            auto_cmd["base_counterclockwise"] = turn_cmd
                            auto_cmd["base_forward"] = forward_cmd

                            wrist_z = transforms.get_wrist_cam_T()[2, 3]
                            if using_wrist_center and abs(wrist_error_y) > args.wrist_lift_deadband:
                                auto_cmd["lift_up"] = clamp(
                                    args.wrist_lift_sign * args.wrist_lift_pixel_kp * wrist_error_y,
                                    -args.wrist_lift_max,
                                    args.wrist_lift_max,
                                )
                            else:
                                auto_cmd["lift_up"] = clamp(kp_lift * (z - wrist_z), -0.6, 0.6)

                            aligned = abs(align_error) <= align_tol and centered and close_for_handoff
                            if aligned and stow_pose.is_at_goal():
                                stable_since = stable_since or now
                                if now - stable_since > args.base_align_stable_seconds:
                                    base_aligned_for_lock = True
                                    phase_started = now
                                    stable_since = None
                                    phase = "base_settle"
                                    if not using_wrist_center:
                                        locked_wrist_centroid = None
                                    last_wrist_slide_cmd = 0.0
                                    close_window("Head RGB", not args.no_display)
                                    print(f"\nPhase: {phase}")
                            else:
                                stable_since = None

                            range_text = f"range {align_range:.2f}m  " if align_range is not None else ""
                            dist_state = "wrist" if wrist_handoff_ok else ("ok" if close_enough else "creep")
                            center_state = "wrist" if using_wrist_center and wrist_centered else ("ok" if centered else "hold")
                            wrist_text = ""
                            if using_wrist_center:
                                wrist_bearing_text = (
                                    f" bearing {math.degrees(wrist_angle_z):+5.1f}deg"
                                    if wrist_angle_z is not None
                                    else ""
                                )
                                wrist_text = (
                                    f" wrist ({wrist_error_x:+.2f},{wrist_error_y:+.2f})"
                                    f"{wrist_bearing_text}"
                                    f" slide {wrist_target_slide_cmd:+.2f}->{wrist_slide_cmd:+.2f} "
                                )
                            print(
                                f"\rBase align dist {distance:.2f}m  bearing {math.degrees(angle_z):+5.1f}deg  "
                                f"target {args.gripper_target_bearing_deg:+5.1f}deg  "
                                f"align {math.degrees(align_error):+5.1f}deg/{align_source}:{align_points}  "
                                f"{range_text}"
                                f"fwd {forward_cmd:+.2f}  turn {turn_cmd:+.2f}  "
                                f"dist {dist_state}  center {center_state}  "
                                f"{wrist_text}"
                                f"score {detection.score:.1f}"
                                f"{'  DRY-RUN freezes base' if args.dry_run else ''}   ",
                                end="",
                                flush=True,
                            )
                    elif now - last_depth_warning > 1.0:
                        print(
                            f"\rBase align sees handle score {detection.score:.1f}, waiting for valid head depth.   ",
                            end="",
                            flush=True,
                        )
                        last_depth_warning = now

                    auto_cmd = merge_proportional(auto_cmd, stow_pose.get_command())

                if not args.no_display and rgb is not None:
                    cv2.imshow("Head RGB", draw_detection(rgb, detection, "base_align"))
                    if head_mask is not None:
                        cv2.imshow("Handle mask", head_mask)
                    if wrist_rgb_base is not None:
                        cv2.imshow("Wrist RGB", draw_detection(wrist_rgb_base, wrist_detection_base, "base_wrist"))
                        if wrist_mask_base is not None:
                            cv2.imshow("Wrist mask", wrist_mask_base)
                last_debug_save = maybe_save_debug_frame(
                    args.debug_dir, last_debug_save, rgb, detection, head_mask, "head_base_align", now
                )

            elif phase == "base_settle":
                auto_cmd["base_forward"] = 0.0
                auto_cmd["base_counterclockwise"] = 0.0
                last_wrist_slide_cmd = 0.0

                settle_elapsed = now - phase_started
                if settle_elapsed >= args.base_settle_seconds:
                    phase = "wrist_align"
                    phase_started = now
                    wrist_slide_cycle_started = now
                    stable_since = None
                    print(f"\nPhase: {phase} (base settled)")
                else:
                    print(
                        f"\rBase settling before wrist servo "
                        f"{settle_elapsed:.1f}/{args.base_settle_seconds:.1f}s   ",
                        end="",
                        flush=True,
                    )

                auto_cmd = merge_proportional(auto_cmd, pre_grasp_pose.get_command())

            elif phase == "wrist_align":
                rgb = WRIST_RGB_CAMERA.get_frame()
                detection = detector.detect(
                    rgb,
                    WRIST_CAMERA,
                    prefer_centroid=locked_wrist_centroid,
                    lock_radius_fraction=args.lock_radius,
                    relax_locked_detection=True,
                )
                if detection is None:
                    last_wrist_slide_cmd = 0.0
                    if now - phase_started > 2.0:
                        phase = "approach"
                        phase_started = now
                        last_approach_seen = now
                        stable_since = None
                        locked_wrist_centroid = None
                        last_wrist_slide_cmd = 0.0
                        print(f"\nPhase: {phase}")
                    auto_cmd = pre_grasp_pose.get_command()
                else:
                    phase_started = now
                    locked_wrist_centroid = smooth_centroid(
                        locked_wrist_centroid,
                        detection.centroid,
                        args.lock_smoothing,
                    )
                    cx, cy = detection.centroid
                    control_cx, control_cy = locked_wrist_centroid
                    frame_cx = rgb.shape[1] / 2
                    frame_cy = rgb.shape[0] / 2
                    error_x = (control_cx - frame_cx) / rgb.shape[1]
                    error_y = (control_cy - frame_cy) / rgb.shape[0]

                    if abs(error_y) > args.wrist_lift_deadband:
                        lift_cmd = args.wrist_lift_sign * args.wrist_lift_pixel_kp * error_y
                    else:
                        lift_cmd = 0.0
                    auto_cmd["lift_up"] = clamp(
                        lift_cmd,
                        -args.wrist_lift_max,
                        args.wrist_lift_max,
                    )
                    if args.enable_wrist_image_joints:
                        auto_cmd["wrist_yaw_counterclockwise"] = clamp(kp_wrist_yaw * error_x, -0.18, 0.18)
                        auto_cmd["wrist_pitch_up"] = clamp(-kp_wrist_pitch * error_y, -0.12, 0.12)
                    target_slide_cmd = tapered_pixel_command(
                        error_x,
                        sign=args.wrist_slide_sign,
                        gain=args.wrist_slide_kp,
                        deadband=args.wrist_slide_deadband,
                        max_command=args.wrist_slide_max,
                    )
                    slide_period = args.wrist_slide_pulse_seconds + args.wrist_slide_pause_seconds
                    slide_elapsed = now - wrist_slide_cycle_started
                    in_slide_pulse = (
                        target_slide_cmd != 0.0
                        and (
                            args.wrist_slide_continuous
                            or (slide_elapsed % slide_period) < args.wrist_slide_pulse_seconds
                        )
                    )
                    if in_slide_pulse:
                        slide_cmd = smooth_command(
                            last_wrist_slide_cmd,
                            target_slide_cmd,
                            args.wrist_slide_command_smoothing,
                        )
                    else:
                        slide_cmd = 0.0
                    last_wrist_slide_cmd = slide_cmd
                    auto_cmd["base_forward"] = clamp(
                        slide_cmd,
                        -args.wrist_slide_max,
                        args.wrist_slide_max,
                    )
                    slide_state = (
                        "move"
                        if args.wrist_slide_continuous and target_slide_cmd != 0.0
                        else ("pulse" if in_slide_pulse else "settle")
                    )

                    _, handle_base_t = locate_object(detection.centroid, WRIST_CAMERA, transforms, sample_radius=5)
                    if handle_base_t is not None:
                        x, y, z = handle_base_t[0:3, 3]
                        angle_z = math.atan2(y, x)
                        wrist_z = transforms.get_wrist_cam_T()[2, 3]
                        lift_error = z - wrist_z

                        gripper_bearing_error = wrap_angle(angle_z - gripper_target_bearing)
                        auto_cmd["base_counterclockwise"] = clamp(
                            -kp_base_angle * gripper_bearing_error,
                            -0.35,
                            0.35,
                        )
                        if abs(gripper_bearing_error) < math.radians(12):
                            auto_cmd["base_forward"] = clamp(
                                slide_cmd,
                                -args.wrist_slide_max,
                                args.wrist_slide_max,
                            )
                        else:
                            auto_cmd["base_forward"] = 0.0
                            last_wrist_slide_cmd = 0.0

                        aligned = (
                            abs(gripper_bearing_error) < math.radians(4)
                            and abs(error_x) < 0.06
                            and abs(error_y) < 0.06
                        )
                        if aligned and pre_grasp_pose.is_at_goal():
                            stable_since = stable_since or now
                            if now - stable_since > 0.5:
                                phase = "reach"
                                phase_started = now
                                stable_since = None
                                print(f"\nPhase: {phase}")
                        else:
                            stable_since = None

                        print(
                            f"\rWrist bearing {math.degrees(angle_z):+5.1f}deg  "
                            f"gripper_err {math.degrees(gripper_bearing_error):+5.1f}deg  "
                            f"lift_err {lift_error:+.3f}m  px_err ({error_x:+.2f},{error_y:+.2f})  "
                            f"target_slide {target_slide_cmd:+.2f}  "
                            f"slide {auto_cmd.get('base_forward', 0.0):+.2f}/"
                            f"{slide_state}  "
                            f"lift {auto_cmd.get('lift_up', 0.0):+.2f}   ",
                            end="",
                            flush=True,
                        )
                    else:
                        centered_for_depth_seek = (
                            abs(error_x) <= args.wrist_depth_seek_center_x
                            and abs(error_y) <= args.wrist_depth_seek_center_y
                        )
                        arm_out = controller.get_state().get("arm_out", 0.0)
                        if centered_for_depth_seek and arm_out < args.wrist_depth_seek_max_arm_out:
                            auto_cmd["arm_out"] = args.wrist_depth_seek_speed
                        else:
                            auto_cmd["arm_out"] = 0.0
                        if now - last_depth_warning > 0.25:
                            reason = "seeking depth" if auto_cmd["arm_out"] > 0.0 else "centering before depth"
                            if arm_out >= args.wrist_depth_seek_max_arm_out:
                                reason = "arm seek limit"
                            print(
                                f"\rWrist RGB lock score {detection.score:.1f}, no depth; {reason}.  "
                                f"px_err ({error_x:+.2f},{error_y:+.2f})  "
                                f"target_slide {target_slide_cmd:+.2f}  "
                                f"slide {auto_cmd.get('base_forward', 0.0):+.2f}/"
                                f"{slide_state}  "
                                f"lift {auto_cmd.get('lift_up', 0.0):+.2f}  "
                                f"arm {auto_cmd.get('arm_out', 0.0):+.2f}/{arm_out:.2f}m   ",
                                end="",
                                flush=True,
                            )
                            last_depth_warning = now

                    auto_cmd = merge_proportional(auto_cmd, pre_grasp_pose.get_command())

                if not args.no_display and rgb is not None:
                    cv2.imshow("Wrist RGB", draw_detection(rgb, detection, "wrist"))
                    if detector.last_mask is not None:
                        cv2.imshow("Handle mask", detector.last_mask)
                last_debug_save = maybe_save_debug_frame(
                    args.debug_dir, last_debug_save, rgb, detection, detector.last_mask, "wrist_align", now
                )

            elif phase == "reach":
                rgb = WRIST_RGB_CAMERA.get_frame()
                detection = detector.detect(
                    rgb,
                    WRIST_CAMERA,
                    prefer_centroid=locked_wrist_centroid,
                    lock_radius_fraction=args.lock_radius,
                    relax_locked_detection=True,
                )
                if detection is not None:
                    phase_started = now
                    locked_wrist_centroid = smooth_centroid(
                        locked_wrist_centroid,
                        detection.centroid,
                        args.lock_smoothing,
                    )
                    cx, cy = detection.centroid
                    control_cx, control_cy = locked_wrist_centroid
                    frame_cx = rgb.shape[1] / 2
                    frame_cy = rgb.shape[0] / 2
                    error_x = (control_cx - frame_cx) / rgb.shape[1]
                    error_y = (control_cy - frame_cy) / rgb.shape[0]

                    if args.enable_wrist_image_joints:
                        auto_cmd["wrist_yaw_counterclockwise"] = clamp(kp_wrist_yaw * error_x, -0.18, 0.18)
                        auto_cmd["wrist_pitch_up"] = clamp(-kp_wrist_pitch * error_y, -0.12, 0.12)
                    target_slide_cmd = tapered_pixel_command(
                        error_x,
                        sign=args.wrist_slide_sign,
                        gain=args.wrist_slide_kp,
                        deadband=args.wrist_slide_deadband,
                        max_command=args.wrist_slide_max,
                    )
                    slide_period = args.wrist_slide_pulse_seconds + args.wrist_slide_pause_seconds
                    slide_elapsed = now - wrist_slide_cycle_started
                    in_slide_pulse = (
                        target_slide_cmd != 0.0
                        and (
                            args.wrist_slide_continuous
                            or (slide_elapsed % slide_period) < args.wrist_slide_pulse_seconds
                        )
                    )
                    if in_slide_pulse:
                        slide_cmd = smooth_command(
                            last_wrist_slide_cmd,
                            target_slide_cmd,
                            args.wrist_slide_command_smoothing,
                        )
                    else:
                        slide_cmd = 0.0
                    last_wrist_slide_cmd = slide_cmd
                    if slide_cmd != 0.0:
                        auto_cmd["base_forward"] = slide_cmd
                    slide_state = (
                        "move"
                        if args.wrist_slide_continuous and target_slide_cmd != 0.0
                        else ("pulse" if in_slide_pulse else "settle")
                    )
                    if abs(error_y) > args.wrist_lift_deadband:
                        auto_cmd["lift_up"] = clamp(
                            args.wrist_lift_sign * args.wrist_lift_pixel_kp * error_y,
                            -args.wrist_lift_max,
                            args.wrist_lift_max,
                        )

                    distance = WRIST_CAMERA.get_depth(detection.centroid, sample_radius=5)
                    if distance is not None:
                        distance_error = distance - args.grasp_depth
                        centered = abs(error_x) <= args.reach_center_x and abs(error_y) <= args.reach_center_y
                        close_enough = distance_error <= args.grasp_depth_tolerance
                        too_close = distance <= args.reach_min_depth
                        if too_close:
                            auto_cmd["arm_out"] = -args.reach_arm_max_in
                            print(
                                f"\rReach depth {distance:.3f}m below floor {args.reach_min_depth:.3f}m; "
                                f"backing off arm {auto_cmd['arm_out']:+.2f}.   ",
                                end="",
                                flush=True,
                            )
                        elif centered and close_enough:
                            auto_cmd["arm_out"] = 0.0
                            phase = "orient_gripper"
                            phase_started = now
                            print(f"\nPhase: {phase}")
                        elif centered or distance_error < 0.0:
                            auto_cmd["arm_out"] = clamp(
                                args.reach_arm_kp * distance_error,
                                -args.reach_arm_max_in,
                                args.reach_arm_max_out,
                            )
                            print(
                                f"\rReach depth {distance:.3f}m target {args.grasp_depth:.3f}m "
                                f"err {distance_error:+.3f}m  arm {auto_cmd['arm_out']:+.2f}  "
                                f"center ok   ",
                                end="",
                                flush=True,
                            )
                        else:
                            auto_cmd["arm_out"] = 0.0
                            print(
                                f"\rReach holding arm until centered.  depth {distance:.3f}m "
                                f"px_err ({error_x:+.2f},{error_y:+.2f})   ",
                                end="",
                                flush=True,
                            )
                    elif now - last_depth_warning > 0.25:
                        centered_for_depth_seek = (
                            abs(error_x) <= args.wrist_depth_seek_center_x
                            and abs(error_y) <= args.wrist_depth_seek_center_y
                        )
                        arm_out = controller.get_state().get("arm_out", 0.0)
                        if centered_for_depth_seek and arm_out < args.wrist_depth_seek_max_arm_out:
                            auto_cmd["arm_out"] = args.wrist_depth_seek_speed
                        print(
                            f"\rReach sees handle but no depth.  px_err ({error_x:+.2f},{error_y:+.2f})  "
                            f"target_slide {target_slide_cmd:+.2f}  "
                            f"slide {auto_cmd.get('base_forward', 0.0):+.2f}/"
                            f"{slide_state}  "
                            f"lift {auto_cmd.get('lift_up', 0.0):+.2f}  "
                            f"arm {auto_cmd.get('arm_out', 0.0):+.2f}/{arm_out:.2f}m   ",
                            end="",
                            flush=True,
                        )
                        last_depth_warning = now

                    auto_cmd = merge_proportional(auto_cmd, pre_grasp_pose.get_command())
                else:
                    last_wrist_slide_cmd = 0.0
                    if now - phase_started > 1.5:
                        phase = "wrist_align"
                        phase_started = now
                        wrist_slide_cycle_started = now
                        stable_since = None
                        locked_wrist_centroid = None
                        print(f"\nPhase: {phase}")
                    auto_cmd = pre_grasp_pose.get_command()

                if not args.no_display and rgb is not None:
                    cv2.imshow("Wrist RGB", draw_detection(rgb, detection, "reach"))
                    if detector.last_mask is not None:
                        cv2.imshow("Handle mask", detector.last_mask)
                last_debug_save = maybe_save_debug_frame(
                    args.debug_dir, last_debug_save, rgb, detection, detector.last_mask, "wrist_reach", now
                )

            elif phase == "orient_gripper":
                auto_cmd = orient_gripper_pose.get_command()
                if orient_gripper_pose.is_at_goal() or now - phase_started > args.gripper_roll_timeout:
                    phase = "grasp"
                    phase_started = now
                    print(f"\nPhase: {phase}")
                else:
                    current_roll = controller.get_state().get("wrist_roll_counterclockwise", 0.0)
                    print(
                        f"\rOrienting gripper to {args.gripper_roll_deg:+.0f}deg before close "
                        f"current {math.degrees(current_roll):+.0f}deg  "
                        f"{now - phase_started:.1f}/{args.gripper_roll_timeout:.1f}s   ",
                        end="",
                        flush=True,
                    )

            elif phase == "grasp":
                auto_cmd["gripper_open"] = -1.0
                auto_cmd["arm_out"] = 0.0
                auto_cmd = merge_proportional(auto_cmd, gripper_roll_pose.get_command())
                if now - phase_started > 2.0:
                    phase = "pull"
                    phase_started = now
                    print(f"\nPhase: {phase}")

            elif phase == "pull":
                auto_cmd["gripper_open"] = -1.0
                auto_cmd["arm_out"] = -0.65
                auto_cmd["base_forward"] = -0.12
                auto_cmd = merge_proportional(auto_cmd, gripper_roll_pose.get_command())
                if now - phase_started > args.pull_seconds:
                    phase = "release"
                    phase_started = now
                    print(f"\nPhase: {phase}")

            elif phase == "release":
                auto_cmd["gripper_open"] = 1.0
                auto_cmd["arm_out"] = -0.25
                auto_cmd = merge_proportional(auto_cmd, gripper_roll_pose.get_command())
                if now - phase_started > 1.2:
                    phase = "done"
                    print(f"\nPhase: {phase}")

            elif phase == "done":
                auto_cmd = {}

            if args.dry_run:
                auto_cmd = perception_only_command(auto_cmd)

            velocities = teleop.get_normalized_velocities()
            velocities = merge_proportional(velocities, auto_cmd)
            controller.set_velocities(velocities)

            key = cv2.waitKey(1) if not args.no_display else -1
            if key in (ord("q"), 27):
                break
            time.sleep(1 / 30)

    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        controller.set_velocities({})
        controller.stop()
        if not args.no_display:
            cv2.destroyAllWindows()
        print("Done.")


if __name__ == "__main__":
    main()
