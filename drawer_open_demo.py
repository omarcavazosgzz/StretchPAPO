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

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            if w <= 0 or h <= 0:
                continue

            aspect = max(w, h) / max(1.0, min(w, h))
            if aspect < self.min_aspect:
                continue

            if self.orientation == "horizontal" and w < h:
                continue
            if self.orientation == "vertical" and h < w:
                continue

            if w > width * 0.85 or h > height * 0.85:
                continue

            if self.orientation == "horizontal":
                edge_margin = int(width * self.edge_margin_fraction)
                locked_tracking = prefer_centroid is not None and relax_locked_detection
                if not locked_tracking and (x <= edge_margin or x + w >= width - edge_margin):
                    continue
                if w < width * 0.035 or w > width * 0.45:
                    continue
                if h > height * 0.18:
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
                if bright_ratio < self.bright_ratio_min:
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
            if score < self.min_score:
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


def tracking_depth_is_consistent(
    distance: float,
    previous_distance: float | None,
    max_jump: float,
) -> bool:
    return previous_distance is None or abs(distance - previous_distance) <= max_jump


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
    parser.add_argument("--grasp-depth", type=float, default=0.12, help="Wrist-camera depth target before closing gripper.")
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
    parser.add_argument("--base-align-distance", type=float, default=0.95, help="Stop creeping and square the base when the head estimate is this close.")
    parser.add_argument("--base-align-angle-deg", type=float, default=4.0, help="Base angle tolerance before wrist handoff.")
    parser.add_argument("--base-align-turn-max", type=float, default=0.16, help="Max normalized base turn during close square-up.")
    parser.add_argument("--base-align-min-turn", type=float, default=0.035, help="Minimum normalized turn when square-up angle is still nonzero.")
    parser.add_argument("--base-align-stable-seconds", type=float, default=0.55, help="How long the base angle must stay aligned.")
    parser.add_argument("--lock-depth-jump", type=float, default=0.35, help="Reject locked detections whose depth jumps by more than this many meters.")
    parser.add_argument("--disable-plane-align", action="store_true", help="Use handle bearing instead of depth plane normal during base_align.")
    parser.add_argument("--plane-align-min-points", type=int, default=24, help="Minimum cabinet depth points for plane-based base alignment.")
    parser.add_argument("--plane-align-depth-band", type=float, default=0.38, help="Depth band around the handle for cabinet plane samples.")
    parser.add_argument("--plane-align-sample-step", type=int, default=6, help="Pixel stride for cabinet plane sampling.")
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
    args.base_align_turn_max = clamp(args.base_align_turn_max, 0.02, args.base_turn_max)
    args.base_align_min_turn = clamp(args.base_align_min_turn, 0.0, args.base_align_turn_max)
    args.base_align_stable_seconds = max(0.05, args.base_align_stable_seconds)
    args.lock_depth_jump = max(0.05, args.lock_depth_jump)
    args.plane_align_min_points = max(6, args.plane_align_min_points)
    args.plane_align_depth_band = max(0.05, args.plane_align_depth_band)
    args.plane_align_sample_step = max(2, args.plane_align_sample_step)
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

    kp_head = args.head_kp
    kp_base_angle = 5.0 / math.pi
    kp_forward = 2.0
    kp_lift = 4.5
    kp_wrist_yaw = 0.7
    kp_wrist_pitch = 0.45
    kp_arm = 8.0

    phase = "search"
    search_dir = 1.0
    last_seen = time.perf_counter()
    phase_started = time.perf_counter()
    stable_since: float | None = None
    last_camera_warning = 0.0
    last_debug_save = 0.0
    last_depth_warning = 0.0
    locked_head_centroid: tuple[float, float] | None = None
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
                        and abs(last_handle_angle) <= math.radians(args.wrist_handoff_angle_deg)
                    )
                    if ready_for_wrist_handoff and lost_elapsed > 0.35:
                        phase = "wrist_align"
                        phase_started = now
                        stable_since = None
                        close_window("Head RGB", not args.no_display)
                        print(f"\nPhase: {phase} (head lost close handle; trying wrist)")
                    elif lost_elapsed > args.approach_lost_seconds:
                        phase = "search"
                        locked_head_centroid = None
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
                                    phase = "wrist_align"
                                    phase_started = now
                                    stable_since = None
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
                    lock_radius_fraction=max(0.06, args.lock_radius * 0.85),
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

                    if lost_elapsed > args.approach_lost_seconds:
                        phase = "search"
                        locked_head_centroid = None
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

                            plane_estimate = None
                            if not args.disable_plane_align:
                                plane_estimate = estimate_surface_yaw_from_depth(
                                    rgb,
                                    detection,
                                    HEAD_CAMERA,
                                    transforms,
                                    sample_step=args.plane_align_sample_step,
                                    min_points=args.plane_align_min_points,
                                    depth_band=args.plane_align_depth_band,
                                )
                            if plane_estimate is None:
                                align_error = angle_z
                                align_source = "bearing"
                                plane_points = 0
                            else:
                                align_error, plane_points = plane_estimate
                                align_source = "plane"

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

                            auto_cmd["base_counterclockwise"] = turn_cmd
                            auto_cmd["base_forward"] = 0.0

                            wrist_z = transforms.get_wrist_cam_T()[2, 3]
                            auto_cmd["lift_up"] = clamp(kp_lift * (z - wrist_z), -0.6, 0.6)

                            centered = abs(error_x) <= args.drive_center_x and abs(error_y) <= args.drive_center_y
                            aligned = abs(align_error) <= align_tol and centered
                            if aligned and stow_pose.is_at_goal():
                                stable_since = stable_since or now
                                if now - stable_since > args.base_align_stable_seconds:
                                    base_aligned_for_lock = True
                                    phase_started = now
                                    stable_since = None
                                    if distance <= args.target_distance + 0.08:
                                        phase = "wrist_align"
                                        close_window("Head RGB", not args.no_display)
                                    else:
                                        phase = "approach"
                                    print(f"\nPhase: {phase}")
                            else:
                                stable_since = None

                            print(
                                f"\rBase align dist {distance:.2f}m  bearing {math.degrees(angle_z):+5.1f}deg  "
                                f"align {math.degrees(align_error):+5.1f}deg/{align_source}:{plane_points}  "
                                f"turn {turn_cmd:+.2f}  center {'ok' if centered else 'hold'}  "
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
                    if detector.last_mask is not None:
                        cv2.imshow("Handle mask", detector.last_mask)
                last_debug_save = maybe_save_debug_frame(
                    args.debug_dir, last_debug_save, rgb, detection, detector.last_mask, "head_base_align", now
                )

            elif phase == "wrist_align":
                rgb = WRIST_RGB_CAMERA.get_frame()
                detection = detector.detect(rgb, WRIST_CAMERA)
                if detection is None:
                    if now - phase_started > 2.0:
                        phase = "approach"
                        phase_started = now
                        last_approach_seen = now
                        stable_since = None
                        print(f"\nPhase: {phase}")
                    auto_cmd = stow_pose.get_command()
                else:
                    phase_started = now
                    cx, cy = detection.centroid
                    frame_cx = rgb.shape[1] / 2
                    frame_cy = rgb.shape[0] / 2
                    error_x = (cx - frame_cx) / rgb.shape[1]
                    error_y = (cy - frame_cy) / rgb.shape[0]

                    _, handle_base_t = locate_object(detection.centroid, WRIST_CAMERA, transforms, sample_radius=5)
                    if handle_base_t is not None:
                        x, y, z = handle_base_t[0:3, 3]
                        angle_z = math.atan2(y, x)
                        wrist_z = transforms.get_wrist_cam_T()[2, 3]
                        lift_error = z - wrist_z

                        auto_cmd["base_counterclockwise"] = clamp(-kp_base_angle * angle_z, -0.35, 0.35)
                        auto_cmd["lift_up"] = clamp(kp_lift * lift_error, -0.5, 0.5)
                        auto_cmd["wrist_yaw_counterclockwise"] = clamp(kp_wrist_yaw * error_x, -0.35, 0.35)
                        auto_cmd["wrist_pitch_up"] = clamp(-kp_wrist_pitch * error_y, -0.25, 0.25)

                        aligned = (
                            abs(angle_z) < math.radians(4)
                            and abs(lift_error) < 0.025
                            and abs(error_x) < 0.06
                            and abs(error_y) < 0.08
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
                            f"\rWrist angle {math.degrees(angle_z):+5.1f}deg  "
                            f"lift_err {lift_error:+.3f}m  px_err ({error_x:+.2f},{error_y:+.2f})   ",
                            end="",
                            flush=True,
                        )
                    elif now - last_depth_warning > 1.0:
                        print(
                            f"\rDetected wrist handle score {detection.score:.1f}, waiting for valid wrist depth.   ",
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
                detection = detector.detect(rgb, WRIST_CAMERA)
                if detection is not None:
                    phase_started = now
                    cx, cy = detection.centroid
                    frame_cx = rgb.shape[1] / 2
                    frame_cy = rgb.shape[0] / 2
                    error_x = (cx - frame_cx) / rgb.shape[1]
                    error_y = (cy - frame_cy) / rgb.shape[0]

                    auto_cmd["wrist_yaw_counterclockwise"] = clamp(kp_wrist_yaw * error_x, -0.35, 0.35)
                    auto_cmd["wrist_pitch_up"] = clamp(-kp_wrist_pitch * error_y, -0.25, 0.25)

                    distance = WRIST_CAMERA.get_depth(detection.centroid, sample_radius=5)
                    if distance is not None:
                        distance_error = distance - args.grasp_depth
                        auto_cmd["arm_out"] = clamp(kp_arm * distance_error, -0.5, 0.6)
                        print(f"\rReach depth {distance:.3f}m  err {distance_error:+.3f}m   ", end="", flush=True)

                        centered = abs(error_x) < 0.07 and abs(error_y) < 0.09
                        if abs(distance_error) < 0.018 and centered:
                            phase = "grasp"
                            phase_started = now
                            print(f"\nPhase: {phase}")

                    auto_cmd = merge_proportional(auto_cmd, pre_grasp_pose.get_command())
                else:
                    if now - phase_started > 1.5:
                        phase = "wrist_align"
                        phase_started = now
                        stable_since = None
                        print(f"\nPhase: {phase}")
                    auto_cmd = pre_grasp_pose.get_command()

                if not args.no_display and rgb is not None:
                    cv2.imshow("Wrist RGB", draw_detection(rgb, detection, "reach"))
                    if detector.last_mask is not None:
                        cv2.imshow("Handle mask", detector.last_mask)
                last_debug_save = maybe_save_debug_frame(
                    args.debug_dir, last_debug_save, rgb, detection, detector.last_mask, "wrist_reach", now
                )

            elif phase == "grasp":
                auto_cmd["gripper_open"] = -1.0
                auto_cmd["arm_out"] = 0.05
                if now - phase_started > 2.0:
                    phase = "pull"
                    phase_started = now
                    print(f"\nPhase: {phase}")

            elif phase == "pull":
                auto_cmd["gripper_open"] = -1.0
                auto_cmd["arm_out"] = -0.65
                auto_cmd["base_forward"] = -0.12
                if now - phase_started > args.pull_seconds:
                    phase = "release"
                    phase_started = now
                    print(f"\nPhase: {phase}")

            elif phase == "release":
                auto_cmd["gripper_open"] = 1.0
                auto_cmd["arm_out"] = -0.25
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
