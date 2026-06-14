"""Real-time 2D top-down LiDAR visualizer using OpenCV."""

import cv2
import numpy as np


class LidarPlotter:
    """Displays a live top-down RViz-style LiDAR scan in an OpenCV window.

    Usage::

        from stretch_toolkit import controller, LidarPlotter

        plotter = LidarPlotter()

        while True:
            ranges = controller.get_lidar_ranges()
            plotter.update(ranges)
            if plotter.should_quit():
                break
    """

    # ── Default display settings ──────────────────────────────────────
    IMG_SIZE = 800
    PIXELS_PER_METER = 55
    MAX_RANGE = 10.0        # metres — match the sensor cutoff in stretch.xml
    ANGLE_OFFSET = 0.0      # rotate the scan if it looks misaligned
    DRAW_HIT_RAYS = False   # draw faint lines from origin to each hit point

    def __init__(self, window_name: str = "LiDAR 2D View"):
        self.window_name = window_name
        self._center = (self.IMG_SIZE // 2, self.IMG_SIZE // 2)

    def update(self, ranges) -> None:
        """Render and display the latest LiDAR scan.

        Args:
            ranges: np.ndarray of distances in metres (np.inf = no hit),
                    as returned by ``controller.get_lidar_ranges()``.
                    Silently skips if None or empty.
        """
        if ranges is None or len(ranges) == 0:
            return
        frame = self._render(np.asarray(ranges, dtype=float))
        cv2.imshow(self.window_name, frame)

    def should_quit(self, wait_ms: int = 30) -> bool:
        """Pump the OpenCV event loop and return True if the user pressed Q or Escape.

        Call this once per loop iteration instead of ``cv2.waitKey`` directly.

        Args:
            wait_ms: Milliseconds to wait (default 30 ≈ 33 Hz).
        """
        key = cv2.waitKey(wait_ms) & 0xFF
        return key in (ord('q'), ord('Q'), 27)

    def close(self) -> None:
        """Destroy the OpenCV window."""
        cv2.destroyWindow(self.window_name)

    # ── Internal rendering ────────────────────────────────────────────

    def _render(self, ranges: np.ndarray) -> np.ndarray:
        """Build and return a BGR image of the LiDAR scan."""
        size = self.IMG_SIZE
        cx, cy = self._center
        ppm = self.PIXELS_PER_METER
        max_r = self.MAX_RANGE
        n_rays = len(ranges)

        # ── Canvas ───────────────────────────────────────────────────
        canvas = np.full((size, size, 3), (18, 18, 18), dtype=np.uint8)

        grid_color = (45, 45, 45)
        axis_color = (120, 120, 120)

        # Metric grid lines
        meters_visible = int(max_r)
        for m in range(-meters_visible, meters_visible + 1):
            offset = int(m * ppm)
            x = cx + offset
            if 0 <= x < size:
                cv2.line(canvas, (x, 0), (x, size), grid_color, 1)
            y = cy + offset
            if 0 <= y < size:
                cv2.line(canvas, (0, y), (size, y), grid_color, 1)

        # Main axes
        cv2.line(canvas, (cx, 0), (cx, size), axis_color, 1)
        cv2.line(canvas, (0, cy), (size, cy), axis_color, 1)

        # Range circles
        for m in range(1, meters_visible + 1):
            r = int(m * ppm)
            cv2.circle(canvas, (cx, cy), r, (35, 35, 35), 1)
            cv2.putText(
                canvas, f"{m}m",
                (cx + r + 4, cy - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (110, 110, 110), 1,
            )

        # ── Robot origin ──────────────────────────────────────────────
        cv2.circle(canvas, (cx, cy), 7, (230, 230, 230), -1)

        front_end = (cx, cy - int(ppm))
        cv2.arrowedLine(canvas, (cx, cy), front_end, (255, 255, 255), 2, tipLength=0.25)
        cv2.putText(
            canvas, "+X / front",
            (front_end[0] + 8, front_end[1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1,
        )

        # ── LiDAR hit points ─────────────────────────────────────────
        # Convention: +X/front → up on screen, +Y/left → left on screen.
        # Ray i spans angle (2π * i / n_rays) + ANGLE_OFFSET, CCW.
        angles = (2.0 * np.pi * np.arange(n_rays) / n_rays) + self.ANGLE_OFFSET

        hit_mask = np.isfinite(ranges)
        hit_angles = angles[hit_mask]
        hit_ranges = ranges[hit_mask]

        # Polar → pixel
        px = (cx - hit_ranges * np.sin(hit_angles) * ppm).astype(int)
        py = (cy - hit_ranges * np.cos(hit_angles) * ppm).astype(int)

        # Colour by distance: close = red, far = cyan
        norm = np.clip(hit_ranges / max_r, 0.0, 1.0)
        r_ch = (255 * (1.0 - norm)).astype(np.uint8)
        g_ch = (180 * norm).astype(np.uint8)
        b_ch = (255 * norm).astype(np.uint8)

        for i in range(len(px)):
            x, y = int(px[i]), int(py[i])
            if 0 <= x < size and 0 <= y < size:
                color = (int(b_ch[i]), int(g_ch[i]), int(r_ch[i]))
                if self.DRAW_HIT_RAYS:
                    cv2.line(canvas, (cx, cy), (x, y), (40, 40, 40), 1)
                cv2.circle(canvas, (x, y), 2, color, -1)

        # ── Stats overlay ─────────────────────────────────────────────
        if len(hit_ranges):
            stats = (
                f"rays={n_rays}  valid={len(hit_ranges)}  "
                f"min={hit_ranges.min():.2f}m  "
                f"mean={hit_ranges.mean():.2f}m  "
                f"max={hit_ranges.max():.2f}m"
            )
        else:
            stats = f"rays={n_rays}  no hits"

        cv2.putText(
            canvas, stats,
            (8, size - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1,
        )

        return canvas
