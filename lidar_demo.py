"""
LiDAR demo — reads 360-ray range data from the simulator and
displays a live top-down 2D plot.

Run with:
    uv run lidar_demo.py
"""

from stretch_toolkit import controller, BACKEND_NAME, LidarPlotter, TeleopProvider
import time


def main():
    print(f"\n=== Running on {BACKEND_NAME} backend ===\n")
    print("Displaying LiDAR scan. Press Q or Escape to quit.\n")

    plotter = LidarPlotter()
    tp = TeleopProvider(is_stretch_env=False)
    while True:
        ranges = controller.get_lidar_ranges()
        velocities = tp.get_normalized_velocities()
        controller.set_velocities(velocities)
        plotter.update(ranges)

        if plotter.should_quit():
            break

    plotter.close()


if __name__ == '__main__':
    main()
