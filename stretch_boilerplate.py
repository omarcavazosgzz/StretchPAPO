"""
Stretch Toolkit boilerplate — starting point for new scripts.
"""
from stretch_toolkit import ( controller, teleop, merge_proportional, BACKEND_NAME, HEAD_CAMERA, WRIST_CAMERA, NAVIGATION_CAMERA, HEAD_RGB_CAMERA, HEAD_DEPTH_CAMERA, WRIST_RGB_CAMERA, WRIST_DEPTH_CAMERA, StateController, RobotTransforms, ObjectPlotter )
import stretch_toolkit.input as inp
import time
import cv2
import numpy as np

print(f"\n=== Running on {BACKEND_NAME} backend ===\n")


def main():
    print("Press Ctrl+C to stop\n")

    try:
        while True:
            # --- Loop setup ---
            t = controller.get_time()
            velocities = {}

            # --- Your logic here ---

            # --- Send commands ---
            controller.set_velocities(velocities)
            cv2.waitKey(1)
            time.sleep(1 / 30)  # 30 Hz

    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        controller.set_velocities({})
        controller.stop()
        cv2.destroyAllWindows()
        print("Done.")


if __name__ == "__main__":
    main()
