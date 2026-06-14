"""
Object plotter demo — detects blue objects via HSV masking and plots their
3D position in the robot base frame using RobotTransforms + ObjectPlotter.
"""
from stretch_toolkit import ( controller, teleop, merge_proportional, locate_object, BACKEND_NAME, HEAD_CAMERA, WRIST_CAMERA, NAVIGATION_CAMERA, HEAD_RGB_CAMERA, HEAD_DEPTH_CAMERA, WRIST_RGB_CAMERA, WRIST_DEPTH_CAMERA, StateController, RobotTransforms, ObjectPlotter )
import time
import cv2
import numpy as np

print(f"\n=== Running on {BACKEND_NAME} backend ===\n")

# HSV range for blue object detection
LOWER_BLUE = np.array([110, 100, 100])
UPPER_BLUE = np.array([130, 255, 255])

# Camera selection: "head" or "wrist"
ACTIVE_CAMERA = "wrist"

if ACTIVE_CAMERA == "head":
    RGB_CAMERA = HEAD_RGB_CAMERA
    DEPTH_CAMERA_INFO = HEAD_CAMERA
elif ACTIVE_CAMERA == "wrist":
    RGB_CAMERA = WRIST_RGB_CAMERA
    DEPTH_CAMERA_INFO = WRIST_CAMERA
else:
    raise ValueError(f"Unknown camera: {ACTIVE_CAMERA!r}")


def find_blue_centroid(rgb_frame):
    """Returns (cx, cy) of the largest blue blob, or None if not found."""
    hsv = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask
    largest = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None, mask
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy), mask


def main():
    print("Press Ctrl+C to stop\n")

    transforms = RobotTransforms(controller)
    plotter = ObjectPlotter()

    try:
        while True:
            # --- Loop setup ---
            t = controller.get_time()
            velocities = teleop.get_normalized_velocities()

            # --- Get camera data ---
            rgb_frame = RGB_CAMERA.get_frame()
            if rgb_frame is None or rgb_frame.size == 0:
                cv2.waitKey(1)
                time.sleep(1 / 30)
                continue
            centroid, mask = find_blue_centroid(rgb_frame)

            # --- Compute transforms ---
            cam_T = transforms.get_cam_T(DEPTH_CAMERA_INFO)
            _, obj_T = locate_object(centroid, DEPTH_CAMERA_INFO, transforms) if centroid is not None else (None, None)

            if obj_T is not None:
                depth = DEPTH_CAMERA_INFO.get_depth(centroid)
                cv2.circle(rgb_frame, centroid, 8, (0, 255, 0), -1)
                cv2.putText(rgb_frame, f"{depth:.2f}m", (centroid[0] + 10, centroid[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # --- Update plotter ---
            plotter.update(cam_T, obj_T)

            # --- Display windows ---
            cv2.imshow("RGB Camera", rgb_frame)
            cv2.imshow("Blue Mask", mask)

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
        plotter.close()
        print("Done.")


if __name__ == "__main__":
    main()
