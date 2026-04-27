"""
Practice 4 Part 3 - Wrist camera visual servoing with 3D object localization.
"""
from stretch_toolkit import ( controller, teleop, merge_proportional, locate_object, BACKEND_NAME, WRIST_CAMERA, WRIST_RGB_CAMERA, StateController )
from stretch_toolkit.robot_transforms import RobotTransforms
import time
import math
import cv2
import numpy as np

print(f"\n=== Running on {BACKEND_NAME} backend ===\n")


def main():
    print("Press Ctrl+C to stop\n")

    transforms = RobotTransforms(controller)

    TARGET_ANGLE = -math.pi / 2  # radians: object directly to the robot's right in base frame
    Kp_angle = 5.0 / math.pi     # maps radian error → normalized velocity (±π → ±1)
    Kp_lift  = 5.0          # maps meter error → normalized velocity

    stow_pose = StateController(controller, {
        "wrist_roll_counterclockwise": 0.0,
        "wrist_yaw_counterclockwise": 0.0,
        "wrist_pitch_up": 0.0,
        "gripper_open": 1.0,
        "arm_out": 0.0,
    })

    try:
        while True:
            velocities = teleop.get_normalized_velocities()

            rgb = WRIST_RGB_CAMERA.get_frame()
            if rgb is not None:
                hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, np.array([110, 100, 100]), np.array([130, 255, 255]))

                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest = max(contours, key=cv2.contourArea)
                    M = cv2.moments(largest)
                    if M['m00'] > 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])

                        error_x = (cx / rgb.shape[1]) - 0.5

                        Kp = 0.5
                        auto_velocities = {
                            # "wrist_yaw_counterclockwise": Kp * error_x,
                        }

                        cv2.circle(rgb, (cx, cy), 10, (0, 255, 0), -1)

                        _, obj2base_T = locate_object((cx, cy), WRIST_CAMERA, transforms)
                        if obj2base_T is not None:
                            x, y, z = obj2base_T[0:3, 3]
                            angle_z = math.atan2(y, x)
                            print(f"\rHeight: {z:+.3f} m   Angle (Z): {math.degrees(angle_z):+.1f} deg   ", end="", flush=True)

                            cam_T = transforms.get_wrist_cam_T()
                            cam_z = cam_T[2, 3]

                            auto_velocities["base_counterclockwise"] = Kp_angle * (TARGET_ANGLE - angle_z + math.radians(3))
                            auto_velocities["lift_up"] = Kp_lift * (z - (cam_z - 0.00))  # target above object

                        velocities = merge_proportional(velocities, auto_velocities)

                cv2.imshow("Wrist RGB", rgb)

            # StateController holds wrist_pitch, wrist_roll, gripper, arm at desired positions
            velocities = merge_proportional(velocities, stow_pose.get_command())

            controller.set_velocities(velocities)
            cv2.waitKey(1)
            time.sleep(1 / 30)

    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        controller.set_velocities({})
        controller.stop()
        cv2.destroyAllWindows()
        print("Done.")


if __name__ == "__main__":
    main()
