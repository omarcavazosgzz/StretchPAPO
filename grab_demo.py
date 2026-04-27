"""
Grab demo — approach with head camera, align and reach with wrist camera, then grab.
"""
from stretch_toolkit import ( controller, teleop, merge_proportional, locate_object, BACKEND_NAME, HEAD_CAMERA, HEAD_RGB_CAMERA, WRIST_CAMERA, WRIST_RGB_CAMERA, StateController )
from stretch_toolkit.robot_transforms import RobotTransforms
import time
import math
import cv2
import numpy as np

print(f"\n=== Running on {BACKEND_NAME} backend ===\n")


def find_blue_object(rgb_frame):
    if rgb_frame is None:
        return None
    hsv = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([110, 100, 100]), np.array([130, 255, 255]))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M['m00'] > 0:
            return (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
    return None


def main():
    print("Press Ctrl+C to stop\n")

    transforms = RobotTransforms(controller)

    stow_pose = StateController(controller, {
        "wrist_roll_counterclockwise": 0.0,
        "wrist_yaw_counterclockwise": 0.0,
        "wrist_pitch_up": 0.0,
        "gripper_open": 0.3,
        "arm_out": 0.0,
    })

    pre_grip_pose = StateController(controller, {
        "wrist_roll_counterclockwise": 0.0,
        "gripper_open": 0.4,
    })

    # --- Tuning constants ---
    Kp_pan    = 1.0
    Kp_tilt   = 1.0
    Kp_angle  = 5.0 / math.pi
    Kp_forward = 2.0
    Kp_lift   = 5.0
    Kp_yaw    = 0.5
    Kp_pitch  = 0.5
    Kp_arm    = 10.0

    # --- Phase state ---
    phase = "approach"
    in_zone = False  # Approach phase hysteresis

    print(f"Phase: {phase}")

    try:
        while True:
            velocities = teleop.get_normalized_velocities()
            auto_velocities = {}

            # ----------------------------------------------------------------
            if phase == "approach":
                rgb = HEAD_RGB_CAMERA.get_frame()
                centroid = find_blue_object(rgb)

                if centroid is not None and rgb is not None:
                    cx, cy = centroid
                    frame_cx = rgb.shape[1] / 2
                    frame_cy = rgb.shape[0] / 2

                    error_x = (cx - frame_cx) / rgb.shape[1]
                    error_y = (cy - frame_cy) / rgb.shape[0]
                    auto_velocities["head_pan_counterclockwise"] = -Kp_pan * error_x
                    auto_velocities["head_tilt_up"] = -Kp_tilt * error_y

                    _, obj2base_T = locate_object((cx, cy), HEAD_CAMERA, transforms)
                    if obj2base_T is not None:
                        x, y, z = obj2base_T[0:3, 3]
                        angle_z = math.atan2(y, x)
                        horizontal_distance = math.sqrt(x**2 + y**2)

                        # Start raising lift toward object height early (10cm above align target)
                        cam_z = transforms.get_wrist_cam_T()[2, 3]
                        auto_velocities["lift_up"] = Kp_lift * (z - (cam_z + 0.01) + 0.10)

                        if not in_zone:
                            if 0.45 <= horizontal_distance <= 0.55:
                                in_zone = True
                        else:
                            if horizontal_distance < 0.4 or horizontal_distance > 0.6:
                                in_zone = False

                        if in_zone:
                            angle_error = -math.pi / 2 - angle_z
                            auto_velocities["base_counterclockwise"] = Kp_angle * angle_error
                            auto_velocities["base_forward"] = 0.0
                            print(f"\rDist: {horizontal_distance:.2f}m  Angle Err: {math.degrees(angle_error):+6.1f}°  [FLANK]   ", end="", flush=True)
                            if abs(angle_error) < math.radians(5):
                                cv2.destroyWindow("Head RGB")
                                phase = "align"
                                print(f"\nPhase: {phase}")
                        else:
                            auto_velocities["base_counterclockwise"] = -Kp_angle * angle_z
                            alignment = 1.0 - (abs(angle_z) / math.pi)
                            travel_auth = max(0.0, min(1.0, (alignment - 0.9) / 0.1))
                            auto_velocities["base_forward"] = Kp_forward * horizontal_distance * travel_auth
                            print(f"\rDist: {horizontal_distance:.2f}m  Align: {alignment:.3f}  Auth: {travel_auth:.2f}  [moving]   ", end="", flush=True)

                    cv2.circle(rgb, (cx, cy), 10, (0, 255, 0), -1)
                    cv2.circle(rgb, (int(frame_cx), int(frame_cy)), 5, (0, 0, 255), 2)

                if rgb is not None:
                    cv2.imshow("Head RGB", rgb)

                velocities = merge_proportional(velocities, auto_velocities)
                velocities = merge_proportional(velocities, stow_pose.get_command())

            # ----------------------------------------------------------------
            elif phase == "align":
                rgb = WRIST_RGB_CAMERA.get_frame()
                centroid = find_blue_object(rgb)

                if centroid is not None:
                    cx, cy = centroid

                    _, obj2base_T = locate_object((cx, cy), WRIST_CAMERA, transforms)
                    if obj2base_T is not None:
                        x, y, z = obj2base_T[0:3, 3]
                        angle_z = math.atan2(y, x)
                        cam_z = transforms.get_wrist_cam_T()[2, 3]

                        auto_velocities["base_counterclockwise"] = Kp_angle * (-math.pi / 2 - angle_z + math.radians(3))
                        auto_velocities["lift_up"] = Kp_lift * (z - (cam_z + 0.01))

                        angle_err = abs(-math.pi / 2 - angle_z)
                        lift_err  = abs(z - (cam_z + 0.01))
                        if stow_pose.is_at_goal() and angle_err < math.radians(5) and lift_err < 0.03:
                            phase = "reach"
                            print(f"\nPhase: {phase}")

                    if rgb is not None:
                        cv2.circle(rgb, (cx, cy), 10, (0, 255, 0), -1)

                if rgb is not None:
                    cv2.imshow("Wrist RGB", rgb)

                velocities = merge_proportional(velocities, auto_velocities)
                velocities = merge_proportional(velocities, stow_pose.get_command())

            # ----------------------------------------------------------------
            elif phase == "reach":
                rgb = WRIST_RGB_CAMERA.get_frame()
                centroid = find_blue_object(rgb)

                if centroid is not None:
                    cx, cy = centroid

                    if rgb is not None:
                        frame_cx = rgb.shape[1] / 2
                        frame_cy = rgb.shape[0] / 2
                        error_x = (cx - frame_cx) / rgb.shape[1]
                        error_y = (cy - frame_cy) / rgb.shape[0]
                        auto_velocities["wrist_yaw_counterclockwise"] = Kp_yaw * error_x
                        auto_velocities["wrist_pitch_up"] = -Kp_pitch * error_y
                        cv2.circle(rgb, (cx, cy), 10, (0, 255, 0), -1)

                    distance = WRIST_CAMERA.get_depth((cx, cy))
                    if distance is not None:
                        distance_error = distance - 0.12
                        auto_velocities["arm_out"] = Kp_arm * distance_error
                        print(f"\rDist: {distance:.3f}m  Err: {distance_error:+.3f}m   ", end="", flush=True)
                        if abs(distance_error) < 0.02 and pre_grip_pose.is_at_goal():
                            cv2.destroyAllWindows()
                            phase = "grab"
                            print(f"\nPhase: {phase}")

                if rgb is not None:
                    cv2.imshow("Wrist RGB", rgb)

                velocities = merge_proportional(velocities, auto_velocities)
                velocities = merge_proportional(velocities, pre_grip_pose.get_command())

            # ----------------------------------------------------------------
            elif phase == "grab":
                auto_velocities["lift_up"] = 0.2
                auto_velocities["gripper_open"] = -1.0
                velocities = merge_proportional(velocities, auto_velocities)

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
