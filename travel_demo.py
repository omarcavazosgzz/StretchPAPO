"""
Travel demo — head camera tracks object; robot approaches directly, then presents right flank when ~50cm away.
"""
from stretch_toolkit import ( controller, teleop, merge_proportional, locate_object, BACKEND_NAME, HEAD_RGB_CAMERA, HEAD_CAMERA, StateController )
from stretch_toolkit.robot_transforms import RobotTransforms
import math
import time
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

    Kp_pan   = 1.0          # normalized pixel error → normalized velocity
    Kp_tilt  = 1.0
    Kp_angle = 5.0 / math.pi  # radian error → normalized velocity (±π → ±1)
    Kp_forward = 2.0          # distance error (m) → normalized velocity
    
    TARGET_ANGLE_FLANK = -math.pi / 2  # Present right flank when in zone
    
    in_zone = False  # Hysteresis state for parking zone
    
    stow_pose = StateController(controller, {
        "wrist_roll_counterclockwise": 0.0,
        "wrist_yaw_counterclockwise": 0.0,
        "wrist_pitch_up": 0.0,
        "gripper_open": 0.5,
        "arm_out": 0.0,
    })

    try:
        while True:
            velocities = teleop.get_normalized_velocities()

            rgb = HEAD_RGB_CAMERA.get_frame()
            centroid = find_blue_object(rgb)

            if centroid is not None and rgb is not None:
                cx, cy = centroid
                frame_cx = rgb.shape[1] / 2
                frame_cy = rgb.shape[0] / 2

                error_x = (cx - frame_cx) / rgb.shape[1]   # normalized [-0.5, 0.5]
                error_y = (cy - frame_cy) / rgb.shape[0]

                auto_velocities = {
                    "head_pan_counterclockwise": -Kp_pan * error_x,  # negate: right in image → clockwise pan
                    "head_tilt_up": -Kp_tilt * error_y,
                }

                # Rotate base to face the object using 3D localization
                _, obj2base_T = locate_object((cx, cy), HEAD_CAMERA, transforms)
                if obj2base_T is not None:
                    x, y, z = obj2base_T[0:3, 3]
                    angle_z = math.atan2(y, x)  # 0 = directly ahead
                    
                    # Calculate horizontal distance (x,y plane only)
                    horizontal_distance = math.sqrt(x**2 + y**2)
                    
                    # Hysteresis: enter zone at 45-55cm, exit at <40cm or >60cm
                    if not in_zone:
                        if 0.45 <= horizontal_distance <= 0.55:
                            in_zone = True
                    else:
                        if horizontal_distance < 0.4 or horizontal_distance > 0.6:
                            in_zone = False
                    
                    if in_zone:
                        # Present right flank when in zone
                        angle_error = TARGET_ANGLE_FLANK - angle_z
                        auto_velocities["base_counterclockwise"] = Kp_angle * angle_error
                        auto_velocities["base_forward"] = 0.0
                        print(f"\rDist: {horizontal_distance:.2f}m  Angle Err: {math.degrees(angle_error):+6.1f}°  [FLANK]   ", end="", flush=True)
                    else:
                        # Face object directly and move forward when outside zone
                        auto_velocities["base_counterclockwise"] = -Kp_angle * angle_z
                        
                        # Calculate alignment factor: 180° away = 0.0, directly facing = 1.0
                        alignment = 1.0 - (abs(angle_z) / math.pi)
                        
                        # Calculate travel authority: alignment 0.9 = 0.0, alignment 1.0 = 1.0
                        travel_auth = max(0.0, min(1.0, (alignment - 0.9) / 0.1))
                        
                        # Move forward based on distance, gated by travel_auth
                        forward_velocity = Kp_forward * horizontal_distance
                        auto_velocities["base_forward"] = forward_velocity * travel_auth
                        
                        print(f"\rDist: {horizontal_distance:.2f}m  Align: {alignment:.3f}  Auth: {travel_auth:.2f}  [moving]   ", end="", flush=True)

                velocities = merge_proportional(velocities, auto_velocities)
                velocities = merge_proportional(velocities, stow_pose.get_command())

                cv2.circle(rgb, (cx, cy), 10, (0, 255, 0), -1)
                cv2.circle(rgb, (int(frame_cx), int(frame_cy)), 5, (0, 0, 255), 2)

            if rgb is not None:
                cv2.imshow("Head RGB", rgb)

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
