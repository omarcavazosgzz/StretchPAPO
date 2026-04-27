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


def find_blue_object(rgb_frame):
    """Detect blue object in RGB frame and return centroid.
    
    Args:
        rgb_frame: BGR image from camera
        
    Returns:
        tuple: (cx, cy) centroid pixel coordinates, or None if not found
    """
    if rgb_frame is None:
        return None
    
    hsv = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([110, 100, 100]), np.array([130, 255, 255]))
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            return (cx, cy)
    
    return None


def align_phase(transforms, stow_pose):
    """Align base rotation and lift to the object, then return when ready to reach."""
    TARGET_ANGLE     = -math.pi / 2
    Kp_angle         = 5.0 / math.pi
    Kp_lift          = 5.0
    ANGLE_THRESHOLD  = math.radians(5)   # ±5 degrees
    LIFT_THRESHOLD   = 0.03              # ±3 cm

    print("Phase 1: aligning...")
    while True:
        velocities = teleop.get_normalized_velocities()

        rgb = WRIST_RGB_CAMERA.get_frame()
        centroid = find_blue_object(rgb)
        
        if centroid is not None:
            cx, cy = centroid
            auto_velocities = {}
            
            _, obj2base_T = locate_object((cx, cy), WRIST_CAMERA, transforms)
            if obj2base_T is not None:
                x, y, z = obj2base_T[0:3, 3]
                angle_z = math.atan2(y, x)

                cam_T = transforms.get_wrist_cam_T()
                cam_z = cam_T[2, 3]

                auto_velocities["base_counterclockwise"] = Kp_angle * (TARGET_ANGLE - angle_z + math.radians(3))
                auto_velocities["lift_up"] = Kp_lift * (z - (cam_z + 0.01))

                angle_err = abs(TARGET_ANGLE - angle_z)
                lift_err  = abs(z - (cam_z + 0.01))
                ready = (
                    stow_pose.is_at_goal() and
                    angle_err < ANGLE_THRESHOLD and
                    lift_err  < LIFT_THRESHOLD
                )
                if ready:
                    print("Phase 1 complete.")
                    return

            velocities = merge_proportional(velocities, auto_velocities)
            
            if rgb is not None:
                cv2.circle(rgb, (cx, cy), 10, (0, 255, 0), -1)
        
        if rgb is not None:
            cv2.imshow("Wrist RGB", rgb)

        velocities = merge_proportional(velocities, stow_pose.get_command())
        controller.set_velocities(velocities)
        cv2.waitKey(1)
        time.sleep(1 / 30)


def reach_phase(transforms, pre_grip_pose):
    """Extend arm to reach the object while tracking with wrist pitch/yaw."""
    Kp_yaw   = 0.5   # normalized pixel error → normalized velocity
    Kp_pitch = 0.5
    Kp_arm   = 5.0   # distance error (m) → normalized velocity
    TARGET_DISTANCE = 0.125  # meters from wrist camera
    DISTANCE_THRESHOLD = 0.02  # 2cm tolerance
    
    print("Phase 2: reaching and tracking...")
    while True:
        velocities = teleop.get_normalized_velocities()
        
        rgb = WRIST_RGB_CAMERA.get_frame()
        centroid = find_blue_object(rgb)
        
        if centroid is not None:
            cx, cy = centroid
            auto_velocities = {}
            
            # Center the object in the frame using wrist servoing
            if rgb is not None:
                frame_center_x = rgb.shape[1] / 2
                frame_center_y = rgb.shape[0] / 2
                
                error_x = (cx - frame_center_x) / rgb.shape[1]   # normalized [-0.5, 0.5]
                error_y = (cy - frame_center_y) / rgb.shape[0]
                
                auto_velocities["wrist_yaw_counterclockwise"] = Kp_yaw * error_x
                auto_velocities["wrist_pitch_up"] = -Kp_pitch * error_y  # negative because image y is inverted
                
                cv2.circle(rgb, (cx, cy), 10, (0, 255, 0), -1)
                cv2.circle(rgb, (int(frame_center_x), int(frame_center_y)), 5, (0, 0, 255), 2)
            
            # Extend arm based on object distance
            distance = WRIST_CAMERA.get_depth((cx, cy))
            if distance is not None:
                distance_error = distance - TARGET_DISTANCE
                auto_velocities["arm_out"] = Kp_arm * distance_error
                print(f"\rDistance: {distance:.3f} m  Target: {TARGET_DISTANCE:.3f} m  Error: {distance_error:+.3f} m   ", end="", flush=True)
                
                if abs(distance_error) < DISTANCE_THRESHOLD and pre_grip_pose.is_at_goal():
                    print("\nPhase 2 complete.")
                    cv2.destroyAllWindows()
                    return
            
            velocities = merge_proportional(velocities, auto_velocities)
        
        if rgb is not None:
            cv2.imshow("Wrist RGB", rgb)
        
        velocities = merge_proportional(velocities, pre_grip_pose.get_command())
        controller.set_velocities(velocities)
        cv2.waitKey(1)
        time.sleep(1 / 30)


def grab_phase():
    """Close gripper and lift object."""
    print("Phase 3: grabbing...")
    
    grab_velocities = {
        "lift_up": 0.2,
        "gripper_open": -1.0,
    }
    
    while True:
        velocities = teleop.get_normalized_velocities()
        velocities = merge_proportional(velocities, grab_velocities)
        controller.set_velocities(velocities)


def main():
    print("Press Ctrl+C to stop\n")

    transforms = RobotTransforms(controller)

    stow_pose = StateController(controller, {
        "wrist_roll_counterclockwise": 0.0,
        "wrist_yaw_counterclockwise": 0.0,
        "wrist_pitch_up": 0.0,
        "gripper_open": 0.5,
        "arm_out": 0.0,
    })

    pre_grip_pose = StateController(controller, {
        "wrist_roll_counterclockwise": 0.0,
        "gripper_open": 0.5,
    })

    try:
        align_phase(transforms, stow_pose)
        reach_phase(transforms, pre_grip_pose)
        grab_phase()

    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        controller.set_velocities({})
        controller.stop()
        cv2.destroyAllWindows()
        print("Done.")


if __name__ == "__main__":
    main()
