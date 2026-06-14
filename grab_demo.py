"""
Grab demo approach with head camera, align and reach with wrist camera, then grab.

Usage:
    uv run grab_demo.py            # default: blue object
    uv run grab_demo.py red
    uv run grab_demo.py green
    uv run grab_demo.py blue
"""
import sys
import time
import math
import cv2
import numpy as np

from stretch_toolkit import (
    controller, teleop, merge_proportional, locate_object,
    BACKEND_NAME, HEAD_CAMERA, HEAD_RGB_CAMERA,
    WRIST_CAMERA, WRIST_RGB_CAMERA, StateController
)
from stretch_toolkit.robot_transforms import RobotTransforms

print(f"\n=== Running on {BACKEND_NAME} backend ===\n")

# -- Color profiles ----------------------------------------------------------
COLOR_PROFILES = {
    "blue":  (np.array([0, 80, 50]), np.array([30, 255, 255])),
    "red":   (np.array([100,   80,  50]), np.array([130,  255, 255])),
    "green": (np.array([40,  70,   70]), np.array([80,  255, 255])),
}

target_color = sys.argv[1].lower() if len(sys.argv) > 1 else "blue"
if target_color not in COLOR_PROFILES:
    print(f"[WARN] Unknown color '{target_color}', defaulting to 'blue'")
    target_color = "blue"
lower_hsv, upper_hsv = COLOR_PROFILES[target_color]
print(f"Target color: {target_color}\n")

MIN_CONTOUR_AREA = 100


def find_object(rgb_frame):
    """Devuelve (cx, cy) del centroide del objeto, o None si no se ve."""
    if rgb_frame is None:
        return None
    # HEAD_RGB_CAMERA devuelve RGB, no BGR
    hsv = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # Para rojo cubre el wrap-around del HSV (170-180)
    if target_color == "red":
        mask2 = cv2.inRange(hsv, np.array([170, 120, 70]), np.array([180, 255, 255]))
        mask = cv2.bitwise_or(mask, mask2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < MIN_CONTOUR_AREA:
        return None
    M = cv2.moments(largest)
    if M['m00'] > 0:
        cx = int(M['m10'] / M['m00'])   
        cy = int(M['m01'] / M['m00'])
        if cy < rgb_frame.shape[0] * 0.20:
        # Ignorar detecciones en la mitad superior del frame (skybox/techo)
            return None
        return (cx, cy)
    return None


def main():
    print("Press Ctrl+C to stop\n")

    transforms = RobotTransforms(controller)

    print("Warming up sim...")
    for _ in range(60):
        controller.set_velocities({})
        cv2.waitKey(1)
        time.sleep(1 / 30)
    print("Ready!\n")

    print("Pointing head camera towards the table...")
    controller.set_velocities({"head_tilt_up": -0.5})
    time.sleep(2.0)
    controller.set_velocities({"head_tilt_up": -0.3})
    time.sleep(1.0)
    controller.set_velocities({})
    time.sleep(0.5)
    print("READY\n")

    stow_pose = StateController(controller, {
        "wrist_roll_counterclockwise": 0.0,
        "wrist_yaw_counterclockwise":  0.0,
        "wrist_pitch_up":              0.0,
        "gripper_open":                0.3,
        "arm_out":                     0.0,
    })

    pre_grip_pose = StateController(controller, {
        "wrist_roll_counterclockwise": 0.0,
        "gripper_open":                0.4,
    })

    # -- Ganancias proporcionales --------------------------------------------
    Kp_pan     = 1.0
    Kp_tilt    = 1.0
    Kp_angle   = 5.0 / math.pi
    Kp_forward = 2.0
    Kp_lift    = 5.0
    Kp_yaw     = 0.5
    Kp_pitch   = 0.5
    Kp_arm     = 10.0

    # -- Estado inicial ------------------------------------------------------
    phase = "approach"
    in_zone = False
    approach_start = time.time()
    APPROACH_SPIN_TIMEOUT = 20.0
    grab_start = None

    # FIX: valores default para evitar NameError si centroid es None
    frame_cx = 0
    frame_cy = 0

    print(f"Phase: {phase}")

    try:
        while True:
            velocities = teleop.get_normalized_velocities()
            auto_velocities = {}

            # -- APPROACH ----------------------------------------------------
            if phase == "approach":
                rgb = HEAD_RGB_CAMERA.get_frame()
                centroid = find_object(rgb)
                if rgb is not None:
                                hsv_debug = cv2.cvtColor(rgb, cv2.COLOR_GRB2HSV)
                                mask_debug = cv2.inRange(hsv_debug, lower_hsv, upper_hsv)
                                if target_color == "red":
                                    mask2_debug = cv2.inRange(hsv_debug, np.array([170, 120, 70]), np.array([180,255,255]))
                                    mask_debug = cv2.bitwise_or(mask_debug, mask2_debug)
                                cv2.imshow("Mask", mask_debug)
                                if centroid is not None and rgb is not None:
                                    approach_start = time.time()
                                    cx, cy = centroid
                                    frame_cx = rgb.shape[1] / 2
                                    frame_cy = rgb.shape[0] / 2

                                    error_x = (cx - frame_cx) / rgb.shape[1]
                                    error_y = (cy - frame_cy) / rgb.shape[0]
                                    auto_velocities["head_pan_counterclockwise"] = -Kp_pan * error_x
                                    auto_velocities["head_tilt_up"]              = -Kp_tilt * error_y

                                    _, obj2base_T = locate_object((cx, cy), HEAD_CAMERA, transforms)
                                    if obj2base_T is not None:
                                        x, y, z = obj2base_T[0:3, 3]
                                        angle_z = math.atan2(y, x)
                                        horizontal_distance = math.sqrt(x**2 + y**2)

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
                                            print(f"\rDist: {horizontal_distance:.2f}m  AngleErr: {math.degrees(angle_error):+6.1f}  [FLANK]   ", end="", flush=True)
                                            if abs(angle_error) < math.radians(5):
                                                cv2.destroyWindow("Head RGB")
                                                phase = "align"
                                                print(f"\nPhase: {phase}")
                                        else:
                                            auto_velocities["base_counterclockwise"] = -Kp_angle * angle_z
                                            alignment   = 1.0 - (abs(angle_z) / math.pi)
                                            travel_auth = max(0.0, min(1.0, (alignment - 0.7) / 0.3))
                                            auto_velocities["base_forward"] = Kp_forward * horizontal_distance * travel_auth
                                            print(f"\rDist: {horizontal_distance:.2f}m  Align: {alignment:.3f}  Auth: {travel_auth:.2f}  [moving]   ", end="", flush=True)
                                    cv2.circle(rgb, (cx, cy), 10, (0, 255, 0), -1)
                                    cv2.circle(rgb, (int(frame_cx), int(frame_cy)), 5, (0, 0, 255), 2)

                                else:
                                    elapsed = time.time() - approach_start
                                    if elapsed > APPROACH_SPIN_TIMEOUT:
                                        auto_velocities["base_counterclockwise"] = 0.3
                                        print(f"\r[SEARCH] No object found, spinning...  t={elapsed:.1f}s   ", end="", flush=True)

                                if rgb is not None:
                                    cv2.imshow("Head RGB", rgb)

                                # FIX: sin stow_pose en approach para evitar conflicto con lift
                                velocities = merge_proportional(velocities, auto_velocities)

            # -- ALIGN -------------------------------------------------------
            elif phase == "align":
                rgb = WRIST_RGB_CAMERA.get_frame()
                centroid = find_object(rgb)

                if centroid is not None:
                    cx, cy = centroid
                    _, obj2base_T = locate_object((cx, cy), WRIST_CAMERA, transforms)
                    if obj2base_T is not None:
                        x, y, z = obj2base_T[0:3, 3]
                        angle_z = math.atan2(y, x)
                        cam_z   = transforms.get_wrist_cam_T()[2, 3]

                        auto_velocities["base_counterclockwise"] = Kp_angle * (-math.pi / 2 - angle_z + math.radians(3))
                        auto_velocities["lift_up"]               = Kp_lift * (z - (cam_z + 0.01))

                        angle_err = abs(-math.pi / 2 - angle_z)
                        lift_err  = abs(z - (cam_z + 0.01))
                        print(f"\rAngleErr: {math.degrees(angle_err):.1f}  LiftErr: {lift_err:.3f}m   ", end="", flush=True)

                        if stow_pose.is_at_goal() and angle_err < math.radians(6) and lift_err < 0.05:
                            phase = "reach"
                            print(f"\nPhase: {phase}")

                    if rgb is not None:
                        cv2.circle(rgb, (cx, cy), 10, (0, 255, 0), -1)

                if rgb is not None:
                    cv2.imshow("Wrist RGB", rgb)

                velocities = merge_proportional(velocities, auto_velocities)
                velocities = merge_proportional(velocities, stow_pose.get_command())

            # -- REACH -------------------------------------------------------
            elif phase == "reach":
                rgb = WRIST_RGB_CAMERA.get_frame()
                centroid = find_object(rgb)

                if centroid is not None:
                    cx, cy = centroid

                    if rgb is not None:
                        frame_cx = rgb.shape[1] / 2
                        frame_cy = rgb.shape[0] / 2
                        error_x = (cx - frame_cx) / rgb.shape[1]
                        error_y = (cy - frame_cy) / rgb.shape[0]
                        auto_velocities["wrist_yaw_counterclockwise"] = Kp_yaw * error_x
                        auto_velocities["wrist_pitch_up"]             = -Kp_pitch * error_y
                        cv2.circle(rgb, (cx, cy), 10, (0, 255, 0), -1)

                    distance = WRIST_CAMERA.get_depth((cx, cy))
                    if distance is not None:
                        distance_error = distance - 0.12
                        auto_velocities["arm_out"] = Kp_arm * distance_error
                        print(f"\rWristDist: {distance:.3f}m  Err: {distance_error:+.3f}m   ", end="", flush=True)
                        if abs(distance_error) < 0.02 and pre_grip_pose.is_at_goal():
                            cv2.destroyAllWindows()
                            phase = "grab"
                            grab_start = time.time()
                            print(f"\nPhase: {phase}")

                if rgb is not None:
                    cv2.imshow("Wrist RGB", rgb)

                velocities = merge_proportional(velocities, auto_velocities)
                velocities = merge_proportional(velocities, pre_grip_pose.get_command())

            # -- GRAB --------------------------------------------------------
            elif phase == "grab":
                elapsed_grab = time.time() - grab_start

                if elapsed_grab < 1.5:
                    auto_velocities["gripper_open"] = -1.0
                    print(f"\r[GRAB] Closing gripper...  t={elapsed_grab:.1f}s   ", end="", flush=True)
                elif elapsed_grab < 3.5:
                    auto_velocities["gripper_open"] = -1.0
                    auto_velocities["lift_up"] = 0.3
                    print(f"\r[GRAB] Lifting...  t={elapsed_grab:.1f}s   ", end="", flush=True)
                else:
                    phase = "done"
                    print(f"\nPhase: {phase}")

                velocities = merge_proportional(velocities, auto_velocities)

            # -- DONE --------------------------------------------------------
            elif phase == "done":
                print("\nObject grabbed successfully!")
                break

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