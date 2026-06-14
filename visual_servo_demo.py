"""
Simple test: robot control with configurable camera displays.
"""
from stretch_toolkit import (
    controller, teleop, BACKEND_NAME, merge_proportional,
    HEAD_CAMERA, WRIST_CAMERA, NAVIGATION_CAMERA,
    HEAD_RGB_CAMERA, HEAD_DEPTH_CAMERA,
    WRIST_RGB_CAMERA, WRIST_DEPTH_CAMERA
)
import time
import cv2
import numpy as np

print(f"\n=== Running on {BACKEND_NAME} backend ===\n")

# Visual servoing configuration - swap these to switch between head and wrist
HEAD_CONFIG = {
    "camera": HEAD_RGB_CAMERA,
    "joint_x": "head_pan_counterclockwise",
    "joint_y": "head_tilt_up",
    "name": "Head RGB"
}

WRIST_CONFIG = {
    "camera": WRIST_RGB_CAMERA,
    "joint_x": "wrist_yaw_counterclockwise",
    "joint_y": "wrist_pitch_up",
    "name": "Wrist RGB"
}

# Active configuration - change this to switch between head and wrist
SERVO_CONFIG = WRIST_CONFIG

def get_obj_center(frame):
    """Find the largest red contour and return its center.
    
    Args:
        frame: BGR image from camera
    
    Returns:
        (x, y) tuple of center coordinates, or None if no red object found
    """
    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Red color wraps around in HSV, so we need two ranges
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    
    # Create masks for both red ranges
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Filter contours by minimum area and get the largest
    MIN_AREA = 100
    valid_contours = [c for c in contours if cv2.contourArea(c) >= MIN_AREA]
    
    if not valid_contours:
        return None
    
    largest_contour = max(valid_contours, key=cv2.contourArea)
    
    # Calculate moments to find center
    M = cv2.moments(largest_contour)
    if M['m00'] == 0:
        return None
    
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    
    return (cx, cy)

def servoing_demo():
    try:
        while True:
            # Get normalized velocities from input devices
            cmd = teleop.get_normalized_velocities()

            # Get frame from configured camera
            frame = SERVO_CONFIG["camera"].get_frame()
            if frame is not None:
                drawing_frame = frame.copy()

                # Find red object center
                center = get_obj_center(frame)
                if center is not None:
                    cv2.circle(drawing_frame, center, 10, (0, 255, 0), -1)
                    norm_center = (center[0] / frame.shape[1], center[1] / frame.shape[0])
                    norm_image_center = (0.5, 0.5)

                    # Draw line from image center to object center
                    cv2.line(drawing_frame, (int(frame.shape[1] * norm_image_center[0]), int(frame.shape[0] * norm_image_center[1])), center, (0, 255, 255), 2)

                    # Simple proportional control to move towards object
                    error_x = norm_center[0] - norm_image_center[0]
                    error_y = norm_center[1] - norm_image_center[1]
                    Kp = 0.5  # Proportional gain
                    velocities = {
                        SERVO_CONFIG["joint_x"]: Kp * error_x,
                        SERVO_CONFIG["joint_y"]: Kp * error_y
                    }
                    cmd = merge_proportional(cmd, velocities) # User input always wins over auto commands

                # Display the drawing frame
                cv2.imshow(SERVO_CONFIG["name"], drawing_frame)

            # Send to robot (physical or simulated)
            controller.set_velocities(cmd)
            
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            
    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        # Stop all motion
        controller.set_velocities({})
        controller.stop()
        cv2.destroyAllWindows()
        print("Demo complete!")

if __name__ == "__main__":
    servoing_demo()
