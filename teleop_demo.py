"""
Simple test: robot control with configurable camera displays.
"""
from stretch_toolkit import (
    controller, teleop, BACKEND_NAME,
    HEAD_CAMERA, WRIST_CAMERA, NAVIGATION_CAMERA,
    HEAD_RGB_CAMERA, HEAD_DEPTH_CAMERA,
    WRIST_RGB_CAMERA, WRIST_DEPTH_CAMERA
)
import stretch_toolkit.input as inp
import time
import cv2
import numpy as np

print(f"\n=== Running on {BACKEND_NAME} backend ===\n")

# Configure which cameras to display
# Set first value to True/False to enable/disable each feed
CAMERA_DISPLAYS = [
    (False,  "Head RGB", HEAD_RGB_CAMERA),
    (False, "Head Depth", HEAD_DEPTH_CAMERA),
    (False,  "Wrist RGB", WRIST_RGB_CAMERA),
    (False, "Wrist Depth", WRIST_DEPTH_CAMERA),
    (False, "Navigation", NAVIGATION_CAMERA),
]

def teleop_demo():
    from stretch_toolkit.input import rising_edge
    """Run teleoperation loop with configurable camera displays."""
    print("Teleop with camera views. Use gamepad/keyboard to control.")
    active_cameras = [name for enabled, name, cam in CAMERA_DISPLAYS if enabled and cam is not None]
    print(f"Displaying {len(active_cameras)} camera feeds: {', '.join(active_cameras)}")
    print("Press Ctrl+C to stop\n")
    
    # Track which windows are currently open
    open_windows = set()
    
    try:
        while True:
            for i in range(1, len(CAMERA_DISPLAYS) + 1):
                if rising_edge(str(i)):
                    enabled, name, cam = CAMERA_DISPLAYS[i-1]
                    CAMERA_DISPLAYS[i-1] = (not enabled, name, cam)
                    state = "ENABLED" if not enabled else "DISABLED"
                    print(f"\n[{i}. {name}] {state}")

            # Get normalized velocities from input devices
            velocities = teleop.get_normalized_velocities()
            
            # Send to robot (physical or simulated)
            controller.set_velocities(velocities)
            
            # Track which windows should be open this frame
            active_windows = set()
            
            # Display all configured camera feeds
            for enabled, window_name, camera in CAMERA_DISPLAYS:
                if enabled and camera is not None:
                    active_windows.add(window_name)
                    try:
                        frame = camera.get_frame()
                        
                        # Colorize depth frames for visualization
                        if "Depth" in window_name and frame is not None:
                            frame_vis = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                            frame = cv2.applyColorMap(frame_vis, cv2.COLORMAP_JET)
                        
                        if frame is not None:
                            cv2.imshow(window_name, frame)
                            open_windows.add(window_name)
                    except Exception as e:
                        pass  # Silently skip errors
            
            # Close windows that should no longer be displayed
            windows_to_close = open_windows - active_windows
            for window_name in windows_to_close:
                try:
                    cv2.destroyWindow(window_name)
                except:
                    pass  # Silently handle window destruction errors
            open_windows = active_windows
            
            cv2.waitKey(1)
            time.sleep(1/30)  # 30 Hz update rate
            
    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        # Stop all motion
        controller.set_velocities({})
        controller.stop()
        cv2.destroyAllWindows()
        print("Demo complete!")

if __name__ == "__main__":
    teleop_demo()
