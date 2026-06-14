"""
Stretch Toolkit boilerplate — starting point for new scripts.
"""
from stretch_toolkit import ( controller, teleop, BACKEND_NAME, HEAD_CAMERA, WRIST_CAMERA, NAVIGATION_CAMERA, HEAD_RGB_CAMERA, HEAD_DEPTH_CAMERA, WRIST_RGB_CAMERA, WRIST_DEPTH_CAMERA, StateController, merge_proportional )
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
            
            # Get user input
            velocities = teleop.get_normalized_velocities()

            # --- Your logic here ---
            frame = HEAD_RGB_CAMERA.get_frame()
            depth_frame = HEAD_DEPTH_CAMERA.get_frame()
            if frame is not None:
                # Convert to HSV and find blue objects
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                lower_blue = np.array([110, 100, 100])
                upper_blue = np.array([130, 255, 255])
                mask = cv2.inRange(hsv, lower_blue, upper_blue)
                
                # Find largest contour
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest = max(contours, key=cv2.contourArea)
                    M = cv2.moments(largest)
                    if M['m00'] > 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        
                        # Calculate normalized error from image center
                        error_x = (cx / frame.shape[1]) - 0.5
                        error_y = (cy / frame.shape[0]) - 0.5
                        
                        # Proportional control
                        Kp = 0.5
                        auto_velocities = {
                            "head_pan_counterclockwise": Kp * error_x,
                            "head_tilt_up": Kp * error_y
                        }
                        
                        # Merge user input with autonomous control (user always wins)
                        velocities = merge_proportional(velocities, auto_velocities)
                        
                        # Draw visualization
                        # cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)
                        
                        if depth_frame is not None:
                            distance = HEAD_CAMERA.get_depth((cx, cy), depth_frame)
                            if distance is not None:
                                print(f"Distance: {distance} m")
                
                cv2.imshow("Head Camera", frame)
                cv2.imshow("Depth", depth_frame if depth_frame is not None else np.zeros_like(frame))
                cv2.imshow("Mask", mask)

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
