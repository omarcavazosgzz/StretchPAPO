"""Base classes for robot control - environment agnostic."""
from . import input as ci

import numpy as np
import math
import json
import os
import time
from pathlib import Path

class CamInfo:
    """Single camera configuration with intrinsics and frame getter."""
    
    def __init__(self, name, frame_getter, camera_matrix=None, 
                 distortion_coeffs=None, distortion_model=None):
        """
        Args:
            name: Camera identifier (e.g., "D435i RGB", "D435i Depth")
            frame_getter: Function that returns a single frame (rgb or depth array)
            camera_matrix: 3x3 numpy array with camera intrinsics (optional)
            distortion_coeffs: Camera distortion coefficients (optional)
            distortion_model: Camera distortion model type (optional)
        """
        self.name = name
        self.get_frame = frame_getter
        self.camera_matrix = camera_matrix
        self.distortion_coeffs = distortion_coeffs
        self.distortion_model = distortion_model
    
    @property
    def has_intrinsics(self):
        """Check if camera intrinsics are available."""
        return self.camera_matrix is not None
    
    @property
    def fx(self):
        if not self.has_intrinsics:
            raise ValueError(f"Camera '{self.name}' has no intrinsics configured")
        return self.camera_matrix[0, 0]
    
    @property
    def fy(self):
        if not self.has_intrinsics:
            raise ValueError(f"Camera '{self.name}' has no intrinsics configured")
        return self.camera_matrix[1, 1]
    
    @property
    def cx(self):
        if not self.has_intrinsics:
            raise ValueError(f"Camera '{self.name}' has no intrinsics configured")
        return self.camera_matrix[0, 2]
    
    @property
    def cy(self):
        if not self.has_intrinsics:
            raise ValueError(f"Camera '{self.name}' has no intrinsics configured")
        return self.camera_matrix[1, 2]
    
    def pixel_to_normalized(self, centroid):
        """Convert pixel coordinates to normalized camera coordinates.
        
        Args:
            centroid: (x, y) pixel coordinates
        
        Returns:
            (x_norm, y_norm): Normalized coordinates where z=1 in camera frame
        """
        if not self.has_intrinsics:
            raise ValueError(f"Camera '{self.name}' requires intrinsics for pixel_to_normalized")
        
        x_norm = (centroid[0] - self.cx) / self.fx
        y_norm = (centroid[1] - self.cy) / self.fy
        
        return x_norm, y_norm
    
    def pixel_to_object_angles(self, centroid):
        """Convert pixel to angular position (pitch, yaw) in degrees relative to camera.
        
        Args:
            centroid: (x, y) pixel coordinates
        
        Returns:
            (pitch, yaw): Angular position in degrees
        """
        x_norm, y_norm = self.pixel_to_normalized(centroid)
        
        yaw_rad = math.atan(x_norm)
        pitch_rad = math.atan(y_norm)
        
        return math.degrees(pitch_rad), math.degrees(yaw_rad)
    
    def object_angles_to_pixel(self, yaw, pitch):
        """Convert angular position (yaw, pitch) to pixel coordinates.
        
        Args:
            yaw: Yaw angle in degrees
            pitch: Pitch angle in degrees
        
        Returns:
            (x, y): Pixel coordinates
        """
        if not self.has_intrinsics:
            raise ValueError(f"Camera '{self.name}' requires intrinsics for object_angles_to_pixel")
        
        yaw_rad = math.radians(yaw)
        pitch_rad = math.radians(pitch)
        
        x_norm = math.tan(yaw_rad)
        y_norm = math.tan(pitch_rad)
        
        x = x_norm * self.fx + self.cx
        y = y_norm * self.fy + self.cy
        
        return x, y


class DepthCamInfo:
    """Depth camera configuration wrapping separate RGB and depth cameras."""
    
    def __init__(self, name, rgb_cam, depth_cam, depth_scale):
        """
        Args:
            name: Camera system identifier (e.g., "D435i Head", "D405 Wrist")
            rgb_cam: CamInfo for RGB camera
            depth_cam: CamInfo for depth camera  
            depth_scale: Meters per depth unit (e.g., 1e-03 for D435i)
        """
        self.name = name
        self.rgb_cam = rgb_cam
        self.depth_cam = depth_cam
        self.depth_scale = depth_scale
    
    def get_frames(self):
        """Get both RGB and depth frames.
        
        Returns:
            tuple: (rgb_frame, depth_frame)
        """
        return self.rgb_cam.get_frame(), self.depth_cam.get_frame()
    
    # Delegate RGB intrinsics to rgb_cam
    @property
    def fx(self):
        return self.rgb_cam.fx
    
    @property
    def fy(self):
        return self.rgb_cam.fy
    
    @property
    def cx(self):
        return self.rgb_cam.cx
    
    @property
    def cy(self):
        return self.rgb_cam.cy
    
    @property
    def camera_matrix(self):
        return self.rgb_cam.camera_matrix
    
    @property
    def distortion_coeffs(self):
        return self.rgb_cam.distortion_coeffs
    
    @property
    def distortion_model(self):
        return self.rgb_cam.distortion_model
    
    # Depth camera intrinsics
    @property
    def depth_fx(self):
        return self.depth_cam.fx
    
    @property
    def depth_fy(self):
        return self.depth_cam.fy
    
    @property
    def depth_cx(self):
        return self.depth_cam.cx
    
    @property
    def depth_cy(self):
        return self.depth_cam.cy
    
    @property
    def depth_camera_matrix(self):
        return self.depth_cam.camera_matrix
    
    @property
    def depth_distortion_coeffs(self):
        return self.depth_cam.distortion_coeffs
    
    @property
    def depth_distortion_model(self):
        return self.depth_cam.distortion_model
    
    # Delegate RGB projection methods
    def pixel_to_normalized(self, centroid):
        """Convert pixel coordinates to normalized camera coordinates (RGB frame)."""
        return self.rgb_cam.pixel_to_normalized(centroid)
    
    def pixel_to_object_angles(self, centroid):
        """Convert pixel to angular position in degrees (RGB frame)."""
        return self.rgb_cam.pixel_to_object_angles(centroid)
    
    def object_angles_to_pixel(self, yaw, pitch):
        """Convert angular position to pixel coordinates (RGB frame)."""
        return self.rgb_cam.object_angles_to_pixel(yaw, pitch)

    def get_depth(self, centroid, depth_image=None, sample_radius=None):
        """Get distance to object in meters using depth camera intrinsics.
        
        Projects RGB pixel coordinate to depth camera coordinate system,
        then samples depth values around that location.
        
        Args:
            centroid: (x, y) pixel coordinates in RGB frame
            depth_image: Depth image array (if None, fetches current frame)
            sample_radius: Radius in pixels to sample around centroid (default: 3)
        
        Returns:
            Median distance in meters, or None if no valid depth samples
        """
        if sample_radius is None:
            sample_radius = 3
        
        if depth_image is None:
            depth_image = self.depth_cam.get_frame()
        
        if depth_image is None or depth_image.size == 0:
            return None

        # Project RGB pixel to depth camera coordinate system
        # 1. Convert RGB pixel to normalized coordinates
        x_norm = (centroid[0] - self.cx) / self.fx
        y_norm = (centroid[1] - self.cy) / self.fy
        
        # 2. Project to depth camera pixel coordinates
        depth_x = x_norm * self.depth_fx + self.depth_cx
        depth_y = y_norm * self.depth_fy + self.depth_cy
        
        x, y = int(depth_x), int(depth_y)
        height, width = depth_image.shape[:2]
        
        # Collect depth samples in a square region
        samples = []
        for dy in range(-sample_radius, sample_radius + 1):
            for dx in range(-sample_radius, sample_radius + 1):
                # Clamp to image bounds
                px = max(0, min(width - 1, x + dx))
                py = max(0, min(height - 1, y + dy))
                
                depth_value = depth_image[py, px]
                if depth_value > 0:  # Only include valid depth values
                    samples.append(depth_value)
        
        if not samples:
            return None
        
        # Use median to reduce noise
        median_depth = np.median(samples)
        
        return median_depth * self.depth_scale


class TeleopProvider:
    """Provides teleoperation commands as normalized joint velocities."""
    def __init__(self, is_stretch_env=False, config_file='teleop_mappings.json'):
        self.is_stretch_env = is_stretch_env
        
        # Store config file in this script's directory
        script_dir = Path(__file__).parent
        self.config_file = script_dir / config_file
        self.last_mtime = None
        
        # Toggle states
        self.dpad_controls_head = False  # False = wrist, True = head
        self.manual_mode_enabled = False  # False = autonomous mode, True = manual override
        
        # Load mappings from config file
        self._load_or_create_config()
        
        self.joint_mappings = {}
        self._update_joint_mappings()

    def _get_default_config(self):
        """Get default teleop mappings configuration."""
        return {
            'irl': {
                'base_mappings': {
                    'base_forward': ['w', 's', 'LY'],
                    'base_counterclockwise': ['d', 'a', 'LX'],
                    'lift_up': ['z', 'x', 'RY'],
                    'arm_out': ['v', 'c', 'RX'],
                    'gripper_open': ['m', 'n', 'B', 'A'],
                    'wrist_yaw_counterclockwise': ['l', 'j', 'RB', 'LB'],
                    'wrist_roll_counterclockwise': ['u', 'o', None, 'DPAD_X'],
                    'wrist_pitch_up': ['i', 'k', None, 'DPAD_Y'],
                },
                'dpad_head_mappings': {
                    'wrist_yaw_counterclockwise': [],
                    'wrist_roll_counterclockwise': [],
                    'wrist_pitch_up': [],
                    'head_pan_counterclockwise': ['l', 'j', 'DPAD_X'],
                    'head_tilt_up': ['i', 'k', None, 'DPAD_Y'],
                },
                'toggle_buttons': {
                    'head_wrist_toggle': ['X', 'h'],
                    'manual_mode_toggle': ['X', 'y']
                }
            },
            'sim': {
                # Sim-specific overrides can go here
            }
        }

    def _load_or_create_config(self):
        """Load configuration from JSON file, creating with defaults if it doesn't exist."""
        if not self.config_file.exists():
            # Create file with defaults
            defaults = self._get_default_config()
            with open(self.config_file, 'w') as f:
                json.dump(defaults, f, indent=2)
            print(f"Created default teleop config: {self.config_file}")
        
        # Load from file
        self._load_config()

    def _load_config(self):
        """Load configuration from JSON file and update modification time."""
        with open(self.config_file, 'r') as f:
            config = json.load(f)
        
        # Start with 'irl' config (base/default)
        irl_config = config.get('irl', config)  # Fallback to root if no 'irl' key
        
        # If not stretch_env (i.e., simulation), recursively override with 'sim' config
        if not self.is_stretch_env and 'sim' in config:
            final_config = self._recursive_merge(irl_config, config['sim'])
        else:
            final_config = irl_config
        
        # Convert lists back to tuples
        self.base_mappings = {k: tuple(v) for k, v in final_config.get('base_mappings', {}).items()}
        self.dpad_head_mappings = {k: tuple(v) for k, v in final_config.get('dpad_head_mappings', {}).items()}
        
        # Load toggle buttons
        self.toggle_buttons = final_config.get('toggle_buttons', {'head_wrist_toggle': ['X', 'h']})
        
        # Update modification time
        self.last_mtime = os.path.getmtime(self.config_file)
    
    def _recursive_merge(self, base, override):
        """Recursively merge override dict into base dict.
        
        Args:
            base: Base dictionary
            override: Override dictionary (values override base)
        
        Returns:
            Merged dictionary
        """
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # Recursively merge nested dicts
                result[key] = self._recursive_merge(result[key], value)
            else:
                # Override value
                result[key] = value
        return result

    def _check_and_reload_config(self):
        """Check if config file has been modified and reload if necessary."""
        if not self.config_file.exists():
            return
        
        current_mtime = os.path.getmtime(self.config_file)
        if current_mtime != self.last_mtime:
            print(f"Teleop config file changed, reloading: {self.config_file}")
            self._load_config()
            self._update_joint_mappings()

    def _update_joint_mappings(self):
        """Update joint mappings based on current toggle states."""
        # Start with base mappings
        self.joint_mappings = self.base_mappings.copy()
        
        # Add head mappings when toggle is active
        if self.dpad_controls_head:
            self.joint_mappings.update(self.dpad_head_mappings)

    def _normalize_mapping(self, mapping):
        """Normalize mapping tuple to 6 elements with defaults.
        
        Args:
            mapping: Tuple of (high_key, low_key, high_game, low_game, [keyboard_scale], [game_scale])
        
        Returns:
            tuple: 6-element tuple with defaults filled in
        """
        if not mapping:
            return (None, None, None, None, 1.0, 1.0)
        
        defaults = (None, None, None, None, 1.0, 1.0)
        return mapping + defaults[len(mapping):]

    def _get_joint_velocity(self, mapping):
        """Get normalized velocity from a joint mapping.
        
        Args:
            mapping: Tuple of (high_key, low_key, high_game, low_game, [keyboard_scale], [game_scale])
        
        Returns:
            float: Normalized velocity from -1.0 to 1.0
        """
        normalized = self._normalize_mapping(mapping)
        return ci.get_bipolar_ctrl(*normalized)

    def _button_pressed(self, button):
        """Check if a button was just pressed (rising edge).
        
        Args:
            button: Button name string
        
        Returns:
            bool: True if button was just pressed
        """
        return ci.rising_edge(button)

    def _check_toggles(self):
        """Check for toggle button presses and update states."""
        # Check head/wrist toggle buttons
        toggle_buttons = self.toggle_buttons.get('head_wrist_toggle', [])
        if any(self._button_pressed(btn) for btn in toggle_buttons if btn):
            self.dpad_controls_head = not self.dpad_controls_head
            mode = "HEAD (override wrist)" if self.dpad_controls_head else "WRIST (default)"
            print(f"Controls: {mode}")
            self._update_joint_mappings()
        
        # Check manual mode toggle buttons
        manual_toggle_buttons = self.toggle_buttons.get('manual_mode_toggle', [])
        if any(self._button_pressed(btn) for btn in manual_toggle_buttons if btn):
            self.manual_mode_enabled = not self.manual_mode_enabled
            mode = "MANUAL" if self.manual_mode_enabled else "AUTONOMOUS"
            print(f"Mode: {mode}")

    def get_normalized_velocities(self):
        """Get normalized joint velocities from input devices.
        
        Returns:
            dict: Normalized velocities (-1.0 to 1.0) for all joints
        """
        # Check for config file updates
        self._check_and_reload_config()
        
        # Check for toggle button presses
        self._check_toggles()
        
        result = {}
        for joint, mapping in self.joint_mappings.items():
            result[joint] = self._get_joint_velocity(mapping)
        return result

    def get_manual_override(self, cmd_autonomous):
        """Merge an autonomous command with teleop input, giving the operator priority.

        When the operator moves a joint, their input proportionally overrides the
        autonomous command for that joint. Joints the operator is not touching
        continue to follow the autonomous command unmodified.

        If manual_mode_enabled is True, ignores cmd_autonomous completely and returns
        pure teleop control.

        Args:
            cmd_autonomous: Dict of normalized velocities from an autonomous controller.

        Returns:
            dict: Merged command to pass directly to controller.set_velocities().
        """
        cmd_teleop = self.get_normalized_velocities()
        if self.manual_mode_enabled:
            # Pure manual control - ignore autonomous command
            return cmd_teleop
        # Proportional blend - operator can override autonomous
        return merge_proportional(cmd_teleop, cmd_autonomous)


class JointController:
    """Base class for joint controllers."""

    def __init__(self):
        self._start_time = time.perf_counter()

    def get_time(self):
        """Get elapsed time in seconds since the controller was initialized.

        Returns:
            float: Elapsed seconds (wall-clock time).
                   Subclasses may override this to return simulator time.
        """
        return time.perf_counter() - self._start_time

    def set_velocities(self, vel_dict):
        """Set normalized joint velocities.
        
        Args:
            vel_dict: Dict mapping joint names to velocities (-1.0 to 1.0)
        """
        raise NotImplementedError("Subclasses must implement set_velocities()")
    
    def get_state(self):
        """Get current joint positions and base odometry.
        
        Returns:
            dict: Joint positions with keys:
                - base_x, base_y, base_theta (odometry in meters/radians)
                - lift_up, arm_out (meters)
                - wrist joints, head joints (radians)
                - gripper_open (radians)
        """
        raise NotImplementedError("Subclasses must implement get_state()")

    def get_lidar_ranges(self):
        """Get sanitized LiDAR range readings.

        Returns:
            np.ndarray: Array of distances in metres.  Invalid/no-hit rays are
            represented as np.inf.  Returns None on error.
        """
        raise NotImplementedError("get_lidar_ranges() is not implemented for this backend.")

    def stop(self):
        """Stop all robot motion."""
        raise NotImplementedError("Subclasses must implement stop()")


def merge_proportional(cmd_primary, cmd_secondary, deadband=0.05):
    """Merge two command dictionaries with proportional blending.
    
    Primary command overrides secondary based on input magnitude.
    When primary input is below deadband, secondary is used.
    Otherwise, primary input strength determines blend between secondary and full output.
    
    Args:
        cmd_primary: Primary command dict (e.g., from teleop)
        cmd_secondary: Secondary command dict (e.g., from autonomous controller)
        deadband: Threshold below which primary is considered inactive (default 0.05)
    
    Returns:
        dict: Merged command with proportional blending
    """
    cmd_final = {}
    
    # Handle all joints from both commands
    all_joints = set(cmd_primary.keys()) | set(cmd_secondary.keys())
    
    for joint in all_joints:
        primary_input = cmd_primary.get(joint, 0.0)
        secondary_input = cmd_secondary.get(joint, 0.0)
        
        if abs(primary_input) < deadband:
            # No primary input - use secondary
            cmd_final[joint] = secondary_input
        else:
            # Primary input interpolates between secondary and desired value
            # abs(primary_input) determines how much override (0 to 1)
            # sign(primary_input) determines direction
            override_strength = abs(primary_input)
            desired_value = 1.0 if primary_input > 0 else -1.0
            cmd_final[joint] = (1 - override_strength) * secondary_input + override_strength * desired_value
    
    return cmd_final

def locate_object(centroid, depth_cam_info, robot_transforms, sample_radius=3):
    """Locate an object in the robot base frame from a pixel centroid.

    Args:
        centroid: (x, y) pixel coordinates in the RGB frame
        depth_cam_info: DepthCamInfo instance with intrinsics and depth getter
        robot_transforms: RobotTransforms instance for the current camera transform
        sample_radius: Pixel radius for depth sampling around centroid (default: 3)

    Returns:
        Tuple (obj2cam_T, obj2base_T): both are 4x4 numpy arrays, or (None, None)
        if no valid depth was found at the centroid.
    """
    distance = depth_cam_info.get_depth(centroid, sample_radius=sample_radius)
    if distance is None:
        return None, None

    x_norm = (centroid[0] - depth_cam_info.cx) / depth_cam_info.fx
    y_norm = (centroid[1] - depth_cam_info.cy) / depth_cam_info.fy

    obj2cam_T = np.eye(4)
    obj2cam_T[0:3, 3] = [distance * x_norm, distance * y_norm, distance]

    cam_T = robot_transforms.get_cam_T(depth_cam_info)
    obj2base_T = cam_T @ obj2cam_T

    return obj2cam_T, obj2base_T
