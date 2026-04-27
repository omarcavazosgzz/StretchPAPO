"""
Robot transformation utilities.

Provides a centralized interface for obtaining transformation matrices
from various robot frames to the robot base frame and world frame.
"""

import numpy as np
import math
import urchin as urdf_loader


try:
    import stretch_body.robot as rb
except ImportError:
    rb = None

# Module-level flag to control angle convention behavior
# True: Use legacy angle conventions (for backward compatibility)
# False: Use correct URDF angle conventions (target behavior)
USE_LEGACY_ANGLES = True

class RobotTransforms:
    """Encapsulates all robot-dependent transformation logic.
    
    Provides transformation matrices for robot components based on
    current joint positions and odometry.
    """
    
    def __init__(self, controller):
        """
        Args:
            controller: JointController instance (sim or physical)
        """
        self.controller = controller
        
        # Load URDF model for forward kinematics
        # Based on stretch_mujoco/utils.py
        model_name = "SE3"  # SE3, RE1V0, RE2V0
        tool_name = "eoa_wrist_dw3_tool_sg3"
        try:
            import stretch_urdf
            pkg_path = str(stretch_urdf.__path__[0])
            urdf_file_path = f"{pkg_path}/{model_name}/stretch_description_{model_name}_{tool_name}.urdf"
            self.urdf = urdf_loader.URDF.load(urdf_file_path, lazy_load_meshes=True)
        except ImportError:
            import importlib.resources
            pkg_path = str(importlib.resources.files("stretch_urdf"))
            urdf_file_path = f"{pkg_path}/{model_name}/stretch_description_{model_name}_{tool_name}.urdf"
            self.urdf = urdf_loader.URDF.load(urdf_file_path, lazy_load_meshes=True)
    
    def get_head_cam_T_1(self):
        """
        Transform from head camera frame to robot base frame.
        Considers current head pan and tilt joint positions.
        
        Camera frame: X-right, Y-down, Z-forward
        Robot frame: X-forward, Y-left, Z-up
        
        Returns:
            4x4 transformation matrix from camera frame to base frame
        """
        # Get current head joint positions
        state = self.controller.get_state()
        head_pan = state['head_pan_counterclockwise']
        head_tilt = -state['head_tilt_up']
        
        # Base rotation: camera frame to robot frame (camera facing forward)
        # Camera Z -> Robot X, Camera X -> Robot -Y, Camera Y -> Robot -Z
        R_cam_to_robot = np.array([
            [0,  0,  1],
            [-1, 0,  0],
            [0, -1,  0]
        ])
        
        # Pan rotation (around robot Z-axis)
        cos_pan = np.cos(head_pan)
        sin_pan = np.sin(head_pan)
        R_pan = np.array([
            [cos_pan, -sin_pan, 0],
            [sin_pan,  cos_pan, 0],
            [0,        0,       1]
        ])
        
        # Tilt rotation (around robot Y-axis after pan)
        cos_tilt = np.cos(head_tilt)
        sin_tilt = np.sin(head_tilt)
        R_tilt = np.array([
            [cos_tilt,  0, sin_tilt],
            [0,         1, 0],
            [-sin_tilt, 0, cos_tilt]
        ])
        
        # Combined rotation: apply pan, then tilt, then camera-to-robot transform
        R = R_pan @ R_tilt @ R_cam_to_robot
        
        # Translation: camera is 1.3m above ground (base frame origin)
        t = np.array([0, 0, 1.3])  # [x, y, z] in meters
        
        T = np.eye(4)
        T[0:3, 0:3] = R
        T[0:3, 3] = t
        return T
    
    def _get_joint_config(self):
        """
        Convert controller state to URDF joint configuration.
        Based on stretch_mujoco/utils.py URDFmodel.get_transform()
        """
        state = self.controller.get_state()
        arm_out = state.get('arm_out', 0.0)
        # State values use user-facing convention (positive = up / counterclockwise).
        # URDF joint angles use the opposite sign for these five joints, so negate them.
        return {
            "joint_head_pan":   -state.get('head_pan_counterclockwise', 0.0),
            "joint_head_tilt":  -state.get('head_tilt_up', 0.0),
            "joint_lift": state.get('lift_up', 0.0),
            "joint_arm_l0": arm_out / 4,  # Telescoping segments
            "joint_arm_l1": arm_out / 4,
            "joint_arm_l2": arm_out / 4,
            "joint_arm_l3": arm_out / 4,
            "joint_wrist_yaw":   -state.get('wrist_yaw_counterclockwise', 0.0),
            "joint_wrist_pitch": -state.get('wrist_pitch_up', 0.0),
            "joint_wrist_roll":  -state.get('wrist_roll_counterclockwise', 0.0),
        }
    
    def _urdf_to_legacy_transform(self, urdf_transform):
        """
        Convert URDF transform to match legacy manual transform conventions.
        
        This is a compatibility layer to maintain backwards compatibility with
        existing object tracking code that was tuned for the manual transform.
        
        The legacy system was designed with these assumptions:
        - Camera platform has no roll axis (always ~0°)
        - Head tilt maps to legacy roll 
        - Head pan maps to legacy yaw
        - Legacy pitch is always 0°
        
        Args:
            urdf_transform: 4x4 transform from URDF forward kinematics
            
        Returns:
            4x4 transform adjusted to match legacy conventions
        """
        # Extract URDF Euler angles from the rotation matrix
        R_urdf = urdf_transform[:3, :3]
        sy = math.sqrt(R_urdf[0,0]**2 + R_urdf[1,0]**2)
        
        if sy > 1e-6:
            urdf_roll = math.atan2(R_urdf[2,1], R_urdf[2,2])
            urdf_pitch = math.atan2(-R_urdf[2,0], sy)
            urdf_yaw = math.atan2(R_urdf[1,0], R_urdf[0,0])
        else:
            urdf_roll = math.atan2(-R_urdf[1,2], R_urdf[1,1])
            urdf_pitch = math.atan2(-R_urdf[2,0], sy)
            urdf_yaw = 0
        
        # Convert URDF angles to legacy conventions based on empirical mapping:
        # URDF roll is constant ~90°, URDF pitch maps to legacy roll, URDF yaw maps to legacy yaw
        legacy_roll = -urdf_pitch - math.pi/2  # Negative URDF pitch becomes legacy roll
        legacy_pitch = 0.0                     # Legacy pitch always 0
        legacy_yaw = urdf_yaw - math.pi/2      # URDF yaw becomes legacy yaw with offset
        
        # Build legacy rotation matrix
        cos_r, sin_r = math.cos(legacy_roll), math.sin(legacy_roll)
        cos_p, sin_p = math.cos(legacy_pitch), math.sin(legacy_pitch)
        cos_y, sin_y = math.cos(legacy_yaw), math.sin(legacy_yaw)
        
        # ZYX Euler rotation matrix
        R_legacy = np.array([
            [cos_y*cos_p, cos_y*sin_p*sin_r - sin_y*cos_r, cos_y*sin_p*cos_r + sin_y*sin_r],
            [sin_y*cos_p, sin_y*sin_p*sin_r + cos_y*cos_r, sin_y*sin_p*cos_r - cos_y*sin_r],
            [-sin_p,      cos_p*sin_r,                      cos_p*cos_r]
        ])
        
        # Use URDF position (more accurate than fixed legacy position)
        legacy_transform = np.eye(4)
        legacy_transform[:3, :3] = R_legacy
        legacy_transform[:3, 3] = urdf_transform[:3, 3]  # Keep accurate URDF position
        
        return legacy_transform
    
    def get_head_cam_T(self):
        """
        Transform from head camera frame to robot base frame using URDF.
        
        Returns:
            4x4 transformation matrix from camera frame to base frame
        """
        joint_config = self._get_joint_config()
        
        # Get transform from base_link to head camera link
        # Based on URDF: d435i camera is "camera_link"
        try:
            urdf_transform = self.get_head_cam_T_urdf_raw()
            
            # Apply legacy compatibility layer if flag is set
            if USE_LEGACY_ANGLES:
                return self._urdf_to_legacy_transform(urdf_transform)
            else:
                return urdf_transform
                
        except Exception as e:
            raise ValueError(f"Could not get head camera transform for link 'camera_link': {e}")
    
    def get_head_cam_T_urdf_raw(self):
        """
        Get raw URDF head camera transform without compatibility layer.
        
        Returns:
            4x4 transformation matrix from URDF forward kinematics
        """
        joint_config = self._get_joint_config()
        
        try:
            transform = self.urdf.link_fk(joint_config, link="camera_link")
            return transform
        except Exception as e:
            raise ValueError(f"Could not get raw URDF head camera transform: {e}")
    
    def get_wrist_cam_T(self):
        # TODO: I suspect this needs a slight adjustment due to inconsistencies between URDF model and real robot.
        # We could perhaps fix this by locating an object from both cameras, then comparing the resulting transforms.
        # Then find a transform for wrist that results in matching object locations.
        """
        Transform from wrist camera frame to robot base frame using URDF.
        
        Returns:
            4x4 transformation matrix from camera frame to base frame
        """
        joint_config = self._get_joint_config()
        
        # Get transform from base_link to wrist camera link
        # Based on URDF: d405 camera is "gripper_camera_link"
        try:
            transform = self.urdf.link_fk(joint_config, link="gripper_camera_link")
            if USE_LEGACY_ANGLES:
                return self._urdf_to_legacy_transform(transform)
            return transform
        except Exception as e:
            raise ValueError(f"Could not get wrist camera transform for link 'gripper_camera_link': {e}")
    
    def get_base2world_T(self):
        """
        Build base-to-world transformation from robot odometry.
        
        Uses current robot position (x, y, theta) from odometry to compute
        the transformation from the robot base frame to a fixed world frame.
        
        Returns:
            4x4 transformation matrix from base frame to world frame
        """
        # Get odometry (x, y in meters, theta in radians)
        state = self.controller.get_state()
        x = state.get('base_x', 0.0)
        y = state.get('base_y', 0.0)
        theta = state.get('base_theta', 0.0)
        
        # Build 2D rotation matrix for theta
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        
        # Create 4x4 homogeneous transform
        # World frame: Z-up (same as base frame)
        base2world_T = np.array([
            [cos_t, -sin_t, 0, x],
            [sin_t,  cos_t, 0, y],
            [0,      0,     1, 0],  # Z unchanged (planar motion)
            [0,      0,     0, 1]
        ])
        
        return base2world_T
    
    def get_cam_T(self, depth_cam_info):
        """
        Get camera-to-base transformation for a given camera.
        
        Automatically selects the appropriate transform method based on
        which camera is provided.
        
        Args:
            depth_cam_info: DepthCamInfo instance (HEAD_CAMERA or WRIST_CAMERA)
        
        Returns:
            4x4 transformation matrix from camera frame to base frame
        
        Raises:
            ValueError: If camera is not recognized
        """
        if "Head" in depth_cam_info.name:
            return self.get_head_cam_T()
        elif "Wrist" in depth_cam_info.name:
            return self.get_wrist_cam_T()
        else:
            raise ValueError(f"Unknown camera: {depth_cam_info.name}")

def print_transform_info(name, transform):
    # TODO: what the hell is going on with the angle conventions???
    """Print transformation matrix in human-readable format."""
    print(f"\n{name}:")
    
    # Extract translation (position)
    translation = transform[:3, 3]
    print(f"  Position: [{translation[0]:.3f}, {translation[1]:.3f}, {translation[2]:.3f}] meters")
    
    # Extract rotation matrix and convert to Euler angles
    R = transform[:3, :3]
    
    # Extract Euler angles (roll, pitch, yaw in degrees)
    # Using ZYX convention (yaw-pitch-roll)
    sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
    singular = sy < 1e-6
    
    if not singular:
        roll = math.atan2(R[2,1], R[2,2])
        pitch = math.atan2(-R[2,0], sy)
        yaw = math.atan2(R[1,0], R[0,0])
    else:
        roll = math.atan2(-R[1,2], R[1,1])
        pitch = math.atan2(-R[2,0], sy)
        yaw = 0
    
    # Convert to degrees
    roll_deg = math.degrees(roll)
    pitch_deg = math.degrees(pitch)
    yaw_deg = math.degrees(yaw)
    
    print(f"  Rotation: Roll={roll_deg:.1f}°, Pitch={pitch_deg:.1f}°, Yaw={yaw_deg:.1f}°")

if __name__ == '__main__':
    print("Testing URDF loading...")
    model_name = "SE3"
    tool_name = "eoa_wrist_dw3_tool_sg3"
    try:
        import stretch_urdf
        pkg_path = str(stretch_urdf.__path__[0])
        urdf_file_path = f"{pkg_path}/{model_name}/stretch_description_{model_name}_{tool_name}.urdf"
        print(f"Loading URDF from: {urdf_file_path}")
        urdf = urdf_loader.URDF.load(urdf_file_path, lazy_load_meshes=True)
        print(f"OK - loaded {len(urdf.links)} links")
        print("\nLinks:")
        for link in sorted(urdf.links, key=lambda x: x.name):
            print(f"  {link.name}")
    except Exception as e:
        print(f"FAILED: {e}")

if __name__ == '__main__' and False:
    try:
        # Initialize robot
        print("Initializing robot...")
        robot = rb.Robot()
        robot.startup()
        
        # Create transform manager
        transforms = RobotTransforms(robot)
        
        # Debug: Print all available links
        print("\nAvailable links in URDF:")
        print("-" * 30)
        for link in sorted(transforms.urdf.links, key=lambda x: x.name):
            print(f"  {link.name}")
        print("-" * 30)
        
        print("\nHead camera transform comparison:")
        print("=" * 60)
        print(f"Legacy angle mode: {USE_LEGACY_ANGLES}")
        print("-" * 60)
        
        # Get and display manual head camera transform
        try:
            manual_head_T = transforms.get_head_cam_T_1()
            print_transform_info("Manual Head Camera Transform (Legacy)", manual_head_T)
        except Exception as e:
            print(f"Manual head camera error: {e}")
        
        # Get and display raw URDF head camera transform
        try:
            urdf_raw_T = transforms.get_head_cam_T_urdf_raw()
            print_transform_info("Raw URDF Head Camera Transform", urdf_raw_T)
        except ValueError as e:
            print(f"Raw URDF head camera error: {e}")
        
        # Get and display current head camera transform (legacy or URDF based on flag)
        try:
            current_head_T = transforms.get_head_cam_T()
            mode = "Legacy Compatible" if USE_LEGACY_ANGLES else "Raw URDF"
            print_transform_info(f"Current Head Camera Transform ({mode})", current_head_T)
        except ValueError as e:
            print(f"Current head camera error: {e}")
        
        print("\n" + "=" * 60)
        
    finally:
        robot.stop()