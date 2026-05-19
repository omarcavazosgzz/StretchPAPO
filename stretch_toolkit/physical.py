"""Physical robot implementation - requires stretch_body."""
import stretch_body.robot as rb
from stretch_body.robot_params import RobotParams
from stretch_body import gamepad_joints
from .base import JointController, DepthCamInfo, CamInfo
from .get_cam_feeds import (
    get_head_rgb_frame, get_head_depth_frame,
    get_wrist_rgb_frame, get_wrist_depth_frame,
    get_wide_cam_frames
)
import numpy as np


class PhysicalJointController(JointController):
    """Controls robot joints using normalized velocities (-1.0 to 1.0) via gamepad Command objects."""
    
    def __init__(self, robot=None, collision_mgmt=True):
        """Initialize controller with Command objects.
        
        Args:
            robot: Robot instance (creates one if None)
            collision_mgmt: Enable collision management
        """
        super().__init__()
        if robot is None:
            self.robot = rb.Robot()
            self.robot.startup()
            self._owns_robot = True
        else:
            self.robot = robot
            self._owns_robot = False
            
        if collision_mgmt:
            self.robot.enable_collision_mgmt()
        
        # Detect end-of-arm tool
        self.end_of_arm_tool = RobotParams().get_params()[1]['robot']['tool']
        
        # Create Command objects
        self.commands = {
            'base_forward': gamepad_joints.CommandBase(),
            'base_counterclockwise': gamepad_joints.CommandBase(),  # Same object
            'lift_up': gamepad_joints.CommandLift(),
            'arm_out': gamepad_joints.CommandArm(),
            'wrist_yaw_counterclockwise': gamepad_joints.CommandWristYaw(),
        }
        
        # Add dexwrist commands if available
        if self._using_dexwrist():
            self.commands['wrist_pitch_up'] = gamepad_joints.CommandWristPitch()
            self.commands['wrist_roll_counterclockwise'] = gamepad_joints.CommandWristRoll()
        
        # Add gripper if available
        if self._using_stretch_gripper():
            self.commands['gripper_open'] = gamepad_joints.CommandGripperPosition()
        
        # Add head commands
        self.commands['head_pan_counterclockwise'] = gamepad_joints.CommandHeadPan()
        self.commands['head_tilt_up'] = gamepad_joints.CommandHeadTilt()
        
        # Base uses same command object for both translation and rotation
        self.commands['base_counterclockwise'] = self.commands['base_forward']
        
        self._i = 0  # Iteration counter for stop_motion throttling
    
    def _using_stretch_gripper(self):
        return self.end_of_arm_tool in ['tool_stretch_dex_wrist', 'eoa_wrist_dw3_tool_sg3', 'tool_stretch_gripper']
    
    def _using_dexwrist(self):
        return self.end_of_arm_tool in ['tool_stretch_dex_wrist', 'eoa_wrist_dw3_tool_sg3', 
                                         'eoa_wrist_dw3_tool_nil', 'eoa_wrist_dw3_tool_tablet_12in']
    
    def set_velocities(self, vel_dict):
        """Set normalized joint velocities.
        
        Args:
            vel_dict: Dict with keys:
                - base_forward: -1.0 to 1.0
                - base_counterclockwise: -1.0 to 1.0
                - lift_up: -1.0 to 1.0
                - arm_out: -1.0 to 1.0
                - wrist_roll_counterclockwise: -1.0 to 1.0
                - wrist_pitch_up: -1.0 to 1.0
                - wrist_yaw_counterclockwise: -1.0 to 1.0
                - head_pan_counterclockwise: -1.0 to 1.0
                - head_tilt_up: -1.0 to 1.0
                - gripper_open: -1.0 to 1.0
        """
        self._i += 1
        
        # Base motion (needs both x and y)
        base_x = vel_dict.get('base_counterclockwise', 0.0)
        base_y = vel_dict.get('base_forward', 0.0)
        if base_x != 0.0 or base_y != 0.0:
            self.commands['base_forward'].command_stick_to_motion(base_x, base_y, self.robot)
        else:
            self.commands['base_forward'].stop_motion(self.robot)
        
        # Lift
        lift_vel = vel_dict.get('lift_up', 0.0)
        if lift_vel != 0.0:
            self.commands['lift_up'].command_stick_to_motion(lift_vel, self.robot)
        else:
            self.commands['lift_up'].stop_motion(self.robot)
        
        # Arm
        arm_vel = vel_dict.get('arm_out', 0.0)
        if arm_vel != 0.0:
            self.commands['arm_out'].command_stick_to_motion(arm_vel, self.robot)
        else:
            self.commands['arm_out'].stop_motion(self.robot)
        
        # Wrist Yaw
        yaw_vel = vel_dict.get('wrist_yaw_counterclockwise', 0.0)
        if yaw_vel != 0.0:
            self.commands['wrist_yaw_counterclockwise'].command_stick_to_motion(yaw_vel, self.robot)
        else:
            if self._i % 3 == 0:  # Throttle stop commands for Dynamixels
                self.commands['wrist_yaw_counterclockwise'].stop_motion(self.robot)
        
        # Wrist Pitch (if available)
        if 'wrist_pitch_up' in self.commands:
            pitch_vel = vel_dict.get('wrist_pitch_up', 0.0)
            if pitch_vel != 0.0:
                self.commands['wrist_pitch_up'].command_stick_to_motion(pitch_vel, self.robot)
            else:
                if self._i % 3 == 0:
                    self.commands['wrist_pitch_up'].stop_motion(self.robot)
        
        # Wrist Roll (if available)
        if 'wrist_roll_counterclockwise' in self.commands:
            roll_vel = vel_dict.get('wrist_roll_counterclockwise', 0.0)
            if roll_vel != 0.0:
                self.commands['wrist_roll_counterclockwise'].command_stick_to_motion(roll_vel, self.robot)
            else:
                if self._i % 3 == 0:
                    self.commands['wrist_roll_counterclockwise'].stop_motion(self.robot)
        
        # Head Pan
        pan_vel = vel_dict.get('head_pan_counterclockwise', 0.0)
        if pan_vel != 0.0:
            self.commands['head_pan_counterclockwise'].command_stick_to_motion(pan_vel, self.robot)
        else:
            if self._i % 3 == 0:
                self.commands['head_pan_counterclockwise'].stop_motion(self.robot)
        
        # Head Tilt
        tilt_vel = vel_dict.get('head_tilt_up', 0.0)
        if tilt_vel != 0.0:
            self.commands['head_tilt_up'].command_stick_to_motion(tilt_vel, self.robot)
        else:
            if self._i % 3 == 0:
                self.commands['head_tilt_up'].stop_motion(self.robot)
        
        # Gripper (if available)
        if 'gripper_open' in self.commands:
            gripper_vel = vel_dict.get('gripper_open', 0.0)
            if gripper_vel > 0.1:
                self.commands['gripper_open'].open_gripper(self.robot)
            elif gripper_vel < -0.1:
                self.commands['gripper_open'].close_gripper(self.robot)
            else:
                self.commands['gripper_open'].stop_gripper(self.robot)
        
        self.robot.push_command()
    
    def get_state(self):
        """Get current joint positions and base odometry.
        
        Returns:
            dict: Joint positions with keys:
                - base_x: meters
                - base_y: meters
                - base_theta: radians
                - lift_up: meters
                - arm_out: meters
                - wrist_yaw_counterclockwise: radians
                - wrist_pitch_up: radians (if available)
                - wrist_roll_counterclockwise: radians (if available)
                - head_pan_counterclockwise: radians
                - head_tilt_up: radians
                - gripper_open: radians (if available)
        """
        state = {
            # Base odometry
            'base_x': self.robot.base.status['x'],
            'base_y': self.robot.base.status['y'],
            'base_theta': self.robot.base.status['theta'],
            
            # Linear joints
            'lift_up': self.robot.lift.status['pos'],
            'arm_out': self.robot.arm.status['pos'],
            
            # Wrist yaw (always available)
            'wrist_yaw_counterclockwise': self.robot.end_of_arm.motors['wrist_yaw'].status['pos'],
            
            # Head
            'head_pan_counterclockwise': self.robot.head.status['head_pan']['pos'],
            'head_tilt_up': self.robot.head.status['head_tilt']['pos'],
        }
        
        # Add dexwrist joints if available
        if self._using_dexwrist():
            state['wrist_pitch_up'] = self.robot.end_of_arm.motors['wrist_pitch'].status['pos']
            state['wrist_roll_counterclockwise'] = self.robot.end_of_arm.motors['wrist_roll'].status['pos']
        
        # Add gripper if available
        if self._using_stretch_gripper():
            state['gripper_open'] = self.robot.end_of_arm.motors['stretch_gripper'].status['pos']
        
        return state
    
    def stop(self):
        """Stop the robot."""
        if self._owns_robot:
            self.robot.stop()

    def get_lidar_ranges(self):
        """LiDAR is not yet implemented for the physical robot.

        Returns:
            None
        """
        print("[LiDAR] get_lidar_ranges() is not yet implemented for the physical robot.")
        return None


# Camera instances for physical robot
# D435i head camera (rotated 90° CW) - separate RGB and depth cameras
HEAD_RGB_CAMERA = CamInfo(
    name="D435i Head RGB",
    frame_getter=get_head_rgb_frame,
    camera_matrix=np.array([
        [303.07223511, 0.0,         122.78679657],
        [0.0,          303.06060791, 210.94392395],
        [0.0,          0.0,          1.0]
    ]),
    distortion_coeffs=np.array([0., 0., 0., 0., 0.]),
    distortion_model="inverse_brown_conrady"
)

HEAD_DEPTH_CAMERA = CamInfo(
    name="D435i Head Depth",
    frame_getter=get_head_depth_frame,
    camera_matrix=np.array([
        [214.76873779, 0.0,         120.41242218],
        [0.0,          214.76873779, 209.7878418],
        [0.0,          0.0,          1.0]
    ]),
    distortion_coeffs=np.array([0., 0., 0., 0., 0.]),
    distortion_model="brown_conrady"
)

HEAD_CAMERA = DepthCamInfo(
    name="D435i Head",
    rgb_cam=HEAD_RGB_CAMERA,
    depth_cam=HEAD_DEPTH_CAMERA,
    depth_scale=1e-03
)

# D405 wrist camera (no rotation) - separate RGB and depth cameras
WRIST_RGB_CAMERA = CamInfo(
    name="D405 Wrist RGB",
    frame_getter=get_wrist_rgb_frame,
    camera_matrix=np.array([
        [385.62329102, 0.0,         314.58789062],
        [0.0,          385.1807251,  243.30551147],
        [0.0,          0.0,          1.0]
    ]),
    distortion_coeffs=np.array([-5.52569292e-02, 5.98766357e-02, -8.58005136e-04,
                                 -9.32277253e-05, -1.93387289e-02]),
    distortion_model="inverse_brown_conrady"
)

WRIST_DEPTH_CAMERA = CamInfo(
    name="D405 Wrist Depth",
    frame_getter=get_wrist_depth_frame,
    camera_matrix=np.array([
        [378.52832031, 0.0,         318.47045898],
        [0.0,          378.52832031, 241.03790283],
        [0.0,          0.0,          1.0]
    ]),
    distortion_coeffs=np.array([0., 0., 0., 0., 0.]),
    distortion_model="brown_conrady"
)

WRIST_CAMERA = DepthCamInfo(
    name="D405 Wrist",
    rgb_cam=WRIST_RGB_CAMERA,
    depth_cam=WRIST_DEPTH_CAMERA,
    depth_scale=1e-04
)

# OV9782 navigation camera (RGB-only, wide-angle)
# Intrinsics not yet calibrated - will be added when available
NAVIGATION_CAMERA = CamInfo(
    name="OV9782 Navigation",
    frame_getter=get_wide_cam_frames,
    # camera_matrix=None,  # TODO: Add intrinsics after calibration
)
