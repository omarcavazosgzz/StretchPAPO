"""Simulated robot implementation - placeholder for simulation backends."""
import time
import threading
import numpy as np
import json
import os
from pathlib import Path
from .base import JointController, CamInfo, DepthCamInfo
from stretch_mujoco import StretchMujocoSimulator
from stretch_mujoco.enums.actuators import Actuators
from stretch_mujoco.enums.stretch_cameras import StretchCameras


class SimulatedJointController(JointController):
    """Controls simulated robot joints using normalized velocities (-1.0 to 1.0)."""
    
    def __init__(self, sim: StretchMujocoSimulator, max_linear_accel: float = 0.15, max_angular_accel: float = 1.78, 
                 config_file: str = 'sim_joint_config.json'):
        """Initialize simulated controller.
        
        Args:
            sim: StretchMujocoSimulator instance to control
            max_linear_accel: Maximum linear acceleration (m/s^2)
            max_angular_accel: Maximum angular acceleration (rad/s^2)
            config_file: JSON file for joint speed/acceleration configuration
        """
        super().__init__()
        self.sim = sim
        
        # Store config file in this script's directory
        script_dir = Path(__file__).parent
        self.config_file = script_dir / config_file
        self.last_mtime = None
        
        # Current velocities for smoothing
        self.current_v_linear = 0.0
        self.current_omega = 0.0
        self.current_joint_vels = {}  # Track current velocity for each joint
        
        # Acceleration limits (m/s^2 and rad/s^2)
        self.max_linear_accel = max_linear_accel  # m/s^2
        self.max_angular_accel = max_angular_accel  # rad/s^2
        
        # Time tracking
        self.last_update_time = None
        
        # Initialize joint speeds and accelerations from file
        self._load_or_create_config()
        
        # Map velocity dict keys to Actuator enums
        self.joint_actuator_map = {
            'lift_up': Actuators.lift,
            'arm_out': Actuators.arm,
            'head_tilt_up': Actuators.head_tilt,
            'head_pan_counterclockwise': Actuators.head_pan,
            'wrist_yaw_counterclockwise': Actuators.wrist_yaw,
            'wrist_pitch_up': Actuators.wrist_pitch,
            'wrist_roll_counterclockwise': Actuators.wrist_roll,
            'gripper_open': Actuators.gripper,
        }

    def _get_default_config(self):
        """Get default joint speed and acceleration configuration."""
        return {
            'joint_max_speeds': {
                'lift_up': 0.2,
                'arm_out': 0.1,
                'head_tilt_up': 0.5,
                'head_pan_counterclockwise': 0.5,
                'wrist_yaw_counterclockwise': 1.0,
                'wrist_pitch_up': 0.05,
                'wrist_roll_counterclockwise': 0.25,
                'gripper_open': 0.07,
                'base_forward': 0.1,
                'base_counterclockwise': 1.77,
            },
            'joint_max_accels': {
                'lift_up': 10,
                'arm_out': 10,
                'head_tilt_up': 1.0,
                'head_pan_counterclockwise': 1.0,
                'wrist_yaw_counterclockwise': 1.0,
                'wrist_pitch_up': 1.0,
                'wrist_roll_counterclockwise': 1.0,
                'gripper_open': 0.35,
                'base_forward': 0.15,
                'base_counterclockwise': 1.78,
            }
        }

    def _load_or_create_config(self):
        """Load configuration from JSON file, creating with defaults if it doesn't exist."""
        if not self.config_file.exists():
            # Create file with defaults
            defaults = self._get_default_config()
            with open(self.config_file, 'w') as f:
                json.dump(defaults, f, indent=2)
            print(f"Created default sim joint config: {self.config_file}")
        
        # Load from file
        self._load_config()

    def _load_config(self):
        """Load configuration from JSON file and update modification time."""
        with open(self.config_file, 'r') as f:
            config = json.load(f)
        
        self.joint_max_speeds = config['joint_max_speeds']
        self.joint_max_accels = config['joint_max_accels']
        
        # Update modification time
        self.last_mtime = os.path.getmtime(self.config_file)

    def _check_and_reload_config(self):
        """Check if config file has been modified and reload if necessary."""
        if not self.config_file.exists():
            return
        
        current_mtime = os.path.getmtime(self.config_file)
        if current_mtime != self.last_mtime:
            print(f"Sim joint config file changed, reloading: {self.config_file}")
            self._load_config()
    
    def _set_base_velocities(self, vel_dict, dt):
        """Set base velocities with acceleration smoothing and unit conversion.
        
        Args:
            vel_dict: Dictionary of normalized velocities
            dt: Time delta since last update (seconds)
        """
        # Get max velocities from config (m/s and rad/s)
        max_linear_vel = abs(self.joint_max_speeds.get('base_forward', 0.1))
        max_angular_vel = abs(self.joint_max_speeds.get('base_counterclockwise', 1.77))
        
        # Sim conversion factors (empirically determined)
        # sim_units = real_units * conversion_factor
        LINEAR_CONVERSION = 15.6  # 4.68 sim units = 0.3 m/s real
        ANGULAR_CONVERSION = 5.0  # Empirically determined
        
        # Calculate target velocities in real-world units
        target_v_linear = vel_dict.get('base_forward', 0.0) * max_linear_vel
        target_omega = vel_dict.get('base_counterclockwise', 0.0) * max_angular_vel
        
        # Get acceleration limits from config (m/s^2 and rad/s^2)
        max_linear_accel = abs(self.joint_max_accels.get('base_forward', 0.15))
        max_angular_accel = abs(self.joint_max_accels.get('base_counterclockwise', 1.78))
        
        # Apply acceleration limits (in real-world units)
        max_linear_delta = max_linear_accel * dt
        max_angular_delta = max_angular_accel * dt
        
        # Ramp linear velocity
        v_linear_diff = target_v_linear - self.current_v_linear
        if abs(v_linear_diff) > max_linear_delta:
            self.current_v_linear += max_linear_delta if v_linear_diff > 0 else -max_linear_delta
        else:
            self.current_v_linear = target_v_linear
        
        # Ramp angular velocity
        omega_diff = target_omega - self.current_omega
        if abs(omega_diff) > max_angular_delta:
            self.current_omega += max_angular_delta if omega_diff > 0 else -max_angular_delta
        else:
            self.current_omega = target_omega

        # Convert to sim units and apply
        sim_v_linear = self.current_v_linear * LINEAR_CONVERSION
        sim_omega = self.current_omega * ANGULAR_CONVERSION
        self.sim.set_base_velocity(sim_v_linear, -sim_omega)
    
    def _set_joint_velocities(self, vel_dict, dt):
        """Set joint velocities via move_by with acceleration smoothing.
        
        Args:
            vel_dict: Dictionary of normalized velocities
            dt: Time delta since last update (seconds)
        """
        for joint_name, max_speed in self.joint_max_speeds.items():
            # Skip base movements - they're handled by _set_base_velocities
            if joint_name in ['base_forward', 'base_counterclockwise']:
                continue
            
            # Get target velocity
            normalized_vel = vel_dict.get(joint_name, 0.0)
            target_vel = normalized_vel * max_speed
            
            # Get current velocity for this joint (initialize if needed)
            if joint_name not in self.current_joint_vels:
                self.current_joint_vels[joint_name] = 0.0
            
            current_vel = self.current_joint_vels[joint_name]
            
            # Apply acceleration limit
            max_accel = self.joint_max_accels[joint_name]
            max_delta = max_accel * dt
            
            vel_diff = target_vel - current_vel
            if abs(vel_diff) > max_delta:
                current_vel += max_delta if vel_diff > 0 else -max_delta
            else:
                current_vel = target_vel
            
            # Store updated velocity
            self.current_joint_vels[joint_name] = current_vel
            
            # Apply movement if velocity is significant
            if abs(current_vel) > 0.001:  # Small deadzone to avoid jitter
                actuator = self.joint_actuator_map[joint_name]
                self.sim.move_by(actuator, current_vel)
    
    def set_velocities(self, vel_dict):
        """Set normalized joint velocities in simulation with acceleration smoothing.
        
        Args:
            vel_dict: Dict mapping joint names to velocities (-1.0 to 1.0)
                     Example: {'base_forward': 0.3, 'base_counterclockwise': 0.1}
        """
        # Check for config file updates
        self._check_and_reload_config()
        
        # Calculate actual time delta
        current_time = time.perf_counter()
        if self.last_update_time is None:
            dt = 1/30  # Default for first call
        else:
            dt = current_time - self.last_update_time
        self.last_update_time = current_time
        
        # Update base velocities
        self._set_base_velocities(vel_dict, dt)
        
        # Update joint positions with acceleration smoothing
        self._set_joint_velocities(vel_dict, dt)
    
    def get_state(self):
        """Get current joint positions and base odometry from simulation.
        
        Returns:
            dict: Joint positions with keys:
                - base_x, base_y, base_theta (odometry)
                - lift_up, arm_out (meters)
                - wrist/head joints (radians)
                - gripper_open (radians)
        """
        status = self.sim.pull_status()
        
        state = {
            # Base odometry
            'base_x': status.base.x,
            'base_y': status.base.y,
            'base_theta': status.base.theta,
            
            # Linear joints
            'lift_up': status.lift.pos,
            'arm_out': status.arm.pos,
            
            # Wrist joints
            'wrist_yaw_counterclockwise': status.wrist_yaw.pos,
            'wrist_pitch_up': status.wrist_pitch.pos,
            'wrist_roll_counterclockwise': status.wrist_roll.pos,
            
            # Head joints
            'head_pan_counterclockwise': status.head_pan.pos,
            'head_tilt_up': status.head_tilt.pos,
            
            # Gripper
            'gripper_open': status.gripper.pos
        }
        return state
    
    def stop(self):
        """Stop the simulated robot."""
        self.sim.set_base_velocity(0.0, 0.0)

    def get_lidar_ranges(self):
        """Get sanitized 360-ray LiDAR range readings from the simulator.

        Returns:
            np.ndarray: 360-element array of distances in metres.
            Invalid/no-hit rays are np.inf.  Returns None on error.
        """
        from stretch_mujoco.enums.stretch_sensors import StretchSensors
        try:
            sensor_data = self.sim.pull_sensor_data()
            ranges = np.asarray(
                sensor_data.get_data(StretchSensors.base_lidar), dtype=float
            ).reshape(-1)

            # MuJoCo returns -1 for rays that exceed the cutoff distance.
            # Also discard NaN and physically impossible near readings.
            range_min = 0.02   # metres
            range_max = 10.0   # matches sensor cutoff in stretch.xml
            invalid = np.isnan(ranges) | (ranges < range_min) | (ranges > range_max)
            ranges[invalid] = np.inf
            return ranges
        except Exception as e:
            print(f"[LiDAR] Error reading sensor data: {e}")
            return None

    def set_object_pose(
        self,
        body_name: str,
        pose: dict,
    ) -> None:
        """Teleport a freejoint object to a new world pose.

        Args:
            body_name: Body name as defined in the scene XML.
            pose: Dict with keys ``x``, ``y``, ``z`` (metres).
                  Rotation is optional — supply either
                  ``qw``/``qx``/``qy``/``qz`` (quaternion) or
                  ``roll``/``pitch``/``yaw`` (radians, intrinsic ZYX).
        """
        self.sim.set_object_pose(body_name, pose)

    def move_object_by(
        self,
        body_name: str,
        delta: dict,
        z_min: float = 0.45,
    ) -> None:
        """Move a freejoint object by a relative offset from its current pose.

        Args:
            body_name: Body name as defined in the scene XML.
            delta: Dict with any of: ``x``, ``y``, ``z`` (metres),
                   ``roll``, ``pitch``, ``yaw`` (radians). Missing keys default to 0.
                   An empty dict is a no-op.
            z_min: Minimum allowed z value (floor clamp). Defaults to 0.45 m.
        """
        if not delta:
            return
        self.sim.move_object_by(body_name, delta, z_min)

    def set_object_gravity(self, body_name: str, enabled: bool | None) -> None:
        """Enable or disable gravity for a freejoint object.

        Args:
            body_name: Body name as defined in the scene XML.
            enabled: True = normal gravity, False = zero gravity (body floats).
                     None is a no-op.
        """
        if enabled is None:
            return
        self.sim.set_object_gravity(body_name, enabled)

    def list_scene_objects(self) -> list:
        """Return names of all freejoint bodies in the active scene XML.

        Parses the scene XML (and any included files) to find bodies that
        have a <freejoint/> child element.

        Returns:
            List of body name strings.
        """
        import xml.etree.ElementTree as ET
        from pathlib import Path
        from stretch_mujoco import utils as mj_utils

        xml_path = Path(self.sim.scene_xml_path or mj_utils.default_scene_xml_path)
        bodies = []

        def _scan(path: Path):
            try:
                tree = ET.parse(path)
            except Exception:
                return
            root = tree.getroot()
            # Follow <include> tags
            for include in root.iter('include'):
                ref = include.get('file', '')
                _scan(path.parent / ref)
            # Collect bodies that contain a freejoint
            for body in root.iter('body'):
                if body.find('freejoint') is not None:
                    name = body.get('name')
                    if name:
                        bodies.append(name)

        _scan(xml_path)
        return bodies


# Camera watchdog system - auto-deregister unused cameras
_camera_last_access = {}  # Track last access time for each camera
CAMERA_TIMEOUT_MS = 1000  # Deregister if not accessed for this time (ms)
_watchdog_thread = None
_watchdog_running = False


def _watchdog_loop():
    """Background thread that monitors and cleans up stale cameras."""
    from . import _sim
    global _watchdog_running, _camera_last_access
    
    while _watchdog_running:
        time.sleep(0.05)  # Check every 50ms
        
        if _sim is None:
            continue
        
        current_time = time.perf_counter()
        
        # Check each tracked camera
        cameras_to_remove = []
        for camera, last_access in list(_camera_last_access.items()):
            time_since_access = (current_time - last_access) * 1000  # Convert to ms
            if time_since_access > CAMERA_TIMEOUT_MS:
                cameras_to_remove.append(camera)
        
        # Deregister stale cameras
        for camera in cameras_to_remove:
            try:
                _sim.remove_camera(camera)
                del _camera_last_access[camera]
            except:
                pass


def _start_watchdog():
    """Start the camera watchdog thread."""
    global _watchdog_thread, _watchdog_running
    
    if _watchdog_thread is None or not _watchdog_thread.is_alive():
        _watchdog_running = True
        _watchdog_thread = threading.Thread(target=_watchdog_loop, daemon=True)
        _watchdog_thread.start()


def _stop_watchdog():
    """Stop the camera watchdog thread."""
    global _watchdog_running
    _watchdog_running = False


def _mark_camera_accessed(camera):
    """Mark a camera as recently accessed."""
    global _camera_last_access
    _camera_last_access[camera] = time.perf_counter()


# Per-camera retry timestamps: if set, don't access camera until this time has passed
_camera_retry_after = {}
CAMERA_RETRY_DELAY_S = 1.5  # Must exceed CAMERA_TIMEOUT_MS / 1000 so watchdog fires first


def _get_sim_camera_frame(camera_enum):
    """Get a camera frame with automatic watchdog-reset retry on None frames.
    
    When a frame is None (camera failed to open), the camera is released from
    the watchdog so it gets deregistered, then re-registered after a cooldown.
    """
    from . import _sim
    if _sim is None:
        return None

    current_time = time.perf_counter()

    # During cooldown: don't access the camera so the watchdog deregisters it
    retry_after = _camera_retry_after.get(camera_enum)
    if retry_after is not None:
        if current_time < retry_after:
            return None
        else:
            del _camera_retry_after[camera_enum]  # Cooldown expired, try again

    # Mark as accessed (keeps watchdog from deregistering a healthy camera)
    _mark_camera_accessed(camera_enum)

    # Auto-register camera if not already active
    if camera_enum not in _sim.get_active_cameras():
        _sim.add_camera(camera_enum)

    try:
        camera_data = _sim.pull_camera_data()
        all_frames = camera_data.get_all(use_depth_color_map=False)
        frame = all_frames.get(camera_enum)

        if frame is None:
            # Trigger watchdog reset: stamp last_access as ancient so watchdog deregisters it
            _camera_last_access[camera_enum] = 0
            _camera_retry_after[camera_enum] = current_time + CAMERA_RETRY_DELAY_S

        return frame
    except:
        return None


# Frame getter functions for simulation cameras
def _get_head_rgb_frame():
    return _get_sim_camera_frame(StretchCameras.cam_d435i_rgb)

def _get_head_depth_frame():
    return _get_sim_camera_frame(StretchCameras.cam_d435i_depth)

def _get_wrist_rgb_frame():
    return _get_sim_camera_frame(StretchCameras.cam_d405_rgb)

def _get_wrist_depth_frame():
    return _get_sim_camera_frame(StretchCameras.cam_d405_depth)

def _get_nav_cam_frame():
    return _get_sim_camera_frame(StretchCameras.cam_nav_rgb)


# Camera instances for simulated robot
# D435i head camera (simulated)
HEAD_CAMERA = DepthCamInfo(
    name="D435i Head (Sim)",
    rgb_cam=CamInfo(
        name="D435i RGB",
        frame_getter=_get_head_rgb_frame,
        camera_matrix=np.array([
            [303.07223511, 0.0,         122.78679657],
            [0.0,          303.06060791, 210.94392395],
            [0.0,          0.0,          1.0]
        ]),
        distortion_coeffs=np.array([0., 0., 0., 0., 0.]),
        distortion_model="inverse_brown_conrady"
    ),
    depth_cam=CamInfo(
        name="D435i Depth",
        frame_getter=_get_head_depth_frame,
        camera_matrix=np.array([
            [214.76873779, 0.0,         120.41242218],
            [0.0,          214.76873779, 209.7878418],
            [0.0,          0.0,          1.0]
        ]),
        distortion_coeffs=np.array([0., 0., 0., 0., 0.]),
        distortion_model="brown_conrady"
    ),
    depth_scale=1.0
)

# D405 wrist camera (simulated)
WRIST_CAMERA = DepthCamInfo(
    name="D405 Wrist (Sim)",
    rgb_cam=CamInfo(
        name="D405 RGB",
        frame_getter=_get_wrist_rgb_frame,
        camera_matrix=np.array([
            [385.62329102, 0.0,         314.58789062],
            [0.0,          385.1807251,  243.30551147],
            [0.0,          0.0,          1.0]
        ]),
        distortion_coeffs=np.array([-5.52569292e-02, 5.98766357e-02, -8.58005136e-04,
                                     -9.32277253e-05, -1.93387289e-02]),
        # distortion_coeffs=np.array([0., 0., 0., 0., 0.]),
        distortion_model="inverse_brown_conrady"
    ),
    depth_cam=CamInfo(
        name="D405 Depth",
        frame_getter=_get_wrist_depth_frame,
        camera_matrix=np.array([
            [378.52832031, 0.0,         318.47045898],
            [0.0,          378.52832031, 241.03790283],
            [0.0,          0.0,          1.0]
        ]),
        distortion_coeffs=np.array([0., 0., 0., 0., 0.]),
        distortion_model="brown_conrady"
    ),
    depth_scale=1.0
)

# OV9782 navigation camera (simulated)
NAVIGATION_CAMERA = CamInfo(
    name="OV9782 Navigation (Sim)",
    frame_getter=_get_nav_cam_frame
)

# Individual camera feed exports
HEAD_RGB_CAMERA = HEAD_CAMERA.rgb_cam
HEAD_DEPTH_CAMERA = HEAD_CAMERA.depth_cam
WRIST_RGB_CAMERA = WRIST_CAMERA.rgb_cam
WRIST_DEPTH_CAMERA = WRIST_CAMERA.depth_cam
