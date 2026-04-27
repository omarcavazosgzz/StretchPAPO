"""State-based position control using P-controllers.

Converts desired joint positions into velocity commands that work with both
physical and simulated backends.
"""
from .base import JointController


class StateController:
    """Position-based controller that generates velocity commands to reach desired states.
    
    Uses proportional control to smoothly drive joints to target positions.
    Works with both physical and simulated JointController backends.
    """
    
    def __init__(self, controller: JointController, desired_state: dict):
        """Initialize state controller.
        
        Args:
            controller: JointController instance (physical or simulated)
            desired_state: Dict of desired joint positions using full toolkit names, e.g.:
                {
                    "arm_out": 0.5,                    # meters
                    "lift_up": 1.1,                    # meters
                    "wrist_yaw_counterclockwise": 0.0, # radians
                    "gripper_open": 1.57,              # radians
                }
        """
        self.controller = controller
        self.desired_state = desired_state
        
        # Individual Kp values for different joint types
        self.Kp = {
            "wrist_roll_counterclockwise": 1.0,   # rad -> normalized velocity
            "wrist_pitch_up": 1.0,                # rad -> normalized velocity
            "wrist_yaw_counterclockwise": 1.0,    # rad -> normalized velocity
            "lift_up": 10.0,                      # m -> normalized velocity
            "arm_out": 5.0,                       # m -> normalized velocity
            "head_pan_counterclockwise": 1.0,     # rad -> normalized velocity
            "head_tilt_up": 1.0,                  # rad -> normalized velocity
            "gripper_open": 5.0                  # rad -> normalized velocity
        }
        
        # Maximum velocity limits (overrides default 1.0)
        self.max_velocity = {
            "lift_up": 0.75,         # Limit lift to 75% max speed
            # Add other joint-specific limits here as needed
        }
        
        # Position tolerance for each joint
        self.tolerance = {
            "wrist_roll_counterclockwise": 0.05,  # rad
            "wrist_pitch_up": 0.05,               # rad
            "wrist_yaw_counterclockwise": 0.05,   # rad
            "lift_up": 0.05,                      # m
            "arm_out": 0.05,                      # m
            "head_pan_counterclockwise": 0.05,    # rad
            "head_tilt_up": 0.05,                 # rad
            "gripper_open": 0.15                  # rad
        }
    
    def get_current_state(self):
        """Get current joint positions from controller.
        
        Returns:
            dict: Current positions using simplified joint names
        """
        full_state = self.controller.get_state()
        return {k: full_state[k] for k in self.desired_state if k in full_state}
    
    def is_at_goal(self):
        """Check if robot is within tolerance of desired state.
        
        Returns:
            bool: True if all joints are within tolerance
        """
        current_state = self.get_current_state()
        for joint, desired_pos in self.desired_state.items():
            if joint in current_state:
                error = abs(current_state[joint] - desired_pos)
                if error > self.tolerance.get(joint, 0.01):
                    return False
        return True
    
    def get_progress(self, previous_state):
        """Calculate progress from previous_state to desired_state.
        
        Args:
            previous_state: Dict of joint positions (simplified names)
        
        Returns:
            dict: Progress (0.0 to 1.0) for each joint
        """
        current_state = self.get_current_state()
        progress = {}
        
        for joint, desired_pos in self.desired_state.items():
            if joint in current_state and joint in previous_state:
                current_pos = current_state[joint]
                prev_pos = previous_state[joint]
                
                total_distance = abs(desired_pos - prev_pos)
                distance_covered = abs(prev_pos - current_pos)
                progress[joint] = distance_covered / total_distance if total_distance > 0 else 1.0
        
        return progress
    
    def get_command(self):
        """Generate velocity commands to reach desired state.
        
        Returns:
            dict: Normalized velocity commands using toolkit naming
        """
        current_state = self.get_current_state()
        command = {}
        
        for joint, desired_pos in self.desired_state.items():
            if joint in current_state:
                # Calculate position error
                error = desired_pos - current_state[joint]
                
                # Set to zero velocity if within tolerance
                if abs(error) <= self.tolerance.get(joint, 0.01):
                    velocity = 0.0
                else:
                    # Calculate proportional velocity using joint-specific Kp
                    kp = self.Kp.get(joint, 1.0)
                    velocity = kp * error
                    
                    # Clamp velocity to joint-specific max (default 1.0)
                    max_vel = self.max_velocity.get(joint, 1.0)
                    velocity = max(-max_vel, min(max_vel, velocity))
                
                command[joint] = velocity
        
        return command
