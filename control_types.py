"""
Real-time demonstration of velocity vs position control.

This script monitors two JSON files and applies their commands to the robot:
- joint_vels.json: Direct velocity control (normalized -1.0 to 1.0)
- joint_pos.json: Position targets (StateController will drive toward them)

Edit the JSON files while the script is running to see the robot respond in real-time.
"""
from stretch_toolkit import controller, BACKEND_NAME, StateController
import json
import time
from pathlib import Path

print(f"\n=== Running on {BACKEND_NAME} backend ===\n")

# File paths
vel_file = Path("joint_vels.json")
pos_file = Path("joint_pos.json")

# Initialize files if they don't exist
if not vel_file.exists():
    vel_file.write_text("{}")
    print(f"Created {vel_file}")

if not pos_file.exists():
    pos_file.write_text("{}")
    print(f"Created {pos_file}")

# Track file modification times
vel_mtime = vel_file.stat().st_mtime
pos_mtime = pos_file.stat().st_mtime

# Current commands
vel_commands = {}
pos_targets = {}
state_controller = None


def load_vel_commands():
    """Load velocity commands from JSON file."""
    global vel_commands, vel_mtime
    try:
        new_mtime = vel_file.stat().st_mtime
        if new_mtime != vel_mtime:
            with open(vel_file, 'r') as f:
                vel_commands = json.load(f)
            vel_mtime = new_mtime
            print(f"Velocity commands updated: {vel_commands}")
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error reading {vel_file}: {e}")


def load_pos_targets():
    """Load position targets from JSON file."""
    global pos_targets, pos_mtime, state_controller
    try:
        new_mtime = pos_file.stat().st_mtime
        if new_mtime != pos_mtime:
            with open(pos_file, 'r') as f:
                new_targets = json.load(f)
            
            # Only recreate StateController if targets changed
            if new_targets != pos_targets:
                pos_targets = new_targets
                if pos_targets:
                    state_controller = StateController(controller, pos_targets)
                    print(f"Position targets updated: {pos_targets}")
                else:
                    state_controller = None
                    print("Position targets cleared")
            
            pos_mtime = new_mtime
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error reading {pos_file}: {e}")


def main():
    print("Monitoring joint_vels.json and joint_pos.json for changes...")
    print("Edit the files to control the robot in real-time.")
    print("\nExamples:")
    print('  joint_vels.json: {"arm_out": 0.5, "lift_up": -0.3}')
    print('  joint_pos.json: {"arm_out": 0.3, "lift_up": 0.8}')
    print("\nPress Ctrl+C to stop\n")
    
    try:
        while True:
            # Check for file updates
            load_vel_commands()
            load_pos_targets()
            
            # Build final command
            final_cmd = {}
            
            # Position control (lower priority)
            if state_controller is not None:
                final_cmd.update(state_controller.get_command())
            
            # Velocity control overrides position control
            final_cmd.update(vel_commands)
            
            # Send command
            controller.set_velocities(final_cmd)
            
            time.sleep(1 / 30)  # 30 Hz
    
    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        controller.set_velocities({})
        controller.stop()
        print("Done.")


if __name__ == "__main__":
    main()
