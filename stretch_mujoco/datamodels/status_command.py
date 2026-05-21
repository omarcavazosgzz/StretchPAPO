"""
Dataclasses that communicate movement commands to Mujoco.
"""

import copy
from dataclasses import asdict, dataclass, field

from stretch_mujoco.enums.actuators import Actuators
from stretch_mujoco.utils import dataclass_from_dict


@dataclass
class CommandMove:
    actuator_name: str
    trigger: bool
    pos: float


@dataclass
class CommandBaseVelocity:
    v_linear: float
    omega: float
    trigger: bool


@dataclass
class CommandKeyframe:
    name: str
    trigger: bool


@dataclass
class CommandCoordinateFrameArrowsViz:
    position: tuple[float, float, float]
    rotation: tuple[float, float, float]
    trigger: bool


@dataclass
class CommandCameraManagement:
    """Command to add or remove cameras dynamically."""
    camera_name: str  # StretchCameras enum name as string
    action: str  # "add" or "remove"
    trigger: bool


@dataclass
class CommandObjectPose:
    """Teleport a freejoint body to an exact world pose."""
    body_name: str
    position: tuple[float, float, float]
    quat: tuple[float, float, float, float]  # (qw, qx, qy, qz)
    trigger: bool


@dataclass
class CommandObjectMoveBy:
    """Move a freejoint body by a relative offset from its current position."""
    body_name: str
    delta: tuple[float, float, float, float, float, float]  # (dx, dy, dz, droll, dpitch, dyaw)
    z_min: float
    trigger: bool


@dataclass
class CommandObjectGravity:
    """Enable or disable gravity compensation for a freejoint body."""
    body_name: str
    enabled: bool   # True = gravity on, False = gravity off (gravcomp=1)
    trigger: bool


@dataclass
class StatusCommand:
    """
    A dataclass to ferry movement commands to the Mujoco server.
    """

    move_to: dict[str, CommandMove] = field(default_factory=dict)
    move_by: dict[str, CommandMove] = field(default_factory=dict)
    base_velocity: CommandBaseVelocity = field(default_factory=lambda:CommandBaseVelocity(0, 0, False))
    keyframe: CommandKeyframe = field(default_factory=lambda:CommandKeyframe("", False))
    coordinate_frame_arrows_viz: list[CommandCoordinateFrameArrowsViz] = field(default_factory=list)
    camera_management: CommandCameraManagement | None = None
    object_pose: CommandObjectPose | None = None
    object_move_by: CommandObjectMoveBy | None = None
    object_gravity: CommandObjectGravity | None = None



    def set_move_to(self, command: CommandMove):
        """Sends a move_to command and removes the move_by command."""
        self.move_to[command.actuator_name] = command

        self.move_by.pop(command.actuator_name, None)

    def set_move_by(self, command: CommandMove):
        """Sends a move_by command and removes the move_to command."""
        self.move_by[command.actuator_name] = command

        self.move_to.pop(command.actuator_name, None)

    def set_base_velocity(self, command: CommandBaseVelocity):
        """Sends the velocity command and removes the move_to and move_by commands."""
        self.base_velocity = command

        for actuator in [
            Actuators.left_wheel_vel,
            Actuators.right_wheel_vel,
            Actuators.base_rotate,
            Actuators.base_translate,
        ]:
            self.move_to.pop(actuator.name, None)
            self.move_by.pop(actuator.name, None)

    def to_dict(self):
        return asdict(self)

    def copy(self):
        return StatusCommand.from_dict(copy.copy(self.to_dict()))

    @staticmethod
    def from_dict(dict_data: dict) -> "StatusCommand":
        command: StatusCommand = dataclass_from_dict(StatusCommand, dict_data)  # type: ignore

        command.move_to = {
            key: dataclass_from_dict(CommandMove, val) for key, val in command.move_to.items()  # type: ignore
        }
        command.move_by = {
            key: dataclass_from_dict(CommandMove, val) for key, val in command.move_by.items()  # type: ignore
        }
        if command.object_pose is not None and isinstance(command.object_pose, dict):
            command.object_pose = dataclass_from_dict(CommandObjectPose, command.object_pose)
        if command.object_move_by is not None and isinstance(command.object_move_by, dict):
            command.object_move_by = dataclass_from_dict(CommandObjectMoveBy, command.object_move_by)
        if command.object_gravity is not None and isinstance(command.object_gravity, dict):
            command.object_gravity = dataclass_from_dict(CommandObjectGravity, command.object_gravity)
        return command

    @staticmethod
    def default():
        """
        Returns an empty instance with None or zeros for properties.
        """
        return StatusCommand()
