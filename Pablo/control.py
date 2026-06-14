"""Servo por POSICION para juntas del Stretch (head/wrist/arm/lift/gripper).

Evita el bug de control multi-junta del toolkit (move_by relativo se desborda y
cancela con 2+ juntas). Usa sim.move_to_many (posicion absoluta, idempotente).

Uso:
    servo = PosServo(controller.sim, controller, model)
    while ...:
        servo.command({"head_pan_counterclockwise": vp, "head_tilt_up": vt}, dt)

Para la BASE seguir usando controller.set_velocities({base_forward, base_counterclockwise}).
"""
import mujoco

# semantico (get_state) -> actuator (move_to)
SEM2ACT = {
    "head_pan_counterclockwise": "head_pan",
    "head_tilt_up": "head_tilt",
    "wrist_yaw_counterclockwise": "wrist_yaw",
    "wrist_pitch_up": "wrist_pitch",
    "lift_up": "lift",
    "arm_out": "arm",
    "gripper_open": "gripper",
}

MAX_LEAD = 0.25   # cuanto puede adelantarse el target respecto a la posicion real


class PosServo:
    def __init__(self, sim, controller, model):
        self.sim = sim
        self.controller = controller
        self.limits = {}
        for act in set(SEM2ACT.values()):
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act)
            if aid >= 0:
                lo, hi = float(model.actuator_ctrlrange[aid][0]), float(model.actuator_ctrlrange[aid][1])
                self.limits[act] = (lo, hi) if hi > lo else (-1e9, 1e9)
            else:
                self.limits[act] = (-1e9, 1e9)
        self.targets = {}
        self.sync()

    def sync(self):
        """Reinicia los targets a la posicion actual real."""
        s = self.controller.get_state()
        for sem, act in SEM2ACT.items():
            if sem in s:
                self.targets[act] = float(s[sem])

    def command(self, vels: dict, dt: float):
        """Integra velocidades deseadas (semantico -> rad/s o m/s) en targets
        absolutos y los envia. Anti-windup: el target no se aleja mas de MAX_LEAD
        de la posicion real."""
        s = self.controller.get_state()
        active = {}
        for sem, v in vels.items():
            act = SEM2ACT.get(sem)
            if act is None or act not in self.targets:
                continue
            cur = float(s.get(sem, self.targets[act]))
            tgt = self.targets[act] + float(v) * dt
            tgt = min(max(tgt, cur - MAX_LEAD), cur + MAX_LEAD)
            lo, hi = self.limits[act]
            tgt = min(max(tgt, lo), hi)
            self.targets[act] = tgt
            active[act] = tgt
        if active:
            self.sim.move_to_many(active)

    def hold(self):
        """Mantiene los targets actuales (re-envia)."""
        if self.targets:
            self.sim.move_to_many(dict(self.targets))

    def move_to(self, sem_targets: dict, settle=0.0):
        """Fija targets absolutos directos por nombre semantico (no integra)."""
        active = {}
        for sem, pos in sem_targets.items():
            act = SEM2ACT.get(sem)
            if act is None:
                continue
            lo, hi = self.limits[act]
            self.targets[act] = min(max(float(pos), lo), hi)
            active[act] = self.targets[act]
        if active:
            self.sim.move_to_many(active)
