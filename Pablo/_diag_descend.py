"""PROBE: con el gripper ya SOBRE el cubo (alto), bajar el lift en pasos y registrar
hasta donde DE VERDAD bajan las puntas (rubber_tip) antes de atorarse, y que tan abajo
queda respecto al cubo. Dice si el agarre lateral es geometricamente posible y a que
altura de la cara del cubo llegan las puntas.

    STRETCH_FIXED_SPAWN="2.25,-0.8,90" uv run Pablo/_diag_descend.py cubo_rojo
"""
import sys
import time
import numpy as np


def tip_z(sim, cube_xy):
    cp = sim.pull_status().camera_poses
    zs = []
    for k in ("rubber_tip_left", "rubber_tip_right"):
        for _ in range(8):
            cp = sim.pull_status().camera_poses
            g = cp.get(k)
            if g is not None:
                p = np.array(g["pos"])
                if 0.0 < p[2] < 2.0 and np.linalg.norm(p[:2] - cube_xy) < 1.0:
                    zs.append(p[2]); break
            time.sleep(0.02)
    return (float(np.mean(zs)) if zs else None)


def main():
    args = [a.lower() for a in sys.argv[1:]]
    target = next((a for a in args if a not in ("ver", "view", "v")), "cubo_rojo")

    from sim_setup import start_kitchen
    from detection import OracleDetector, resolve_name
    from control import PosServo
    from grasp_lib import position_for_grasp, _grasp_center, _wait_joint, _wrist_helpers, APPROACH_OVERSHOOT
    from positioning import _base_pose
    from stretch_mujoco.enums.stretch_cameras import StretchCameras

    log = lambda m: print(m, flush=True)
    controller, model = start_kitchen(cameras=("cam_d435i_rgb", "cam_d435i_depth",
                                               "cam_d405_rgb", "cam_d405_depth", "cam_nav_rgb"),
                                      headless=True)
    sim = controller.sim
    det = OracleDetector(sim, model)
    servo = PosServo(sim, controller, model)
    body = resolve_name(target)
    HEAD, HEAD_D = StretchCameras.cam_d435i_rgb, StretchCameras.cam_d435i_depth
    WRIST, WRIST_D = StretchCameras.cam_d405_rgb, StretchCameras.cam_d405_depth

    obj, ok = position_for_grasp(controller, sim, det, model, servo, body,
                                 HEAD, HEAD_D, WRIST, method="lateral", log=log)
    cube_xy = np.array(obj[:2])
    cube = sim.pull_status().object_poses[body]
    log(f"[probe] cubo verdad z={cube[2]:.3f} (top={cube[2]+0.025:.3f} bottom={cube[2]-0.025:.3f})")

    # muneca horizontal + abierto, subir alto
    servo.move_to({"wrist_pitch_up": 0.0, "wrist_yaw_counterclockwise": 0.0,
                   "wrist_roll_counterclockwise": 0.0, "gripper_open": 0.5})
    _wait_joint(controller, "wrist_pitch_up", 0.0, tol=0.05, timeout=4, servo=servo)
    HIGH = obj[2] + 0.06
    for _ in range(20):
        gc = _grasp_center(controller, "lateral"); dz = HIGH - gc[2]
        if abs(dz) < 0.01: break
        lf = controller.get_state()["lift_up"]; nl = float(np.clip(lf+dz,.12,1.05))
        if abs(nl-lf) < .005: break
        servo.move_to({"lift_up": nl}); _wait_joint(controller, "lift_up", nl, tol=.015, timeout=2.5, servo=servo)

    # extender sobre el cubo (radial)
    for _ in range(8):
        gc = _grasp_center(controller, "lateral")
        off = cube_xy - gc[:2]; bx, by, th = _base_pose(controller)
        rad = np.array([np.sin(th), -np.cos(th)]); d_rad = float(off @ rad)
        if d_rad <= -APPROACH_OVERSHOOT: break
        a = controller.get_state()["arm_out"]; na = float(np.clip(a + d_rad + APPROACH_OVERSHOOT, 0, .5))
        if abs(na-a) < .005: break
        servo.move_to({"arm_out": na}); _wait_joint(controller, "arm_out", na, tol=.02, timeout=3, servo=servo)

    log("[probe] DESCENSO paso a paso (lift cmd, lift real, tip_z verdad):")
    prev_lift = None
    for i in range(30):
        lf = controller.get_state()["lift_up"]
        tz = tip_z(sim, cube_xy)
        tzs = f"{tz:.3f}" if tz is not None else "NA"
        rel = f"{(tz-cube[2])*100:+.1f}cm" if tz is not None else "NA"
        log(f"  step{i:02d} lift_real={lf:.3f} tip_z={tzs} (tip - cubo_centro={rel})")
        if prev_lift is not None and abs(lf - prev_lift) < 0.003:
            log(f"  -> LIFT ATORADO en {lf:.3f} (tip_z={tzs}). No baja mas.")
            break
        prev_lift = lf
        new_lf = float(np.clip(lf - 0.02, 0.12, 1.05))   # bajar 2cm decisivo
        servo.move_to({"lift_up": new_lf})
        _wait_joint(controller, "lift_up", new_lf, tol=0.01, timeout=2.0, servo=servo)
        time.sleep(0.1)

    controller.stop()


if __name__ == "__main__":
    main()
