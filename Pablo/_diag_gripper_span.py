"""Caracteriza el rango REAL del gripper: separacion de las PUNTAS de goma
(rubber_tip_left/right) a varios valores de gripper_open. Responde: ?puede el
gripper cerrarse lo suficiente para apretar un cubo de 5 cm? No necesita la cocina;
arranca el sim y mueve solo el gripper.

    uv run Pablo/_diag_gripper_span.py
"""
import time
import numpy as np


def main():
    from sim_setup import start_kitchen
    from control import PosServo
    controller, model = start_kitchen(cameras=("cam_d435i_rgb", "cam_d405_rgb"))
    sim = controller.sim
    servo = PosServo(sim, controller, model)
    time.sleep(0.5)

    # poner el brazo en una pose comoda y la muneca horizontal
    servo.move_to({"lift_up": 0.9, "arm_out": 0.2, "wrist_pitch_up": 0.0,
                   "wrist_yaw_counterclockwise": 0.0})
    t = time.time()
    while time.time() - t < 3:
        servo.hold(); time.sleep(1/30)

    def tips():
        cp = sim.pull_status().camera_poses
        l = cp.get("rubber_tip_left"); r = cp.get("rubber_tip_right")
        gc = cp.get("link_grasp_center")
        if l is None or r is None:
            return None
        L = np.array(l["pos"]); R = np.array(r["pos"])
        sep = float(np.linalg.norm(L - R))
        return L, R, sep, (np.array(gc["pos"]) if gc else None)

    for g in (0.5, 0.2, 0.0, -0.1, -0.2, -0.35):
        servo.move_to({"gripper_open": g})
        t = time.time()
        while time.time() - t < 2.0:
            servo.hold(); time.sleep(1/30)
        st = controller.get_state().get("gripper_open")
        info = tips()
        if info is None:
            print(f"gripper_open cmd={g:+.2f} estado={st:+.3f}  (sin rubber_tip en status)", flush=True)
            continue
        L, R, sep, gc = info
        gcz = f"{gc[2]:.3f}" if gc is not None else "NA"
        print(f"gripper_open cmd={g:+.2f} estado={st:+.3f}  SEPARACION_PUNTAS={sep*100:.1f}cm  "
              f"tipL_z={L[2]:.3f} tipR_z={R[2]:.3f} grasp_center_z={gcz}", flush=True)

    print("\n[i] El cubo mide 5.0 cm. Para apretarlo, la separacion CERRADA debe ser < 5 cm.", flush=True)
    controller.stop()


if __name__ == "__main__":
    main()
