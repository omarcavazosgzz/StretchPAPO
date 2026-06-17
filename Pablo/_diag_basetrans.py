"""Caracteriza la PRECISION de move_by(base_translate, d): mueve la base distancias
conocidas y mide el desplazamiento real (y el sobrepaso). Decide si sirve para
centrar el eje A-LO-LARGO con precision (~<1.5cm).

    uv run Pablo/_diag_basetrans.py
"""
import time
import numpy as np


def base_xy(controller):
    s = controller.get_state()
    return np.array([s["base_x"], s["base_y"]])


def move_and_measure(sim, controller, dist):
    from stretch_mujoco.enums.actuators import Actuators
    p0 = base_xy(controller)
    sim.move_by(Actuators.base_translate, dist)
    # esperar a que pare (la base recorre la distancia y se detiene)
    prev = None
    t0 = time.time()
    while time.time() - t0 < 8:
        time.sleep(0.15)
        p = base_xy(controller)
        if prev is not None and np.linalg.norm(p - prev) < 0.002:
            break
        prev = p
    p1 = base_xy(controller)
    moved = float(np.linalg.norm(p1 - p0))
    return moved


def main():
    from sim_setup import start_kitchen
    from control import PosServo
    log = lambda m: print(m, flush=True)
    controller, model = start_kitchen(cameras=("cam_d435i_rgb", "cam_d405_rgb"))
    sim = controller.sim
    servo = PosServo(sim, controller, model)
    time.sleep(0.5)
    # brazo recogido y arriba (como en el agarre, gripper alto)
    servo.move_to({"lift_up": 0.95, "arm_out": 0.0, "wrist_pitch_up": 0.0})
    t = time.time()
    while time.time() - t < 2:
        servo.hold(); time.sleep(1/30)

    for d in (0.10, -0.05, 0.03, -0.02, 0.05):
        moved = move_and_measure(sim, controller, d)
        log(f"comando={d:+.3f}m  movido_real={moved:.3f}m  error={(moved-abs(d))*100:+.1f}cm")
        time.sleep(0.3)
    controller.stop()


if __name__ == "__main__":
    main()
