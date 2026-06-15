"""Diagnostico de la LENTITUD de la base: mide la velocidad real de la base
(1) recien arrancado, (2) tras usar PosServo (mover juntas por posicion),
(3) tras controller.stop()+servo.sync(). Asi vemos si PosServo deja la base lenta."""
import time
import numpy as np


def main():
    from sim_setup import start_kitchen
    from control import PosServo
    controller, model = start_kitchen(cameras=("cam_d435i_rgb",))
    sim = controller.sim
    servo = PosServo(sim, controller, model)
    time.sleep(0.5)

    def base_xy():
        s = controller.get_state()
        return np.array([s["base_x"], s["base_y"]])

    def measure(label, secs=1.5):
        p0 = base_xy(); t0 = time.time()
        n = 0
        while time.time() - t0 < secs:
            controller.set_velocities({"base_forward": 1.0, "base_counterclockwise": 0.0})
            time.sleep(1/30); n += 1
        controller.set_velocities({"base_forward": 0.0, "base_counterclockwise": 0.0})
        time.sleep(0.5)
        d = float(np.hypot(*(base_xy() - p0)))
        print(f"[base] {label:28s} avanzo {d:.3f} m en {secs}s -> {d/secs:.3f} m/s ({n} cmds)", flush=True)

    measure("1) base fresca")

    # ejercitar PosServo (mover cabeza por posicion, como en aim_head)
    print("[base] ...moviendo cabeza con PosServo 4s...", flush=True)
    t = time.time()
    while time.time() - t < 4:
        servo.command({"head_pan_counterclockwise": 0.6, "head_tilt_up": -0.3}, 1/30)
        time.sleep(1/30)
    measure("2) tras PosServo (sin reset)")

    controller.stop(); time.sleep(0.4); servo.sync()
    measure("3) tras stop()+sync()")

    # probar tambien comandar base muchas veces seguidas (warm-up)
    for _ in range(10):
        controller.set_velocities({"base_forward": 0.0}); time.sleep(1/30)
    measure("4) tras warm-up de set_velocities")

    controller.stop()


if __name__ == "__main__":
    main()
