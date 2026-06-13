"""
Calibracion de las convenciones de la base (entorno de bloques, rapido).
Pulsa comandos y mide como responde la odometria, para arreglar navigation.py.

Uso:
    uv run Pablo/set_env.py block
    uv run Pablo/test_nav_calib.py
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ["STRETCH_SIM_HEADLESS"] = "1"
os.environ["STRETCH_SIM_CAMERAS"] = "cam_d405_rgb"

import time
import numpy as np


def rest(controller, secs=0.6):
    t = time.time() + secs
    while time.time() < t:
        controller.set_velocities({})
        time.sleep(1 / 30)


def pulse(controller, cmd, secs=1.2):
    rest(controller)
    a = controller.get_state()
    t = time.time() + secs
    while time.time() < t:
        controller.set_velocities(cmd)
        time.sleep(1 / 30)
    controller.set_velocities({})
    rest(controller)
    b = controller.get_state()
    return a, b


def main():
    from stretch_toolkit import controller
    print("[cal] arrancando (bloques)...", flush=True)
    controller.get_state()
    time.sleep(0.3)

    # FORWARD
    a, b = pulse(controller, {"base_forward": 0.5})
    dx, dy = b["base_x"] - a["base_x"], b["base_y"] - a["base_y"]
    th = a["base_theta"]
    motion_dir = np.arctan2(dy, dx)
    diff = (motion_dir - th + np.pi) % (2 * np.pi) - np.pi
    print(f"[cal] +base_forward: d=({dx:+.3f},{dy:+.3f}) |d|={np.hypot(dx,dy):.3f} "
          f"theta={th:+.2f} dir_mov={motion_dir:+.2f} (dir-theta)={diff:+.2f} "
          f"-> {'ADELANTE alineado' if abs(diff)<0.5 else 'INVERTIDO (~pi)' if abs(abs(diff)-np.pi)<0.6 else 'lateral?'}", flush=True)

    # ROTATION
    a, b = pulse(controller, {"base_counterclockwise": 0.5})
    dth = (b["base_theta"] - a["base_theta"] + np.pi) % (2 * np.pi) - np.pi
    print(f"[cal] +base_counterclockwise: dtheta={dth:+.3f} "
          f"-> {'CCW (+theta) OK' if dth > 0 else 'CW (theta baja) INVERTIDO'}", flush=True)

    try:
        controller.stop()
    except Exception:
        pass
    print("[cal] fin", flush=True)


if __name__ == "__main__":
    main()
