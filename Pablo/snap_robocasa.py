"""
Snapshot de RoboCasa desde las camaras del robot, para VER donde estan los
objetos y de que color son. Restaura sim_config.json al terminar.

Uso:
    uv run Pablo/snap_robocasa.py
Salida: Pablo/snaps/rc_*.png
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ["STRETCH_SIM_HEADLESS"] = "1"
os.environ["STRETCH_SIM_CAMERAS"] = "cam_d435i_rgb,cam_d435i_depth,cam_d405_rgb,cam_d405_depth,cam_nav_rgb"

import json
import time
from pathlib import Path
import numpy as np
import cv2

OUT = Path(__file__).resolve().parent / "snaps"
OUT.mkdir(exist_ok=True)
CONFIG = Path(__file__).resolve().parent.parent / "stretch_toolkit" / "sim_config.json"


def move_head_to(controller, tilt=None, pan=None, timeout=8.0):
    t_end = time.time() + timeout
    while time.time() < t_end:
        st = controller.get_state()
        cmd = {}
        if tilt is not None:
            e = tilt - st["head_tilt_up"]
            if abs(e) > 0.02:
                cmd["head_tilt_up"] = float(np.clip(3.0 * e, -1, 1))
        if pan is not None:
            e = pan - st["head_pan_counterclockwise"]
            if abs(e) > 0.02:
                cmd["head_pan_counterclockwise"] = float(np.clip(3.0 * e, -1, 1))
        if not cmd:
            break
        controller.set_velocities(cmd)
        time.sleep(1 / 30)
    controller.set_velocities({})
    time.sleep(0.2)


def main():
    original = CONFIG.read_text()
    cfg = json.loads(original)
    cfg.setdefault("robocasa", {})["enabled"] = True
    CONFIG.write_text(json.dumps(cfg, indent=2))

    try:
        from stretch_toolkit import controller, HEAD_RGB_CAMERA, WRIST_RGB_CAMERA, NAVIGATION_CAMERA

        print("[rcsnap] arrancando RoboCasa (~40s)...", flush=True)
        controller.get_state()
        for _ in range(80):
            if HEAD_RGB_CAMERA.get_frame() is not None:
                break
            time.sleep(0.2)

        # Nav cam (gran angular) para ubicar todo
        nav = NAVIGATION_CAMERA.get_frame()
        if nav is not None:
            cv2.imwrite(str(OUT / "rc_nav.png"), nav)
            print(f"[rcsnap] nav {nav.shape}", flush=True)
        w = WRIST_RGB_CAMERA.get_frame()
        if w is not None:
            cv2.imwrite(str(OUT / "rc_wrist_home.png"), w)
            print(f"[rcsnap] wrist {w.shape}", flush=True)

        # Rejilla de cabeza mirando al frente y a los lados
        for pan in [-0.8, -0.3, 0.0, 0.3, 0.8]:
            for tilt in [0.0, -0.4, -0.8]:
                move_head_to(controller, tilt=tilt, pan=pan)
                f = HEAD_RGB_CAMERA.get_frame()
                if f is None:
                    continue
                st = controller.get_state()
                fname = f"rc_head_p{st['head_pan_counterclockwise']:+.1f}_t{st['head_tilt_up']:+.1f}.png"
                cv2.imwrite(str(OUT / fname), f)
                print(f"[rcsnap] {fname}", flush=True)

    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        CONFIG.write_text(original)
        print("[rcsnap] config restaurado", flush=True)
        try:
            controller.stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()
