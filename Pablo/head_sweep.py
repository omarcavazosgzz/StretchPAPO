"""
Barrido pan x tilt de la camara HEAD para localizar los bloques (que estan al
COSTADO del robot, donde llega el brazo, no enfrente). Guarda frames y reporta
deteccion de azul/rojo.

Uso:
    uv run Pablo/head_sweep.py
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ["STRETCH_SIM_HEADLESS"] = "1"
os.environ["STRETCH_SIM_CAMERAS"] = "cam_d435i_rgb,cam_d435i_depth,cam_d405_rgb,cam_d405_depth"

import time
from pathlib import Path
import numpy as np
import cv2
import vision

OUT = Path(__file__).resolve().parent / "snaps"
OUT.mkdir(exist_ok=True)


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
    time.sleep(0.15)


def main():
    from stretch_toolkit import controller, HEAD_RGB_CAMERA

    print("[sweep] arrancando...", flush=True)
    controller.get_state()
    for _ in range(50):
        if HEAD_RGB_CAMERA.get_frame() is not None:
            break
        time.sleep(0.1)

    blue = vision.blue_target()
    red = vision.red_target()
    hits = []

    for pan in [-1.6, -1.2, 1.2, 1.6, 0.8, -0.8]:
        for tilt in [-0.6, -0.9, -1.2]:
            move_head_to(controller, tilt=tilt, pan=pan)
            st = controller.get_state()
            f = HEAD_RGB_CAMERA.get_frame()
            if f is None:
                continue
            ob = vision.find_object(f, blue, min_area=15)
            orr = vision.find_object(f, red, min_area=15)
            tag = f"p{st['head_pan_counterclockwise']:+.2f}_t{st['head_tilt_up']:+.2f}"
            bstr = f"azul a={ob['area']:.0f} c={ob['centroid']}" if ob else "azul:no"
            rstr = f"rojo a={orr['area']:.0f} c={orr['centroid']}" if orr else "rojo:no"
            if ob or orr:
                vis = f.copy()
                if ob:
                    cv2.circle(vis, ob["centroid"], 6, (0, 255, 0), -1)
                if orr:
                    cv2.circle(vis, orr["centroid"], 6, (0, 255, 255), -1)
                fname = f"hg_{tag}.png"
                cv2.imwrite(str(OUT / fname), vis)
                hits.append((tag, bstr, rstr, fname))
                print(f"[sweep] {tag} -> {bstr} | {rstr}  [HIT {fname}]", flush=True)
            else:
                print(f"[sweep] {tag} -> nada", flush=True)

    print(f"\n[sweep] {len(hits)} poses con deteccion:", flush=True)
    for h in hits:
        print(f"    {h}", flush=True)

    try:
        controller.stop()
    except Exception:
        pass
    print("[sweep] listo.", flush=True)


if __name__ == "__main__":
    main()
