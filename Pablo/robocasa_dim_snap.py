"""
Prueba la atenuacion de luces en RoboCasa y reporta estadisticas RGB + color
dominante en head y wrist. Sirve para confirmar si STRETCH_DIM_LIGHTS arregla
la sobreexposicion.

Uso:
    STRETCH_DIM_LIGHTS=0.3 uv run Pablo/robocasa_dim_snap.py
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ["STRETCH_SIM_HEADLESS"] = "1"
os.environ["STRETCH_SIM_CAMERAS"] = "cam_d435i_rgb,cam_d435i_depth,cam_d405_rgb,cam_d405_depth"

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
        if tilt is not None and abs(tilt - st["head_tilt_up"]) > 0.02:
            cmd["head_tilt_up"] = float(np.clip(3.0 * (tilt - st["head_tilt_up"]), -1, 1))
        if pan is not None and abs(pan - st["head_pan_counterclockwise"]) > 0.02:
            cmd["head_pan_counterclockwise"] = float(np.clip(3.0 * (pan - st["head_pan_counterclockwise"]), -1, 1))
        if not cmd:
            break
        controller.set_velocities(cmd)
        time.sleep(1 / 30)
    controller.set_velocities({})
    time.sleep(0.2)


def stats(f):
    return f"min={f.reshape(-1,3).min(0)} max={f.reshape(-1,3).max(0)} mean={f.reshape(-1,3).mean(0).round(0)}"


def dominant_color(frame_bgr, s_min=70, v_lo=40, v_hi=235, min_area=40):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]; v = hsv[:, :, 2]
    mask = ((s >= s_min) & (v >= v_lo) & (v <= v_hi)).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if cv2.contourArea(c) >= min_area]
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    M = cv2.moments(c)
    cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
    hue = int(np.median(hsv[max(0, cy-3):cy+3, max(0, cx-3):cx+3, 0]))
    return {"area": float(cv2.contourArea(c)), "centroid": (cx, cy), "hue": hue}


def main():
    dim = os.getenv("STRETCH_DIM_LIGHTS", "(off)")
    original = CONFIG.read_text()
    cfg = json.loads(original)
    cfg.setdefault("robocasa", {})["enabled"] = True
    CONFIG.write_text(json.dumps(cfg, indent=2))
    try:
        from stretch_toolkit import controller, HEAD_RGB_CAMERA, WRIST_RGB_CAMERA

        print(f"[dim] STRETCH_DIM_LIGHTS={dim}. arrancando RoboCasa (~40s)...", flush=True)
        controller.get_state()
        for _ in range(80):
            if HEAD_RGB_CAMERA.get_frame() is not None:
                break
            time.sleep(0.2)

        # HEAD: stats + barrido con color dominante
        move_head_to(controller, tilt=-0.5, pan=0.0)
        f = HEAD_RGB_CAMERA.get_frame()
        print(f"[dim] HEAD pan0 tilt-0.5 stats: {stats(f)}", flush=True)
        cv2.imwrite(str(OUT / f"dim_head_{dim}.png"), f)
        for pan in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            move_head_to(controller, tilt=-0.5, pan=pan)
            f = HEAD_RGB_CAMERA.get_frame()
            dc = dominant_color(f)
            print(f"[dim] HEAD pan={pan:+.1f} color={dc}", flush=True)

        # WRIST: baja el lift para mirar la barra y reporta
        t_end = time.time() + 2.0
        while time.time() < t_end:
            controller.set_velocities({"lift_up": -0.5})
            time.sleep(1/30)
        controller.set_velocities({})
        time.sleep(0.3)
        w = WRIST_RGB_CAMERA.get_frame()
        print(f"[dim] WRIST stats: {stats(w)}", flush=True)
        print(f"[dim] WRIST color={dominant_color(w)}", flush=True)
        cv2.imwrite(str(OUT / f"dim_wrist_{dim}.png"), w)

    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        CONFIG.write_text(original)
        print("[dim] config restaurado", flush=True)
        try:
            controller.stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()
