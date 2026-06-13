"""
Experimento de metodo de deteccion en RoboCasa (donde el color no sirve:
fondo blanco + objetos palidos). Prueba:
  1) ArUco (marcadores) en head/wrist/nav.
  2) Deteccion por PROFUNDIDAD: el objeto sobresale del mostrador.
Guarda frames anotados y reporta que funciona.

Uso:
    uv run Pablo/robocasa_detect.py
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


def aruco_detect(frame_bgr):
    if not hasattr(cv2, "aruco"):
        return "cv2.aruco no disponible"
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    found = {}
    for dname in ["DICT_4X4_50", "DICT_5X5_50", "DICT_6X6_250", "DICT_APRILTAG_36h11"]:
        try:
            d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dname))
            corners, ids, _ = cv2.aruco.detectMarkers(gray, d)
            if ids is not None and len(ids):
                found[dname] = ids.ravel().tolist()
        except Exception:
            pass
    return found if found else "sin marcadores"


def depth_protrusion(depth):
    """Encuentra el blob mas cercano a la camara (objeto que sobresale)."""
    d = depth.astype(np.float32)
    valid = d[(d > 0) & np.isfinite(d)]
    if valid.size < 100:
        return None
    near = np.percentile(valid, 5)   # lo mas cercano
    med = np.percentile(valid, 60)   # superficie dominante
    if med - near < 0.01:
        return None
    mask = ((d > 0) & (d < near + 0.4 * (med - near))).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if cv2.contourArea(c) >= 60]
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    M = cv2.moments(c)
    cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
    return {"centroid": (cx, cy), "area": float(cv2.contourArea(c)),
            "near_m": float(near), "surf_m": float(med)}


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


def main():
    original = CONFIG.read_text()
    cfg = json.loads(original)
    cfg.setdefault("robocasa", {})["enabled"] = True
    CONFIG.write_text(json.dumps(cfg, indent=2))
    try:
        from stretch_toolkit import (controller, HEAD_RGB_CAMERA, WRIST_RGB_CAMERA,
                                      HEAD_DEPTH_CAMERA, WRIST_DEPTH_CAMERA, NAVIGATION_CAMERA)
        print("[det] arrancando RoboCasa (~40s)...", flush=True)
        controller.get_state()
        for _ in range(80):
            if WRIST_RGB_CAMERA.get_frame() is not None:
                break
            time.sleep(0.2)

        print(f"[det] cv2.aruco disponible: {hasattr(cv2,'aruco')}", flush=True)

        # En home y con la cabeza mirando abajo
        for label, setup in [("home", None), ("head_down", lambda: move_head_to(controller, tilt=-0.7, pan=0.0))]:
            if setup:
                setup()
            hr = HEAD_RGB_CAMERA.get_frame()
            wr = WRIST_RGB_CAMERA.get_frame()
            nv = NAVIGATION_CAMERA.get_frame()
            hd = HEAD_DEPTH_CAMERA.get_frame()
            wd = WRIST_DEPTH_CAMERA.get_frame()
            print(f"\n[det] === {label} ===", flush=True)
            print(f"[det] ArUco HEAD : {aruco_detect(hr) if hr is not None else 'None'}", flush=True)
            print(f"[det] ArUco WRIST: {aruco_detect(wr) if wr is not None else 'None'}", flush=True)
            print(f"[det] ArUco NAV  : {aruco_detect(nv) if nv is not None else 'None'}", flush=True)
            print(f"[det] depth WRIST: {depth_protrusion(wd) if wd is not None else 'None'}", flush=True)
            print(f"[det] depth HEAD : {depth_protrusion(hd) if hd is not None else 'None'}", flush=True)
            if wd is not None:
                vis = cv2.applyColorMap(cv2.normalize(wd, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U), cv2.COLORMAP_JET)
                dp = depth_protrusion(wd)
                if dp:
                    cv2.circle(vis, dp["centroid"], 7, (255, 255, 255), 2)
                cv2.imwrite(str(OUT / f"det_wrist_depth_{label}.png"), vis)
            if wr is not None:
                cv2.imwrite(str(OUT / f"det_wrist_rgb_{label}.png"), wr)

    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        CONFIG.write_text(original)
        print("\n[det] config restaurado", flush=True)
        try:
            controller.stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()
