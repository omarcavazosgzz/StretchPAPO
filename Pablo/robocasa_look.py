"""
Mirada panoramica dentro de la cocina RoboCasa actual (sin cambiar de escenario).
Barre el pan de la cabeza y, en cada vista, busca el blob MAS SATURADO
(ignorando el blanco sobreexpuesto) para ubicar objetos reales y conocer su
color/hue verdadero. Guarda frames anotados.

Uso:
    uv run Pablo/robocasa_look.py
Salida: Pablo/snaps/look_*.png
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


def dominant_color(frame_bgr, s_min=70, v_lo=40, v_hi=235, min_area=40):
    """Blob mas saturado, ignorando blanco (S bajo) y zonas quemadas (V alto)."""
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
    original = CONFIG.read_text()
    cfg = json.loads(original)
    cfg.setdefault("robocasa", {})["enabled"] = True
    CONFIG.write_text(json.dumps(cfg, indent=2))
    try:
        from stretch_toolkit import controller, HEAD_RGB_CAMERA

        print("[look] arrancando RoboCasa (~40s)...", flush=True)
        controller.get_state()
        for _ in range(80):
            if HEAD_RGB_CAMERA.get_frame() is not None:
                break
            time.sleep(0.2)

        # Posicion de los objetos (ground-truth) solo para referencia
        for name in controller.list_scene_objects():
            p = controller.get_object_pose(name)
            if p:
                print(f"[look] obj {name}: x={p['x']:.2f} y={p['y']:.2f} z={p['z']:.2f}", flush=True)
        bx, by, bt = controller.get_state()["base_x"], controller.get_state()["base_y"], controller.get_state()["base_theta"]
        print(f"[look] base odom x={bx:.2f} y={by:.2f} theta={bt:.2f}", flush=True)

        for pan in [-1.8, -1.4, -1.0, -0.6, -0.2, 0.2, 0.6, 1.0, 1.4, 1.8]:
            move_head_to(controller, tilt=-0.55, pan=pan)
            f = HEAD_RGB_CAMERA.get_frame()
            if f is None:
                continue
            st = controller.get_state()
            dc = dominant_color(f)
            fname = f"look_p{st['head_pan_counterclockwise']:+.1f}.png"
            vis = f.copy()
            if dc:
                cv2.circle(vis, dc["centroid"], 7, (0, 255, 0), 2)
                cv2.putText(vis, f"hue={dc['hue']} a={dc['area']:.0f}", (5, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imwrite(str(OUT / fname), vis)
            msg = f"hue={dc['hue']} area={dc['area']:.0f} c={dc['centroid']}" if dc else "sin color saturado"
            print(f"[look] pan={st['head_pan_counterclockwise']:+.2f} -> {msg}  [{fname}]", flush=True)

    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        CONFIG.write_text(original)
        print("[look] config restaurado", flush=True)
        try:
            controller.stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()
