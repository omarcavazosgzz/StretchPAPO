"""
Verificacion de la Fase 1 en RoboCasa (entorno objetivo real).

Arranca la cocina headless, barre la cabeza buscando objetos coloridos
(cupcake, hot_dog, etc.), elige el mejor blob de color y hace head_servo para
centrarlo. Confirma que el seguimiento por color funciona con objetos reales
de RoboCasa (no solo bloques). Restaura sim_config.json al terminar.

Uso:
    uv run Pablo/test_fase1_robocasa.py
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

import vision
import servo

OUT = Path(__file__).resolve().parent / "snaps"
OUT.mkdir(exist_ok=True)
CONFIG = Path(__file__).resolve().parent.parent / "stretch_toolkit" / "sim_config.json"

# Conjunto de colores a escanear (objetos de cocina suelen ser saturados)
def scan_targets():
    T = vision.ColorTarget
    return {
        "rojo": vision.red_target(),
        "verde": T(np.array([35, 80, 60]), np.array([85, 255, 255]), None, None, (60, 180, 180)),
        "amarillo": T(np.array([20, 90, 90]), np.array([35, 255, 255]), None, None, (27, 200, 200)),
        "naranja": T(np.array([10, 110, 110]), np.array([20, 255, 255]), None, None, (15, 200, 200)),
        "rosa/magenta": T(np.array([140, 70, 70]), np.array([170, 255, 255]), None, None, (155, 180, 180)),
    }


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

    ok = False
    try:
        from stretch_toolkit import controller, HEAD_RGB_CAMERA

        print("[rc1] arrancando RoboCasa headless (~40s)...", flush=True)
        controller.get_state()
        for _ in range(80):
            if HEAD_RGB_CAMERA.get_frame() is not None:
                break
            time.sleep(0.2)

        targets = scan_targets()
        best = None  # (area, color_name, pan, tilt, centroid)

        print("[rc1] barriendo cabeza buscando objetos de color...", flush=True)
        for pan in [-1.5, -0.9, -0.3, 0.3, 0.9, 1.5]:
            for tilt in [-0.3, -0.6, -0.9]:
                move_head_to(controller, tilt=tilt, pan=pan)
                f = HEAD_RGB_CAMERA.get_frame()
                if f is None:
                    continue
                for cname, tgt in targets.items():
                    obj = vision.find_object(f, tgt, min_area=40)
                    if obj and (best is None or obj["area"] > best[0]):
                        best = (obj["area"], cname, pan, tilt, obj["centroid"])
                        vis = f.copy()
                        cv2.circle(vis, obj["centroid"], 6, (0, 255, 0), -1)
                        cv2.imwrite(str(OUT / f"rc_best_{cname}.png"), vis)

        if best is None:
            print("[rc1] no se hallaron objetos de color en el barrido (revisar manualmente)", flush=True)
        else:
            area, cname, pan, tilt, cen = best
            print(f"[rc1] mejor objeto: '{cname}' area={area:.0f} en pan={pan} tilt={tilt} c={cen}", flush=True)
            tgt = targets[cname]
            move_head_to(controller, tilt=tilt, pan=pan)

            # Lazo cerrado: centrar el objeto en la head con head_servo
            errs = []
            t_end = time.time() + 9
            while time.time() < t_end:
                f = HEAD_RGB_CAMERA.get_frame()
                if f is not None:
                    obj = vision.find_object(f, tgt, min_area=40)
                    if obj is not None:
                        cmd, (ex, ey) = servo.head_servo(obj["centroid"], f.shape)
                        controller.set_velocities(cmd)
                        errs.append(servo.error_norm(ex, ey))
                    else:
                        controller.set_velocities({})
                time.sleep(1 / 30)
            controller.set_velocities({})
            if errs:
                e0, e1 = errs[0], float(np.median(errs[-5:]))
                print(f"[rc1] HEAD |err| inicio={e0:.3f} final={e1:.3f} (n={len(errs)})", flush=True)
                ok = e1 < 0.12 and e1 <= e0

    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        CONFIG.write_text(original)
        print("[rc1] sim_config.json restaurado", flush=True)
        try:
            controller.stop()
        except Exception:
            pass

    print(f"\n[rc1] ===== {'PASS' if ok else 'REVISAR'} =====", flush=True)


if __name__ == "__main__":
    main()
