"""
Prueba de navegacion de base en RoboCasa: acercarse al objeto monitoreando
LiDAR (requisito "que no choque"). Tambien vuelca la estructura del LiDAR para
calibrar cual indice es el frente.

Uso:
    uv run Pablo/test_nav.py
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ["STRETCH_SIM_HEADLESS"] = "1"
os.environ["STRETCH_SIM_CAMERAS"] = "cam_d405_rgb"

import json
import time
from pathlib import Path
import numpy as np

import navigation as nav

CONFIG = Path(__file__).resolve().parent.parent / "stretch_toolkit" / "sim_config.json"


def main():
    original = CONFIG.read_text()
    cfg = json.loads(original)
    cfg.setdefault("robocasa", {})["enabled"] = True
    CONFIG.write_text(json.dumps(cfg, indent=2))
    try:
        from stretch_toolkit import controller

        print("[nav] arrancando RoboCasa (~40s)...", flush=True)
        st = controller.get_state()
        time.sleep(0.3)

        # objetivo: android_lego (target fiable)
        obj = controller.get_object_pose("android_lego")
        target = (obj["x"], obj["y"])
        st = controller.get_state()
        print(f"[nav] base inicio=({st['base_x']:.2f},{st['base_y']:.2f},th={st['base_theta']:.2f}) "
              f"-> objetivo=({target[0]:.2f},{target[1]:.2f})", flush=True)

        # Volcado de LiDAR para calibrar el frente
        r = controller.get_lidar_ranges()
        if r is not None:
            r = np.asarray(r)
            finite = np.where(np.isfinite(r))[0]
            print(f"[nav] LiDAR n={len(r)} validos={len(finite)} "
                  f"argmin={int(np.nanargmin(np.where(np.isfinite(r), r, np.inf)))} "
                  f"min={np.nanmin(np.where(np.isfinite(r), r, np.inf)):.2f}", flush=True)
            for i in (0, 45, 90, 135, 180, 225, 270, 315):
                print(f"[nav]   ray[{i}]={r[i] if np.isfinite(r[i]) else 'inf'}", flush=True)

        print("[nav] navegando (standoff=0.6)...", flush=True)
        res = nav.go_to_xy(controller, target, standoff=0.6, max_time=25.0,
                           front_index=0, stop_dist=0.3, log=lambda m: print("[nav]" + m, flush=True))
        st = controller.get_state()
        print(f"[nav] base fin=({st['base_x']:.2f},{st['base_y']:.2f},th={st['base_theta']:.2f})", flush=True)
        print(f"[nav] RESULTADO: {res}", flush=True)

    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        CONFIG.write_text(original)
        try:
            controller.set_velocities({})
            controller.stop()
        except Exception:
            pass
        print("[nav] fin", flush=True)


if __name__ == "__main__":
    main()
