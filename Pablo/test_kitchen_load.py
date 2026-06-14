"""Hito: verifica que una cocina RoboCasa se genera, carga y RENDERIZA camaras
en Windows nativo (sin egl). Self-contained: no pasa por los scripts que fuerzan
MUJOCO_GL=egl. Guarda head+wrist RGB y profundidad para inspeccion.

Uso:
    uv run Pablo/test_kitchen_load.py
Salida: Pablo/snaps/kitchen_*.png
"""
# IMPORTANTE: NO forzar MUJOCO_GL=egl (eso es Linux). En Windows el backend
# default (wgl) funciona. Dejamos que mujoco elija.
import time
from pathlib import Path
import numpy as np
import cv2

OUT = Path(__file__).resolve().parent / "snaps"
OUT.mkdir(exist_ok=True)


def main():
    t0 = time.time()
    print("[kitchen] generando cocina RoboCasa (task=PnPCounterToCab, layout=0, style=0)...", flush=True)
    from stretch_mujoco.robocasa_gen import model_generation_wizard
    from stretch_mujoco import StretchMujocoSimulator
    from stretch_mujoco.enums.stretch_cameras import StretchCameras

    model, xml, info = model_generation_wizard(task="PnPCounterToCab", layout=0, style=0)
    print(f"[kitchen] modelo generado en {time.time()-t0:.1f}s. objetos de tarea:", flush=True)
    for k, v in info.items():
        print(f"    {k}: cat={v['cat']} pos={np.round(v['pos'],2)}", flush=True)

    cams = [StretchCameras.cam_d435i_rgb, StretchCameras.cam_d435i_depth,
            StretchCameras.cam_d405_rgb, StretchCameras.cam_nav_rgb]
    sim = StretchMujocoSimulator(model=model, cameras_to_use=cams)
    sim.start(headless=True)
    print(f"[kitchen] sim arrancado headless en {time.time()-t0:.1f}s", flush=True)

    # esperar primeros frames
    got = {}
    deadline = time.time() + 30
    while time.time() < deadline and len(got) < 3:
        data = sim.pull_camera_data()
        allf = data.get_all(use_depth_color_map=False)
        for c in cams:
            if c not in got and allf.get(c) is not None:
                got[c] = allf[c]
                print(f"[kitchen] frame {c.name} shape={allf[c].shape}", flush=True)
        time.sleep(0.05)

    # guardar
    data = sim.pull_camera_data()
    allf = data.get_all(use_depth_color_map=False)
    for c in cams:
        f = allf.get(c)
        if f is None:
            print(f"[kitchen] FALTA frame {c.name}", flush=True)
            continue
        if "depth" in c.name:
            vis = cv2.applyColorMap(cv2.normalize(f, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U), cv2.COLORMAP_JET)
            cv2.imwrite(str(OUT / f"kitchen_{c.name}.png"), vis)
        else:
            cv2.imwrite(str(OUT / f"kitchen_{c.name}.png"), f)
        print(f"[kitchen] guardado kitchen_{c.name}.png", flush=True)

    ok = len(got) >= 3
    print(f"\n[kitchen] ===== {'PASS' if ok else 'FAIL'} ===== ({len(got)} camaras, {time.time()-t0:.1f}s)", flush=True)
    sim.stop()


if __name__ == "__main__":
    main()
