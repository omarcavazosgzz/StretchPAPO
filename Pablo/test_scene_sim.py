"""Verifica la escena custom en el SIM REAL: bootea la cocina con los objetos
garantizados, deja asentar la fisica, apunta la cabeza al mostrador y guarda lo
que ven las camaras del robot (nav, head, wrist). Confirma que los objetos
inyectados sobreviven la fisica y son visibles.

    uv run Pablo/test_scene_sim.py
Salida: Pablo/snaps/sim_{nav,head,wrist}.png
"""
import time
from pathlib import Path
import numpy as np
import cv2

OUT = Path(__file__).resolve().parent / "snaps"
OUT.mkdir(exist_ok=True)


def main():
    from scene import build_pablo_kitchen, DEFAULT_OBJECTS
    from stretch_mujoco import StretchMujocoSimulator
    from stretch_mujoco.enums.stretch_cameras import StretchCameras

    t0 = time.time()
    print("[sim] generando cocina custom + objetos...", flush=True)
    model, xml, info = build_pablo_kitchen()

    cams = [StretchCameras.cam_d435i_rgb, StretchCameras.cam_d405_rgb, StretchCameras.cam_nav_rgb]
    sim = StretchMujocoSimulator(model=model, cameras_to_use=cams)
    sim.start(headless=True)
    print(f"[sim] arrancado en {time.time()-t0:.1f}s; dejando asentar fisica 2s...", flush=True)
    time.sleep(2.0)

    # Apuntar la cabeza hacia la derecha (hacia los objetos) y abajo al mostrador
    sim.move_to("head_pan", -0.9)
    sim.move_to("head_tilt", -0.5)
    time.sleep(1.5)

    # Reportar poses de los objetos inyectados (ground-truth del sim = "oraculo")
    print("[sim] poses de objetos inyectados (mundo):", flush=True)
    for label, *_ in DEFAULT_OBJECTS:
        p = sim.get_object_pose(label)
        if p:
            print(f"    {label:10s} x={p['x']:.2f} y={p['y']:.2f} z={p['z']:.2f}", flush=True)
        else:
            print(f"    {label:10s} -> NO ENCONTRADO", flush=True)

    # Guardar frames
    deadline = time.time() + 10
    while time.time() < deadline:
        data = sim.pull_camera_data()
        allf = data.get_all(use_depth_color_map=False)
        if all(allf.get(c) is not None for c in cams):
            break
        time.sleep(0.1)
    data = sim.pull_camera_data()
    allf = data.get_all(use_depth_color_map=False)
    tags = {StretchCameras.cam_nav_rgb: "nav", StretchCameras.cam_d435i_rgb: "head", StretchCameras.cam_d405_rgb: "wrist"}
    for c, tag in tags.items():
        f = allf.get(c)
        if f is not None:
            cv2.imwrite(str(OUT / f"sim_{tag}.png"), f)
            print(f"[sim] guardado sim_{tag}.png shape={f.shape}", flush=True)
        else:
            print(f"[sim] FALTA {tag}", flush=True)

    print(f"\n[sim] ===== DONE ({time.time()-t0:.1f}s) =====", flush=True)
    sim.stop()


if __name__ == "__main__":
    main()
