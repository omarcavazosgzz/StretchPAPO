"""
Diagnostico visual: arranca el sim (entorno de bloques, headless+EGL) y guarda
a disco lo que ven las camaras, en varias poses de cabeza. Sirve para INSPECCIONAR
(yo leo los PNG) el color real (orden de canales), que es visible y el efecto de
la rotacion 90 de la camara head.

Uso:
    uv run Pablo/snap.py
Salida: Pablo/snaps/*.png
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ["STRETCH_SIM_HEADLESS"] = "1"
os.environ["STRETCH_SIM_CAMERAS"] = "cam_d435i_rgb,cam_d435i_depth,cam_d405_rgb,cam_d405_depth"

import time
from pathlib import Path
import numpy as np
import cv2

OUT = Path(__file__).resolve().parent / "snaps"
OUT.mkdir(exist_ok=True)


def save_rgb(cam, name):
    f = cam.get_frame()
    if f is None:
        print(f"[snap] {name}: None", flush=True)
        return None
    cv2.imwrite(str(OUT / f"{name}.png"), f)
    # Tambien guardo una version con los canales invertidos para comparar BGR/RGB
    cv2.imwrite(str(OUT / f"{name}_swapped.png"), f[:, :, ::-1])
    print(f"[snap] {name}: shape={f.shape} guardado", flush=True)
    return f


def save_depth(cam, name):
    d = cam.get_frame()
    if d is None:
        print(f"[snap] {name}: None", flush=True)
        return
    vis = cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
    cv2.imwrite(str(OUT / f"{name}.png"), vis)
    print(f"[snap] {name}: shape={d.shape} guardado", flush=True)


def main():
    from stretch_toolkit import (
        controller, HEAD_RGB_CAMERA, WRIST_RGB_CAMERA,
        HEAD_DEPTH_CAMERA, WRIST_DEPTH_CAMERA,
    )

    print("[snap] arrancando...", flush=True)
    st = controller.get_state()
    print(f"[snap] objetos: {controller.list_scene_objects()}", flush=True)

    # Espera a que las camaras entreguen frame
    for _ in range(50):
        if HEAD_RGB_CAMERA.get_frame() is not None and WRIST_RGB_CAMERA.get_frame() is not None:
            break
        time.sleep(0.1)

    # Pose 0: inicial (home)
    save_rgb(HEAD_RGB_CAMERA, "00_head_home")
    save_rgb(WRIST_RGB_CAMERA, "00_wrist_home")
    save_depth(HEAD_DEPTH_CAMERA, "00_head_depth_home")

    # Tilt de cabeza hacia abajo para mirar la mesa
    print("[snap] inclinando cabeza hacia abajo...", flush=True)
    t_end = time.time() + 2.5
    while time.time() < t_end:
        controller.set_velocities({"head_tilt_up": -0.6})
        time.sleep(1/30)
    controller.set_velocities({})
    time.sleep(0.4)
    st = controller.get_state()
    print(f"[snap] head_tilt={st['head_tilt_up']:.2f} head_pan={st['head_pan_counterclockwise']:.2f}", flush=True)
    save_rgb(HEAD_RGB_CAMERA, "01_head_tiltdown")
    save_depth(HEAD_DEPTH_CAMERA, "01_head_depth_tiltdown")

    # Baja un poco el brazo/lift y mira con la wrist
    print("[snap] bajando lift y mirando con wrist...", flush=True)
    t_end = time.time() + 2.0
    while time.time() < t_end:
        controller.set_velocities({"lift_up": -0.5})
        time.sleep(1/30)
    controller.set_velocities({})
    time.sleep(0.4)
    save_rgb(WRIST_RGB_CAMERA, "02_wrist_liftdown")
    save_depth(WRIST_DEPTH_CAMERA, "02_wrist_depth")

    try:
        controller.stop()
    except Exception:
        pass
    print("[snap] listo. Revisa Pablo/snaps/", flush=True)


if __name__ == "__main__":
    main()
