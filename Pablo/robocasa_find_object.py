"""
Exploracion: encontrar una pose de brazo/muneca donde la camara del BRAZO
(wrist) vea el mostrador y el objeto de la tarea en RoboCasa. Guarda wrist
RGB + profundidad en varias poses para inspeccionar.

Uso:
    uv run Pablo/robocasa_find_object.py
Salida: Pablo/snaps/find_*.png
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

# velocidad maxima de cada junta (de sim_joint_config) para escalar el control
GAINS = {"lift_up": 3.0, "arm_out": 3.0, "wrist_pitch_up": 5.0,
         "wrist_yaw_counterclockwise": 3.0, "head_tilt_up": 3.0, "head_pan_counterclockwise": 3.0}


def move_joints_to(controller, targets, timeout=8.0, tol=0.02):
    """Lleva varias juntas a posiciones absolutas por control de velocidad."""
    t_end = time.time() + timeout
    while time.time() < t_end:
        st = controller.get_state()
        cmd = {}
        for j, tgt in targets.items():
            e = tgt - st[j]
            if abs(e) > tol:
                cmd[j] = float(np.clip(GAINS.get(j, 3.0) * e, -1, 1))
        if not cmd:
            break
        controller.set_velocities(cmd)
        time.sleep(1 / 30)
    controller.set_velocities({})
    time.sleep(0.25)


def save_views(HEAD_RGB, WRIST_RGB, WRIST_DEPTH, tag):
    wr = WRIST_RGB.get_frame()
    wd = WRIST_DEPTH.get_frame()
    if wr is not None:
        cv2.imwrite(str(OUT / f"find_{tag}_wrgb.png"), wr)
    if wd is not None:
        vis = cv2.applyColorMap(cv2.normalize(wd, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U), cv2.COLORMAP_JET)
        cv2.imwrite(str(OUT / f"find_{tag}_wdepth.png"), vis)
        d = wd.astype(np.float32)
        v = d[(d > 0) & np.isfinite(d)]
        if v.size:
            print(f"[find] {tag}: wrist depth p5={np.percentile(v,5):.2f} p50={np.percentile(v,50):.2f} p95={np.percentile(v,95):.2f} m", flush=True)
    print(f"[find] {tag}: guardado", flush=True)


def main():
    original = CONFIG.read_text()
    cfg = json.loads(original)
    cfg.setdefault("robocasa", {})["enabled"] = True
    CONFIG.write_text(json.dumps(cfg, indent=2))
    try:
        from stretch_toolkit import controller, HEAD_RGB_CAMERA, WRIST_RGB_CAMERA, WRIST_DEPTH_CAMERA

        print("[find] arrancando RoboCasa (~40s)...", flush=True)
        st = controller.get_state()
        for _ in range(80):
            if WRIST_RGB_CAMERA.get_frame() is not None:
                break
            time.sleep(0.2)

        st = controller.get_state()
        print(f"[find] home: lift={st['lift_up']:.2f} arm={st['arm_out']:.2f} "
              f"wpitch={st['wrist_pitch_up']:.2f} wyaw={st['wrist_yaw_counterclockwise']:.2f}", flush=True)
        for name in controller.list_scene_objects():
            p = controller.get_object_pose(name)
            if p and name not in ("base_link", "link_docking_station"):
                print(f"[find] obj {name}: x={p['x']:.2f} y={p['y']:.2f} z={p['z']:.2f}", flush=True)

        save_views(HEAD_RGB_CAMERA, WRIST_RGB_CAMERA, WRIST_DEPTH_CAMERA, "00_home")

        # Mira hacia abajo con la muneca (la camara wrist apunta al mostrador)
        move_joints_to(controller, {"wrist_pitch_up": -1.3})
        save_views(HEAD_RGB_CAMERA, WRIST_RGB_CAMERA, WRIST_DEPTH_CAMERA, "01_wpitch_down")

        # Extiende el brazo sobre el mostrador
        move_joints_to(controller, {"arm_out": st["arm_out"] + 0.3, "wrist_pitch_up": -1.3})
        save_views(HEAD_RGB_CAMERA, WRIST_RGB_CAMERA, WRIST_DEPTH_CAMERA, "02_arm_out")

        # Baja el lift hacia la altura del mostrador
        move_joints_to(controller, {"lift_up": max(0.4, st["lift_up"] - 0.3), "wrist_pitch_up": -1.3})
        save_views(HEAD_RGB_CAMERA, WRIST_RGB_CAMERA, WRIST_DEPTH_CAMERA, "03_lift_down")

        # Sube el lift por si el objeto esta mas alto
        move_joints_to(controller, {"lift_up": st["lift_up"] + 0.2, "wrist_pitch_up": -1.0})
        save_views(HEAD_RGB_CAMERA, WRIST_RGB_CAMERA, WRIST_DEPTH_CAMERA, "04_lift_up")

    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        CONFIG.write_text(original)
        print("[find] config restaurado", flush=True)
        try:
            controller.stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()
