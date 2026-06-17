"""Graba un VIDEO del agarre lateral: corre el pipeline real y captura, en un hilo de
fondo, las vistas nav + cabeza + muneca lado a lado -> mp4.

    STRETCH_FIXED_SPAWN="2.25,-0.8,90" uv run Pablo/_record_grasp.py cubo_rojo
Salida: Pablo/snaps/agarre_lateral.mp4
"""
import sys
import time
import threading
from pathlib import Path
import numpy as np
import cv2

OUT = Path(__file__).resolve().parent / "snaps"
OUT.mkdir(exist_ok=True)
FONT = cv2.FONT_HERSHEY_SIMPLEX
TILE = {"nav": (480, 360), "cabeza": (270, 360), "muneca": (480, 360)}


def main():
    args = [a.lower() for a in sys.argv[1:]]
    target = next((a for a in args if a not in ("ver", "view", "v", "top", "lateral")), "cubo_rojo")

    from sim_setup import start_kitchen
    from detection import OracleDetector, resolve_name
    from control import PosServo
    from grasp_lib import grasp_with_retries
    from stretch_mujoco.enums.stretch_cameras import StretchCameras

    log = lambda m: print(m, flush=True)
    controller, model = start_kitchen(cameras=("cam_d435i_rgb", "cam_d435i_depth",
                                               "cam_d405_rgb", "cam_d405_depth", "cam_nav_rgb"),
                                      headless=True)
    sim = controller.sim
    det = OracleDetector(sim, model)
    servo = PosServo(sim, controller, model)
    body = resolve_name(target)
    HEAD, HEAD_D = StretchCameras.cam_d435i_rgb, StretchCameras.cam_d435i_depth
    WRIST, WRIST_D = StretchCameras.cam_d405_rgb, StretchCameras.cam_d405_depth
    NAV = StretchCameras.cam_nav_rgb
    cams = [(NAV, "nav"), (HEAD, "cabeza"), (WRIST, "muneca")]

    frames = []
    state = {"rec": True, "phase": "inicio"}

    def grab():
        while state["rec"]:
            try:
                data = sim.pull_camera_data().get_all(use_depth_color_map=False)
            except Exception:
                time.sleep(0.05); continue
            tiles = []
            for cam, label in cams:
                f = data.get(cam)
                w, h = TILE[label]
                if f is None:
                    tiles.append(np.zeros((h, w, 3), np.uint8)); continue
                f = np.ascontiguousarray(f)
                try:
                    d = det.detect(cam, body)
                    if d and d.in_frame:
                        cv2.circle(f, (int(d.centroid[0]), int(d.centroid[1])), 10, (0, 0, 255), 2)
                except Exception:
                    pass
                f = cv2.resize(f, (w, h))
                cv2.rectangle(f, (0, 0), (w - 1, h - 1), (60, 60, 60), 1)
                cv2.putText(f, label, (8, 26), FONT, 0.8, (0, 255, 0), 2)
                tiles.append(f)
            frame = np.hstack(tiles)
            bar = np.zeros((34, frame.shape[1], 3), np.uint8)
            cv2.putText(bar, f"Stretch agarre LATERAL  |  objeto={target}  |  {state['phase']}",
                        (8, 24), FONT, 0.6, (255, 255, 255), 1)
            frames.append(np.vstack([bar, frame]))
            time.sleep(1 / 12.0)

    th = threading.Thread(target=grab, daemon=True); th.start()
    time.sleep(0.5)

    log(f"[rec] objetivo='{target}'")
    state["phase"] = "Localizar por camara + agarre lateral (verifica firmeza, reintenta)"
    obj, res = grasp_with_retries(controller, sim, det, model, servo, body,
                                  HEAD, HEAD_D, WRIST, WRIST_D, method="lateral",
                                  retries=3, log=log)
    state["phase"] = "AGARRADO OK" if res else "REVISAR"
    time.sleep(1.5)
    state["rec"] = False; th.join(timeout=3)

    if frames:
        h, w = frames[0].shape[:2]
        path = str(OUT / "agarre_lateral.mp4")
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 12, (w, h))
        for fr in frames:
            vw.write(fr)
        vw.release()
        log(f"[rec] VIDEO guardado: {path}  ({len(frames)} frames, {len(frames)/12:.0f}s)  resultado={state['phase']}")
    else:
        log("[rec] sin frames!")
    controller.stop()


if __name__ == "__main__":
    main()
