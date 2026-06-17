"""
FASE 2 - AGARRE.

El robot localiza el objeto POR CAMARA, se posiciona sin chocar y, con la CAMARA
DEL BRAZO, centra fino + baja + cierra el gripper + levanta. Verifica el agarre.

    uv run Pablo/fase2.py huevo            # headless
    uv run Pablo/fase2.py huevo ver        # con ventana
"""
import sys
import time
from pathlib import Path
import numpy as np
import cv2

OUT = Path(__file__).resolve().parent / "snaps"
OUT.mkdir(exist_ok=True)


def main():
    args = [a.lower() for a in sys.argv[1:]]
    view = any(a in ("ver", "view", "v") for a in args)
    method = "top" if any(a in ("top", "arriba") for a in args) else "lateral"
    objs = [a for a in args if a not in ("ver", "view", "v", "top", "arriba", "lateral")]
    target = objs[0] if objs else "huevo"

    from sim_setup import start_kitchen
    from detection import OracleDetector, resolve_name
    from control import PosServo
    from grasp_lib import grasp_with_retries
    from stretch_mujoco.enums.stretch_cameras import StretchCameras

    log = lambda m: print(m, flush=True)
    controller, model = start_kitchen(cameras=("cam_d435i_rgb", "cam_d435i_depth",
                                               "cam_d405_rgb", "cam_d405_depth", "cam_nav_rgb"),
                                      headless=not view)
    sim = controller.sim
    det = OracleDetector(sim, model)
    servo = PosServo(sim, controller, model)
    body = resolve_name(target)
    HEAD, HEAD_D = StretchCameras.cam_d435i_rgb, StretchCameras.cam_d435i_depth
    WRIST, WRIST_D = StretchCameras.cam_d405_rgb, StretchCameras.cam_d405_depth

    def snap(tag):
        allf = sim.pull_camera_data().get_all(use_depth_color_map=False)
        for cam, t in {WRIST: f"f2_{tag}_wrist", StretchCameras.cam_nav_rgb: f"f2_{tag}_nav"}.items():
            f = allf.get(cam)
            if f is None: continue
            f = np.ascontiguousarray(f)
            d = det.detect(cam, body)
            if d and d.in_frame:
                cv2.circle(f, (int(d.centroid[0]), int(d.centroid[1])), 9, (0, 0, 255), 2)
            cv2.imwrite(str(OUT / f"{t}.png"), f)

    log(f"[fase2] objetivo='{target}' metodo='{method}'. Agarrando (con reintentos)...")
    obj, ok = grasp_with_retries(controller, sim, det, model, servo, body,
                                 HEAD, HEAD_D, WRIST, WRIST_D, method=method, retries=3, log=log)
    snap("post")
    log(f"\n[fase2] ===== {'AGARRE OK' if ok else 'REVISAR'} =====")

    if view:
        log("[fase2] ventana abierta; cierra con X o Ctrl+C.")
        try:
            while sim.is_running(): time.sleep(0.2)
        except KeyboardInterrupt: pass
    controller.stop()


if __name__ == "__main__":
    main()
