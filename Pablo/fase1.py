"""
FASE 1a - Busqueda + centrado con la camara de la CABEZA.

El robot orienta su base hacia el objeto, se ACERCA (freno LiDAR), barre la cabeza
hasta encontrarlo y lo CENTRA en la camara de la cabeza (deteccion por oraculo).

    uv run Pablo/fase1.py                 # objetivo por defecto: huevo, headless
    uv run Pablo/fase1.py tomate          # otro objeto (huevo/tomate/cubo_rojo/cuchillo/plato)
    uv run Pablo/fase1.py huevo ver       # ABRE LA VENTANA del visor para verlo en vivo

Salida: error de centrado por consola + Pablo/snaps/fase1_head.png (marcador en objeto).
En modo 'ver' la ventana queda abierta al final hasta que cierres con la X o Ctrl+C.
"""
import sys
import time
from pathlib import Path
import cv2

OUT = Path(__file__).resolve().parent / "snaps"
OUT.mkdir(exist_ok=True)


def main():
    args = [a.lower() for a in sys.argv[1:]]
    view = any(a in ("ver", "view", "v", "--ver") for a in args)
    objs = [a for a in args if a not in ("ver", "view", "v", "--ver")]
    target = objs[0] if objs else "huevo"

    from sim_setup import start_kitchen
    from detection import OracleDetector, resolve_name
    from phase1_lib import aim_head
    from control import PosServo
    from stretch_mujoco.enums.stretch_cameras import StretchCameras

    print(f"[fase1] objetivo = '{target}'  (modo {'VISOR' if view else 'headless'}). Arrancando cocina...", flush=True)
    controller, model = start_kitchen(cameras=("cam_d435i_rgb", "cam_d405_rgb", "cam_nav_rgb"),
                                      headless=not view)
    HEAD = StretchCameras.cam_d435i_rgb
    det = OracleDetector(controller.sim, model)
    servo = PosServo(controller.sim, controller, model)

    res = aim_head(controller, det, controller.sim, servo, target, HEAD,
                   body=resolve_name(target), log=lambda m: print(m, flush=True))

    # imagenes anotadas: cabeza (con marcador) + nav (vista general)
    allf = controller.sim.pull_camera_data().get_all(use_depth_color_map=False)
    frame = allf.get(HEAD)
    if frame is not None and res.get("centroid"):
        x, y = int(res["centroid"][0]), int(res["centroid"][1])
        H, W = frame.shape[:2]
        cv2.circle(frame, (x, y), 10, (0, 0, 255), 2)
        cv2.drawMarker(frame, (W // 2, H // 2), (0, 255, 0), cv2.MARKER_CROSS, 16, 1)
        cv2.putText(frame, target, (max(0, x - 20), max(15, y - 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imwrite(str(OUT / "fase1_head.png"), frame)
    nav = allf.get(StretchCameras.cam_nav_rgb)
    if nav is not None:
        cv2.imwrite(str(OUT / "fase1_nav.png"), nav)

    status = "PASS" if res.get("ok") else "REVISAR"
    print(f"\n[fase1] ===== {status} ===== error_final={res.get('error', 9):.3f} "
          f"in_frame={res.get('in_frame')}  (guardado fase1_head.png)", flush=True)

    if view:
        print("[fase1] Ventana abierta: el objeto quedo centrado en la camara de la cabeza.", flush=True)
        print("[fase1] Cierra con la X de la ventana o Ctrl+C aqui.", flush=True)
        try:
            while controller.sim.is_running():
                time.sleep(0.2)
        except KeyboardInterrupt:
            pass
    controller.stop()


if __name__ == "__main__":
    main()
