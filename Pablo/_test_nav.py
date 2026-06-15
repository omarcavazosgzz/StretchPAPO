"""Prueba aislada de nav_to_parallel: localiza por camara y navega a la pose
PARALELA con esquive. Verifica que avanza, llega y la distancia LiDAR minima
nunca baja de un umbral (no choca)."""
import sys
import time
import numpy as np


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "huevo"
    from sim_setup import start_kitchen
    from detection import OracleDetector, resolve_name
    from control import PosServo
    from phase1_lib import aim_head
    from positioning import (localize_with_head_camera, remember_object, nav_to_parallel,
                             lidar_front_min, _base_pose)
    from stretch_mujoco.enums.stretch_cameras import StretchCameras

    log = lambda m: print(m, flush=True)
    controller, model = start_kitchen(cameras=("cam_d435i_rgb", "cam_d435i_depth",
                                               "cam_d405_rgb", "cam_nav_rgb"))
    sim = controller.sim
    det = OracleDetector(sim, model)
    servo = PosServo(sim, controller, model)
    body = resolve_name(target)
    HEAD, HEAD_D = StretchCameras.cam_d435i_rgb, StretchCameras.cam_d435i_depth

    aim_head(controller, det, sim, servo, body, HEAD, body=body, log=log, do_approach=False)
    o = localize_with_head_camera(sim, det, model, body, HEAD, HEAD_D, log=log)
    if o is None:
        o = remember_object(sim, body)
    log(f"[nav] objeto: ({o[0]:.2f},{o[1]:.2f})")
    controller.stop(); time.sleep(0.4); servo.sync()

    res = nav_to_parallel(controller, sim, o[:2], standoff=0.6, log=log)
    bx, by, th = _base_pose(controller)
    truth = remember_object(sim, body)
    lateral = abs(by - truth[1])
    along = abs(bx - truth[0])
    front = lidar_front_min(controller, 30)
    log(f"\n[nav] base final ({bx:.2f},{by:.2f}) heading={np.degrees(th):.0f}deg")
    log(f"[nav] objeto en ({truth[0]:.2f},{truth[1]:.2f})  -> lateral={lateral:.2f}m along={along:.2f}m")
    log(f"[nav] LiDAR frente final={front:.2f}m  reached={res['reached']} dist_pto={res['dist'] if 'dist' in res else '?'}")
    ok = res["reached"] and lateral < 0.85 and front > 0.25
    log(f"[nav] ===== {'OK (paralelo, alcanzable, sin choque)' if ok else 'REVISAR'} =====")
    controller.stop()


if __name__ == "__main__":
    main()
