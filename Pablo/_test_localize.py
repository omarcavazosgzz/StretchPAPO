"""Verifica la localizacion 3D POR CAMARA (profundidad de la cabeza) vs el oraculo.
Centra el objeto en la cabeza (Fase 1a), estima su posicion por back-projection de
profundidad, y compara con la verdad-de-tierra."""
import sys
import numpy as np


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "huevo"
    from sim_setup import start_kitchen
    from detection import OracleDetector, resolve_name
    from control import PosServo
    from phase1_lib import aim_head
    from positioning import localize_with_head_camera, remember_object
    from stretch_mujoco.enums.stretch_cameras import StretchCameras

    log = lambda m: print(m, flush=True)
    controller, model = start_kitchen(cameras=("cam_d435i_rgb", "cam_d435i_depth",
                                               "cam_d405_rgb", "cam_nav_rgb"))
    sim = controller.sim
    det = OracleDetector(sim, model)
    servo = PosServo(sim, controller, model)
    body = resolve_name(target)
    HEAD = StretchCameras.cam_d435i_rgb
    HEAD_D = StretchCameras.cam_d435i_depth

    log(f"[loc] centrando '{body}' en la cabeza (Fase 1a)...")
    res = aim_head(controller, det, sim, servo, target, HEAD, body=body, log=log)
    log(f"[loc] cabeza centrada ok={res.get('ok')} error={res.get('error'):.3f}")

    cam_xyz = localize_with_head_camera(sim, det, model, body, HEAD, HEAD_D, log=log)
    oracle = remember_object(sim, body)
    if cam_xyz is None:
        log("[loc] FAIL: no pude localizar por camara"); controller.stop(); return
    err = float(np.linalg.norm(cam_xyz - oracle))
    log(f"[loc] CAMARA(profundidad) = ({cam_xyz[0]:.2f},{cam_xyz[1]:.2f},{cam_xyz[2]:.2f})")
    log(f"[loc] ORACULO(verdad)     = ({oracle[0]:.2f},{oracle[1]:.2f},{oracle[2]:.2f})")
    log(f"[loc] ===== error camara vs oraculo = {err*100:.1f} cm {'OK' if err < 0.12 else 'REVISAR'} =====")
    controller.stop()


if __name__ == "__main__":
    main()
