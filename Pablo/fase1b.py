"""
FASE 1b - Posicionar con altura segura + centrar con la CAMARA DEL BRAZO.

Flujo:
  1) Fase 1a: la cabeza encuentra y centra el objeto.
  2) LOCALIZA por CAMARA: estima la pos 3D por profundidad de la cabeza y la guarda.
  3) Navega a la pose de agarre (paralelo, sin chocar, freno LiDAR).
  4) ALTURA SEGURA: sube el gripper POR ENCIMA del mostrador, extiende por arriba y
     apunta la muneca hacia ABAJO (asi no choca con la barra del mostrador).
  5) Centra el objeto en la camara del BRAZO (servo) y se queda ahi (listo para agarrar).

    uv run Pablo/fase1b.py huevo            # headless
    uv run Pablo/fase1b.py huevo ver        # con ventana
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
    objs = [a for a in args if a not in ("ver", "view", "v")]
    target = objs[0] if objs else "huevo"

    from sim_setup import start_kitchen
    from detection import OracleDetector, resolve_name
    from control import PosServo
    from phase1_lib import aim_head
    from positioning import (localize_with_head_camera, remember_object, compute_grasp_xy,
                             goto_pose, face_arm_at_object, coarse_align_gripper,
                             _base_pose, GRIPPER_HOME_OFFSET)
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
    WRIST = StretchCameras.cam_d405_rgb

    # 1) Fase 1a: centrar en la cabeza
    log(f"[1b] objetivo='{target}'. Fase 1a (cabeza)...")
    aim_head(controller, det, sim, servo, target, HEAD, body=body, log=log)

    # 2) LOCALIZAR por camara (profundidad) y guardar
    obj = localize_with_head_camera(sim, det, model, body, HEAD, HEAD_D, log=log)
    truth = remember_object(sim, body)
    if obj is None:
        log("[1b] no pude localizar por camara; uso oraculo de respaldo"); obj = truth
    log(f"[1b] objeto GUARDADO (por camara): ({obj[0]:.2f},{obj[1]:.2f},{obj[2]:.2f})  "
        f"[verdad: ({truth[0]:.2f},{truth[1]:.2f},{truth[2]:.2f})]")

    # Reset de la base tras la Fase 1a (el control de velocidad de base queda lento
    # despues del centrado de cabeza; lo reseteamos antes de navegar).
    controller.stop(); time.sleep(0.5)
    servo.sync()

    # 3) SUBIR EL BRAZO (RECOGIDO) ANTES de navegar/rotar -> al rotar libra el borde
    #    salido de la barra del mostrador (si rota con el brazo abajo, choca).
    lift_safe = float(np.clip(obj[2] + 0.18, 0.3, 1.05))
    log(f"[1b] subo el brazo a altura segura (lift={lift_safe:.2f}) ANTES de rotar")
    servo.move_to({"wrist_pitch_up": 0.0, "wrist_yaw_counterclockwise": 0.0,
                   "arm_out": 0.0, "lift_up": lift_safe})
    t = time.time()
    while time.time() - t < 5 and (abs(controller.get_state()["lift_up"] - lift_safe) > 0.03
                                   or abs(controller.get_state()["arm_out"]) > 0.03):
        servo.hold(); time.sleep(1/30)

    # 4) Navegar (si lejos) y girar para apuntar el brazo al objeto (ya en alto).
    bx, by, _ = _base_pose(controller)
    if float(np.hypot(obj[0] - bx, obj[1] - by)) > 0.75:
        tx, ty = compute_grasp_xy(obj[:2], (bx, by), grasp_dist=0.55)
        goto_pose(controller, tx, ty, None, log=log)
    face_arm_at_object(controller, obj[:2], log=log)

    # 5) Ya rotado y en alto: extender el brazo por ARRIBA y apuntar la muneca abajo.
    bx, by, _ = _base_pose(controller)
    base_obj = float(np.hypot(obj[0] - bx, obj[1] - by))
    arm_target = float(np.clip(base_obj - GRIPPER_HOME_OFFSET, 0.0, 0.5))
    log(f"[1b] extiendo el brazo por ARRIBA: arm_out={arm_target:.2f}")
    servo.move_to({"arm_out": arm_target})
    t = time.time()
    while time.time() - t < 5 and abs(controller.get_state()["arm_out"] - arm_target) > 0.03:
        servo.hold(); time.sleep(1/30)
    log("[1b] apunto la muneca hacia ABAJO para ver el objeto")
    servo.move_to({"wrist_pitch_up": -1.5})
    t = time.time()
    while time.time() - t < 4 and abs(controller.get_state()["wrist_pitch_up"] - (-1.5)) > 0.05:
        servo.hold(); time.sleep(1/30)
    time.sleep(0.5)

    # 5) ALINEAR el gripper SOBRE el objeto (3D: objeto localizado por camara + pose
    #    de la muneca) -> el objeto entra a la camara del brazo.
    log("[1b] alineando el gripper sobre el objeto...")
    off = coarse_align_gripper(controller, sim, servo, obj[:2], log=log)
    log(f"[1b] offset gripper-objeto final = {off*100:.0f} cm")

    # 6) CENTRAR con la CAMARA DEL BRAZO: llevar el objeto al punto de agarre (entre
    #    las pinzas, abajo-centro del frame de la muneca). arm_out mueve el objeto en
    #    vertical (acerca/aleja); base a lo largo del mostrador para el horizontal.
    TGT = (0.50, 0.62)   # objetivo normalizado (x,y) en la imagen de la muneca
    from positioning import drive_forward, _base_pose as _bp

    def werr():
        d = det.detect(WRIST, body)
        if d is None or not d.in_frame:
            return None, None
        H, W = d.frame_shape
        return np.array([d.centroid[0] / W - TGT[0], d.centroid[1] / H - TGT[1]]), d

    # calibrar signo de d(ey)/d(arm_out) con un nudge corto
    e0, _ = werr()
    if e0 is not None:
        a0 = controller.get_state()["arm_out"]
        an = float(np.clip(a0 + 0.05, 0, 0.5))
        servo.move_to({"arm_out": an})
        t = time.time()
        while time.time() - t < 3 and abs(controller.get_state()["arm_out"] - an) > 0.02:
            servo.hold(); time.sleep(1/30)
        e1, _ = werr()
        g_arm = (e1[1] - e0[1]) / (an - a0) if (e1 is not None and abs(an - a0) > 1e-3) else 4.0
        if abs(g_arm) < 1.0:
            g_arm = 4.0
        log(f"[1b] centrado muneca: d_ey/d_arm={g_arm:+.2f}")
        log("[1b] centrando objeto en la camara del brazo...")
        for it in range(12):
            e, d = werr()
            if e is None:
                log("[1b]   objeto fuera de la muneca; detengo"); break
            em = float(np.hypot(*e))
            log(f"[1b]   it{it} wrist_err={em:.3f} (ex={e[0]:+.3f} ey={e[1]:+.3f})")
            if em < 0.06:
                break
            a = controller.get_state()["arm_out"]
            d_arm = float(np.clip(-0.7 * e[1] / g_arm, -0.08, 0.08))
            servo.move_to({"arm_out": float(np.clip(a + d_arm, 0, 0.5))})
            t = time.time()
            while time.time() - t < 2.5 and abs(controller.get_state()["arm_out"] - (a + d_arm)) > 0.02:
                servo.hold(); time.sleep(1/30)
            if abs(e[0]) > 0.10:                       # horizontal: pequeño avance de base
                drive_forward(controller, float(np.clip(-0.12 * np.sign(e[0]), -0.06, 0.06)))
            time.sleep(0.1)

    # Reporte: gripper por encima del objeto? objeto en la camara del brazo?
    wpos = np.array(sim.pull_status().camera_poses["d405_rgb"]["pos"])
    dW = det.detect(WRIST, body)
    en_brazo = bool(dW and dW.in_frame)
    log(f"[1b] muneca(gripper) z={wpos[2]:.2f} (objeto z={obj[2]:.2f}) -> "
        f"{'POR ENCIMA (ok, no choca)' if wpos[2] > obj[2] + 0.03 else 'BAJO (riesgo)'}")
    log(f"[1b] objeto en camara del BRAZO: {en_brazo}"
        + (f"  centroide={tuple(int(c) for c in dW.centroid)} de {dW.frame_shape}" if en_brazo else ""))

    # Guardar fotos
    allf = sim.pull_camera_data().get_all(use_depth_color_map=False)
    for cam, tag in {WRIST: "1b_wrist", StretchCameras.cam_nav_rgb: "1b_nav", HEAD: "1b_head"}.items():
        f = allf.get(cam)
        if f is None: continue
        f = np.ascontiguousarray(f)
        d = det.detect(cam, body)
        if d and d.in_frame:
            x, y = int(d.centroid[0]), int(d.centroid[1])
            cv2.circle(f, (x, y), 9, (0, 0, 255), 2)
            cv2.drawMarker(f, (f.shape[1]//2, f.shape[0]//2), (0, 255, 0), cv2.MARKER_CROSS, 14, 1)
        cv2.imwrite(str(OUT / f"{tag}.png"), f)
    log("[1b] guardadas snaps/1b_wrist/nav/head.png")

    if view:
        log("[1b] ventana abierta; cierra con X o Ctrl+C.")
        try:
            while sim.is_running(): time.sleep(0.2)
        except KeyboardInterrupt: pass
    controller.stop()


if __name__ == "__main__":
    main()
