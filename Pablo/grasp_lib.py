"""
Fase 2 - AGARRE con la camara del brazo (y la profundidad de la muneca).

position_for_grasp(): localiza el objeto POR CAMARA (profundidad de la cabeza),
navega sin chocar, sube el gripper por ENCIMA del mostrador, lo alinea sobre el
objeto y apunta la muneca abajo. Deja el objeto en la camara del brazo.

grasp_object(): con la CAMARA DEL BRAZO hace el centrado FINO (mueve arm_out para
llevar el objeto al punto de agarre), BAJA usando la profundidad de la muneca,
CIERRA el gripper y LEVANTA. Verifica que el objeto subio (agarrado).
"""
import time
import numpy as np

DT = 1 / 30
GRASP_PX = (0.50, 0.60)   # punto de agarre (x,y normalizado) en la imagen de la muneca
FINGER_DEPTH = 0.11       # distancia muneca->objeto (m) a la que cerrar


def _wait_joint(controller, key, target, tol=0.03, timeout=5.0, servo=None):
    t = time.time()
    while time.time() - t < timeout and abs(controller.get_state()[key] - target) > tol:
        if servo:
            servo.hold()
        time.sleep(DT)


def position_for_grasp(controller, sim, det, model, servo, body, HEAD, HEAD_D, WRIST,
                       log=print):
    """Deja el robot posicionado con el objeto en la camara del brazo. Returns
    (obj_world_xyz, objeto_en_wrist:bool)."""
    from phase1_lib import aim_head
    from positioning import (localize_with_head_camera, remember_object, compute_grasp_xy,
                             goto_pose, face_arm_at_object, coarse_align_gripper,
                             _base_pose, GRIPPER_HOME_OFFSET)

    log("[g] Fase 1a (cabeza) para ver el objeto...")
    aim_head(controller, det, sim, servo, body, HEAD, body=body, log=log)
    obj = localize_with_head_camera(sim, det, model, body, HEAD, HEAD_D, log=log)
    truth = remember_object(sim, body)
    if obj is None:
        obj = truth
    log(f"[g] objeto localizado por camara: ({obj[0]:.2f},{obj[1]:.2f},{obj[2]:.2f}) "
        f"[verdad ({truth[0]:.2f},{truth[1]:.2f},{truth[2]:.2f})]")

    controller.stop(); time.sleep(0.4); servo.sync()
    bx, by, _ = _base_pose(controller)
    if float(np.hypot(obj[0] - bx, obj[1] - by)) > 0.75:
        tx, ty = compute_grasp_xy(obj[:2], (bx, by), grasp_dist=0.55)
        goto_pose(controller, tx, ty, None, log=log)
    face_arm_at_object(controller, obj[:2], log=log)

    # altura segura: subir gripper por ENCIMA del mostrador, extender, muneca abajo
    bx, by, _ = _base_pose(controller)
    base_obj = float(np.hypot(obj[0] - bx, obj[1] - by))
    lift_safe = float(np.clip(obj[2] + 0.16, 0.3, 1.05))
    arm0 = float(np.clip(base_obj - GRIPPER_HOME_OFFSET, 0.0, 0.5))
    servo.move_to({"gripper_open": 0.5, "wrist_yaw_counterclockwise": 0.0,
                   "wrist_pitch_up": 0.0, "lift_up": lift_safe})
    _wait_joint(controller, "lift_up", lift_safe, servo=servo)
    servo.move_to({"arm_out": arm0})
    _wait_joint(controller, "arm_out", arm0, servo=servo)
    servo.move_to({"wrist_pitch_up": -1.5})
    _wait_joint(controller, "wrist_pitch_up", -1.5, tol=0.05, timeout=4, servo=servo)
    time.sleep(0.4)

    log("[g] alineando el gripper sobre el objeto...")
    coarse_align_gripper(controller, sim, servo, obj[:2], log=log)
    d = det.detect(WRIST, body)
    en_wrist = bool(d and d.in_frame)
    if not en_wrist:                       # buscar barriendo arm_out (lateral)
        log("[g] no en wrist; busco con arm_out...")
        cur = controller.get_state()["arm_out"]
        for a in np.linspace(max(0.0, cur - 0.18), min(0.5, cur + 0.18), 14):
            servo.move_to({"arm_out": float(a)})
            _wait_joint(controller, "arm_out", float(a), tol=0.02, timeout=2, servo=servo)
            d = det.detect(WRIST, body)
            if d and d.in_frame:
                en_wrist = True; break
    log(f"[g] objeto en camara del brazo: {en_wrist}")
    return obj, en_wrist


def grasp_object(controller, sim, det, model, servo, body, WRIST, WRIST_D, obj, log=print):
    """Centrado fino con la camara del brazo + descenso + cierre + levantar."""
    def wdet():
        d = det.detect(WRIST, body)
        if d is None or not d.in_frame:
            return None, None
        H, W = d.frame_shape
        e = np.array([d.centroid[0] / W - GRASP_PX[0], d.centroid[1] / H - GRASP_PX[1]])
        return e, d

    def wdepth(d):
        depth = sim.pull_camera_data().get_all(use_depth_color_map=False).get(WRIST_D)
        if depth is None or d is None:
            return None
        ix, iy = int(d.centroid[0]), int(d.centroid[1])
        H, W = depth.shape[:2]
        p = depth[max(0, iy-3):min(H, iy+4), max(0, ix-3):min(W, ix+4)].astype(float)
        v = p[(p > 0.02) & np.isfinite(p)]
        return float(np.median(v)) if v.size else None

    # calibrar signo d(ey)/d(arm_out)
    e0, _ = wdet()
    if e0 is None:
        log("[g] FAIL: el objeto no esta en la camara del brazo"); return False
    a0 = controller.get_state()["arm_out"]
    an = float(np.clip(a0 + 0.05, 0, 0.5)); servo.move_to({"arm_out": an}); _wait_joint(controller, "arm_out", an, servo=servo)
    e1, _ = wdet()
    g_arm = (e1[1] - e0[1]) / (an - a0) if (e1 is not None) else 4.0
    if abs(g_arm) < 1.0:
        g_arm = 4.0

    log("[g] centrando con la camara del brazo + descendiendo...")
    for it in range(28):
        e, d = wdet()
        if e is None:
            log("[g]   objeto fuera de la muneca"); break
        dep = wdepth(d)
        em = float(np.hypot(*e))
        log(f"[g]   it{it} wrist_err={em:.3f} (ex={e[0]:+.2f} ey={e[1]:+.2f}) depth={dep if dep else -1:.2f} lift={controller.get_state()['lift_up']:.2f}")
        # centrado lateral con arm_out
        if abs(e[1]) > 0.05:
            a = controller.get_state()["arm_out"]
            da = float(np.clip(-0.6 * e[1] / g_arm, -0.06, 0.06))
            servo.move_to({"arm_out": float(np.clip(a + da, 0, 0.5))})
            _wait_joint(controller, "arm_out", float(np.clip(a + da, 0, 0.5)), tol=0.02, timeout=2, servo=servo)
        # descenso si esta razonablemente centrado y aun alto
        centered = abs(e[0]) < 0.18 and abs(e[1]) < 0.12
        if dep is not None and dep > FINGER_DEPTH and centered:
            lf = controller.get_state()["lift_up"]
            servo.move_to({"lift_up": float(np.clip(lf - 0.03, 0.2, 1.05))})
            _wait_joint(controller, "lift_up", float(np.clip(lf - 0.03, 0.2, 1.05)), tol=0.015, timeout=2, servo=servo)
        if dep is not None and dep <= FINGER_DEPTH and centered:
            log(f"[g]   objeto al alcance (depth={dep:.2f}) y centrado -> cerrar"); break
        time.sleep(0.05)

    # CERRAR el gripper y LEVANTAR
    obj_z_before = sim.pull_status().object_poses[body][2]
    log("[g] cerrando gripper...")
    servo.move_to({"gripper_open": -0.05})
    _wait_joint(controller, "gripper_open", -0.05, tol=0.05, timeout=3, servo=servo)
    time.sleep(0.5)
    log("[g] levantando...")
    lf = controller.get_state()["lift_up"]
    servo.move_to({"lift_up": float(np.clip(lf + 0.18, 0.2, 1.1))})
    _wait_joint(controller, "lift_up", float(np.clip(lf + 0.18, 0.2, 1.1)), tol=0.03, timeout=4, servo=servo)
    time.sleep(0.6)

    obj_z_after = sim.pull_status().object_poses[body][2]
    grabbed = (obj_z_after - obj_z_before) > 0.05
    log(f"[g] objeto z: {obj_z_before:.2f} -> {obj_z_after:.2f}  "
        f"=> {'AGARRADO ✓' if grabbed else 'no subio (fallo)'}")
    return grabbed
