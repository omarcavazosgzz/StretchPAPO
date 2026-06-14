"""Rutinas de apuntado de camara para la Fase 1 (reutilizables).

aim_head(): apunta la camara de la CABEZA a un objeto. Dos etapas:
  1) ADQUIRIR: panea con servo en mundo (azimut) hasta tener el objeto en cuadro.
     El pan converge bien aunque el giro sea grande.
  2) CENTRAR FINO: servo en espacio-imagen con signos auto-calibrados (nudge corto
     manteniendo el objeto en cuadro). Robusto a la rotacion portrait de la head.

Devuelve dict con resultado (ok, error_centrado, centroid, frame_shape).
"""
import time
import numpy as np


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _cam_azel(cam_pose, obj_xyz):
    c = np.array(cam_pose["pos"]); R = np.array(cam_pose["xmat"]).reshape(3, 3)
    f = -R[:, 2]
    d = np.array(obj_xyz) - c; d /= (np.linalg.norm(d) + 1e-9)
    az = _wrap(np.arctan2(d[1], d[0]) - np.arctan2(f[1], f[0]))
    el = np.arcsin(np.clip(d[2], -1, 1)) - np.arcsin(np.clip(f[2], -1, 1))
    return az, el


def _bounded_err(d):
    H, W = d.frame_shape
    ex = float(np.clip(d.centroid[0] / W - 0.5, -0.5, 0.5))
    ey = float(np.clip(d.centroid[1] / H - 0.5, -0.5, 0.5))
    return ex, ey


def aim_head(controller, det, sim, servo, target, HEAD, HEAD_MJCF="d435i_camera_rgb",
             body=None, log=print, KP=1.3, DB=0.025, V=0.9, timeout=60):
    body = body or target
    t0 = time.time()
    DT = 1 / 30
    def cmd(vels):
        servo.command(vels, DT)
    def stop():
        pass  # con control por posicion, dejar de comandar mantiene la pose

    def detect():
        return det.detect(HEAD, target)

    def obj_cam():
        st = sim.pull_status()
        o = st.object_poses.get(body)
        cam = st.camera_poses.get(HEAD_MJCF)
        return (o[:3] if o else None), cam

    def head_angles():
        s = controller.get_state()
        return s["head_pan_counterclockwise"], s["head_tilt_up"]

    def err_now(settle=0.0):
        if settle:
            t = time.time()
            while time.time() - t < settle:
                servo.hold(); time.sleep(DT)
        d = detect()
        if d is None:
            return None, None
        return np.array(_bounded_err(d)), d

    def wait_reach(getter, target, tol=0.06, timeout=8.0):
        """Espera (sosteniendo el servo) a que un valor llegue al target. La cabeza
        es lenta, asi que hay que esperar antes de re-medir para no hacer windup."""
        tw = time.time()
        while time.time() - tw < timeout:
            servo.hold()
            if abs(getter() - target) < tol:
                return True
            time.sleep(DT)
        return False

    # ── 0) ORIENTAR LA BASE hacia el objeto + ACERCARSE ──────────────────────
    # RoboCasa randomiza la pose del robot (a veces ~2m del objeto). A esa distancia
    # el objeto cae a ~90deg de la camara y el centrado se acopla/inestabiliza. Por
    # eso: orientar la base -> avanzar (freno LiDAR) -> re-orientar -> ya cerca.
    BASE_YAW_SIGN = -1.0   # +base_counterclockwise gira CW en este toolkit (navigation.py)

    def orient_base(tol=0.12, timeout=10):
        tb = time.time()
        while time.time() - tb < timeout:
            o, _ = obj_cam()
            if o is None:
                return None
            st = controller.get_state()
            herr = _wrap(np.arctan2(o[1] - st["base_y"], o[0] - st["base_x"]) - st["base_theta"])
            if abs(herr) < tol:
                break
            controller.set_velocities({"base_counterclockwise": float(np.clip(BASE_YAW_SIGN * 1.1 * herr, -1, 1))})
            time.sleep(DT)
        controller.set_velocities({"base_counterclockwise": 0.0})
        time.sleep(0.3)
        return True

    log("[head] orientando la base hacia el objeto...")
    if orient_base() is None:
        log(f"[head] objeto '{body}' no existe"); return {"ok": False}

    o, _ = obj_cam()
    st = controller.get_state()
    dist = float(np.hypot(o[0] - st["base_x"], o[1] - st["base_y"]))
    if dist > 0.95:
        from navigation import go_to_xy
        log(f"[head] acercandose al objeto (dist={dist:.2f}m, freno LiDAR)...")
        go_to_xy(controller, (o[0], o[1]), standoff=0.7, max_time=18, log=log)
        orient_base()
        st = controller.get_state()
        log(f"[head] base ahora en ({st['base_x']:.2f},{st['base_y']:.2f}); re-orientada.")

    # ── 1) ADQUIRIR: tilt fijo al mostrador + BARRIDO de pan hasta in_frame ────
    # Sin matematica de azimut (inestable con la cabeza inclinada): barre el pan a
    # velocidad constante y para apenas el oraculo reporta el objeto en cuadro.
    log("[head] adquiriendo (barrido de pan)...")
    servo.sync()
    servo.move_to({"head_tilt_up": -0.55})        # mirar hacia abajo al mostrador
    wait_reach(lambda: head_angles()[1], -0.55, tol=0.05, timeout=4)
    found = False
    for direction in (+1.0, -1.0):                # primero un lado, luego el otro
        tsw = time.time()
        entered = False
        best = 9.0
        while time.time() - tsw < 11:
            d = detect()
            inframe = d is not None and d.in_frame
            if inframe:
                entered = True
                e = float(np.hypot(*_bounded_err(d)))
                # ya en cuadro: barrer LENTO hasta minimizar el error (centrar en pan)
                if e < best - 0.01:
                    best = e
                elif e > best + 0.06:
                    found = True            # paso el minimo -> parar
                    break
                spd = 0.22
            else:
                if entered:                 # estaba en cuadro y se salio -> ya paso
                    break
                spd = 0.7                   # buscando: barrer rapido
            servo.command({"head_pan_counterclockwise": direction * spd}, DT)
            time.sleep(DT)
        servo.sync()                              # detener / sostener pose actual
        if found:
            break
    if not found:
        log("[head] FAIL: no encontre el objeto al barrer"); return {"ok": False}
    log("[head] objeto en cuadro, centrando...")

    # ── 3) CENTRADO FINO: Gauss-Newton DISCRETO (paso + espera), sin integrar
    # velocidad -> sin overshoot con la cabeza lenta. Mapeo PORTRAIT: pan->ey,
    # tilt->ex. Mide la ganancia diagonal una vez (probe corto con espera).
    def goto(p, t, timeout=3.5):
        servo.move_to({"head_pan_counterclockwise": p, "head_tilt_up": t})
        tw = time.time()
        while time.time() - tw < timeout:
            servo.hold()
            pa, ta = head_angles()
            if abs(pa - p) < 0.03 and abs(ta - t) < 0.03:
                break
            time.sleep(DT)
        time.sleep(0.08)
        d = detect()
        return (np.array(_bounded_err(d)) if (d and d.in_frame) else None)

    # CENTRADO FINO: servo visual con Jacobiano completo 2x2 aprendido EN LINEA
    # (Broyden). Maneja el acoplamiento cruzado de la camara portrait (que hacia
    # girar el error) y aprende los signos solo. J = d[ex,ey]/d[pan,tilt].
    log("[head] centrando fino (Broyden visual servo)...")
    J = np.array([[0.0, -1.5], [1.5, 0.0]])   # guess: tilt->ex, pan->ey
    best_em = 9.0; best_pose = head_angles(); em = 9.0
    e_prev = q_prev = None
    for it in range(26):
        d = detect()
        if d is None or not d.in_frame:
            o, cam = obj_cam(); az, _ = _cam_azel(cam, o)
            pan, _t = head_angles(); goto(pan + float(az), -0.55)
            e_prev = q_prev = None; continue
        e = np.array(_bounded_err(d)); em = float(np.hypot(*e))
        if em < best_em:
            best_em, best_pose = em, head_angles()
        log(f"[head]   it{it} error_centrado={em:.3f} (ex={e[0]:+.3f} ey={e[1]:+.3f})")
        if em < 0.05:
            break
        pan, tilt = head_angles()
        q = np.array([pan, tilt])
        if e_prev is not None:                 # actualizacion de Broyden con el ultimo paso
            dq = q - q_prev; de = e - e_prev
            if dq @ dq > 1e-5:
                J = J + np.outer(de - J @ dq, dq) / (dq @ dq)
        try:
            step = -0.6 * np.linalg.solve(J, e)
        except np.linalg.LinAlgError:
            step = -0.6 * (np.linalg.pinv(J) @ e)
        step = np.clip(step, -0.20, 0.20)
        e_prev, q_prev = e.copy(), q.copy()
        goto(pan + float(step[0]), tilt + float(step[1]))
    if best_em < em - 0.01:                     # volver a la mejor pose si oscilo
        goto(best_pose[0], best_pose[1])
        log(f"[head] vuelvo a mejor pose (error={best_em:.3f})")

    d = detect()
    ex, ey = _bounded_err(d) if d else (9, 9)
    e = float(np.hypot(ex, ey))
    return {"ok": bool(d and d.in_frame and e < 0.07), "error": e,
            "centroid": d.centroid if d else None, "frame_shape": d.frame_shape if d else None,
            "in_frame": bool(d and d.in_frame)}
