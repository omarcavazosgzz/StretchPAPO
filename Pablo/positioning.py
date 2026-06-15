"""
Posicionamiento del Stretch para agarrar (Fase 1.5).

Idea: recordamos la posicion mundo del objeto (oraculo; en robot real seria por
profundidad de la camara de la cabeza) y calculamos una POSE DE AGARRE comoda:
el Stretch tiene el brazo lateral (se extiende hacia base -Y), asi que la mejor
pose es PARALELO al mostrador, en el pasillo, con el lado del brazo hacia el
objeto, a una distancia donde el brazo alcanza. Luego navegamos ahi por el
pasillo ABIERTO (no hacia el mostrador) con freno LiDAR -> no choca.

Datos calibrados (Pablo/_diag_kin.py):
  - El brazo se extiende hacia base -Y (lateral). Gripper ~0.34 m de la base aun
    retraido; alcanza mas con arm_out.
  - LiDAR: frente = indice 94 (base +x_local). 360 rayos.
"""
import time
import numpy as np

LIDAR_FRONT = 94          # indice del rayo que apunta al frente de la base
GRIPPER_HOME_OFFSET = 0.34
DT = 1 / 30
BASE_YAW_SIGN = -1.0      # +base_counterclockwise gira CW en este toolkit


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def remember_object(sim, body):
    """Verdad-de-tierra del sim (oraculo). Solo para comparar/verificar."""
    o = sim.pull_status().object_poses.get(body)
    return None if o is None else np.array(o[:3])


def localize_with_head_camera(sim, det, model, body, head_cam, head_depth_cam,
                              head_mjcf="d435i_camera_rgb", log=print):
    """ESTIMA la posicion 3D del objeto USANDO LA CAMARA: toma el pixel del objeto
    (detector) en la camara de la cabeza, lee la PROFUNDIDAD en ese pixel y
    back-proyecta a mundo con la pose de la camara. Esto es lo que se "guarda".
    El objeto debe estar visible/centrado en la cabeza (correr Fase 1a antes).

    Returns world (x,y,z) o None.
    """
    from detection import _CamModel
    cm = _CamModel(head_cam, model)             # W,H,f,cx,cy nativos del render
    d = det.detect(head_cam, body)
    if d is None or not d.in_frame:
        return None
    depth = sim.pull_camera_data().get_all(use_depth_color_map=False).get(head_depth_cam)
    if depth is None:
        return None
    Hd, Wd = depth.shape[:2]
    ix, iy = int(round(d.centroid[0])), int(round(d.centroid[1]))
    y0, y1 = max(0, iy - 3), min(Hd, iy + 4)
    x0, x1 = max(0, ix - 3), min(Wd, ix + 4)
    patch = depth[y0:y1, x0:x1].astype(float)
    valid = patch[(patch > 0.05) & np.isfinite(patch)]
    if valid.size < 3:
        return None
    z = float(np.median(valid))                 # profundidad (m) al objeto
    # pixel MOSTRADO -> pixel NATIVO (head rota np.rot90(-1)): disp_x=H_nat-1-v, disp_y=u
    u = float(d.centroid[1])
    v = float((cm.H - 1) - d.centroid[0])
    Xc = (u - cm.cx) * z / cm.f
    Yc = -(v - cm.cy) * z / cm.f
    Zc = -z                                     # MuJoCo mira por -z
    cam = sim.pull_status().camera_poses.get(head_mjcf)
    if cam is None:
        return None
    R = np.array(cam["xmat"]).reshape(3, 3)
    world = np.array(cam["pos"]) + R @ np.array([Xc, Yc, Zc])
    return world


def compute_grasp_xy(obj_xy, robot_xy, grasp_dist=0.55):
    """Punto en el PASILLO a 'grasp_dist' del objeto, del lado del robot (el
    mostrador corre en x, el pasillo esta en y mas baja). Returns (tx, ty)."""
    ox, oy = float(obj_xy[0]), float(obj_xy[1])
    side = -1.0 if robot_xy[1] < oy else 1.0
    return ox, oy + side * grasp_dist


def face_arm_at_object(controller, obj_xy, log=print, iters=4):
    """Gira la base para que el BRAZO (sale hacia base -Y_local) apunte al objeto.
    ITERA porque la base DERIVA al girar (recalcula el heading desde la pose real
    en cada vuelta hasta converger). -Y_local=u(dir base->obj) => th=atan2(ux,-uy)."""
    ttheta = None
    for _ in range(iters):
        bx, by, th = _base_pose(controller)
        ux, uy = obj_xy[0] - bx, obj_xy[1] - by
        n = np.hypot(ux, uy)
        if n < 1e-6:
            break
        ttheta = float(np.arctan2(ux / n, -uy / n))
        if abs(_wrap(ttheta - th)) < 0.06:
            break
        turn_to(controller, ttheta, log=log)
    log(f"[pos] brazo apuntando al objeto (heading={np.degrees(ttheta):.0f}deg)")
    return ttheta


def nav_to_parallel(controller, sim, obj_xy, standoff=0.6, max_time=35, log=print):
    """Navega a una pose PARALELA al mostrador (en el pasillo, a 'standoff' del objeto)
    con ESQUIVE reactivo por LiDAR: NO va recto al objeto; si detecta choque al frente,
    GIRA hacia el lado mas abierto y sigue hacia el punto. El brazo extendido alcanza,
    asi que un standoff comodo (~0.6 m) esta bien."""
    obj_xy = np.array(obj_xy, float)
    AVOID, SLOW = 0.45, 0.75
    t0 = time.time(); last = 0.0
    while time.time() - t0 < max_time:
        bx, by, th = _base_pose(controller)
        tx, ty = compute_grasp_xy(obj_xy, (bx, by), standoff)     # punto en el pasillo
        dist = float(np.hypot(tx - bx, ty - by))
        if dist < 0.12:
            break
        herr = _wrap(np.arctan2(ty - by, tx - bx) - th)
        front = lidar_front_min(controller, 30)
        L = lidar_front_min(controller, 25, front=(LIDAR_FRONT + 55) % 360)
        R = lidar_front_min(controller, 25, front=(LIDAR_FRONT - 55) % 360)
        if front < AVOID:                       # obstaculo al frente -> ESQUIVAR
            steer = 1.0 if L >= R else -1.0     # girar hacia el lado mas abierto
            controller.set_velocities({"base_counterclockwise": BASE_YAW_SIGN * 0.7 * steer,
                                       "base_forward": 0.06})
            mode = "ESQUIVA"
        else:
            # COMBINADO con magnitudes MINIMAS decisivas: el sim tiene friccion estatica,
            # si el comando es chico la base se CONGELA. Por eso los comandos no-cero van
            # por encima de un piso. Lejos avanza decidido; cerca frena para no sobrepasar.
            turn = float(np.clip(BASE_YAW_SIGN * 2.0 * herr, -1.0, 1.0))
            if abs(herr) > 0.06:
                turn = float(np.sign(turn) * max(0.45, abs(turn)))   # piso para girar
            else:
                turn = 0.0
            align = max(0.0, 1.0 - abs(herr) / 0.6)
            cap = 0.25 if front < SLOW else 1.0
            fwd = align * float(np.clip(2.2 * dist, 0.0, 1.0)) * cap
            if dist > 0.3 and fwd > 0.03:
                fwd = max(0.35, fwd)             # piso para avanzar (vencer friccion)
            controller.set_velocities({"base_forward": fwd, "base_counterclockwise": turn})
            mode = "avanza" if align > 0.3 else "gira"
        if time.time() - last > 1.0:
            log(f"[pos]   nav-paralelo dist={dist:.2f} herr={np.degrees(herr):+.0f}deg front={front:.2f} [{mode}]")
            last = time.time()
        time.sleep(DT)
    stop_base(controller)
    bx, by, th = _base_pose(controller)
    tx, ty = compute_grasp_xy(obj_xy, (bx, by), standoff)
    d = float(np.hypot(tx - bx, ty - by))
    log(f"[pos] pose paralela: base ({bx:.2f},{by:.2f}) dist_al_punto={d:.2f}")
    return {"reached": d < 0.25, "final": (bx, by, th)}


def drive_forward(controller, dist, speed=0.3, max_time=6.0, brake_lidar=True):
    """Avanza (o retrocede si dist<0) ~dist metros, con freno LiDAR al frente."""
    b0 = np.array(_base_pose(controller)[:2])
    sign = 1.0 if dist >= 0 else -1.0
    t0 = time.time()
    while time.time() - t0 < max_time:
        cur = np.array(_base_pose(controller)[:2])
        if np.hypot(*(cur - b0)) >= abs(dist):
            break
        if brake_lidar and sign > 0 and lidar_front_min(controller) < 0.30:
            break
        controller.set_velocities({"base_forward": sign * speed, "base_counterclockwise": 0.0})
        time.sleep(DT)
    stop_base(controller)


def _wrist_xy(sim):
    return np.array(sim.pull_status().camera_poses["d405_rgb"]["pos"])[:2]


def coarse_align_gripper(controller, sim, servo, obj_xy, iters=5, log=print):
    """Pone el GRIPPER sobre el objeto (objeto localizado por camara + pose de la
    muneca). DESACOPLADO: el LATERAL con arm_out (preciso) y el eje DEL MOSTRADOR
    con la base en lazo cerrado (drive_forward + odometria). Returns offset final."""
    obj_xy = np.array(obj_xy)
    # 1) LATERAL con arm_out (en pasos, por si necesita >rango)
    for _ in range(3):
        bx, by, th = _base_pose(controller)
        ymloc = np.array([np.sin(th), -np.cos(th)])
        d_arm = float((obj_xy - _wrist_xy(sim)) @ ymloc)
        if abs(d_arm) < 0.02:
            break
        cur = controller.get_state()["arm_out"]
        new = float(np.clip(cur + d_arm, 0.0, 0.5))
        servo.move_to({"arm_out": new})
        t = time.time()
        while time.time() - t < 3 and abs(controller.get_state()["arm_out"] - new) > 0.02:
            servo.hold(); time.sleep(DT)
        if new in (0.0, 0.5):     # arm en su limite: no puede mas lateral
            break
    # 2) El eje A LO LARGO del mostrador ya viene alineado de nav_to_parallel; NO
    #    movemos la base aqui (su control fino es impreciso y divergia). Si hace falta
    #    un ajuste fino en ese eje, lo hace el centrado con la camara del brazo.
    off_mag = float(np.hypot(*(obj_xy - _wrist_xy(sim))))
    log(f"[pos]   alineado (lateral con brazo): offset={off_mag:.2f}m")
    return off_mag


def lidar_front_min(controller, half_cone=25, front=LIDAR_FRONT):
    """Distancia minima en el cono frontal de la base."""
    r = controller.get_lidar_ranges()
    if r is None:
        return np.inf
    r = np.asarray(r, float)
    n = len(r)
    idx = [(front + d) % n for d in range(-half_cone, half_cone + 1)]
    v = r[idx]
    v = v[np.isfinite(v)]
    return float(v.min()) if v.size else np.inf


def _base_pose(controller):
    s = controller.get_state()
    return s["base_x"], s["base_y"], s["base_theta"]


def stop_base(controller, timeout=4.0):
    """Frena la base y ESPERA a que realmente se detenga (la velocidad tiene rampa
    de aceleracion, asi que no para de golpe -> si no esperamos, deriva al girar)."""
    prev = None
    t0 = time.time()
    while time.time() - t0 < timeout:
        controller.set_velocities({"base_forward": 0.0, "base_counterclockwise": 0.0})
        bx, by, bth = _base_pose(controller)
        if prev is not None and np.hypot(bx - prev[0], by - prev[1]) < 0.003 \
                and abs(_wrap(bth - prev[2])) < 0.006:
            break
        prev = (bx, by, bth)
        time.sleep(0.1)


def turn_to(controller, target_theta, tol=0.05, timeout=18.0, log=None):
    """Gira la base a un heading absoluto. Proporcional (frena cerca del objetivo
    para no oscilar/derivar); timeout amplio porque la base gira lento."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        _, _, th = _base_pose(controller)
        herr = _wrap(target_theta - th)
        if abs(herr) < tol:
            break
        # base_forward=0 EXPLICITO: nada de avance mientras gira (evita deriva)
        controller.set_velocities({"base_counterclockwise": float(np.clip(BASE_YAW_SIGN * 1.5 * herr, -1, 1)),
                                   "base_forward": 0.0})
        time.sleep(DT)
    stop_base(controller, timeout=2.0)
    return abs(_wrap(target_theta - _base_pose(controller)[2])) < tol * 2.5


def goto_pose(controller, tx, ty, ttheta, stop_dist=0.35, slow_dist=0.7,
              reach_tol=0.10, max_time=25.0, log=print):
    """Navega la base a (tx,ty,ttheta): gira hacia el waypoint -> avanza con freno
    LiDAR -> gira a la orientacion final. El waypoint esta en el pasillo abierto,
    asi que no se mete al mostrador. Devuelve dict con resultado."""
    bx, by, bth = _base_pose(controller)
    log(f"[pos] navegando al punto ({tx:.2f},{ty:.2f}) desde ({bx:.2f},{by:.2f})")

    # 1) girar hacia el waypoint
    turn_to(controller, np.arctan2(ty - by, tx - bx), log=log)

    # 2) avanzar con freno LiDAR
    t0 = time.time()
    blocked = False
    last = 0
    while time.time() - t0 < max_time:
        bx, by, bth = _base_pose(controller)
        dist = float(np.hypot(tx - bx, ty - by))
        if dist <= reach_tol:
            break
        desired = np.arctan2(ty - by, tx - bx)
        herr = _wrap(desired - bth)
        front = lidar_front_min(controller)
        if abs(herr) > 0.3:
            controller.set_velocities({"base_counterclockwise": float(np.clip(BASE_YAW_SIGN * 1.0 * herr, -0.6, 0.6)),
                                       "base_forward": 0.0})
        else:
            if front < stop_dist:
                controller.set_velocities({"base_forward": 0.0, "base_counterclockwise": 0.0})
                blocked = True
                log(f"[pos] obstaculo al frente (LiDAR {front:.2f}m) -> freno")
                break
            fwd = 0.18 if front < slow_dist else float(np.clip(0.8 * dist, 0.12, 0.55))
            controller.set_velocities({"base_forward": fwd,
                                       "base_counterclockwise": float(np.clip(BASE_YAW_SIGN * 0.8 * herr, -0.3, 0.3))})
        if time.time() - last > 1.0:
            log(f"[pos]   nav dist={dist:.2f} herr={np.degrees(herr):+.0f}deg front_lidar={front:.2f}")
            last = time.time()
        time.sleep(DT)
    stop_base(controller)        # frenar del todo antes de cualquier giro (evita deriva)

    # 3) (opcional) girar a una orientacion final
    if ttheta is not None:
        turn_to(controller, ttheta, log=log)

    bx, by, bth = _base_pose(controller)
    dist = float(np.hypot(tx - bx, ty - by))
    log(f"[pos] llegada a ({bx:.2f},{by:.2f},th={np.degrees(bth):.0f}deg) dist_al_objetivo={dist:.2f}m blocked={blocked}")
    return {"reached": dist <= reach_tol + 0.15, "blocked": blocked, "final": (bx, by, bth), "dist": dist}
