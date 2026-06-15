"""
Fase 2 - AGARRE con la camara del brazo (y la profundidad de la muneca).

position_for_grasp(): localiza el objeto POR CAMARA (profundidad de la cabeza),
navega sin chocar a una pose PARALELA, sube el gripper por ENCIMA del mostrador y
gira para que el brazo apunte al objeto. Deja el robot ARRIBA + PARALELO + RECOGIDO,
listo para el agarre.

grasp_object(): dos metodos.
  - "lateral" (DEFAULT, recomendado, mas general): apunta la muneca HORIZONTAL,
    BAJA el gripper a la ALTURA del objeto estando RECOGIDO (sobre el pasillo -> sin
    chocar), centra con la camara del brazo y EXTIENDE el brazo hacia el objeto
    vigilando la PROFUNDIDAD de la muneca; cierra cuando esta al alcance. Sirve para
    objetos con volumen (huevo, cubo, tomate).
  - "top": agarre desde ARRIBA (muneca hacia abajo, desciende vertical). Util para
    objetos planos (cuchillo, plato).
"""
import time
import numpy as np

DT = 1 / 30
GRASP_PX = (0.50, 0.60)        # punto de agarre (x,y normalizado) en la imagen de la muneca
FINGER_DEPTH = 0.11            # (top) distancia muneca->objeto (m) a la que cerrar
FINGER_DEPTH_LAT = 0.14        # (lateral) profundidad muneca->objeto a la que cerrar
LIFT_CLEAR = 0.03             # (lateral) altura del centro del gripper sobre el centro del objeto


def _wait_joint(controller, key, target, tol=0.03, timeout=5.0, servo=None):
    t = time.time()
    while time.time() - t < timeout and abs(controller.get_state()[key] - target) > tol:
        if servo:
            servo.hold()
        time.sleep(DT)


def position_for_grasp(controller, sim, det, model, servo, body, HEAD, HEAD_D, WRIST,
                       method="lateral", log=print):
    """Deja el robot ARRIBA + PARALELO + apuntando al objeto con el brazo. Returns
    (obj_world_xyz, listo:bool)."""
    from phase1_lib import aim_head
    from positioning import (localize_with_head_camera, remember_object,
                             nav_to_parallel, face_arm_at_object, coarse_align_gripper,
                             _base_pose, GRIPPER_HOME_OFFSET)

    def localize(tag):
        o = localize_with_head_camera(sim, det, model, body, HEAD, HEAD_D, log=log)
        t = remember_object(sim, body)
        if o is None:
            o = t
        log(f"[g] localizado por camara ({tag}): ({o[0]:.2f},{o[1]:.2f},{o[2]:.2f}) "
            f"[verdad ({t[0]:.2f},{t[1]:.2f},{t[2]:.2f})]")
        return o

    # 1) Buscar/centrar el objeto con la CABEZA (SIN ir recto al objeto) y localizar.
    log("[g] buscando/centrando el objeto con la cabeza...")
    aim_head(controller, det, sim, servo, body, HEAD, body=body, log=log, do_approach=False)
    obj = localize("inicial")
    controller.stop(); time.sleep(0.4); servo.sync()

    # 2) SUBIR el brazo (recogido) a altura segura ANTES de mover/rotar.
    lift_safe = float(np.clip(obj[2] + 0.16, 0.3, 1.05))
    log(f"[g] subo el brazo a altura segura (lift={lift_safe:.2f}) ANTES de mover/rotar")
    servo.move_to({"gripper_open": 0.5, "wrist_yaw_counterclockwise": 0.0,
                   "wrist_pitch_up": 0.0, "arm_out": 0.0, "lift_up": lift_safe})
    _wait_joint(controller, "lift_up", lift_safe, servo=servo)
    _wait_joint(controller, "arm_out", 0.0, servo=servo)

    # 3) Navegar a una pose PARALELA al mostrador (pasillo) con ESQUIVE reactivo.
    #    standoff mas corto -> base_obj menor -> el brazo alcanza con margen.
    nav_to_parallel(controller, sim, obj[:2], standoff=0.5, log=log)
    controller.stop(); time.sleep(0.3); servo.sync()

    # 4) Girar para que el brazo apunte al objeto (ya en alto -> libra el borde).
    face_arm_at_object(controller, obj[:2], log=log)

    if method == "top":
        # ---- preparacion AGARRE DESDE ARRIBA: muneca abajo, RECOGIDO y ALTO. _grasp_top
        #      alinea a-lo-largo con la base (recogido+alto -> seguro), baja y extiende.
        servo.move_to({"arm_out": 0.0, "wrist_pitch_up": -1.5,
                       "wrist_yaw_counterclockwise": 0.0})
        _wait_joint(controller, "arm_out", 0.0, servo=servo)
        _wait_joint(controller, "wrist_pitch_up", -1.5, tol=0.05, timeout=4, servo=servo)
        time.sleep(0.3)
        log("[g] listo para agarre DESDE ARRIBA (arriba, paralelo, recogido, muneca abajo)")
        return obj, True

    # ---- preparacion AGARRE LATERAL: dejar ARRIBA + RECOGIDO + muneca horizontal ----
    # NO extendemos ni bajamos aqui; grasp_object_lateral baja a la altura del objeto
    # estando recogido (sobre el pasillo, sin chocar) y luego extiende.
    servo.move_to({"wrist_pitch_up": 0.0, "wrist_yaw_counterclockwise": 0.0,
                   "wrist_roll_counterclockwise": 0.0, "arm_out": 0.0})
    _wait_joint(controller, "arm_out", 0.0, servo=servo)
    log("[g] listo para agarre LATERAL (arriba, paralelo, apuntando, recogido)")
    return obj, True


# ----------------------------------------------------------------------------------
def _wrist_helpers(det, sim, body, WRIST, WRIST_D):
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

    def wdepth_center(half=8):
        """Profundidad en la REGION CENTRAL fija (punto de agarre), SIN depender del
        RGB. Devuelve la mediana de lo MAS CERCANO (percentil 25) que hay enfrente."""
        depth = sim.pull_camera_data().get_all(use_depth_color_map=False).get(WRIST_D)
        if depth is None:
            return None
        H, W = depth.shape[:2]
        ix, iy = int(GRASP_PX[0] * W), int(GRASP_PX[1] * H)
        p = depth[max(0, iy-half):min(H, iy+half+1), max(0, ix-half):min(W, ix+half+1)].astype(float)
        v = p[(p > 0.02) & np.isfinite(p)]
        return float(np.percentile(v, 25)) if v.size else None

    return wdet, wdepth, wdepth_center


def _gripper_z(sim):
    """Altura mundial del gripper (~ camara de la muneca). Cinematica propia del robot."""
    return float(sim.pull_status().camera_poses["d405_rgb"]["pos"][2])


# CENTRO DE AGARRE de los dedos por CINEMATICA PURA (get_state: base, lift_up, arm_out),
# calibrado con _diag_gripper.py. NO usa la pose de camara d405 (que llega STALE por la
# concurrencia del pull_status y corrompia la geometria). Robusto.
#   radial desde la base = GC_RADIAL0[mode] + arm_out ;  z = lift_up + GC_ZLIFT[mode]
#   muneca HORIZONTAL (lateral): los dedos apuntan AL FRENTE (entrar recto).
#   muneca ABAJO (top): los dedos apuntan ABAJO.
GC_RADIAL0 = {"lateral": 0.415, "top": 0.139}
GC_ZLIFT   = {"lateral": 0.112, "top": -0.133}


def _grasp_center(controller, mode="lateral"):
    """Posicion mundial (x,y,z) del CENTRO DE AGARRE de los dedos por CINEMATICA propia
    del robot (base + lift_up + arm_out). mode='lateral' (muneca horizontal) o 'top'."""
    s = controller.get_state()
    bx, by, th = s["base_x"], s["base_y"], s["base_theta"]
    rad = np.array([np.sin(th), -np.cos(th)])        # direccion de extension del brazo
    radial = GC_RADIAL0[mode] + s["arm_out"]
    gc = np.empty(3)
    gc[:2] = np.array([bx, by]) + radial * rad
    gc[2] = s["lift_up"] + GC_ZLIFT[mode]
    return gc


def _close_and_lift(controller, sim, servo, body, log):
    obj_z_before = sim.pull_status().object_poses[body][2]
    log("[g] cerrando gripper (firme)...")
    servo.move_to({"gripper_open": -0.35})            # cierre firme (el clip lo lleva al min)
    _wait_joint(controller, "gripper_open", -0.35, tol=0.08, timeout=3, servo=servo)
    time.sleep(0.6)
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


def grasp_object(controller, sim, det, model, servo, body, WRIST, WRIST_D, obj,
                 method="lateral", log=print):
    """Dispatcher: 'lateral' (default) o 'top'."""
    if method == "top":
        return _grasp_top(controller, sim, det, model, servo, body, WRIST, WRIST_D, obj, log)
    return _grasp_lateral(controller, sim, det, model, servo, body, WRIST, WRIST_D, obj, log)


# ----------------------------------------------------------------------------------
def _grasp_lateral(controller, sim, det, model, servo, body, WRIST, WRIST_D, obj, log=print):
    """Agarre LATERAL (de lado). El robot ya esta ARRIBA + PARALELO + apuntando al
    objeto con el brazo + RECOGIDO. Pasos:
      1) muneca HORIZONTAL (apunta a lo largo del brazo = al objeto), gripper abierto.
      2) BAJAR a la ALTURA del objeto estando RECOGIDO (gripper sobre el pasillo, sin choque).
      3) centrar con la camara del brazo (vertical via lift, horizontal via wrist_yaw), SIN extender.
      4) EXTENDER el brazo hacia el objeto vigilando la PROFUNDIDAD de la muneca; cerrar al alcance.
    """
    from positioning import _base_pose
    obj_xy = np.array(obj[:2])

    def _gc_fresh():
        for _ in range(6):
            gc = _grasp_center(controller, mode="lateral")
            if gc[2] > obj[2] - 0.30:           # descarta lecturas stale (z implausible)
                return gc
            time.sleep(0.05)
        return _grasp_center(controller, mode="lateral")

    def _decomp():
        gc = _gc_fresh()
        off = obj_xy - gc[:2]
        bx, by, th = _base_pose(controller)
        rad = np.array([np.sin(th), -np.cos(th)])
        along = np.array([np.cos(th), np.sin(th)])
        return gc, float(off @ rad), float(off @ along)

    # 1) muneca HORIZONTAL (los dedos apuntan al frente = ENTRAR RECTO) + gripper abierto
    log("[gL] muneca horizontal (entrar recto) + gripper abierto")
    servo.move_to({"wrist_pitch_up": 0.0, "wrist_yaw_counterclockwise": 0.0,
                   "wrist_roll_counterclockwise": 0.0, "gripper_open": 0.5})
    _wait_joint(controller, "wrist_pitch_up", 0.0, tol=0.05, timeout=4, servo=servo)

    _, d_rad0, d_along0 = _decomp()
    log(f"[gL] offset inicial: radial={d_rad0:+.3f} a-lo-largo={d_along0:+.3f}")

    # 2) BAJAR el CENTRO DE AGARRE a la ALTURA del objeto, estando RECOGIDO (los dedos,
    #    horizontales, quedan a la altura del objeto sobre el pasillo -> sin chocar).
    log("[gL] bajando el centro de agarre a la altura del objeto...")
    for _ in range(25):
        gc = _gc_fresh()
        dz = obj[2] - gc[2]
        if abs(dz) < 0.02:
            break
        lf = controller.get_state()["lift_up"]
        new_lf = float(np.clip(lf + dz, 0.12, 1.05))
        if abs(new_lf - lf) < 0.005:
            break
        servo.move_to({"lift_up": new_lf})
        _wait_joint(controller, "lift_up", new_lf, tol=0.015, timeout=2.5, servo=servo)

    # 3) ENTRAR RECTO: extender arm_out (horizontal) hasta SOBREPASAR ~3cm el objeto, para
    #    que el objeto quede ENTRE los dedos (no en la punta). El eje a-lo-largo lo dejo
    #    face_arm. NO movemos la base.
    OVERSHOOT = 0.03
    log("[gL] entrando recto: extiendo arm_out sobrepasando el objeto (radial)...")
    for _ in range(16):
        gc, d_rad, _ = _decomp()
        if d_rad <= -OVERSHOOT:                 # ya sobrepaso el objeto
            break
        a = controller.get_state()["arm_out"]
        new_a = float(np.clip(a + d_rad + OVERSHOOT, 0.0, 0.5))
        if abs(new_a - a) < 0.003:
            break                               # brazo en el tope -> cierra igual
        servo.move_to({"arm_out": new_a})
        _wait_joint(controller, "arm_out", new_a, tol=0.02, timeout=3, servo=servo)

    gc, d_rad, d_along = _decomp()
    log(f"[gL] offset CENTRO-objeto: radial={d_rad:+.3f} a-lo-largo={d_along:+.3f} dz={obj[2]-gc[2]:+.3f}")

    # 3.5) AFINAR el centrado A-LO-LARGO con la CAMARA DEL BRAZO via wrist_yaw (junta
    #      PRECISA, no la base): ex (horizontal en la wrist cam) ~ eje a-lo-largo = eje de
    #      la abertura de los dedos. Centrar ex pone el objeto alineado con la abertura.
    wdet, _, _ = _wrist_helpers(det, sim, body, WRIST, WRIST_D)
    e0, _ = wdet()
    if e0 is not None:
        yw0 = controller.get_state()["wrist_yaw_counterclockwise"]
        yw1 = float(np.clip(yw0 + 0.12, -0.6, 0.6)); servo.move_to({"wrist_yaw_counterclockwise": yw1})
        _wait_joint(controller, "wrist_yaw_counterclockwise", yw1, tol=0.02, timeout=2, servo=servo)
        e1, _ = wdet()
        s_yaw = float(np.sign((e1[0] - e0[0]) / (yw1 - yw0))) if (e1 is not None and abs(e1[0]-e0[0]) > 0.01) else 1.0
        servo.move_to({"wrist_yaw_counterclockwise": yw0})
        _wait_joint(controller, "wrist_yaw_counterclockwise", yw0, tol=0.02, timeout=2, servo=servo)
        log(f"[gL] centrando a-lo-largo con wrist_yaw (camara del brazo, s={s_yaw:+.0f})...")
        for it in range(8):
            e, _ = wdet()
            if e is None:
                break
            log(f"[gL]   yaw it{it} ex={e[0]:+.3f}")
            if abs(e[0]) < 0.03:
                break
            yw = controller.get_state()["wrist_yaw_counterclockwise"]
            new_yw = float(np.clip(yw - s_yaw * 0.6 * e[0], -0.6, 0.6))
            if abs(new_yw - yw) < 0.004:
                break
            servo.move_to({"wrist_yaw_counterclockwise": new_yw})
            _wait_joint(controller, "wrist_yaw_counterclockwise", new_yw, tol=0.02, timeout=2, servo=servo)

    # 4) CERRAR firme y LEVANTAR (el objeto esta entre los dedos a esta pose)
    return _close_and_lift(controller, sim, servo, body, log)


# ----------------------------------------------------------------------------------
def _base_pulse(controller, servo, sign, dur=0.16, speed=0.5):
    """Pulso DECISIVO de la base a lo largo (+/-) para vencer la friccion estatica, y
    frena. El brazo se mantiene con servo.hold() (PosServo) durante el pulso."""
    from positioning import stop_base
    t = time.time()
    while time.time() - t < dur:
        controller.set_velocities({"base_forward": sign * speed, "base_counterclockwise": 0.0})
        if servo:
            servo.hold()
        time.sleep(DT)
    stop_base(controller)
    if servo:
        servo.sync()


def _grasp_top(controller, sim, det, model, servo, body, WRIST, WRIST_D, obj, log=print):
    """Agarre DESDE ARRIBA. Con la muneca mirando ABAJO: centra el objeto en la camara
    del brazo nullando AMBOS ejes -- el eje RADIAL con arm_out y el eje A LO LARGO del
    mostrador con PULSOS de la BASE (servo visual por camara) -- y DESCIENDE por
    profundidad hasta cerrar. Para objetos sobre superficie plana (huevo, cubo, tomate, plato)."""
    from positioning import _base_pose
    obj_xy = np.array(obj[:2])

    def _gc_fresh():
        """Centro de agarre, descartando lecturas basura (pull_status a veces
        desactualizado tras mover la base): exige gripper por ENCIMA del objeto-0.3."""
        for _ in range(6):
            gc = _grasp_center(controller, mode="top")
            if gc[2] > obj[2] - 0.30:
                return gc
            time.sleep(0.05)
        return _grasp_center(controller, mode="top")

    def _decomp():
        gc = _gc_fresh()
        off = obj_xy - gc[:2]
        bx, by, th = _base_pose(controller)
        rad = np.array([np.sin(th), -np.cos(th)])
        along = np.array([np.cos(th), np.sin(th)])
        return gc, float(off @ rad), float(off @ along)

    # NOTA: el eje A LO LARGO del mostrador lo deja face_arm (giro en sitio). La base del
    # sim NO permite ajuste fino (se congela en comandos chicos y rota/deriva), asi que NO
    # la movemos aqui: abrimos bien el gripper (span ~9cm) para tolerar el residuo a-lo-largo.
    _, d_rad0, d_along0 = _decomp()
    log(f"[g] offset inicial: radial={d_rad0:+.3f} a-lo-largo={d_along0:+.3f}")

    # --- 1) EXTENDER el brazo (ALTO, por encima del mostrador) para llevar el CENTRO DE
    #     AGARRE sobre el objeto en el eje RADIAL. Extender ALTO -> no choca con el mostrador.
    log("[g] extendiendo arm_out (alto) para alinear el centro de agarre (radial)...")
    for _ in range(14):
        gc, d_rad, _ = _decomp()
        if abs(d_rad) <= 0.015:
            break
        a = controller.get_state()["arm_out"]
        new_a = float(np.clip(a + d_rad, 0.0, 0.5))
        if abs(new_a - a) < 0.003:
            break
        servo.move_to({"arm_out": new_a})
        _wait_joint(controller, "arm_out", new_a, tol=0.02, timeout=3, servo=servo)

    # --- 2) BAJAR EL CENTRO DE AGARRE a la altura del objeto (cinematica). Desciende
    #     sobre el objeto; se satura ~al nivel del mostrador (dedos rozan), objeto en span.
    log("[g] bajando el centro de agarre a la altura del objeto...")
    for _ in range(25):
        gc = _gc_fresh()
        dz = obj[2] - gc[2]
        if abs(dz) < 0.02 or dz > 0:
            break
        lf = controller.get_state()["lift_up"]
        new_lf = float(np.clip(lf + dz, 0.12, 1.05))
        if abs(new_lf - lf) < 0.005:
            break
        servo.move_to({"lift_up": new_lf})
        _wait_joint(controller, "lift_up", new_lf, tol=0.015, timeout=2.5, servo=servo)

    gc, d_rad, d_along = _decomp()
    log(f"[g] offset CENTRO-objeto: radial={d_rad:+.3f} a-lo-largo={d_along:+.3f} dz={obj[2]-gc[2]:+.3f}")

    # --- 3) CERRAR y LEVANTAR (el objeto esta entre los dedos a esta pose) ---
    return _close_and_lift(controller, sim, servo, body, log)
