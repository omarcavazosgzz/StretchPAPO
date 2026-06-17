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
# (lateral) bajar el centro de agarre un poco POR DEBAJO del centro del objeto: asi los
# dedos ABIERTOS abrazan el CUERPO del objeto (no rozan la tapa) y el brazo avanza sin
# atorarse contra la cara superior. Tambien compensa el sesgo +~1.5cm de la localizacion
# por la camara de la cabeza (mide el objeto un poco mas alto de lo que esta).
LIFT_BIAS_LAT = 0.025
HIGH_CLEAR = 0.07              # (lateral) altura del centro de agarre SOBRE el objeto para
                              # extender el brazo por encima del mostrador sin rozar/empujar
APPROACH_OVERSHOOT = 0.015     # (lateral) pasar el centro de agarre apenas detras del objeto
# Inclinacion de la muneca hacia ABAJO para el agarre lateral. Un agarre con la muneca
# perfectamente HORIZONTAL es IMPOSIBLE aqui: el brazo se atora ~1.5cm por ENCIMA de la
# tapa del cubo (el gripper choca con el mostrador al bajar). Con una inclinacion moderada
# los dedos ALCANZAN el cuerpo del objeto pasando por encima del borde, mientras la muneca
# queda alta y libra el mostrador. Sigue siendo un agarre DE LADO (los dedos aprietan los
# costados), solo que entrando un poco inclinado, no recto-horizontal.
PITCH_LAT = 0.6
# Tolerancia de FIRMEZA: si el objeto, justo antes de cerrar, esta a mas de esto del centro
# del eje de cierre (entre las puntas), el agarre saldria DEBIL (de orilla) y se soltaria al
# mover la base. _grasp_lateral aborta (reintento) si se supera. ~0.7cm.
FIRM_TOL = 0.007


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

    # 4) Girar para que el brazo apunte al objeto (ya en alto -> libra el borde). La CAMARA
    #    DE LA CABEZA SIGUE al objeto mientras la base gira (no lo pierde de vista).
    face_arm_at_object(controller, obj[:2], log=log, sim=sim, servo=servo, obj_z=obj[2])

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
    del robot (base + lift_up + arm_out). mode='lateral' (muneca horizontal) o 'top'.
    Aproximacion por constantes calibradas (sirve cuando la muneca esta horizontal o
    abajo; con la muneca INCLINADA usa _grasp_center_true)."""
    s = controller.get_state()
    bx, by, th = s["base_x"], s["base_y"], s["base_theta"]
    rad = np.array([np.sin(th), -np.cos(th)])        # direccion de extension del brazo
    radial = GC_RADIAL0[mode] + s["arm_out"]
    gc = np.empty(3)
    gc[:2] = np.array([bx, by]) + radial * rad
    gc[2] = s["lift_up"] + GC_ZLIFT[mode]
    return gc


def _grasp_center_true(controller, sim, fallback_mode="lateral", retries=10):
    """CENTRO DE AGARRE por CINEMATICA DIRECTA del propio robot (pose-mundo del body
    'link_grasp_center' que expone el sim). Es propiocepcion (el robot sabe donde tiene
    su gripper por sus encoders + URDF; existe igual en hardware real), NO sensado del
    objeto. Mas exacto que las constantes (que erran ~2cm y no valen con la muneca
    inclinada). Reintenta si el pull_status llega con un valor implausible (concurrencia).
    Cae a _grasp_center si no hay lectura buena."""
    s = controller.get_state()
    base = np.array([s["base_x"], s["base_y"]])
    for _ in range(retries):
        cp = sim.pull_status().camera_poses
        g = cp.get("link_grasp_center")
        if g is not None:
            p = np.array(g["pos"], dtype=float)
            if 0.0 < p[2] < 2.0 and np.linalg.norm(p[:2] - base) < 1.5:
                return p
        time.sleep(0.03)
    return _grasp_center(controller, fallback_mode)


def _closing_axis_offset(sim, obj_xyz, base_xy):
    """Offset del objeto sobre el EJE DE CIERRE real (entre las puntas de goma), en metros.
    0 = el objeto esta centrado entre los dedos -> al cerrar lo aprietan (no lo empujan).
    Usa las pose-mundo de rubber_tip_left/right (cinematica directa = propiocepcion).
    Devuelve (offset, valido). Reintenta si el pull_status llega implausible."""
    for _ in range(10):
        cp = sim.pull_status().camera_poses
        L, R = cp.get("rubber_tip_left"), cp.get("rubber_tip_right")
        if L is not None and R is not None:
            L = np.array(L["pos"]); R = np.array(R["pos"])
            if (0.0 < L[2] < 2.0 and 0.0 < R[2] < 2.0
                    and np.linalg.norm(L[:2] - base_xy) < 1.5
                    and np.linalg.norm(R[:2] - base_xy) < 1.5):
                mid = (L + R) / 2.0
                axis = R - L
                n = np.linalg.norm(axis)
                if n > 1e-6:
                    return float((np.array(obj_xyz) - mid) @ (axis / n)), True
        time.sleep(0.03)
    return 0.0, False


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


def grasp_with_retries(controller, sim, det, model, servo, body, HEAD, HEAD_D, WRIST, WRIST_D,
                       method="lateral", retries=3, log=print):
    """Posiciona y agarra, REINTENTANDO si el agarre falla (el cierre lateral del cubo es
    marginal en el sim: ~mitad de las veces sale a la primera). Al fallar: abre el gripper,
    recoge el brazo y vuelve a localizar + posicionar + agarrar. Con ~50% por intento, 3
    reintentos dan ~90% de exito. Returns (obj_world, grabbed)."""
    obj = None
    for attempt in range(1, retries + 1):
        log(f"[grasp] ===== intento de agarre {attempt}/{retries} =====")
        obj, ok = position_for_grasp(controller, sim, det, model, servo, body,
                                     HEAD, HEAD_D, WRIST, method=method, log=log)
        if ok and grasp_object(controller, sim, det, model, servo, body, WRIST, WRIST_D, obj,
                               method=method, log=log):
            log(f"[grasp] AGARRADO en el intento {attempt}")
            return obj, True
        # fallo -> soltar/recoger y reintentar (re-localiza por si el objeto se movio)
        log("[grasp] fallo; abro, recojo y reintento...")
        servo.move_to({"gripper_open": 0.5, "wrist_pitch_up": 0.0, "arm_out": 0.0})
        _wait_joint(controller, "arm_out", 0.0, tol=0.03, timeout=4, servo=servo)
    return obj, False


# ----------------------------------------------------------------------------------
def _grasp_lateral(controller, sim, det, model, servo, body, WRIST, WRIST_D, obj, log=print):
    """Agarre LATERAL (de lado: los dedos aprietan los COSTADOS del objeto, no desde
    arriba) ENTRANDO UN POCO INCLINADO. Un agarre con la muneca perfectamente horizontal
    es IMPOSIBLE en esta cocina: el brazo se atora ~1.5cm por ENCIMA de la tapa del cubo
    (el gripper choca con el mostrador al bajar), asi que se inclina la muneca lo justo
    para que los dedos alcancen el cuerpo del objeto. Secuencia ('cuando esta encima del
    mostrador, baja a la altura del objeto y lo agarra'):
      1) muneca horizontal + gripper abierto.
      2) SUBIR el centro de agarre por ENCIMA del mostrador.
      3) centrar a-lo-largo con la camara del brazo (wrist_yaw), objeto adelante.
      4) EXTENDER el brazo ALTO hasta quedar SOBRE el objeto (radial ~ 0).
      5) INCLINAR la muneca hacia abajo (PITCH_LAT) para alcanzar el cuerpo del objeto.
      6) POSICIONAR fino el CENTRO DE AGARRE VERDADERO (link_grasp_center) en el objeto
         (arm_out=radial, lift=altura; la muneca alta libra el mostrador).
      7) CERRAR firme y LEVANTAR.
    """
    from positioning import _base_pose
    obj_xyz = np.array(obj[:3])
    obj_xy = obj_xyz[:2]

    def gct():
        # CENTRO DE AGARRE por cinematica DIRECTA del robot (link_grasp_center). Exacto y
        # valido tambien con la muneca inclinada (las constantes calibradas no lo son).
        return _grasp_center_true(controller, sim, fallback_mode="lateral")

    def errs():
        gc = gct()
        bx, by, th = _base_pose(controller)
        rad = np.array([np.sin(th), -np.cos(th)])
        along = np.array([np.cos(th), np.sin(th)])
        return (gc, float((obj_xy - gc[:2]) @ rad),
                float((obj_xy - gc[:2]) @ along), float(obj_xyz[2] - gc[2]))

    def move_lift_for_z(z_target, tag):
        for _ in range(20):
            gc = gct(); dz = z_target - gc[2]
            if abs(dz) < 0.012:
                break
            lf = controller.get_state()["lift_up"]
            nl = float(np.clip(lf + dz, 0.12, 1.05))
            if abs(nl - lf) < 0.005:
                break
            servo.move_to({"lift_up": nl})
            _wait_joint(controller, "lift_up", nl, tol=0.015, timeout=2.5, servo=servo)
        log(f"[gL] {tag}: gc_z={gct()[2]:.3f} (objetivo {z_target:.3f})")

    # 1) muneca HORIZONTAL + gripper ABIERTO (para extender limpio sobre el mostrador).
    log("[gL] muneca horizontal + gripper abierto")
    servo.move_to({"wrist_pitch_up": 0.0, "wrist_yaw_counterclockwise": 0.0,
                   "wrist_roll_counterclockwise": 0.0, "gripper_open": 0.5})
    _wait_joint(controller, "wrist_pitch_up", 0.0, tol=0.05, timeout=4, servo=servo)
    gc, d_rad0, d_along0, dz0 = errs()
    log(f"[gL] offset inicial: radial={d_rad0:+.3f} a-lo-largo={d_along0:+.3f} dz={dz0:+.3f}")

    # 2) SUBIR el centro de agarre por ENCIMA del mostrador antes de extender.
    move_lift_for_z(obj_xyz[2] + HIGH_CLEAR, "arriba del mostrador")

    # 4) EXTENDER el brazo ALTO hasta quedar SOBRE el objeto (radial ~ 0). Alto -> no roza.
    log("[gL] extendiendo el brazo (alto) hasta quedar sobre el objeto...")
    for _ in range(8):
        gc, d_rad, _, _ = errs()
        if abs(d_rad) < 0.02:
            break
        a = controller.get_state()["arm_out"]
        na = float(np.clip(a + d_rad, 0.0, 0.5))
        if abs(na - a) < 0.005:
            break
        servo.move_to({"arm_out": na})
        _wait_joint(controller, "arm_out", na, tol=0.02, timeout=3, servo=servo)

    # 5) CENTRAR A-LO-LARGO con move_by(base_translate). Es el UNICO DOF que TRASLADA el
    #    centro de agarre a lo largo del mostrador (wrist_yaw solo GIRA en el sitio -> su eje
    #    pasa por el centro de agarre, sondeo dio ganancia ~0; el brazo solo extiende radial).
    #    base_translate es DISTANCIA controlada y PRECISA (~1mm en pasos <=5cm), a diferencia
    #    de los pulsos de velocidad que sobrepasaban metros. Robot PARALELO -> mueve a-lo-largo.
    #    Gripper ALTO -> SEGURO (libra objetos). CLAVE: el objeto debe quedar centrado < ~2cm
    #    o al cerrar un dedo lo empuja (cubo 5cm, abertura cerrada ~3.9cm; el cierre re-centra
    #    un descentramiento moderado, pero >~2.5cm un dedo lo choca al bajar).
    from stretch_mujoco.enums.actuators import Actuators

    def _base_stop():
        prev = None; t0 = time.time()
        while time.time() - t0 < 6:
            time.sleep(0.15); servo.hold()
            p = np.array([controller.get_state()["base_x"], controller.get_state()["base_y"]])
            if prev is not None and np.linalg.norm(p - prev) < 0.0015:
                break
            prev = p

    # base_translate (heading ~ eje de cierre, muneca horizontal) mueve las PUNTAS ~1:1 a lo
    # largo del mostrador. Centro sobre el EJE DE CIERRE REAL (entre las puntas de goma), que
    # es lo que decide si el cierre APRIETA -- el "along" cinematico vs el centro de agarre
    # no lo predecia bien (diferia ~1cm) y el agarre salia 50/50. Ganancia conocida -1 (un
    # +move baja el offset); pasos = offset (tope 4cm), los moves parciales se autocorrigen.
    def _bxy():
        s = controller.get_state()
        return np.array([s["base_x"], s["base_y"]])

    log("[gL] centrando sobre el eje de cierre con base_translate (gripper alto)...")
    # Sondeo SOLO del signo (un +move sube o baja el offset). El eje de cierre (R-L) puede
    # apuntar en +heading o -heading; el sondeo lo resuelve. Es robusto a moves parciales:
    # solo necesita el signo, no la magnitud (la ganancia |.|~1 se asume y los parciales se
    # autocorrigen al re-medir).
    off0, ok0 = _closing_axis_offset(sim, obj_xyz, _bxy())
    sim.move_by(Actuators.base_translate, 0.025); _base_stop(); servo.sync()
    off1, ok1 = _closing_axis_offset(sim, obj_xyz, _bxy())
    sign_s = 1.0 if (off1 - off0) >= 0 else -1.0     # signo de d(offset)/d(+move)
    log(f"[gL]   sondeo signo: offset {off0:+.3f}->{off1:+.3f} (sign_s={sign_s:+.0f})")
    for it in range(8):
        off, ok = _closing_axis_offset(sim, obj_xyz, _bxy())
        log(f"[gL]   cierre it{it}: offset_eje_cierre={off:+.3f} (ok={ok})")
        if not ok or abs(off) < 0.008:
            break
        step = float(np.clip(-off * sign_s, -0.04, 0.04))   # mueve para reducir |offset|
        if abs(step) < 0.005:
            break
        sim.move_by(Actuators.base_translate, step)
        _base_stop(); servo.sync()

    # 6) INCLINAR la muneca hacia abajo: los dedos pueden ALCANZAR el cuerpo del objeto
    #    pasando por encima del borde del mostrador (horizontal puro se atora arriba).
    log(f"[gL] inclinando la muneca hacia abajo (pitch=-{PITCH_LAT:.2f}) para alcanzar de lado")
    servo.move_to({"wrist_pitch_up": -PITCH_LAT})
    _wait_joint(controller, "wrist_pitch_up", -PITCH_LAT, tol=0.05, timeout=4, servo=servo)

    # 7) POSICIONAR FINO con el CENTRO DE AGARRE VERDADERO (link_grasp_center): RADIAL->arm,
    #    ALTURA->lift (la muneca alta libra el mostrador). El a-lo-largo ya quedo centrado con
    #    la base y no cambia al inclinar/extender. Comandos directos (friccion).
    # El RADIAL se lleva a un pequeno SOBREPASO (el centro de agarre justo pasado el objeto
    # -> el objeto queda ENTRE los dedos, no en la punta). Importante: el comando de brazo
    # incluye el sobrepaso, asi es lo bastante GRANDE para vencer la friccion estatica (un
    # incremento chico NO mueve el brazo telescopico) y la correccion final si "agarra".
    # RAD_DONE negativo -> el centro de agarre SOBREPASA el objeto: el objeto queda DENTRO de
    # los dedos (cerca de la palma), no en la punta -> al cerrar/levantar no se escapa.
    log("[gL] posicionando fino el centro de agarre EN el objeto (radial+altura)...")
    Z_TOL, RAD_DONE, RAD_OVERSHOOT = 0.015, -0.010, 0.022
    prev = None
    for it in range(14):
        gc, d_rad, d_al, dz = errs()
        log(f"[gL]   pos it{it}: gc=({gc[0]:.3f},{gc[1]:.3f},{gc[2]:.3f}) radial={d_rad:+.3f} along={d_al:+.3f} dz={dz:+.3f}")
        rad_ok = d_rad < RAD_DONE          # el centro de agarre alcanzo (o paso) el objeto
        z_ok = abs(dz) < Z_TOL
        if rad_ok and z_ok:
            break
        st = (round(d_rad, 3), round(dz, 3))
        if prev is not None and prev == st:
            log("[gL]   sin avance -> cierro como esta")
            break
        prev = st
        cmd = {}
        if not z_ok:
            cmd["lift_up"] = float(np.clip(controller.get_state()["lift_up"] + dz, 0.12, 1.05))
        if not rad_ok:
            cmd["arm_out"] = float(np.clip(controller.get_state()["arm_out"] + d_rad + RAD_OVERSHOOT, 0.0, 0.5))
        if not cmd:
            break
        servo.move_to(cmd)
        for k, v in cmd.items():
            _wait_joint(controller, "lift_up" if k == "lift_up" else "arm_out", v,
                        tol=0.02, timeout=2.5, servo=servo)

    gc, d_rad, d_al, dz = errs()
    log(f"[gL] pre-cierre: radial={d_rad:+.3f} a-lo-largo={d_al:+.3f} dz={dz:+.3f}")

    # 6.5) VERIFICAR que el agarre quedara FIRME: el objeto debe estar CENTRADO entre las
    #      puntas (eje de cierre). Si quedo descentrado (>FIRM_TOL) el cierre lo agarraria por
    #      una orilla -> aguanta el levanton pero se SUELTA al mover la base. Mejor avisar
    #      (return False) para que grasp_with_retries suelte y reintente -> solo cargamos
    #      agarres firmes. (Verificacion por posiciones del sim; el control es por camara/FK.)
    try:
        cube_now = np.array(sim.pull_status().object_poses[body][:3])
        off, ok = _closing_axis_offset(sim, cube_now,
                                       np.array(_base_pose(controller)[:2]))
        if ok and abs(off) > FIRM_TOL:
            log(f"[gL] agarre quedaria DEBIL (offset eje de cierre {off*100:+.1f}cm > "
                f"{FIRM_TOL*100:.1f}cm) -> reintento")
            return False
        log(f"[gL] agarre centrado (offset eje de cierre {off*100:+.1f}cm) -> cierro")
    except Exception as e:
        log(f"[gL] (no pude verificar firmeza: {e})")

    # 7) CERRAR firme y LEVANTAR (el objeto esta entre los dedos a esta pose)
    return _close_and_lift(controller, sim, servo, body, log)


def place_object_lateral(controller, sim, servo, place_xyz, log=print):
    """Coloca el objeto SOSTENIDO en place_xyz sobre el mostrador. El robot ya debe estar
    PARALELO y apuntando al punto. CLAVE: mantiene la muneca INCLINADA todo el tiempo (NO la
    rota a horizontal, eso gira el cubo y lo zafa). Pasos: sube ALTO (puntas inclinadas libran
    el mostrador) -> extiende sobre el punto -> baja a la altura -> ABRE -> retrae."""
    from positioning import _base_pose
    place = np.array(place_xyz, float)
    place_xy = place[:2]

    def gct():
        return _grasp_center_true(controller, sim, fallback_mode="lateral")

    def errs():
        gc = gct()
        bx, by, th = _base_pose(controller)
        rad = np.array([np.sin(th), -np.cos(th)])
        return gc, float((place_xy - gc[:2]) @ rad), float(place[2] - gc[2])

    def move_lift(z_target):
        for _ in range(20):
            gc = gct(); dz = z_target - gc[2]
            if abs(dz) < 0.012:
                break
            lf = controller.get_state()["lift_up"]; nl = float(np.clip(lf + dz, 0.12, 1.05))
            if abs(nl - lf) < 0.005:
                break
            servo.move_to({"lift_up": nl})
            _wait_joint(controller, "lift_up", nl, tol=0.015, timeout=2.5, servo=servo)

    # 1) SUBIR ALTO manteniendo la muneca INCLINADA (las puntas inclinadas libran el mostrador
    #    porque el centro de agarre va alto). No se toca wrist_pitch (sigue en -PITCH_LAT).
    log("[place] subo sobre el mostrador (muneca sigue inclinada, sin soltar)")
    move_lift(place[2] + HIGH_CLEAR + 0.04)

    # 2) extender sobre el punto (radial ~ 0). Alto -> las puntas inclinadas no chocan.
    log("[place] extiendo sobre el punto de colocacion...")
    for _ in range(8):
        gc, d_rad, _ = errs()
        if abs(d_rad) < 0.02:
            break
        a = controller.get_state()["arm_out"]; na = float(np.clip(a + d_rad, 0.0, 0.5))
        if abs(na - a) < 0.005:
            break
        servo.move_to({"arm_out": na})
        _wait_joint(controller, "arm_out", na, tol=0.02, timeout=3, servo=servo)

    # 3) bajar EN VERTICAL a la altura de colocar (un poco arriba para soltar suave)
    log("[place] bajando a colocar...")
    prev = None
    for _ in range(14):
        gc, _, dz = errs()
        if abs(dz) < 0.02 or prev == round(dz, 3):
            break
        prev = round(dz, 3)
        lf = controller.get_state()["lift_up"]; nl = float(np.clip(lf + dz, 0.12, 1.05))
        if abs(nl - lf) < 0.004:
            break
        servo.move_to({"lift_up": nl})
        _wait_joint(controller, "lift_up", nl, tol=0.015, timeout=2.5, servo=servo)

    # 4) ABRIR (soltar el objeto)
    log("[place] abriendo el gripper (soltar)...")
    servo.move_to({"gripper_open": 0.5})
    _wait_joint(controller, "gripper_open", 0.5, tol=0.1, timeout=3, servo=servo)
    time.sleep(0.6)

    # 5) RETRAER el brazo + subir (ya solto -> ya puede rotar la muneca)
    log("[place] retraigo el brazo")
    servo.move_to({"arm_out": 0.0})
    _wait_joint(controller, "arm_out", 0.0, tol=0.03, timeout=4, servo=servo)
    lf = controller.get_state()["lift_up"]
    servo.move_to({"lift_up": float(np.clip(lf + 0.10, 0.2, 1.05)), "wrist_pitch_up": 0.0})
    log("[place] objeto colocado")


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
