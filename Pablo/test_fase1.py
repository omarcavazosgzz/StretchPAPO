"""
Verificacion headless de la Fase 1: el visual servoing centra el objeto en
ambas camaras (head y wrist) usando SOLO la imagen.

Estrategia (robusta a la rotacion 90 de la head y a signos desconocidos):
  1) Identificacion (sysID): se mueve cada junta un poquito y se mide hacia
     donde se desplaza el objeto en la imagen -> deduce swap de ejes y signos.
  2) Lazo cerrado: usa servo.head_servo / servo.wrist_servo con la config
     descubierta y verifica que el error de centrado baja de un umbral.

Corre en el entorno de bloques (rapido). Imprime la config descubierta para
fijarla en servo.py.

Uso:
    uv run Pablo/test_fase1.py
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ["STRETCH_SIM_HEADLESS"] = "1"
os.environ["STRETCH_SIM_CAMERAS"] = "cam_d435i_rgb,cam_d435i_depth,cam_d405_rgb,cam_d405_depth"

import time
import numpy as np

import vision
import servo


def drive(controller, cmd, dur, hz=30):
    """Aplica un comando de velocidad por 'dur' segundos."""
    t_end = time.time() + dur
    while time.time() < t_end:
        controller.set_velocities(cmd)
        time.sleep(1.0 / hz)
    controller.set_velocities({})


def move_head_to(controller, tilt=None, pan=None, timeout=8.0):
    """Lleva head_tilt/head_pan a un valor absoluto por control de velocidad."""
    t_end = time.time() + timeout
    while time.time() < t_end:
        st = controller.get_state()
        cmd = {}
        if tilt is not None:
            e = tilt - st["head_tilt_up"]
            if abs(e) > 0.02:
                cmd["head_tilt_up"] = float(np.clip(3.0 * e, -1, 1))
        if pan is not None:
            e = pan - st["head_pan_counterclockwise"]
            if abs(e) > 0.02:
                cmd["head_pan_counterclockwise"] = float(np.clip(3.0 * e, -1, 1))
        if not cmd:
            break
        controller.set_velocities(cmd)
        time.sleep(1 / 30)
    controller.set_velocities({})
    time.sleep(0.2)


def detect(cam, target, tries=10):
    """Devuelve (centroid, frame_shape) del objeto, o (None, shape)."""
    shape = None
    for _ in range(tries):
        f = cam.get_frame()
        if f is not None:
            shape = f.shape
            obj = vision.find_object(f, target)
            if obj is not None:
                return obj["centroid"], shape
        time.sleep(0.03)
    return None, shape


def acquire_head_view(controller, cam, target):
    """Apunta la cabeza al costado derecho (donde llega el brazo) para ver la mesa.

    Geometria descubierta con head_sweep.py: los bloques estan a pan ~ -1.25,
    tilt ~ -0.65. Se prueba una pequena rejilla alrededor por robustez.
    """
    for pan, tilt in [(-1.25, -0.65), (-1.4, -0.6), (-1.2, -0.95), (-1.5, -1.0)]:
        move_head_to(controller, tilt=tilt, pan=pan)
        c, shape = detect(cam, target, tries=4)
        if c is not None:
            return c, shape
    return detect(cam, target, tries=5)


def sysid_axis(controller, cam, target, joint, vel=0.4, dur=0.45, reacquire=None):
    """Mueve 'joint' y mide el desplazamiento (dx, dy) del objeto en la imagen.

    Robusto: si pierde el objeto, intenta revertir y, si hay 'reacquire',
    re-apunta la camara para no quedarse ciego.
    """
    c0, _ = detect(cam, target)
    if c0 is None and reacquire:
        reacquire()
        c0, _ = detect(cam, target)
    if c0 is None:
        return None

    drive(controller, {joint: vel}, dur)
    c1, _ = detect(cam, target)
    if c1 is not None:
        return (c1[0] - c0[0], c1[1] - c0[1])

    # Se salio de vista: revierte un poco mas para recuperarlo
    drive(controller, {joint: -vel}, dur * 1.6)
    c1, _ = detect(cam, target)
    if c1 is not None:
        return (-(c1[0] - c0[0]), -(c1[1] - c0[1]))

    if reacquire:
        reacquire()
    return None


def discover_cfg(controller, cam, target, joint_h, joint_v, kp=0.9, deadband=0.035, reacquire=None):
    """Descubre swap y signos para un par (junta_horizontal, junta_vertical)."""
    d_h = sysid_axis(controller, cam, target, joint_h, reacquire=reacquire)
    if reacquire:
        reacquire()
    d_v = sysid_axis(controller, cam, target, joint_v, reacquire=reacquire)
    if d_h is None or d_v is None:
        return None, (d_h, d_v)

    # Que eje de imagen mueve dominantemente cada junta:
    swap = abs(d_h[1]) > abs(d_h[0])  # si la junta "horizontal" mueve mas en Y -> swap

    # Eje de imagen controlado por cada junta tras el swap
    # swap=False: joint_h controla X, joint_v controla Y
    # swap=True : joint_h controla Y, joint_v controla X
    if not swap:
        resp_h = d_h[0]   # respuesta de joint_h sobre X
        resp_v = d_v[1]   # respuesta de joint_v sobre Y
    else:
        resp_h = d_h[1]   # joint_h sobre Y
        resp_v = d_v[0]   # joint_v sobre X

    # Para que cmd = sign*kp*err reduzca err: sign = -sign(respuesta ante +vel)
    sign_h = -1.0 if resp_h > 0 else 1.0
    sign_v = -1.0 if resp_v > 0 else 1.0
    cfg = dict(swap=swap, sign_h=sign_h, sign_v=sign_v, kp=kp, deadband=deadband)
    return cfg, (d_h, d_v)


def make_servo_cfg(disc, kind):
    """Convierte la config descubierta al formato de servo.HEAD/WRIST."""
    if kind == "head":
        return dict(swap=disc["swap"], sign_pan=disc["sign_h"], sign_tilt=disc["sign_v"],
                    kp_pan=disc["kp"], kp_tilt=disc["kp"], deadband=disc["deadband"])
    else:
        return dict(swap=disc["swap"], sign_yaw=disc["sign_h"], sign_pitch=disc["sign_v"],
                    kp_yaw=disc["kp"], kp_pitch=disc["kp"], deadband=disc["deadband"])


def closed_loop(controller, cam, target, servo_fn, cfg, secs=8.0, hz=20):
    """Corre el lazo cerrado y devuelve la trayectoria de |error|."""
    errs = []
    t_end = time.time() + secs
    while time.time() < t_end:
        f = cam.get_frame()
        if f is not None:
            obj = vision.find_object(f, target)
            if obj is not None:
                cmd, (ex, ey) = servo_fn(obj["centroid"], f.shape, cfg)
                errs.append(servo.error_norm(ex, ey))
                controller.set_velocities(cmd)
            else:
                controller.set_velocities({})
        time.sleep(1.0 / hz)
    controller.set_velocities({})
    return errs


def main():
    from stretch_toolkit import controller, HEAD_RGB_CAMERA, WRIST_RGB_CAMERA

    print("[t1] arrancando (entorno bloques, headless+egl)...", flush=True)
    controller.get_state()
    for _ in range(50):
        if WRIST_RGB_CAMERA.get_frame() is not None:
            break
        time.sleep(0.1)

    results = {}
    blue = vision.blue_target()

    # ================= WRIST (camara del brazo) =================
    print("\n[t1] === WRIST: la caja azul ya es visible ===", flush=True)
    c, shape = detect(WRIST_RGB_CAMERA, blue)
    if c is None:
        print("[t1] WRIST: no detecto azul de inicio (revisar)", flush=True)
        results["wrist"] = False
    else:
        print(f"[t1] WRIST azul en {c} (img {shape[1]}x{shape[0]})", flush=True)
        # Descentrar a proposito para tener un error inicial real
        drive(controller, {"wrist_yaw_counterclockwise": 0.5}, 0.7)
        disc, deltas = discover_cfg(controller, WRIST_RGB_CAMERA, blue,
                                    "wrist_yaw_counterclockwise", "wrist_pitch_up")
        print(f"[t1] WRIST sysID deltas={deltas} -> cfg={disc}", flush=True)
        if disc is None:
            results["wrist"] = False
        else:
            cfg = make_servo_cfg(disc, "wrist")
            errs = closed_loop(controller, WRIST_RGB_CAMERA, blue, servo.wrist_servo, cfg, secs=10.0)
            if errs:
                e0, e1 = errs[0], np.median(errs[-5:])
                print(f"[t1] WRIST |err| inicio={e0:.3f} final={e1:.3f} (n={len(errs)})", flush=True)
                results["wrist"] = e1 < 0.10 and e1 <= e0
                results["wrist_cfg"] = cfg
            else:
                results["wrist"] = False

    # ================= HEAD (camara de arriba) =================
    print("\n[t1] === HEAD: inclinar para ver la mesa y centrar ===", flush=True)
    c, shape = acquire_head_view(controller, HEAD_RGB_CAMERA, blue)
    if c is None:
        print("[t1] HEAD: no logro ver el objeto al inclinar (revisar)", flush=True)
        results["head"] = False
    else:
        print(f"[t1] HEAD azul en {c} (img {shape[1]}x{shape[0]})", flush=True)
        reacquire_head = lambda: move_head_to(controller, tilt=-0.65, pan=-1.25)
        disc, deltas = discover_cfg(controller, HEAD_RGB_CAMERA, blue,
                                    "head_pan_counterclockwise", "head_tilt_up",
                                    reacquire=reacquire_head)
        reacquire_head()
        print(f"[t1] HEAD sysID deltas={deltas} -> cfg={disc}", flush=True)
        if disc is None:
            results["head"] = False
        else:
            cfg = make_servo_cfg(disc, "head")
            errs = closed_loop(controller, HEAD_RGB_CAMERA, blue, servo.head_servo, cfg, secs=8.0)
            if errs:
                e0, e1 = errs[0], np.median(errs[-5:])
                print(f"[t1] HEAD |err| inicio={e0:.3f} final={e1:.3f} (n={len(errs)})", flush=True)
                results["head"] = e1 < 0.10 and e1 <= e0
                results["head_cfg"] = cfg
            else:
                results["head"] = False

    try:
        controller.stop()
    except Exception:
        pass

    print("\n[t1] ===== RESULTADOS =====", flush=True)
    print(f"  WRIST: {'PASS' if results.get('wrist') else 'FAIL'}  cfg={results.get('wrist_cfg')}", flush=True)
    print(f"  HEAD : {'PASS' if results.get('head') else 'FAIL'}  cfg={results.get('head_cfg')}", flush=True)
    overall = results.get("wrist") and results.get("head")
    print(f"\n[t1] ===== {'PASS' if overall else 'FAIL'} =====", flush=True)


if __name__ == "__main__":
    main()
