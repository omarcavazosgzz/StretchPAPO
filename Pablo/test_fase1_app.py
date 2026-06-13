"""
Auto-test headless de la clase Fase1 (la MISMA logica del script interactivo
fase1_seguimiento.py), sin GUI. Verifica que compute() centra el objeto en
ambas camaras. Entorno de bloques.

Uso:
    uv run Pablo/test_fase1_app.py
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ["STRETCH_SIM_HEADLESS"] = "1"
os.environ["STRETCH_SIM_CAMERAS"] = "cam_d435i_rgb,cam_d435i_depth,cam_d405_rgb,cam_d405_depth"

import time
import numpy as np
import vision
import servo
from fase1_seguimiento import Fase1


def drive(controller, cmd, dur, hz=30):
    t_end = time.time() + dur
    while time.time() < t_end:
        controller.set_velocities(cmd)
        time.sleep(1.0 / hz)
    controller.set_velocities({})


def move_head_to(controller, tilt, pan, timeout=8.0):
    t_end = time.time() + timeout
    while time.time() < t_end:
        st = controller.get_state()
        cmd = {}
        for j, tgt in (("head_tilt_up", tilt), ("head_pan_counterclockwise", pan)):
            e = tgt - st[j]
            if abs(e) > 0.02:
                cmd[j] = float(np.clip(3.0 * e, -1, 1))
        if not cmd:
            break
        controller.set_velocities(cmd)
        time.sleep(1 / 30)
    controller.set_velocities({})
    time.sleep(0.2)


def main():
    from stretch_toolkit import (
        controller, teleop, merge_proportional,
        HEAD_RGB_CAMERA, HEAD_DEPTH_CAMERA, WRIST_RGB_CAMERA, WRIST_DEPTH_CAMERA,
    )
    cams = (HEAD_RGB_CAMERA, HEAD_DEPTH_CAMERA, WRIST_RGB_CAMERA, WRIST_DEPTH_CAMERA)
    app = Fase1(controller, teleop, cams, merge_proportional)
    app.head_search = True  # probamos el mecanismo real de busqueda de cabeza

    print("[app] arrancando...", flush=True)
    controller.get_state()
    for _ in range(50):
        if WRIST_RGB_CAMERA.get_frame() is not None:
            break
        time.sleep(0.1)

    app.target = vision.blue_target()
    print("[app] objetivo azul inyectado", flush=True)

    def run_phase(cam, secs):
        """Corre el loop de compute() y registra el error en 'cam'."""
        errs = []
        t_end = time.time() + secs
        while time.time() < t_end:
            hf = HEAD_RGB_CAMERA.get_frame()
            wf = WRIST_RGB_CAMERA.get_frame()
            auto, _, _ = app.compute(hf, wf)
            controller.set_velocities(auto)  # sin teleop en el test
            f = cam.get_frame()
            if f is not None:
                o = vision.find_object(f, app.target)
                if o:
                    errs.append(servo.error_norm(*vision.centering_error(o["centroid"], f.shape)))
            time.sleep(1 / 30)
        controller.set_velocities({})
        return errs

    # ----- WRIST: descentra el brazo y deja que compute() lo recentre -----
    drive(controller, {"wrist_yaw_counterclockwise": 0.5}, 0.7)
    wrist_errs = run_phase(WRIST_RGB_CAMERA, 10)

    # ----- HEAD: desde pan=0, la BUSQUEDA debe barrer, encontrar y centrar -----
    move_head_to(controller, tilt=-0.5, pan=0.0)
    head_errs = run_phase(HEAD_RGB_CAMERA, 14)

    controller.set_velocities({})
    try:
        controller.stop()
    except Exception:
        pass

    def verdict(errs):
        if len(errs) < 5:
            return False, errs[:1], errs[-1:]
        return (np.median(errs[-5:]) < 0.10), errs[0], np.median(errs[-5:])

    hok, h0, h1 = verdict(head_errs)
    wok, w0, w1 = verdict(wrist_errs)
    print(f"[app] HEAD  err {h0} -> {h1}  {'PASS' if hok else 'FAIL'}", flush=True)
    print(f"[app] WRIST err {w0} -> {w1}  {'PASS' if wok else 'FAIL'}", flush=True)
    print(f"\n[app] ===== {'PASS' if (hok and wok) else 'FAIL'} =====", flush=True)


if __name__ == "__main__":
    main()
