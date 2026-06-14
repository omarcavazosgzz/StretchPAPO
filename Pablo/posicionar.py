"""
FASE 1.5 - Posicionamiento de agarre (sin chocar).

El robot recuerda donde esta el objeto, calcula una pose comoda para agarrar
(paralelo al mostrador, brazo hacia el objeto) y navega ahi por el pasillo con
freno LiDAR. Luego extiende el brazo para verificar que ALCANZA el objeto.

    uv run Pablo/posicionar.py            # huevo
    uv run Pablo/posicionar.py tomate
    uv run Pablo/posicionar.py huevo ver  # con ventana del visor

Salida: Pablo/snaps/pos_{nav,head,wrist}.png + reporte de alcance.
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
    from positioning import (remember_object, compute_grasp_xy, goto_pose,
                             face_arm_at_object, GRIPPER_HOME_OFFSET, _base_pose)
    from stretch_mujoco.enums.stretch_cameras import StretchCameras

    log = lambda m: print(m, flush=True)
    log(f"[pos] objetivo='{target}' ({'VISOR' if view else 'headless'}). Arrancando...")
    controller, model = start_kitchen(cameras=("cam_d435i_rgb", "cam_d405_rgb", "cam_nav_rgb"),
                                      headless=not view)
    sim = controller.sim
    servo = PosServo(sim, controller, model)
    det = OracleDetector(sim, model)
    body = resolve_name(target)
    HEAD, WRIST = StretchCameras.cam_d435i_rgb, StretchCameras.cam_d405_rgb
    time.sleep(0.5)

    # 1) RECORDAR el objeto
    obj = remember_object(sim, body)
    if obj is None:
        log(f"[pos] objeto '{body}' no existe"); controller.stop(); return
    log(f"[pos] objeto recordado en mundo: ({obj[0]:.2f},{obj[1]:.2f},{obj[2]:.2f})")

    # 2) Navegar al PUNTO de agarre (en el pasillo, sin chocar) y luego GIRAR para
    #    que el brazo apunte exactamente al objeto.
    st = controller.get_state()
    tx, ty = compute_grasp_xy(obj[:2], (st["base_x"], st["base_y"]), grasp_dist=0.48)
    res = goto_pose(controller, tx, ty, None, log=log)   # ttheta=None: lo fija face_arm_at
    face_arm_at_object(controller, obj[:2], log=log)

    # 3) Preparar brazo hacia el objeto: lift a la altura del objeto + extender lo
    #    necesario segun la distancia REAL base->objeto. (El gripper queda ~0.12 m
    #    por encima del setpoint de lift, asi que bajamos el lift para compensar.)
    bx, by, _ = _base_pose(controller)
    base_obj_dist = float(np.hypot(obj[0] - bx, obj[1] - by))
    lift_target = float(np.clip(obj[2] - 0.12, 0.2, 1.05))
    arm_target = float(np.clip(base_obj_dist - GRIPPER_HOME_OFFSET, 0.0, 0.5))
    log(f"[pos] base->objeto={base_obj_dist:.2f}m -> lift={lift_target:.2f} arm_out={arm_target:.2f}")
    servo.move_to({"lift_up": lift_target, "wrist_yaw_counterclockwise": 0.0, "wrist_pitch_up": -0.3})
    t = time.time()
    while time.time() - t < 4 and abs(controller.get_state()["lift_up"] - lift_target) > 0.03:
        servo.hold(); time.sleep(1/30)
    servo.move_to({"arm_out": arm_target})
    t = time.time()
    while time.time() - t < 5 and abs(controller.get_state()["arm_out"] - arm_target) > 0.03:
        servo.hold(); time.sleep(1/30)
    time.sleep(0.5)

    # 4) Verificar alcance: distancia gripper(muneca) -> objeto
    wpos = np.array(sim.pull_status().camera_poses["d405_rgb"]["pos"])
    reach = float(np.linalg.norm(wpos - obj))
    log(f"[pos] muneca(gripper) en ({wpos[0]:.2f},{wpos[1]:.2f},{wpos[2]:.2f}); "
        f"distancia al objeto = {reach:.2f} m")

    # 5) Orientar la camara del BRAZO hacia el mostrador y verificar que VE el objeto
    #    (esta es la metrica real: listo para la deteccion con la camara del brazo).
    servo.move_to({"wrist_pitch_up": 0.1, "wrist_yaw_counterclockwise": 0.0})
    time.sleep(1.0)
    dW = det.detect(WRIST, body)
    obj_en_brazo = bool(dW and dW.in_frame)
    log(f"[pos] objeto en la camara del BRAZO: {obj_en_brazo}")

    allf = sim.pull_camera_data().get_all(use_depth_color_map=False)
    tags = {HEAD: "head", WRIST: "wrist", StretchCameras.cam_nav_rgb: "nav"}
    for cam, tag in tags.items():
        f = allf.get(cam)
        if f is None: continue
        f = np.ascontiguousarray(f)   # los frames rotados (rot90) no son escribibles
        d = det.detect(cam, body)
        if d and d.in_frame:
            x, y = int(d.centroid[0]), int(d.centroid[1])
            cv2.circle(f, (x, y), 9, (0, 0, 255), 2)
            cv2.putText(f, target, (max(0,x-18), max(14,y-12)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,255), 1)
        cv2.imwrite(str(OUT / f"pos_{tag}.png"), f)
    log(f"[pos] guardadas pos_head/wrist/nav.png")

    ok = (not res['blocked']) and (obj_en_brazo or reach < 0.6)
    log(f"\n[pos] ===== {'OK posicionado' if ok else 'REVISAR'} ===== "
        f"sin_choque={not res['blocked']}  reach={reach:.2f}m  objeto_en_brazo={obj_en_brazo}")

    if view:
        log("[pos] ventana abierta; cierra con X o Ctrl+C.")
        try:
            while sim.is_running(): time.sleep(0.2)
        except KeyboardInterrupt: pass
    controller.stop()


if __name__ == "__main__":
    main()
