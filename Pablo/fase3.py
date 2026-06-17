"""FASE 3 - AGARRAR y LLEVAR el objeto a otra zona (p.ej. cerca del FREGADERO).

Agarra el objeto (Fase 2, lateral), RETRAE el brazo (para no barrer el mostrador),
navega en PARALELO a la zona destino y lo COLOCA. Graba un video de todo.

    STRETCH_FIXED_SPAWN="2.25,-0.8,90" uv run Pablo/fase3.py cubo_rojo
Destino por defecto: cerca del fregadero (x<2.28). Cambialo con  dx,dy  en argv:
    uv run Pablo/fase3.py cubo_rojo 2.15 -0.33
Salida: Pablo/snaps/fase3_llevar.mp4
"""
import sys
import time
import threading
from pathlib import Path
import numpy as np
import cv2

OUT = Path(__file__).resolve().parent / "snaps"
OUT.mkdir(exist_ok=True)
FONT = cv2.FONT_HERSHEY_SIMPLEX
TILE = {"nav": (480, 360), "cabeza": (270, 360), "muneca": (480, 360)}


def main():
    args = sys.argv[1:]
    target = next((a for a in args if not _isnum(a) and a not in ("ver", "view")), "cubo_rojo")
    nums = [float(a) for a in args if _isnum(a)]
    place_x = nums[0] if len(nums) >= 1 else 2.13      # cerca del fregadero (x<2.28)
    place_y = nums[1] if len(nums) >= 2 else -0.33
    place_z = nums[2] if len(nums) >= 3 else 0.95

    from sim_setup import start_kitchen
    from detection import OracleDetector, resolve_name
    from control import PosServo
    from grasp_lib import grasp_with_retries, place_object_lateral, _wait_joint
    from positioning import nav_to_parallel, face_arm_at_object, _base_pose
    from stretch_mujoco.enums.stretch_cameras import StretchCameras

    log = lambda m: print(m, flush=True)
    controller, model = start_kitchen(cameras=("cam_d435i_rgb", "cam_d435i_depth",
                                               "cam_d405_rgb", "cam_d405_depth", "cam_nav_rgb"),
                                      headless=True)
    sim = controller.sim
    det = OracleDetector(sim, model)
    servo = PosServo(sim, controller, model)
    body = resolve_name(target)
    HEAD, HEAD_D = StretchCameras.cam_d435i_rgb, StretchCameras.cam_d435i_depth
    WRIST, WRIST_D = StretchCameras.cam_d405_rgb, StretchCameras.cam_d405_depth
    NAV = StretchCameras.cam_nav_rgb
    cams = [(NAV, "nav"), (HEAD, "cabeza"), (WRIST, "muneca")]

    frames = []
    state = {"rec": True, "phase": "inicio"}

    def grab():
        while state["rec"]:
            try:
                data = sim.pull_camera_data().get_all(use_depth_color_map=False)
            except Exception:
                time.sleep(0.05); continue
            tiles = []
            for cam, label in cams:
                f = data.get(cam); w, h = TILE[label]
                if f is None:
                    tiles.append(np.zeros((h, w, 3), np.uint8)); continue
                f = np.ascontiguousarray(f)
                try:
                    d = det.detect(cam, body)
                    if d and d.in_frame:
                        cv2.circle(f, (int(d.centroid[0]), int(d.centroid[1])), 10, (0, 0, 255), 2)
                except Exception:
                    pass
                f = cv2.resize(f, (w, h))
                cv2.putText(f, label, (8, 26), FONT, 0.8, (0, 255, 0), 2)
                tiles.append(f)
            frame = np.hstack(tiles)
            bar = np.zeros((34, frame.shape[1], 3), np.uint8)
            cv2.putText(bar, f"Fase 3: llevar {target} cerca del fregadero  |  {state['phase']}",
                        (8, 24), FONT, 0.6, (255, 255, 255), 1)
            frames.append(np.vstack([bar, frame]))
            time.sleep(1 / 12.0)

    th = threading.Thread(target=grab, daemon=True); th.start()
    time.sleep(0.5)

    # ---- AGARRAR (Fase 2, con REINTENTOS) ----
    state["phase"] = "Fase 1/2: localizar + agarrar (con reintentos)"
    obj, grabbed = grasp_with_retries(controller, sim, det, model, servo, body,
                                      HEAD, HEAD_D, WRIST, WRIST_D, method="lateral",
                                      retries=3, log=log)
    log(f"[fase3] agarre: {'OK' if grabbed else 'FALLO'}")

    if grabbed:
        # ---- RETRAER el brazo y subir (para no barrer el mostrador al moverse) ----
        # IMPORTANTE: NO rotar la muneca (se queda INCLINADA como quedo al agarrar). Rotarla
        # a horizontal gira el cubo y lo zafa del agarre (marginal) -> lo dejaba caer.
        state["phase"] = "recoger brazo (sostiene el objeto)"
        log("[fase3] recojo el brazo (alto) sosteniendo el objeto, sin rotar la muneca...")
        lf = controller.get_state()["lift_up"]
        servo.move_to({"lift_up": float(np.clip(lf + 0.05, 0.6, 1.05))})
        _wait_joint(controller, "lift_up", float(np.clip(lf + 0.05, 0.6, 1.05)), tol=0.03, timeout=3, servo=servo)
        servo.move_to({"arm_out": 0.0})
        _wait_joint(controller, "arm_out", 0.0, tol=0.03, timeout=4, servo=servo)

        # ---- TRASLADAR a lo largo del mostrador con un manejo RECTO y SUAVE de la base
        #      (una sola aceleracion/frenado, con el brazo SOSTENIDO). base_translate iba a
        #      PASOS (varios arranques/frenones) y zafaba el cubo; el nav reactivo (esquive)
        #      tambien. El robot esta PARALELO -> base_forward corre A-LO-LARGO del mostrador.
        from positioning import stop_base
        state["phase"] = "trasladar (suave) cerca del fregadero"
        log(f"[fase3] trasladando (suave, con rampa) a lo largo del mostrador hacia x={place_x:.2f}...")
        # CLAVE: RAMPEAR la velocidad (acelerar y frenar GRADUAL). Comandar la velocidad de
        # golpe da un tiron: el mastil/brazo da LATIGAZO y avienta el cubo. Subir/bajar la
        # velocidad poco a poco = aceleracion suave = el cubo no se zafa.
        DT = 1.0 / 30
        VMAX = 0.42          # velocidad de crucero (normalizada)
        VMIN = 0.16          # velocidad minima de avance (vence friccion)
        RAMP = 0.9           # segundos para llegar a VMAX (rampa)
        v = 0.0
        t0 = time.time(); last = 0.0
        while time.time() - t0 < 24:
            s = controller.get_state()
            bx = s["base_x"]; dx = place_x - bx
            if abs(dx) < 0.05:
                break
            c = float(np.cos(s["base_theta"]))      # base_forward+ mueve base_x en +cos(th)
            sgn = 1.0 if (dx * c) >= 0 else -1.0     # signo de base_forward para acercarse
            if abs(dx) < 0.22:                       # cerca -> FRENAR gradual
                v = max(VMIN, v - VMAX * DT / RAMP)
            else:                                    # lejos -> ACELERAR gradual hasta VMAX
                v = min(VMAX, v + VMAX * DT / RAMP)
            controller.set_velocities({"base_forward": sgn * v, "base_counterclockwise": 0.0})
            servo.hold()                            # mantiene brazo+gripper (cubo sujeto)
            if time.time() - last > 1.0:
                log(f"[fase3]   traslado: base_x={bx:.3f} v={v:.2f} (objetivo {place_x:.2f})")
                last = time.time()
            time.sleep(DT)
        stop_base(controller); servo.sync()
        # NO re-orientar con face_arm aqui: rotar la base EN SITIO con el cubo sostenido lo
        # AVIENTA (fuerza centripeta en el gripper extendido). Tras el manejo RECTO el robot
        # sigue PARALELO al mostrador, asi que ya apunta al punto de colocacion.

        # ---- COLOCAR ----
        state["phase"] = "colocar el objeto"
        place_object_lateral(controller, sim, servo, (place_x, place_y, place_z), log=log)

        # ---- verificar ----
        time.sleep(0.8)
        pos = np.array(sim.pull_status().object_poses[body][:3])
        d_place = float(np.hypot(pos[0] - place_x, pos[1] - place_y))
        ok_place = pos[2] > 0.6 and d_place < 0.35
        state["phase"] = "LISTO" if ok_place else "revisar colocacion"
        log(f"[fase3] objeto en ({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}); "
            f"dist al destino={d_place:.2f}m -> {'COLOCADO OK' if ok_place else 'REVISAR'}")
    else:
        state["phase"] = "no se pudo agarrar"

    time.sleep(1.2)
    state["rec"] = False; th.join(timeout=3)
    if frames:
        h, w = frames[0].shape[:2]
        path = str(OUT / "fase3_llevar.mp4")
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 12, (w, h))
        for fr in frames:
            vw.write(fr)
        vw.release()
        log(f"[fase3] VIDEO: {path} ({len(frames)} frames)")
    controller.stop()


def _isnum(s):
    try:
        float(s); return True
    except ValueError:
        return False


if __name__ == "__main__":
    main()
