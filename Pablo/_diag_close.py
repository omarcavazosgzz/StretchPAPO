"""DIAGNOSTICO: capturar la pose donde el brazo se ATORA al extender (agarre lateral).

Corre el pipeline REAL (sim_setup + position_for_grasp + grasp_object lateral) pero
intercepta el cierre para volcar la verdad del sim (base, lift, arm, centro de agarre
y dedos reales con reintento) y GUARDAR vistas (cabeza, muneca, nav) en ese instante.

    STRETCH_FIXED_SPAWN="2.25,-0.8,90" uv run Pablo/_diag_close.py cubo_rojo
"""
import sys
import time
from pathlib import Path
import numpy as np
import cv2

OUT = Path(__file__).resolve().parent / "snaps"
OUT.mkdir(exist_ok=True)


def read_truth(sim, body):
    st = sim.pull_status()
    cube = np.array(st.object_poses[body][:3])
    out = {"cube": cube}
    for k in ("link_grasp_center", "rubber_tip_left", "rubber_tip_right",
              "link_gripper_finger_left", "link_gripper_finger_right"):
        for _ in range(10):
            st = sim.pull_status()
            g = st.camera_poses.get(k)
            if g is not None:
                p = np.array(g["pos"])
                if 0.0 < p[2] < 2.0 and np.linalg.norm(p[:2] - cube[:2]) < 2.5:
                    out[k] = p
                    break
            time.sleep(0.03)
    return out


def analyze_tips(t, log):
    cube = t["cube"]
    L = t.get("rubber_tip_left"); R = t.get("rubber_tip_right")
    if L is None or R is None:
        log("  (sin rubber_tip)")
        return
    mid = (L + R) / 2.0
    axis = R - L
    sep = float(np.linalg.norm(axis))
    ax = axis / (sep + 1e-9)               # eje de CIERRE (entre las puntas)
    d = cube - mid
    along_close = float(d @ ax)            # offset del cubo sobre el eje de cierre
    log(f"  PUNTAS: L={L}  R={R}")
    log(f"  separacion puntas={sep*100:.1f}cm  midpoint=({mid[0]:.3f},{mid[1]:.3f},{mid[2]:.3f})")
    log(f"  cubo sobre eje de CIERRE = {along_close*100:+.1f}cm (0=centrado; |.|<~0.6cm para apretar)")
    log(f"  cubo vs midpoint puntas: dz={d[2]*100:+.1f}cm  |xy|={np.linalg.norm(d[:2])*100:.1f}cm")


def main():
    args = [a.lower() for a in sys.argv[1:]]
    target = next((a for a in args if a not in ("ver", "view", "v", "top", "lateral")), "cubo_rojo")

    from sim_setup import start_kitchen
    from detection import OracleDetector, resolve_name
    from control import PosServo
    import grasp_lib
    from grasp_lib import position_for_grasp, grasp_object
    from stretch_mujoco.enums.stretch_cameras import StretchCameras

    log = lambda m: print(m, flush=True)
    controller, model = start_kitchen(cameras=("cam_d435i_rgb", "cam_d435i_depth",
                                               "cam_d405_rgb", "cam_d405_depth", "cam_nav_rgb"),
                                      headless=True)
    sim = controller.sim
    det = OracleDetector(sim, model)
    servo = PosServo(sim, controller, model)
    body = resolve_name(target)
    HEAD = StretchCameras.cam_d435i_rgb
    WRIST = StretchCameras.cam_d405_rgb
    NAV = StretchCameras.cam_nav_rgb
    HEAD_D, WRIST_D = StretchCameras.cam_d435i_depth, StretchCameras.cam_d405_depth

    def snap(tag):
        allf = sim.pull_camera_data().get_all(use_depth_color_map=False)
        for cam, nm in {HEAD: f"stall_{tag}_head", WRIST: f"stall_{tag}_wrist",
                        NAV: f"stall_{tag}_nav"}.items():
            f = allf.get(cam)
            if f is None:
                continue
            f = np.ascontiguousarray(f)
            d = det.detect(cam, body)
            if d and d.in_frame:
                cv2.circle(f, (int(d.centroid[0]), int(d.centroid[1])), 9, (0, 0, 255), 2)
            cv2.imwrite(str(OUT / f"{nm}.png"), f)

    orig_close = grasp_lib._close_and_lift
    def patched(controller, sim, servo, body, log):
        time.sleep(0.4)
        t = read_truth(sim, body)
        s = controller.get_state()
        log("================ POSE PRE-CIERRE (verdad) ================")
        log(f"base=({s['base_x']:.3f},{s['base_y']:.3f},{np.degrees(s['base_theta']):.1f}deg) "
            f"lift={s['lift_up']:.3f} arm={s['arm_out']:.3f} yaw={s['wrist_yaw_counterclockwise']:.3f}")
        log(f"cubo={t['cube']}")
        gc = t.get("link_grasp_center")
        fl = t.get("link_gripper_finger_left")
        fr = t.get("link_gripper_finger_right")
        log(f"grasp_center VERDAD={gc}")
        log(f"finger_left={fl}  finger_right={fr}")
        if gc is not None:
            log(f"cubo - grasp_center = {t['cube'] - gc}")
        analyze_tips(t, log)
        snap("preclose")
        r = orig_close(controller, sim, servo, body, log)
        time.sleep(0.4)
        t2 = read_truth(sim, body)
        log("---- despues de cerrar ----")
        analyze_tips(t2, log)
        return r
    grasp_lib._close_and_lift = patched

    log(f"[diag] objetivo='{target}'. Posicionando (lateral)...")
    obj, ok = position_for_grasp(controller, sim, det, model, servo, body,
                                 HEAD, HEAD_D, WRIST, method="lateral", log=log)
    if ok:
        grasp_object(controller, sim, det, model, servo, body, WRIST, WRIST_D, obj,
                     method="lateral", log=log)
    controller.stop()


if __name__ == "__main__":
    main()
