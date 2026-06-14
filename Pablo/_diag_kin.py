"""Diagnostico de cinematica/sensores para disenar el posicionamiento de agarre:
 - pose de base y del objeto (oraculo)
 - hacia que lado (frame base) se EXTIENDE el brazo (mide la camara de muneca al
   extender el brazo) y el offset del gripper
 - indice 'frente' del LiDAR (mueve la base hacia ATRAS y ve que rayo crece)
 - guarda nav snapshot
"""
import time
import numpy as np


def main():
    from sim_setup import start_kitchen
    from control import PosServo
    controller, model = start_kitchen(cameras=("cam_d435i_rgb", "cam_d405_rgb", "cam_nav_rgb"))
    sim = controller.sim
    servo = PosServo(sim, controller, model)
    time.sleep(0.5)

    st0 = controller.get_state()
    base = np.array([st0["base_x"], st0["base_y"], st0["base_theta"]])
    obj = np.array(sim.pull_status().object_poses["huevo"][:3])
    print(f"[kin] base x={base[0]:.2f} y={base[1]:.2f} theta={base[2]:+.2f} rad", flush=True)
    print(f"[kin] huevo x={obj[0]:.2f} y={obj[1]:.2f} z={obj[2]:.2f}", flush=True)
    dx, dy = obj[0] - base[0], obj[1] - base[1]
    print(f"[kin] objeto respecto base: dist={np.hypot(dx,dy):.2f}m  "
          f"angulo_mundo={np.degrees(np.arctan2(dy,dx)):+.0f}deg  "
          f"angulo_rel_base={np.degrees((np.arctan2(dy,dx)-base[2]+np.pi)%(2*np.pi)-np.pi):+.0f}deg", flush=True)

    # ── lado del brazo: posicion de la muneca (d405) en home vs brazo extendido ──
    def wrist_pos():
        return np.array(sim.pull_status().camera_poses["d405_rgb"]["pos"])
    w0 = wrist_pos()
    servo.move_to({"arm_out": st0["arm_out"] + 0.25, "lift_up": st0["lift_up"]})
    t = time.time()
    while time.time() - t < 4:
        servo.hold()
        if abs(controller.get_state()["arm_out"] - (st0["arm_out"] + 0.25)) < 0.03:
            break
        time.sleep(1/30)
    w1 = wrist_pos()
    dW = w1 - w0
    th = base[2]
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    dW_base = R.T @ dW[:2]
    print(f"[kin] al EXTENDER brazo 0.25: muneca se movio mundo=({dW[0]:+.2f},{dW[1]:+.2f},{dW[2]:+.2f})", flush=True)
    print(f"[kin]   -> en frame BASE: forward(x)={dW_base[0]:+.2f}  lateral(y)={dW_base[1]:+.2f}", flush=True)
    print(f"[kin]   => el brazo se extiende hacia base-{'+' if dW_base[1] > 0 else '-'}Y (lateral)", flush=True)
    print(f"[kin]   wrist en home (mundo): ({w0[0]:.2f},{w0[1]:.2f},{w0[2]:.2f})", flush=True)
    servo.move_to({"arm_out": st0["arm_out"]})  # retraer
    time.sleep(1.0)

    # ── LiDAR: indice frontal (mover base hacia ATRAS y ver que rayo crece) ──────
    r0 = controller.get_lidar_ranges()
    if r0 is not None:
        r0 = np.asarray(r0, float)
        print(f"[kin] LiDAR: {len(r0)} rayos. min={np.nanmin(r0[np.isfinite(r0)]):.2f} en idx={int(np.nanargmin(np.where(np.isfinite(r0),r0,np.inf)))}", flush=True)
        # mover hacia atras ~0.2m
        b0 = np.array([controller.get_state()["base_x"], controller.get_state()["base_y"]])
        t = time.time()
        while np.hypot(*(np.array([controller.get_state()["base_x"], controller.get_state()["base_y"]]) - b0)) < 0.2 and time.time()-t < 5:
            controller.set_velocities({"base_forward": -0.5})
            time.sleep(1/30)
        controller.set_velocities({"base_forward": 0.0}); time.sleep(0.4)
        r1 = np.asarray(controller.get_lidar_ranges(), float)
        delta = np.where(np.isfinite(r0) & np.isfinite(r1), r1 - r0, -np.inf)
        fwd_idx = int(np.argmax(delta))   # el rayo que MAS crecio al retroceder = frente
        print(f"[kin] al retroceder, el rayo que mas crecio (=FRENTE) es idx={fwd_idx} "
              f"(r0={r0[fwd_idx]:.2f}->r1={r1[fwd_idx]:.2f})", flush=True)
        # muestrear alrededor
        for off in (-90, -45, 0, 45, 90, 180):
            i = (fwd_idx + off) % len(r0)
            print(f"[kin]   rayo frente{off:+4d}deg (idx {i}): {r1[i]:.2f} m", flush=True)

    import cv2
    nav = sim.pull_camera_data().get_all(use_depth_color_map=False).get(__import__("stretch_mujoco.enums.stretch_cameras", fromlist=["StretchCameras"]).StretchCameras.cam_nav_rgb)
    if nav is not None:
        from pathlib import Path
        cv2.imwrite(str(Path(__file__).resolve().parent / "snaps" / "diag_kin_nav.png"), nav)
        print("[kin] guardado diag_kin_nav.png", flush=True)
    controller.stop()


if __name__ == "__main__":
    main()
