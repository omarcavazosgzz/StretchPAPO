"""Diagnostico: encontrar el CENTRO DE AGARRE real de los dedos vs la camara d405.
Imprime bodies/sites del gripper y su posicion mundial, y el offset camara->dedos en
el frame de la base (radial/along/vertical), para posicionar los DEDOS (no la camara)
sobre el objeto."""
import numpy as np
import mujoco


def main():
    from sim_setup import start_kitchen
    from control import PosServo
    controller, model = start_kitchen(cameras=("cam_d435i_rgb", "cam_d405_rgb"))
    sim = controller.sim
    servo = PosServo(sim, controller, model)
    import time
    time.sleep(0.4)

    import sys
    pitch = 0.0 if any(a in ("h", "horizontal", "lat") for a in sys.argv[1:]) else -1.5
    print(f"\n### wrist_pitch={pitch} ({'HORIZONTAL/lateral' if pitch==0 else 'ABAJO/top'}) ###", flush=True)
    servo.move_to({"lift_up": 0.9, "arm_out": 0.2, "wrist_pitch_up": pitch,
                   "wrist_yaw_counterclockwise": 0.0, "gripper_open": 0.4})
    t = time.time()
    while time.time() - t < 4:
        servo.hold(); time.sleep(1/30)

    # listar bodies con nombres de gripper/dedos/agarre
    print("\n=== bodies con grip/finger/grasp/hand ===", flush=True)
    data = sim.mjdata if hasattr(sim, "mjdata") else None
    names = []
    for i in range(model.nbody):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if nm and any(k in nm.lower() for k in ("grip", "finger", "grasp", "hand", "d405", "wrist")):
            names.append((i, nm))
    st = sim.pull_status()
    s = controller.get_state()
    bx, by, th = s["base_x"], s["base_y"], s["base_theta"]
    lift, arm = s["lift_up"], s["arm_out"]
    base = np.array([bx, by])
    rad = np.array([np.sin(th), -np.cos(th)])      # extension del brazo (-Y_local)
    along = np.array([np.cos(th), np.sin(th)])     # a lo largo del mostrador (x_local)
    print(f"base=({bx:.3f},{by:.3f},th={np.degrees(th):.1f}deg) lift={lift:.3f} arm={arm:.3f}", flush=True)
    cam = np.array(st.camera_poses["d405_rgb"]["pos"])
    print(f"d405_rgb cam pos = {cam}", flush=True)
    for key in ("link_grasp_center", "link_gripper_finger_left", "link_gripper_finger_right"):
        if key in st.camera_poses:
            p = np.array(st.camera_poses[key]["pos"])
            radial_dist = float((p[:2] - base) @ rad)        # dist base->grasp center (radial)
            along_dist = float((p[:2] - base) @ along)
            print(f"  {key:30s} pos={p}", flush=True)
            print(f"     desde BASE: radial={radial_dist:+.3f} along={along_dist:+.3f}  z={p[2]:.3f}  z-lift={p[2]-lift:+.3f}  radial-(home+arm)={radial_dist-(0.34+arm):+.3f}", flush=True)
    controller.stop()


if __name__ == "__main__":
    main()
