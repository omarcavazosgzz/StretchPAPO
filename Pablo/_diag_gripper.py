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

    # poner el brazo extendido + muneca abajo (pose de agarre top)
    servo.move_to({"lift_up": 0.9, "arm_out": 0.2, "wrist_pitch_up": -1.5,
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
    cam = np.array(st.camera_poses["d405_rgb"]["pos"])
    print(f"d405_rgb cam pos = {cam}", flush=True)
    bx, by, th = controller.get_state()["base_x"], controller.get_state()["base_y"], controller.get_state()["base_theta"]
    rad = np.array([np.sin(th), -np.cos(th)])      # extension del brazo (-Y_local)
    along = np.array([np.cos(th), np.sin(th)])     # a lo largo del mostrador (x_local)
    for key in ("link_grasp_center", "link_gripper_finger_left", "link_gripper_finger_right"):
        if key in st.camera_poses:
            p = np.array(st.camera_poses[key]["pos"])
            off = p - cam
            print(f"  {key:30s} pos={p}", flush=True)
            print(f"     offset (centro-camara) mundo={off}  radial={float(off[:2]@rad):+.3f} along={float(off[:2]@along):+.3f} dz={off[2]:+.3f}", flush=True)
    controller.stop()


if __name__ == "__main__":
    main()
