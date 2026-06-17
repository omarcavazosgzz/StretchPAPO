"""TEST DECISIVO: ?puede el gripper sujetar el cubo de 5cm con posicionamiento PERFECTO?

Coloca el cubo EXACTAMENTE entre los dedos abiertos (en el centro de agarre, sin
gravedad), cierra, reactiva gravedad y sube. Si el cubo sube con el gripper -> el
gripper SI puede; el problema es solo el posicionamiento. Si NO sube -> el gripper no
cierra lo suficiente / no aprieta (problema de hardware del modelo).

    uv run Pablo/_diag_canclose.py
"""
import time
import numpy as np


def read_pt(sim, key, near=None):
    for _ in range(10):
        cp = sim.pull_status().camera_poses
        g = cp.get(key)
        if g is not None:
            p = np.array(g["pos"])
            if 0.0 < p[2] < 2.0 and (near is None or np.linalg.norm(p[:2] - near) < 1.5):
                return p
        time.sleep(0.03)
    return None


def tips_sep(sim, near):
    L = read_pt(sim, "rubber_tip_left", near)
    R = read_pt(sim, "rubber_tip_right", near)
    if L is None or R is None:
        return None, None, None
    return L, R, float(np.linalg.norm(L - R))


def main():
    from sim_setup import start_kitchen
    from control import PosServo
    from grasp_lib import _wait_joint
    body = "cubo_rojo"
    log = lambda m: print(m, flush=True)
    controller, model = start_kitchen(cameras=("cam_d435i_rgb", "cam_d405_rgb"))
    sim = controller.sim
    servo = PosServo(sim, controller, model)
    time.sleep(0.5)

    # pose de brazo limpia, muneca horizontal, gripper ABIERTO
    servo.move_to({"lift_up": 0.7, "arm_out": 0.25, "wrist_pitch_up": 0.0,
                   "wrist_yaw_counterclockwise": 0.0, "gripper_open": 0.5})
    t = time.time()
    while time.time() - t < 4:
        servo.hold(); time.sleep(1/30)

    base = np.array([controller.get_state()["base_x"], controller.get_state()["base_y"]])
    gc = read_pt(sim, "link_grasp_center", base)
    log(f"grasp_center (entre los dedos abiertos) = {gc}")
    L, R, sep = tips_sep(sim, base)
    log(f"separacion puntas ABIERTO = {sep*100:.1f}cm")

    # poner el cubo EXACTAMENTE en el centro de agarre, sin gravedad
    sim.set_object_gravity(body, False)
    time.sleep(0.3)
    sim.set_object_pose(body, {"x": float(gc[0]), "y": float(gc[1]), "z": float(gc[2])})
    time.sleep(0.6)
    c = np.array(sim.pull_status().object_poses[body][:3])
    log(f"cubo teletransportado a = {c} (deberia ~= grasp_center)")

    # CERRAR
    log("cerrando...")
    servo.move_to({"gripper_open": -0.35})
    _wait_joint(controller, "gripper_open", -0.35, tol=0.08, timeout=3, servo=servo)
    time.sleep(0.8)
    L, R, sep = tips_sep(sim, base)
    c2 = np.array(sim.pull_status().object_poses[body][:3])
    log(f"separacion puntas CERRADO = {sep*100:.1f}cm   (cubo = 5.0cm)")
    log(f"cubo tras cerrar = {c2}")

    # reactivar gravedad y SUBIR
    sim.set_object_gravity(body, True)
    time.sleep(0.3)
    z_before = float(sim.pull_status().object_poses[body][2])
    lf = controller.get_state()["lift_up"]
    servo.move_to({"lift_up": lf + 0.2})
    _wait_joint(controller, "lift_up", lf + 0.2, tol=0.03, timeout=4, servo=servo)
    time.sleep(0.8)
    z_after = float(sim.pull_status().object_poses[body][2])
    log(f"cubo z {z_before:.3f} -> {z_after:.3f}  => {'SUJETADO ✓ (el gripper SI puede)' if z_after - z_before > 0.05 else 'SE CAYO (el gripper NO aprieta lo suficiente)'}")
    controller.stop()


if __name__ == "__main__":
    main()
