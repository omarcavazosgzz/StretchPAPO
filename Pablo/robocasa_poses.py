"""
Geometria de RoboCasa para posicionar el gripper sobre el objeto:
imprime pose del efector (EE), de la base y de los objetos, y limites de juntas.

Uso:
    uv run Pablo/robocasa_poses.py
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ["STRETCH_SIM_HEADLESS"] = "1"
os.environ["STRETCH_SIM_CAMERAS"] = "cam_d405_rgb,cam_d405_depth"

import json
import time
from pathlib import Path
import numpy as np

CONFIG = Path(__file__).resolve().parent.parent / "stretch_toolkit" / "sim_config.json"


def main():
    original = CONFIG.read_text()
    cfg = json.loads(original)
    cfg.setdefault("robocasa", {})["enabled"] = True
    CONFIG.write_text(json.dumps(cfg, indent=2))
    try:
        import stretch_toolkit
        from stretch_toolkit import controller

        print("[pose] arrancando RoboCasa (~40s)...", flush=True)
        st = controller.get_state()
        time.sleep(0.5)
        sim = stretch_toolkit._sim

        np.set_printoptions(precision=3, suppress=True)
        ee = sim.get_ee_pose()
        print(f"[pose] EE (gripper) world pos = {ee[:3, 3]}", flush=True)
        bx, by, bt = sim.get_base_pose()
        print(f"[pose] base world = x={bx:.3f} y={by:.3f} theta={bt:.3f}", flush=True)
        print(f"[pose] joints: lift={st['lift_up']:.3f} arm={st['arm_out']:.3f} "
              f"wyaw={st['wrist_yaw_counterclockwise']:.3f} wpitch={st['wrist_pitch_up']:.3f}", flush=True)

        for name in controller.list_scene_objects():
            if name in ("base_link", "link_docking_station"):
                continue
            p = controller.get_object_pose(name)
            if p:
                d = np.array([p['x'] - ee[0, 3], p['y'] - ee[1, 3], p['z'] - ee[2, 3]])
                print(f"[pose] obj {name}: world=({p['x']:.3f},{p['y']:.3f},{p['z']:.3f})  "
                      f"EE->obj delta=({d[0]:.2f},{d[1]:.2f},{d[2]:.2f}) dist={np.linalg.norm(d):.2f}m", flush=True)

        # limites de lift/arm para saber el alcance
        try:
            from stretch_mujoco.enums.actuators import Actuators
            lims = sim.pull_joint_limits()
            for a in (Actuators.lift, Actuators.arm):
                print(f"[pose] limite {a.name}: {lims.get(a)}", flush=True)
        except Exception as e:
            print(f"[pose] limites n/d: {e}", flush=True)

    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        CONFIG.write_text(original)
        print("[pose] config restaurado", flush=True)
        try:
            controller.stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()
