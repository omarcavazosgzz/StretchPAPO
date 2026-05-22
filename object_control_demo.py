"""
Object control demo — cycle through scene objects and move them up/down.

Controls:
    RB / m  →  next object
    LB / n  →  previous object
    z / DPAD_UP    →  move up
    x / DPAD_DOWN  →  move down
    w / s          →  move +y / -y
    a / d          →  move -x / +x
    i / k          →  pitch +/-
    j / l          →  yaw   +/-
    u / o          →  roll  +/-
    Y / g          →  gravity OFF (float)
    B / f          →  gravity ON  (fall)

Run with:
    uv run object_control_demo.py
"""

from stretch_toolkit import controller, BACKEND_NAME, ObjectControlProvider
import stretch_toolkit.input as inp
import time

MOVE_STEP = 0.02    # metres per tick
ROT_STEP  = 0.05    # radians per tick (~3°)
obj_ctrl  = ObjectControlProvider(move_step=MOVE_STEP, rot_step=ROT_STEP)


def main():
    print(f"\n=== Running on {BACKEND_NAME} backend ===\n")

    objects = controller.list_scene_objects()

    if not objects:
        print("No movable objects found in the scene.")
        return

    selected = 0
    print(f"Found {len(objects)} object(s): {objects}")
    print(f"Selected: {objects[selected]}")
    print("RB/m = next, LB/n = prev | z/x=up/down | wasd=xy | uiojkl=rot | Y/g=float B/f=fall | Ctrl+C to quit.\n")

    try:
        while True:
            # ── Cycle selection ───────────────────────────────────────
            prev = selected
            selected = (selected + inp.rising_edge("RB", "m") - inp.rising_edge("LB", "n")) % len(objects)
            if selected != prev:
                print(f"Selected: {objects[selected]}")
                pose = controller.get_object_pose(objects[selected])
                if pose:
                    print(f"  x={pose['x']:.3f}  y={pose['y']:.3f}  z={pose['z']:.3f}  "
                          f"qw={pose['qw']:.3f}  qx={pose['qx']:.3f}  qy={pose['qy']:.3f}  qz={pose['qz']:.3f}")

            # ── Move / rotate + gravity ────────────────────────────────────
            delta, gravity = obj_ctrl.get_delta()
            controller.move_object_by(objects[selected], delta)
            controller.set_object_gravity(objects[selected], gravity)

            time.sleep(1 / 30)

    except KeyboardInterrupt:
        print("\nDone.")


if __name__ == "__main__":
    main()

