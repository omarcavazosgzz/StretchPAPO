"""Aisla el control por POSICION absoluta (move_to_many): 1 junta vs 2 juntas,
SIN usar la emulacion de velocidad del toolkit (para descartar residuos)."""
import time


def main():
    from sim_setup import start_kitchen
    controller, model = start_kitchen()
    sim = controller.sim

    def st():
        s = controller.get_state()
        return s["head_pan_counterclockwise"], s["head_tilt_up"]

    def hold(targets, secs=1.6):
        end = time.time() + secs
        while time.time() < end:
            sim.move_to_many(targets)
            time.sleep(1 / 60)
        time.sleep(0.4)

    time.sleep(1.0)
    p0, t0 = st()

    hold({"head_pan": p0 + 0.5})
    p1, t1 = st(); print(f"[mj] move_to pan solo     dpan={p1-p0:+.3f} (obj +0.50) dtilt={t1-t0:+.3f}", flush=True)

    p0, t0 = st()
    hold({"head_tilt": t0 - 0.4})
    p1, t1 = st(); print(f"[mj] move_to tilt solo    dtilt={t1-t0:+.3f} (obj -0.40) dpan={p1-p0:+.3f}", flush=True)

    p0, t0 = st()
    hold({"head_pan": p0 + 0.5, "head_tilt": t0 + 0.3})
    p1, t1 = st(); print(f"[mj] move_to pan+tilt     dpan={p1-p0:+.3f} (obj +0.50) dtilt={t1-t0:+.3f} (obj +0.30)", flush=True)

    controller.stop()


if __name__ == "__main__":
    main()
