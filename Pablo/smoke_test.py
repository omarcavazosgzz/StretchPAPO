"""
Smoke test (headless desde mi punto de vista): verifica que en este entorno
el simulador arranca, las camaras entregan frames y el control responde.

No abre ventanas de OpenCV. El viewer de MuJoCo si se abre (DISPLAY=:1),
pero la verificacion es 100% por consola.

IMPORTANTE: el simulador usa multiprocessing 'spawn', que re-importa este
modulo. Por eso TODA la ejecucion va dentro de if __name__ == "__main__".

Uso:
    uv run Pablo/smoke_test.py
"""
import time
import sys
from stretch_toolkit import (
    controller, BACKEND_NAME,
    HEAD_RGB_CAMERA, WRIST_RGB_CAMERA, HEAD_DEPTH_CAMERA,
)


def wait_for_frame(camera, label, timeout=20.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        frame = camera.get_frame()
        if frame is not None:
            return frame
        time.sleep(0.1)
    return None


def main():
    import numpy as np
    t0 = time.time()
    print(f"[smoke] backend = {BACKEND_NAME}", flush=True)
    ok = True

    def fail(msg):
        nonlocal ok
        ok = False
        print(f"[smoke] FAIL: {msg}", flush=True)

    try:
        # 1) Forzar arranque del sim accediendo al controller
        print("[smoke] arrancando simulador (puede tardar)...", flush=True)
        state = controller.get_state()
        print(f"[smoke] sim arrancado en {time.time()-t0:.1f}s. "
              f"base=({state['base_x']:.2f},{state['base_y']:.2f},th={state['base_theta']:.2f})", flush=True)
        print(f"[smoke] head_pan inicial = {state['head_pan_counterclockwise']:.3f} rad", flush=True)

        # 2) Frame de HEAD RGB
        print("[smoke] esperando frame de HEAD RGB...", flush=True)
        head_frame = wait_for_frame(HEAD_RGB_CAMERA, "HEAD RGB")
        if head_frame is None:
            fail("HEAD RGB no entrego frame en 20s")
        else:
            print(f"[smoke] HEAD RGB shape={head_frame.shape} dtype={head_frame.dtype}", flush=True)

        # 3) Frame de WRIST RGB
        print("[smoke] esperando frame de WRIST RGB...", flush=True)
        wrist_frame = wait_for_frame(WRIST_RGB_CAMERA, "WRIST RGB")
        if wrist_frame is None:
            fail("WRIST RGB no entrego frame en 20s")
        else:
            print(f"[smoke] WRIST RGB shape={wrist_frame.shape} dtype={wrist_frame.dtype}", flush=True)

        # 4) Profundidad
        depth = HEAD_DEPTH_CAMERA.get_frame()
        if depth is not None:
            print(f"[smoke] HEAD DEPTH shape={depth.shape} dtype={depth.dtype} "
                  f"min={float(depth.min()):.3f} max={float(depth.max()):.3f}", flush=True)
        else:
            print("[smoke] aviso: HEAD DEPTH None (no critico)", flush=True)

        # 5) Control: mover head_pan y medir cambio real
        print("[smoke] comandando head_pan +0.6 por ~1.5s y midiendo...", flush=True)
        pan0 = controller.get_state()['head_pan_counterclockwise']
        t_end = time.time() + 1.5
        while time.time() < t_end:
            controller.set_velocities({"head_pan_counterclockwise": 0.6})
            time.sleep(1/30)
        controller.set_velocities({})
        time.sleep(0.3)
        pan1 = controller.get_state()['head_pan_counterclockwise']
        dpan = pan1 - pan0
        print(f"[smoke] head_pan {pan0:.3f} -> {pan1:.3f} (delta={dpan:+.3f} rad)", flush=True)
        if abs(dpan) < 0.02:
            fail(f"head_pan apenas se movio (delta={dpan:+.3f})")

        # 6) LiDAR (base para anti-colision)
        ranges = controller.get_lidar_ranges()
        if ranges is not None:
            finite = ranges[np.isfinite(ranges)]
            mn = finite.min() if len(finite) else float('inf')
            print(f"[smoke] LiDAR: {len(ranges)} rayos, {len(finite)} validos, min={mn:.2f} m", flush=True)
        else:
            print("[smoke] aviso: LiDAR None", flush=True)

    except Exception as e:
        import traceback
        traceback.print_exc()
        fail(f"excepcion: {e}")

    finally:
        try:
            controller.set_velocities({})
            controller.stop()
        except Exception:
            pass

    print(f"\n[smoke] ===== {'PASS' if ok else 'FAIL'} ===== (total {time.time()-t0:.1f}s)", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
