"""
Benchmark headless de camaras: arranca el sim SIN viewer, con las camaras
pre-registradas, y mide (1) cuanto tarda el primer frame de cada camara y
(2) la tasa real de render (contando avances de camera_data.time).

Esto evita el watchdog/auto-registro fragil del toolkit y prueba si en este
entorno headless las camaras entregan frames de forma estable.

Uso (opcionalmente con backend GL):
    uv run Pablo/test_headless_cams.py
    MUJOCO_GL=egl    uv run Pablo/test_headless_cams.py
    MUJOCO_GL=osmesa uv run Pablo/test_headless_cams.py
"""
import os
import time


def main():
    from stretch_mujoco import StretchMujocoSimulator
    from stretch_mujoco.enums.stretch_cameras import StretchCameras

    cams = [StretchCameras.cam_d435i_rgb, StretchCameras.cam_d405_rgb]
    print(f"[hl] MUJOCO_GL={os.environ.get('MUJOCO_GL','(default)')}", flush=True)
    print(f"[hl] camaras pre-registradas: {[c.name for c in cams]}", flush=True)

    sim = StretchMujocoSimulator(cameras_to_use=cams)
    t0 = time.time()
    sim.start(headless=True)
    print(f"[hl] sim arrancado (headless) en {time.time()-t0:.1f}s", flush=True)

    got = {c.name: None for c in cams}
    shapes = {}
    deadline = time.time() + 30
    while time.time() < deadline and not all(v is not None for v in got.values()):
        data = sim.pull_camera_data()
        allf = data.get_all(use_depth_color_map=False)
        for c in cams:
            if got[c.name] is None and allf.get(c) is not None:
                got[c.name] = time.time() - t0
                shapes[c.name] = allf[c].shape
                print(f"[hl] primer frame {c.name} a los {got[c.name]:.1f}s shape={shapes[c.name]}", flush=True)
        time.sleep(0.05)

    for name, val in got.items():
        if val is None:
            print(f"[hl] FAIL: {name} nunca entrego frame en 30s", flush=True)

    # Medir tasa real de render contando avances de camera_data.time
    print("[hl] midiendo tasa de render por ~5s...", flush=True)
    last_t = -1.0
    frames = 0
    t1 = time.time()
    while time.time() - t1 < 5.0:
        ct = sim.pull_camera_data().time
        if ct != last_t:
            frames += 1
            last_t = ct
        time.sleep(0.005)
    fps = frames / 5.0
    print(f"[hl] tasa de frames nuevos ~ {fps:.1f} FPS", flush=True)

    ok = all(v is not None for v in got.values())
    print(f"\n[hl] ===== {'PASS' if ok else 'FAIL'} ===== first_frames={got} fps~{fps:.1f}", flush=True)
    sim.stop()


if __name__ == "__main__":
    main()
