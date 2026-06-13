"""
Verifica que la escena RoboCasa carga en este entorno (headless + EGL) y
lista los objetos manipulables, ademas de confirmar que las camaras entregan
frames dentro de la cocina.

Activa robocasa.enabled=true temporalmente en sim_config.json y lo restaura
al terminar (try/finally).

Uso:
    uv run Pablo/test_robocasa_boot.py
"""
import os
import json
import time
from pathlib import Path

# Render por GPU headless ANTES de importar mujoco/toolkit.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ["STRETCH_SIM_HEADLESS"] = "1"
os.environ["STRETCH_SIM_CAMERAS"] = "cam_d435i_rgb,cam_d435i_depth,cam_d405_rgb,cam_d405_depth"

CONFIG_PATH = Path(__file__).resolve().parent.parent / "stretch_toolkit" / "sim_config.json"


def main():
    original = CONFIG_PATH.read_text()
    cfg = json.loads(original)
    cfg.setdefault("robocasa", {})
    cfg["robocasa"]["enabled"] = True
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
    print(f"[rc] robocasa.enabled=true (task={cfg['robocasa'].get('task')}, "
          f"layout={cfg['robocasa'].get('layout')}, style={cfg['robocasa'].get('style')})", flush=True)

    t0 = time.time()
    ok = True
    try:
        from stretch_toolkit import controller, HEAD_RGB_CAMERA, WRIST_RGB_CAMERA

        print("[rc] arrancando RoboCasa headless (puede tardar bastante: genera la cocina)...", flush=True)
        state = controller.get_state()
        print(f"[rc] sim listo en {time.time()-t0:.1f}s. base=({state['base_x']:.2f},{state['base_y']:.2f})", flush=True)

        # Objetos manipulables en la escena
        objs = controller.list_scene_objects()
        print(f"[rc] objetos manipulables ({len(objs)}): {objs}", flush=True)
        for name in objs[:10]:
            pose = controller.get_object_pose(name)
            if pose:
                print(f"    - {name}: x={pose['x']:.2f} y={pose['y']:.2f} z={pose['z']:.2f}", flush=True)

        # Frames de camara en la cocina
        h = None
        deadline = time.time() + 15
        while time.time() < deadline and h is None:
            h = HEAD_RGB_CAMERA.get_frame()
            time.sleep(0.1)
        w = WRIST_RGB_CAMERA.get_frame()
        print(f"[rc] HEAD frame: {None if h is None else h.shape} | WRIST frame: {None if w is None else w.shape}", flush=True)
        if h is None:
            ok = False
            print("[rc] FAIL: HEAD no entrego frame en RoboCasa", flush=True)

    except Exception as e:
        import traceback
        traceback.print_exc()
        ok = False
    finally:
        CONFIG_PATH.write_text(original)
        print("[rc] sim_config.json restaurado", flush=True)
        try:
            controller.stop()
        except Exception:
            pass

    print(f"\n[rc] ===== {'PASS' if ok else 'FAIL'} ===== (total {time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
