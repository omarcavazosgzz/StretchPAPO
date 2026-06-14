"""Abre el VISOR de MuJoCo con la cocina custom + objetos garantizados, para
verla en vivo y orbitarla con el mouse.

    uv run Pablo/ver_cocina.py            # abre la ventana; cierra con la X o Ctrl+C
    uv run Pablo/ver_cocina.py 10         # se cierra solo a los 10 s (para pruebas)

Controles del visor (free camera, independiente de las camaras del robot):
    - Arrastrar boton IZQ  : orbitar
    - Arrastrar boton DER  : desplazar (pan)
    - Rueda                : zoom

Apunta la cabeza del robot hacia el mostrador de objetos (a su derecha) para que
la vista del robot tambien tenga sentido. Imprime las poses reales (oraculo).
"""
import sys
import time


def main():
    auto_close = None
    if len(sys.argv) > 1:
        try:
            auto_close = float(sys.argv[1])
        except ValueError:
            pass

    from scene import build_pablo_kitchen, DEFAULT_OBJECTS
    from stretch_mujoco import StretchMujocoSimulator
    from stretch_mujoco.enums.stretch_cameras import StretchCameras

    print("[ver] generando cocina custom + objetos (~12s)...", flush=True)
    model, xml, info = build_pablo_kitchen()

    cams = [StretchCameras.cam_d435i_rgb, StretchCameras.cam_d405_rgb]
    sim = StretchMujocoSimulator(model=model, cameras_to_use=cams)
    print("[ver] abriendo visor (headless=False)...", flush=True)
    sim.start(headless=False)

    # Dejar asentar y apuntar la cabeza a la derecha (hacia los objetos / estufa)
    time.sleep(1.5)
    sim.move_to("head_pan", -0.9)   # girar a la derecha del robot
    sim.move_to("head_tilt", -0.5)  # mirar hacia abajo al mostrador

    print("[ver] poses de objetos (oraculo):", flush=True)
    for label, *_ in DEFAULT_OBJECTS:
        p = sim.get_object_pose(label)
        if p:
            print(f"    {label:10s} x={p['x']:.2f} y={p['y']:.2f} z={p['z']:.2f}", flush=True)

    if auto_close:
        print(f"[ver] cerrando automaticamente en {auto_close:.0f}s...", flush=True)
        t_end = time.time() + auto_close
        while time.time() < t_end and sim.is_running():
            time.sleep(0.2)
    else:
        print("[ver] Ventana abierta. Cierra con la X de la ventana o Ctrl+C aqui.", flush=True)
        try:
            while sim.is_running():
                time.sleep(0.2)
        except KeyboardInterrupt:
            pass

    sim.stop()


if __name__ == "__main__":
    main()
