"""Diagnostico OFFLINE rapido: renderiza la cocina+objetos desde arriba (vista
de pajaro) y desde un angulo frontal, para mapear el mostrador y ver donde
quedan/ocultan los objetos. Tambien dibuja una cuadricula de coordenadas x,y.

    uv run Pablo/_diag_scene.py
Salida: Pablo/snaps/diag_top.png , Pablo/snaps/diag_front.png
"""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

CACHE = Path(__file__).resolve().parent / "cache" / "base_kitchen.xml"
SNAPS = Path(__file__).resolve().parent / "snaps"


def _add_cam(xml, name, pos, xyaxes, fovy=60):
    root = ET.fromstring(xml)
    world = root.find("worldbody")
    cam = ET.SubElement(world, "camera")
    cam.set("name", name)
    cam.set("pos", " ".join(str(v) for v in pos))
    cam.set("xyaxes", " ".join(str(v) for v in xyaxes))
    cam.set("fovy", str(fovy))
    return ET.tostring(root, encoding="unicode")


def main():
    import mujoco
    import imageio.v2 as imageio
    from scene import inject_objects, DEFAULT_OBJECTS
    SNAPS.mkdir(exist_ok=True)

    xml = inject_objects(CACHE.read_text())
    # camara cenital sobre el centro del mostrador, mirando hacia abajo (-z)
    xml = _add_cam(xml, "diag_top", pos=(2.6, -0.25, 2.35), xyaxes=(1, 0, 0, 0, 1, 0), fovy=70)
    # camara frontal-elevada
    xml = _add_cam(xml, "diag_front", pos=(2.6, -1.35, 1.55), xyaxes=(1, 0, 0, 0, 0.5, 0.87), fovy=70)

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    for cam, tag in [("diag_top", "top"), ("diag_front", "front")]:
        with mujoco.Renderer(model, height=480, width=640) as r:
            r.update_scene(data, camera=cam)
            img = r.render()
        imageio.imwrite(str(SNAPS / f"diag_{tag}.png"), img)
        print(f"[diag] guardado diag_{tag}.png")

    print("[diag] posiciones COLOCADAS (antes de fisica):")
    for label, cat, inst, pos, kind in DEFAULT_OBJECTS:
        print(f"    {label:10s} x={pos[0]:.2f} y={pos[1]:.2f} z={pos[2]:.2f}  ({kind})")


if __name__ == "__main__":
    main()
