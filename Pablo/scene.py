"""
Escena de cocina custom para el proyecto (Fase 0 - infraestructura).

Genera una cocina RoboCasa simple y DETERMINISTA e inyecta un conjunto FIJO de
objetos que siempre estan presentes para pruebas: cubo rojo, huevo, tomate,
cuchillo, plato. Asi podemos probar deteccion/agarre por nombre sin depender de
la aleatoriedad de la tarea de RoboCasa.

Los fixtures (mostrador, gabinetes, 4+ cajones, estufa) vienen del layout fijo.

Funciones clave:
    build_pablo_kitchen(...)  -> (model, xml, placements)  genera cocina + inyecta
    inject_objaverse_object(...) -> xml                     inyecta 1 objeto real
    inject_box(...)              -> xml                     inyecta una primitiva

Verificacion offline rapida (sin bootear el sim ~40s):
    uv run Pablo/scene.py
    -> Pablo/snaps/scene_check.png  (render de la cocina + objetos)

Nota Windows/Ubuntu: NO se fuerza MUJOCO_GL aqui (en Windows el backend default
renderiza; en Linux/headless se puede exportar MUJOCO_GL=egl externamente).
"""
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

# ── Catalogo de objetos garantizados (categoria objaverse, instancia, etiqueta) ──
# label = nombre por el que el usuario los pide (ES/EN). pos = sobre el mostrador.
DEFAULT_OBJECTS = [
    # (label,    category,   instance, pos (x,y,z),         tipo)
    # Con el mostrador ya limpio: 4 objetos chicos en fila FRONTAL (y=-0.35) y el
    # plato (ancho) ATRAS (y=-0.13), zona despejada y probada estable. El fregadero
    # esta a la izquierda (x<2.28), por eso todo va en x>=2.30. Estufa en x>2.8.
    #   Fila frontal (faciles de ver y agarrar):
    ("cuchillo", "knife",     0,       (2.30, -0.35, 0.93), "objaverse"),
    ("tomate",   "tomato",    0,       (2.42, -0.35, 0.95), "objaverse"),
    ("cubo_rojo", None,       None,    (2.54, -0.35, 0.94), "box"),
    ("huevo",    "egg",       0,       (2.66, -0.35, 0.95), "objaverse"),
    #   Plato atras (visible ahora que no hay rollo de papel delante):
    ("plato",    "plate",     0,       (2.64, -0.13, 0.925), "objaverse"),
]


def objaverse_root() -> str:
    import robocasa
    return os.path.join(robocasa.__path__[0], "models", "assets", "objects", "objaverse")


def _object_model_xml(category: str, instance: int) -> Path:
    """Ruta al model.xml de una instancia de objeto objaverse."""
    root = Path(objaverse_root()) / category
    # carpetas tipo egg_0, egg_1, ... ; elegimos <category>_<instance>
    cand = root / f"{category}_{instance}"
    if not cand.exists():
        # fallback: primera carpeta disponible
        subdirs = sorted([d for d in root.iterdir() if d.is_dir()])
        if not subdirs:
            raise FileNotFoundError(f"No hay instancias para categoria '{category}' en {root}")
        cand = subdirs[min(instance, len(subdirs) - 1)]
    return cand / "model.xml"


def inject_objaverse_object(
    xml_string: str,
    name: str,
    category: str,
    instance: int = 0,
    pos=(0.0, 0.0, 1.0),
    quat=(1.0, 0.0, 0.0, 0.0),
    gravcomp: float = 0.0,
) -> str:
    """Inyecta un objeto objaverse real (multi-mesh + textura) en el XML de cocina.

    Hace absolutas TODAS las rutas (mesh y textura), renombra los assets con un
    prefijo unico y los referencia desde un body nuevo con freejoint en `pos`.
    """
    model_xml = _object_model_xml(category, instance)
    obj_dir = model_xml.parent
    obj_root = ET.fromstring(model_xml.read_text())

    kroot = ET.fromstring(xml_string)
    kasset = kroot.find("asset")
    if kasset is None:
        kasset = ET.SubElement(kroot, "asset")
    kworld = kroot.find("worldbody")

    prefix = f"pablo_{name}"

    def _abs(rel: str) -> str:
        return str((obj_dir / rel).resolve()).replace("\\", "/")

    # ── Copiar y renombrar assets (mesh / texture / material) ────────────────
    name_map = {}  # nombre_original -> nombre_prefijado
    obj_asset = obj_root.find("asset")
    if obj_asset is not None:
        for el in list(obj_asset):
            tag = el.tag
            old = el.get("name")
            if old is not None:
                new = f"{prefix}_{old}"
                name_map[old] = new
                el.set("name", new)
            # absolutizar archivos
            if el.get("file"):
                el.set("file", _abs(el.get("file")))
            # re-apuntar material->texture
            if tag == "material" and el.get("texture") in name_map:
                el.set("texture", name_map[el.get("texture")])
            kasset.append(el)

    # ── Construir el body con freejoint y copiar los geoms del objeto ────────
    body = ET.SubElement(kworld, "body")
    body.set("name", name)
    body.set("pos", " ".join(str(float(v)) for v in pos))
    body.set("quat", " ".join(str(float(v)) for v in quat))
    body.set("gravcomp", str(gravcomp))
    ET.SubElement(body, "freejoint")

    # recolectar todos los geoms del objeto (visual group 1 + colision group 0)
    for geom in obj_root.iter("geom"):
        g = ET.SubElement(body, "geom")
        for k, v in geom.attrib.items():
            g.set(k, v)
        # re-apuntar referencias a assets renombrados
        if g.get("mesh") in name_map:
            g.set("mesh", name_map[g.get("mesh")])
        if g.get("material") in name_map:
            g.set("material", name_map[g.get("material")])
        # Friccion "pegajosa" (slide, torsion, roll) para que los objetos
        # redondos no rueden y se queden donde los pongo (escena de prueba).
        g.set("friction", "1.0 0.1 0.05")

    return ET.tostring(kroot, encoding="unicode")


def inject_box(
    xml_string: str,
    name: str,
    pos=(0.0, 0.0, 1.0),
    half_size=(0.025, 0.025, 0.025),
    rgba=(1.0, 0.0, 0.0, 1.0),
    mass: float = 0.05,
    gravcomp: float = 0.0,
) -> str:
    """Inyecta una primitiva box con freejoint (p.ej. el cubo rojo de prueba)."""
    kroot = ET.fromstring(xml_string)
    kworld = kroot.find("worldbody")
    body = ET.SubElement(kworld, "body")
    body.set("name", name)
    body.set("pos", " ".join(str(float(v)) for v in pos))
    body.set("gravcomp", str(gravcomp))
    ET.SubElement(body, "freejoint")
    g = ET.SubElement(body, "geom")
    g.set("name", f"{name}_geom")
    g.set("type", "box")
    g.set("size", " ".join(str(float(v)) for v in half_size))
    g.set("rgba", " ".join(str(float(v)) for v in rgba))
    g.set("mass", str(mass))
    g.set("friction", "1.0 0.1 0.05")
    return ET.tostring(kroot, encoding="unicode")


# Fixtures/distractores que ensucian el mostrador y tapan los objetos de prueba.
# Se quitan para dejar un mostrador limpio (cocina "simple, nada raro").
CLUTTER_TO_REMOVE = [
    "knife_block_main_group_main",   # bloque de cuchillos (accesorio del set)
    "paper_towel_main_group_main",   # portarrollos de papel (accesorio del set)
    "distr_counter_main",            # distractor random sobre el mostrador
    "distr_cab_main",                # distractor random en el gabinete
    "obj_main",                      # objeto-objetivo random de la tarea
]


def remove_clutter(xml_string: str, names=CLUTTER_TO_REMOVE) -> str:
    """Quita los fixtures/distractores que estorban del XML de cocina."""
    from stretch_mujoco.utils import xml_remove_tag_by_name
    xml = xml_string
    for name in names:
        xml, _ = xml_remove_tag_by_name(xml, "body", name)
    return xml


def inject_objects(xml_string: str, objects=DEFAULT_OBJECTS, clean=True) -> str:
    """Limpia el mostrador e inyecta todos los objetos del catalogo."""
    xml = remove_clutter(xml_string) if clean else xml_string
    for label, category, instance, pos, kind in objects:
        if kind == "box":
            xml = inject_box(xml, label, pos=pos)
        else:
            xml = inject_objaverse_object(xml, label, category, instance, pos=pos)
    return xml


def build_pablo_kitchen(task="PnPCounterToCab", layout=0, style=0,
                        objects=DEFAULT_OBJECTS, write_to_file=None):
    """Genera la cocina RoboCasa e inyecta los objetos garantizados.

    Returns: (mujoco.MjModel, xml_str, dict de placements de la tarea).
    """
    import mujoco
    from stretch_mujoco.robocasa_gen import model_generation_wizard
    _model, xml, info = model_generation_wizard(task=task, layout=layout, style=style)
    xml = inject_objects(xml, objects)
    model = mujoco.MjModel.from_xml_string(xml)
    if write_to_file:
        Path(write_to_file).write_text(xml)
    return model, xml, info


# ── Verificacion offline (rapida, sin sim) ───────────────────────────────────
def _add_debug_camera(xml_string: str, pos, xyaxes) -> str:
    root = ET.fromstring(xml_string)
    world = root.find("worldbody")
    cam = ET.SubElement(world, "camera")
    cam.set("name", "pablo_debug")
    cam.set("pos", " ".join(str(v) for v in pos))
    cam.set("xyaxes", " ".join(str(v) for v in xyaxes))
    return ET.tostring(root, encoding="unicode")


def _offline_check():
    import mujoco
    import imageio.v2 as imageio
    cache = Path(__file__).resolve().parent / "cache" / "base_kitchen.xml"
    snaps = Path(__file__).resolve().parent / "snaps"
    snaps.mkdir(exist_ok=True)
    if not cache.exists():
        print("Falta cache/base_kitchen.xml. Corre primero: uv run Pablo/_gen_base_kitchen.py")
        return
    print("[scene] inyectando objetos en cocina cacheada...")
    xml = inject_objects(cache.read_text())
    # camara debug mirando el mostrador desde arriba-frente
    xml = _add_debug_camera(xml, pos=(2.35, -1.15, 1.45),
                            xyaxes=(1, 0, 0, 0, 0.6, 0.8))
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    with mujoco.Renderer(model, height=480, width=640) as r:
        r.update_scene(data, camera="pablo_debug")
        img = r.render()
    out = snaps / "scene_check.png"
    imageio.imwrite(str(out), img)
    print(f"[scene] guardado {out}")
    # reporte: confirmar que los bodies existen en el modelo
    for label, *_ in DEFAULT_OBJECTS:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, label)
        print(f"[scene] body '{label}': id={bid} {'OK' if bid >= 0 else 'FALTA'}")
    print("[scene] OK" if all(
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, l) >= 0 for l, *_ in DEFAULT_OBJECTS
    ) else "[scene] FALTAN objetos")


if __name__ == "__main__":
    _offline_check()
