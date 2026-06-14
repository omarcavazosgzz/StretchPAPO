"""Arranque del sim con la COCINA CUSTOM via el toolkit (controller).

Hace monkeypatch a model_generation_wizard para que, cuando el toolkit construya
la cocina RoboCasa, inyecte nuestros objetos garantizados + limpie el mostrador.
Asi reusamos controller.set_velocities (multi-junta), get_state, LiDAR, etc.

    from sim_setup import start_kitchen
    controller, model = start_kitchen()
"""
import os
import json
from pathlib import Path

CONFIG = Path(__file__).resolve().parent.parent / "stretch_toolkit" / "sim_config.json"


def start_kitchen(cameras=("cam_d435i_rgb", "cam_d405_rgb"),
                  headless=True, layout=0, style=0, dim_lights=None):
    """Arranca el controller del toolkit con la cocina custom. Devuelve (controller, model)."""
    os.environ["USE_SIM"] = "1"
    if headless:
        os.environ["STRETCH_SIM_HEADLESS"] = "1"
    os.environ["STRETCH_SIM_CAMERAS"] = ",".join(cameras)
    if dim_lights:
        os.environ["STRETCH_DIM_LIGHTS"] = str(dim_lights)

    # Habilitar robocasa en la config (modo de trabajo del proyecto)
    cfg = json.loads(CONFIG.read_text())
    cfg.setdefault("robocasa", {})
    cfg["robocasa"]["enabled"] = True
    cfg["robocasa"]["layout"] = layout
    cfg["robocasa"]["style"] = style
    # Nosotros inyectamos nuestros objetos en el monkeypatch; vaciar custom_objects
    # del toolkit (el android_lego de ejemplo apunta a un asset no descargado).
    cfg["robocasa"]["custom_objects"] = []
    CONFIG.write_text(json.dumps(cfg, indent=2))

    # Monkeypatch: inyectar nuestra escena en el modelo que genera el toolkit
    import mujoco
    import stretch_mujoco.robocasa_gen as rg
    from scene import inject_objects
    _orig = rg.model_generation_wizard

    def _patched(*a, **k):
        model, xml, info = _orig(*a, **k)
        xml2 = inject_objects(xml)
        return mujoco.MjModel.from_xml_string(xml2), xml2, info

    rg.model_generation_wizard = _patched

    from stretch_toolkit import controller
    controller.get_state()           # fuerza la construccion del sim
    model = controller.sim.model     # nuestro modelo inyectado
    return controller, model
