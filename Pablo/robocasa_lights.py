"""
Inspecciona las LUCES del modelo RoboCasa generado, para saber que atenuar y
arreglar la sobreexposicion de la camara de arriba. Solo construye el modelo
(no corre el sim completo).

Uso:
    uv run Pablo/robocasa_lights.py
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")


def main():
    import numpy as np
    from stretch_mujoco.robocasa_gen import model_generation_wizard

    print("[luz] generando modelo RoboCasa (PnPCounterToCab, layout0, style0)...", flush=True)
    model, xml, objects_info = model_generation_wizard(
        task="PnPCounterToCab", layout=0, style=0, custom_objects=None,
    )

    print(f"[luz] nlight = {model.nlight}", flush=True)
    np.set_printoptions(precision=2, suppress=True)
    hl = model.vis.headlight
    print(f"[luz] headlight ambient={np.array(hl.ambient)} diffuse={np.array(hl.diffuse)} specular={np.array(hl.specular)} active={hl.active}", flush=True)
    if model.nlight > 0:
        print(f"[luz] light_directional = {np.array(model.light_directional).ravel()}", flush=True)
        print(f"[luz] light_castshadow  = {np.array(model.light_castshadow).ravel()}", flush=True)
        print(f"[luz] light_ambient (sum/axis) =\n{np.array(model.light_ambient)}", flush=True)
        print(f"[luz] light_diffuse =\n{np.array(model.light_diffuse)}", flush=True)
        print(f"[luz] light_specular =\n{np.array(model.light_specular)}", flush=True)
    # materiales emisivos pueden tambien quemar la imagen
    try:
        em = np.array(model.mat_emission).ravel()
        print(f"[luz] mat_emission: n={em.size} max={em.max():.2f} (#>0: {(em>0).sum()})", flush=True)
    except Exception as e:
        print(f"[luz] mat_emission n/d: {e}", flush=True)
    print(f"[luz] global ambient (model.vis.global_?)...", flush=True)


if __name__ == "__main__":
    main()
