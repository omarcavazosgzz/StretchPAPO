"""Genera y cachea un XML base de cocina RoboCasa (sin bootear el sim) para
iterar la inyeccion de objetos con render offline rapido.

    uv run Pablo/_gen_base_kitchen.py
Salida: Pablo/cache/base_kitchen.xml
"""
from pathlib import Path

OUT = Path(__file__).resolve().parent / "cache"
OUT.mkdir(exist_ok=True)


def main():
    from stretch_mujoco.robocasa_gen import model_generation_wizard
    dest = OUT / "base_kitchen.xml"
    model, xml, info = model_generation_wizard(
        task="PnPCounterToCab", layout=0, style=0, write_to_file=str(dest)
    )
    print("OBJ PLACEMENTS:")
    for k, v in info.items():
        print(f"  {k}: cat={v['cat']} pos={[round(float(x),3) for x in v['pos']]}")
    print("Saved:", dest, "chars:", len(xml))


if __name__ == "__main__":
    main()
