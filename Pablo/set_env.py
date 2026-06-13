"""
Cambia el entorno del simulador editando stretch_toolkit/sim_config.json.

    uv run Pablo/set_env.py            # muestra el estado actual
    uv run Pablo/set_env.py block      # entorno simple de bloques
    uv run Pablo/set_env.py robocasa   # cocina RoboCasa (PnPCounterToCab)
"""
import sys
import json
from pathlib import Path

CONFIG = Path(__file__).resolve().parent.parent / "stretch_toolkit" / "sim_config.json"


def main():
    cfg = json.loads(CONFIG.read_text())
    cfg.setdefault("robocasa", {})
    arg = sys.argv[1].lower() if len(sys.argv) > 1 else None

    if arg == "robocasa":
        cfg["robocasa"]["enabled"] = True
    elif arg == "block":
        cfg["robocasa"]["enabled"] = False
    else:
        estado = "RoboCasa" if cfg["robocasa"].get("enabled") else "bloques"
        print(f"Entorno actual: {estado} (robocasa.enabled={cfg['robocasa'].get('enabled')})")
        print("Cambia con:  uv run Pablo/set_env.py [block|robocasa]")
        return

    CONFIG.write_text(json.dumps(cfg, indent=2))
    estado = "RoboCasa" if cfg["robocasa"]["enabled"] else "bloques"
    print(f"Entorno cambiado a: {estado}")


if __name__ == "__main__":
    main()
