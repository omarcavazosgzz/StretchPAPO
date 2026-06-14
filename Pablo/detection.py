"""
Capa de DETECCION por nombre, con dos backends intercambiables (decision del
usuario: oraculo por default, YOLO activable por config):

  - "oracle": usa la verdad-de-tierra del simulador. Proyecta la pose 3D real del
    objeto al pixel de cada camara (head/wrist). 100% confiable en simulacion.
    Es informacion derivada de la camara (como la profundidad) + pose conocida.
  - "yolo": YOLO-World (open-vocab) sobre el RGB, prompt = nombre del objeto.
    Para migrar al robot real. (Stub listo; se carga ultralytics solo si se usa.)

Seleccion por sim_config.json:  {"detector": "oracle" | "yolo"}  (default oracle).

La proyeccion del oraculo es EXACTA: lee el fovy real de la camara del modelo
MuJoCo (-> focal del render) y aplica la rotacion que el toolkit hace a cada
frame (head: rot90 CW, wrist: ninguna). Verificado con el cubo rojo por color.
"""
from __future__ import annotations
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from stretch_mujoco.enums.stretch_cameras import StretchCameras

CONFIG = Path(__file__).resolve().parent.parent / "stretch_toolkit" / "sim_config.json"

# Nombre del objeto (lo que escribe el usuario, ES/EN) -> body name en el sim.
# Los objetos inyectados por scene.py tienen body name == label. Aliases por comodidad.
ALIASES = {
    "egg": "huevo", "tomato": "tomate", "knife": "cuchillo", "plate": "plato",
    "cube": "cubo_rojo", "red_cube": "cubo_rojo", "cubo": "cubo_rojo",
    "red cube": "cubo_rojo",
}

# Nombre de la camara en el MJCF (para camera_poses del status)
MJCF_CAM = {
    StretchCameras.cam_d435i_rgb: "d435i_camera_rgb",   # cabeza
    StretchCameras.cam_d405_rgb: "d405_rgb",            # muneca/brazo
}


@dataclass
class Detection:
    centroid: tuple[float, float]   # (x=col, y=row) en el frame YA MOSTRADO (auto_rotate)
    depth: float                    # distancia camara->objeto en metros
    in_frame: bool                  # True si cae dentro de los limites de la imagen
    frame_shape: tuple[int, int]    # (H, W) del frame mostrado
    name: str
    source: str                     # "oracle" | "yolo"


def resolve_name(name: str) -> str:
    n = name.strip().lower()
    return ALIASES.get(n, n)


# ──────────────────────────────────────────────────────────────────────────
#  Modelo de camara: focal del RENDER (desde fovy real) + mapeo a frame mostrado
# ──────────────────────────────────────────────────────────────────────────
class _CamModel:
    def __init__(self, camera: StretchCameras, mjmodel):
        import mujoco
        s = camera.initial_camera_settings
        self.W = s.width    # ancho NATIVO del render (antes de rotar)
        self.H = s.height   # alto NATIVO
        mjcf = camera.camera_name_in_mjcf
        cam_id = mujoco.mj_name2id(mjmodel, mujoco.mjtObj.mjOBJ_CAMERA, mjcf)
        fovy = float(mjmodel.cam_fovy[cam_id])            # FOV vertical real del render
        self.f = 0.5 * self.H / math.tan(math.radians(fovy) / 2.0)
        self.cx = self.W / 2.0
        self.cy = self.H / 2.0
        self.camera = camera

    def project(self, world_xyz, cam_pos, cam_xmat):
        """Proyecta un punto mundo al pixel del frame MOSTRADO. Devuelve
        (x, y, depth, in_frame, (H_disp, W_disp)) o None si esta detras."""
        R = np.array(cam_xmat, dtype=float).reshape(3, 3)  # cols = ejes cam en mundo
        p = np.array(world_xyz, dtype=float) - np.array(cam_pos, dtype=float)
        Xc, Yc, Zc = R.T @ p                                # a frame camara
        if Zc >= -1e-6:                                     # MuJoCo mira por -Z
            return None
        depth = -Zc
        u = self.cx + self.f * (Xc / depth)                # col nativo (derecha +)
        v = self.cy - self.f * (Yc / depth)                # row nativo (abajo +)
        in_native = (0 <= u < self.W) and (0 <= v < self.H)
        # Mapear a frame mostrado segun la rotacion que aplica get_all(auto_rotate)
        if self.camera == StretchCameras.cam_d435i_rgb:    # head: np.rot90(-1) (CW)
            x_disp = self.H - 1 - v
            y_disp = u
            shape = (self.W, self.H)                        # (H_disp, W_disp)
        else:                                              # wrist d405: sin rotacion
            x_disp = u
            y_disp = v
            shape = (self.H, self.W)
        return x_disp, y_disp, depth, in_native, shape


# ──────────────────────────────────────────────────────────────────────────
#  Backends
# ──────────────────────────────────────────────────────────────────────────
class OracleDetector:
    source = "oracle"

    def __init__(self, sim, mjmodel):
        self.sim = sim
        self.cams = {cam: _CamModel(cam, mjmodel) for cam in MJCF_CAM}

    def detect(self, camera: StretchCameras, name: str) -> Detection | None:
        if camera not in self.cams:        # camara no soportada por el oraculo (p.ej. nav)
            return None
        body = resolve_name(name)
        status = self.sim.pull_status()
        if body not in status.object_poses:
            return None
        world = status.object_poses[body][:3]
        cps = status.camera_poses.get(MJCF_CAM[camera])
        if cps is None:
            return None
        res = self.cams[camera].project(world, cps["pos"], cps["xmat"])
        if res is None:
            return None
        x, y, depth, in_frame, shape = res
        return Detection((float(x), float(y)), float(depth), bool(in_frame), shape, body, self.source)


class YoloDetector:
    source = "yolo"
    # Mapa nombre-interno -> clase/prompt para YOLO-World (open vocabulary)
    PROMPTS = {
        "huevo": "egg", "tomate": "tomato", "cuchillo": "knife",
        "plato": "plate", "cubo_rojo": "red cube",
    }

    def __init__(self, sim, mjmodel, weights="yolov8s-world.pt"):
        self.sim = sim
        self._model = None
        self._weights = weights

    def _lazy(self):
        if self._model is None:
            from ultralytics import YOLOWorld  # se importa solo si se usa yolo
            self._model = YOLOWorld(self._weights)
        return self._model

    def detect(self, camera: StretchCameras, name: str) -> Detection | None:
        body = resolve_name(name)
        prompt = self.PROMPTS.get(body, body)
        model = self._lazy()
        model.set_classes([prompt])
        data = self.sim.pull_camera_data().get_all(use_depth_color_map=False)
        frame = data.get(camera)
        if frame is None:
            return None
        res = model.predict(frame, verbose=False)[0]
        if len(res.boxes) == 0:
            return None
        b = res.boxes[0].xyxy[0].tolist()
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        H, W = frame.shape[:2]
        return Detection((cx, cy), 0.0, True, (H, W), body, self.source)


def get_detector_mode() -> str:
    try:
        cfg = json.loads(CONFIG.read_text())
        return str(cfg.get("detector", "oracle")).lower()
    except Exception:
        return "oracle"


def make_detector(sim, mjmodel):
    """Crea el detector segun sim_config.json ('detector': 'oracle'|'yolo')."""
    mode = get_detector_mode()
    if mode == "yolo":
        print("[detection] backend = YOLO (open-vocab)")
        return YoloDetector(sim, mjmodel)
    print("[detection] backend = ORACLE (sim ground-truth)")
    return OracleDetector(sim, mjmodel)
