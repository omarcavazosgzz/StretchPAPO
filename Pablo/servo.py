"""
Servoing: convierte el error de centrado (en la imagen) en velocidades de
junta para centrar el objeto en cada camara.

El mapeo eje-imagen -> junta puede requerir swap/signo porque la camara head
de la Stretch se renderiza rotada 90 (portrait). Esos parametros se confirman
empiricamente con test_fase1_head.py y se fijan aqui.
"""
import numpy as np
from vision import centering_error

# --- Config de mapeo (confirmar/ajustar con los tests de la Fase 1) ---
# swap=True  -> el eje horizontal de la imagen controla tilt/pitch (no pan/yaw)
# sign_*     -> +1 o -1 segun la orientacion de la camara
HEAD = dict(swap=False, sign_pan=-1.0, sign_tilt=-1.0,
            kp_pan=0.9, kp_tilt=0.9, deadband=0.035)
WRIST = dict(swap=False, sign_yaw=-1.0, sign_pitch=-1.0,
             kp_yaw=0.9, kp_pitch=0.9, deadband=0.035)


def _split_axes(ex, ey, swap):
    """Devuelve (eje_para_pan/yaw, eje_para_tilt/pitch)."""
    return (ey, ex) if swap else (ex, ey)


def _p(value, sign, kp, deadband):
    if abs(value) < deadband:
        return 0.0
    return float(np.clip(sign * kp * value, -1.0, 1.0))


def head_servo(centroid, frame_shape, cfg=HEAD):
    """Velocidades para centrar el objeto en la camara HEAD (arriba)."""
    ex, ey = centering_error(centroid, frame_shape)
    a_pan, a_tilt = _split_axes(ex, ey, cfg["swap"])
    cmd = {
        "head_pan_counterclockwise": _p(a_pan, cfg["sign_pan"], cfg["kp_pan"], cfg["deadband"]),
        "head_tilt_up": _p(a_tilt, cfg["sign_tilt"], cfg["kp_tilt"], cfg["deadband"]),
    }
    return cmd, (ex, ey)


def wrist_servo(centroid, frame_shape, cfg=WRIST):
    """Velocidades para que la camara del BRAZO (wrist) vea de frente al objeto."""
    ex, ey = centering_error(centroid, frame_shape)
    a_yaw, a_pitch = _split_axes(ex, ey, cfg["swap"])
    cmd = {
        "wrist_yaw_counterclockwise": _p(a_yaw, cfg["sign_yaw"], cfg["kp_yaw"], cfg["deadband"]),
        "wrist_pitch_up": _p(a_pitch, cfg["sign_pitch"], cfg["kp_pitch"], cfg["deadband"]),
    }
    return cmd, (ex, ey)


def error_norm(ex, ey):
    return float(np.hypot(ex, ey))
