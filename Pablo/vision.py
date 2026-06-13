"""
Vision pura para la Fase 1 (sin dependencias del simulador, facil de testear).

Idea: el usuario "pica" (click) un objeto en la ventana de la camara. Tomamos
el color HSV de ese pixel y construimos un rango adaptativo. A partir de ahi
detectamos el objeto SOLO con informacion de la imagen (centroide del mayor
blob de ese color). Esto cumple "solo con lo que suelta la camara".
"""
from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class ColorTarget:
    """Rango de color HSV a seguir, derivado de un pixel muestreado."""
    lower1: np.ndarray
    upper1: np.ndarray
    lower2: np.ndarray | None  # segundo rango para el rojo (wraparound del hue)
    upper2: np.ndarray | None
    hsv: tuple  # color medio muestreado (h, s, v) para debug/legibilidad

    def mask(self, frame_bgr) -> np.ndarray:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        m = cv2.inRange(hsv, self.lower1, self.upper1)
        if self.lower2 is not None:
            m = cv2.bitwise_or(m, cv2.inRange(hsv, self.lower2, self.upper2))
        return m


def make_color_target(frame_bgr, x, y, patch=5,
                      hue_tol=10, sat_min=70, val_min=60) -> ColorTarget:
    """Construye un ColorTarget a partir del pixel (x, y) y su vecindad.

    Maneja el wraparound del hue rojo (cerca de 0/180) generando dos rangos.
    """
    h_img, w_img = frame_bgr.shape[:2]
    x = int(np.clip(x, 0, w_img - 1))
    y = int(np.clip(y, 0, h_img - 1))

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    x0, x1 = max(0, x - patch), min(w_img, x + patch + 1)
    y0, y1 = max(0, y - patch), min(h_img, y + patch + 1)
    region = hsv[y0:y1, x0:x1].reshape(-1, 3)

    h_med = int(np.median(region[:, 0]))
    s_med = int(np.median(region[:, 1]))
    v_med = int(np.median(region[:, 2]))

    s_lo = int(max(sat_min, s_med - 90))
    v_lo = int(max(val_min, v_med - 90))
    s_hi, v_hi = 255, 255

    lower2 = upper2 = None
    lo_h = h_med - hue_tol
    hi_h = h_med + hue_tol

    if lo_h < 0:  # wraparound por abajo (rojo)
        lower1 = np.array([0, s_lo, v_lo]);            upper1 = np.array([hi_h, s_hi, v_hi])
        lower2 = np.array([180 + lo_h, s_lo, v_lo]);   upper2 = np.array([180, s_hi, v_hi])
    elif hi_h > 180:  # wraparound por arriba (rojo)
        lower1 = np.array([lo_h, s_lo, v_lo]);         upper1 = np.array([180, s_hi, v_hi])
        lower2 = np.array([0, s_lo, v_lo]);            upper2 = np.array([hi_h - 180, s_hi, v_hi])
    else:
        lower1 = np.array([lo_h, s_lo, v_lo]);         upper1 = np.array([hi_h, s_hi, v_hi])

    return ColorTarget(lower1, upper1, lower2, upper2, (h_med, s_med, v_med))


def find_object(frame_bgr, target: ColorTarget, min_area=60):
    """Encuentra el mayor blob del color objetivo.

    Returns:
        dict con centroid (cx, cy), area, bbox (x,y,w,h) y mask; o None.
    """
    mask = target.mask(frame_bgr)
    # Limpieza morfologica para quitar ruido y cerrar huecos
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not valid:
        return None

    largest = max(valid, key=cv2.contourArea)
    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    x, y, w, h = cv2.boundingRect(largest)
    return {
        "centroid": (cx, cy),
        "area": float(cv2.contourArea(largest)),
        "bbox": (x, y, w, h),
        "mask": mask,
    }


def centering_error(centroid, frame_shape):
    """Error normalizado del centroide respecto al centro de la imagen.

    Returns (ex, ey) en [-0.5, 0.5]. ex>0: objeto a la derecha; ey>0: abajo.
    """
    h_img, w_img = frame_shape[:2]
    cx, cy = centroid
    ex = (cx / w_img) - 0.5
    ey = (cy / h_img) - 0.5
    return ex, ey


# Colores predefinidos utiles para pruebas en el entorno de bloques simple
# (cilindro rojo, caja azul). Permiten testear sin click.
def red_target() -> ColorTarget:
    return ColorTarget(
        lower1=np.array([0, 100, 80]),  upper1=np.array([10, 255, 255]),
        lower2=np.array([160, 100, 80]), upper2=np.array([180, 255, 255]),
        hsv=(0, 200, 200),
    )


def blue_target() -> ColorTarget:
    return ColorTarget(
        lower1=np.array([105, 100, 80]), upper1=np.array([130, 255, 255]),
        lower2=None, upper2=None,
        hsv=(118, 200, 200),
    )
