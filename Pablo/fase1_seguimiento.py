"""
FASE 1 - Seguimiento del objeto con las camaras (solo informacion visual).

Que hace:
  - Muestra las camaras HEAD (arriba) y WRIST (brazo).
  - "Picas" (click izquierdo) un objeto en cualquiera de las dos ventanas:
    se muestrea su color y queda como objetivo.
  - La camara del BRAZO se queda viendo de frente al objeto (wrist yaw/pitch).
  - La camara de ARRIBA tambien apunta al objeto (head pan/tilt). Si no lo ve,
    hace una busqueda barriendo la cabeza hasta encontrarlo.
  - Puedes sobreescribir en cualquier momento con el teclado/gamepad (teleop):
    tu input siempre gana (merge proporcional).

Controles:
  - Click izquierdo en una ventana: seleccionar objeto (picar).
  - r: olvidar objetivo
  - g: activar/desactivar busqueda de cabeza
  - teclas de teleop (w/a/s/d, etc.): mover el robot (te sobrepones al auto)
  - q o Ctrl+C: salir

Entorno (bloques o RoboCasa): se controla en stretch_toolkit/sim_config.json
(robocasa.enabled). Usa  Pablo/set_env.py  para cambiarlo facil.

Uso:
    uv run Pablo/fase1_seguimiento.py
"""
import os
# Camaras siempre-activas (evita el thrash del watchdog incluso con viewer).
os.environ.setdefault("STRETCH_SIM_CAMERAS",
                      "cam_d435i_rgb,cam_d435i_depth,cam_d405_rgb,cam_d405_depth")

import time
import numpy as np
import cv2

import vision
import servo

# --- Parametros de busqueda de la cabeza ---
SEARCH_PAN_RANGE = (-1.8, 1.8)   # rango de barrido de pan
SEARCH_TILT = -0.6               # tilt fijo mientras busca
SEARCH_PAN_SPEED = 0.5           # velocidad normalizada del barrido


class HeadSearcher:
    """Barre la cabeza de lado a lado hasta que el objetivo aparece."""
    def __init__(self):
        self.dir = -1  # empieza hacia el lado del brazo (pan negativo)

    def step(self, state):
        pan = state["head_pan_counterclockwise"]
        if pan <= SEARCH_PAN_RANGE[0]:
            self.dir = 1
        elif pan >= SEARCH_PAN_RANGE[1]:
            self.dir = -1
        tilt_err = SEARCH_TILT - state["head_tilt_up"]
        return {
            "head_pan_counterclockwise": SEARCH_PAN_SPEED * self.dir,
            "head_tilt_up": float(np.clip(2.0 * tilt_err, -0.5, 0.5)),
        }


def draw_overlay(frame, obj, label, err=None, dist=None):
    vis = frame.copy()
    h, w = vis.shape[:2]
    cv2.drawMarker(vis, (w // 2, h // 2), (200, 200, 200), cv2.MARKER_CROSS, 16, 1)
    if obj is not None:
        c = obj["centroid"]
        x, y, bw, bh = obj["bbox"]
        cv2.rectangle(vis, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        cv2.circle(vis, c, 5, (0, 255, 0), -1)
        cv2.line(vis, (w // 2, h // 2), c, (0, 255, 255), 1)
        txt = label
        if err is not None:
            txt += f" e={err:.2f}"
        if dist is not None:
            txt += f" {dist:.2f}m"
        cv2.putText(vis, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(vis, f"{label}: buscando...", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return vis


class Fase1:
    def __init__(self, controller, teleop, cams, merge_proportional):
        self.controller = controller
        self.teleop = teleop
        self.HEAD_RGB, self.HEAD_DEPTH, self.WRIST_RGB, self.WRIST_DEPTH = cams
        self.merge = merge_proportional
        self.target = None
        self.head_search = True
        self.searcher = HeadSearcher()
        self.latest = {"Head (arriba)": None, "Wrist (brazo)": None}

    def on_mouse(self, event, x, y, flags, win):
        if event == cv2.EVENT_LBUTTONDOWN:
            frame = self.latest.get(win)
            if frame is not None:
                self.target = vision.make_color_target(frame, x, y)
                print(f"[fase1] objetivo: HSV~{self.target.hsv} (click en {win} @ {x},{y})", flush=True)

    def compute(self, head_frame, wrist_frame):
        """Calcula el comando automatico y los overlays. Devuelve (auto_cmd, vis_head, vis_wrist)."""
        auto = {}
        head_obj = wrist_obj = None
        head_err = wrist_err = None
        wrist_dist = None

        if self.target is not None:
            # --- WRIST: ver de frente al objeto ---
            if wrist_frame is not None:
                wrist_obj = vision.find_object(wrist_frame, self.target)
                if wrist_obj is not None:
                    cmd, (ex, ey) = servo.wrist_servo(wrist_obj["centroid"], wrist_frame.shape)
                    auto.update(cmd)
                    wrist_err = servo.error_norm(ex, ey)
                    d = self.WRIST_DEPTH.get_frame()
                    if d is not None:
                        from stretch_toolkit import WRIST_CAMERA
                        wrist_dist = WRIST_CAMERA.get_depth(wrist_obj["centroid"], d)

            # --- HEAD: apuntar al objeto (o buscarlo) ---
            if head_frame is not None:
                head_obj = vision.find_object(head_frame, self.target)
                if head_obj is not None:
                    cmd, (ex, ey) = servo.head_servo(head_obj["centroid"], head_frame.shape)
                    auto.update(cmd)
                    head_err = servo.error_norm(ex, ey)
                elif self.head_search:
                    auto.update(self.searcher.step(self.controller.get_state()))

        vis_head = draw_overlay(head_frame, head_obj, "Head", head_err) if head_frame is not None else None
        vis_wrist = draw_overlay(wrist_frame, wrist_obj, "Wrist", wrist_err, wrist_dist) if wrist_frame is not None else None
        return auto, vis_head, vis_wrist


def main():
    from stretch_toolkit import (
        controller, teleop, merge_proportional, BACKEND_NAME,
        HEAD_RGB_CAMERA, HEAD_DEPTH_CAMERA, WRIST_RGB_CAMERA, WRIST_DEPTH_CAMERA,
    )
    print(f"\n=== Fase 1 (seguimiento) en backend: {BACKEND_NAME} ===", flush=True)
    print("Click izquierdo en una ventana para picar un objeto. r=reset g=busqueda q=salir", flush=True)

    cams = (HEAD_RGB_CAMERA, HEAD_DEPTH_CAMERA, WRIST_RGB_CAMERA, WRIST_DEPTH_CAMERA)
    app = Fase1(controller, teleop, cams, merge_proportional)

    # Espera frames iniciales
    controller.get_state()
    for _ in range(50):
        if WRIST_RGB_CAMERA.get_frame() is not None:
            break
        time.sleep(0.1)

    win_head, win_wrist = "Head (arriba)", "Wrist (brazo)"
    cv2.namedWindow(win_head); cv2.setMouseCallback(win_head, app.on_mouse, win_head)
    cv2.namedWindow(win_wrist); cv2.setMouseCallback(win_wrist, app.on_mouse, win_wrist)

    try:
        while True:
            head_frame = HEAD_RGB_CAMERA.get_frame()
            wrist_frame = WRIST_RGB_CAMERA.get_frame()
            app.latest[win_head] = head_frame
            app.latest[win_wrist] = wrist_frame

            auto, vis_head, vis_wrist = app.compute(head_frame, wrist_frame)

            # Teleop siempre gana sobre el comando automatico
            cmd_teleop = teleop.get_normalized_velocities()
            final = merge_proportional(cmd_teleop, auto)
            controller.set_velocities(final)

            if vis_head is not None:
                cv2.imshow(win_head, vis_head)
            if vis_wrist is not None:
                cv2.imshow(win_wrist, vis_wrist)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                app.target = None
                print("[fase1] objetivo olvidado", flush=True)
            elif key == ord('g'):
                app.head_search = not app.head_search
                print(f"[fase1] busqueda de cabeza: {app.head_search}", flush=True)

            time.sleep(1 / 30)
    except KeyboardInterrupt:
        print("\n[fase1] saliendo...", flush=True)
    finally:
        controller.set_velocities({})
        controller.stop()
        cv2.destroyAllWindows()
        print("[fase1] listo.", flush=True)


if __name__ == "__main__":
    main()
