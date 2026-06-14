"""Verifica el DETECTOR oraculo: proyecta cada objeto a las camaras head/wrist,
dibuja el marcador sobre el frame real y CALIBRA con el cubo rojo (detectable por
color) -> reporta el error en pixeles entre la proyeccion y el color.

Si el marcador cae sobre los objetos y el error del cubo rojo es pequeno (<~8px),
el oraculo es 100% confiable.

    uv run Pablo/test_detection.py
Salida: Pablo/snaps/det_head.png , Pablo/snaps/det_wrist.png
"""
import time
from pathlib import Path
import numpy as np
import cv2

OUT = Path(__file__).resolve().parent / "snaps"
OUT.mkdir(exist_ok=True)


def red_cube_pixel_by_color(frame_bgr):
    """Detecta el cubo rojo por color en el frame (ground-truth independiente)."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (0, 120, 80), (10, 255, 255)) | cv2.inRange(hsv, (170, 120, 80), (180, 255, 255))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if cv2.contourArea(c) >= 25]
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    M = cv2.moments(c)
    return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))


def main():
    from scene import build_pablo_kitchen, DEFAULT_OBJECTS
    from detection import OracleDetector
    from stretch_mujoco import StretchMujocoSimulator
    from stretch_mujoco.enums.stretch_cameras import StretchCameras

    print("[det] generando cocina + objetos...", flush=True)
    model, xml, info = build_pablo_kitchen()
    cams = [StretchCameras.cam_d435i_rgb, StretchCameras.cam_d405_rgb]
    sim = StretchMujocoSimulator(model=model, cameras_to_use=cams)
    sim.start(headless=True)
    time.sleep(1.5)
    # Apuntar la cabeza a los objetos (y el brazo un poco al frente y abajo)
    sim.move_to("head_pan", -0.75)
    sim.move_to("head_tilt", -0.6)
    sim.move_to("wrist_yaw", 0.0)
    sim.move_to("wrist_pitch", -0.8)
    sim.move_to("arm", 0.2)
    time.sleep(2.0)

    det = OracleDetector(sim, model)
    labels = [o[0] for o in DEFAULT_OBJECTS]

    data = sim.pull_camera_data().get_all(use_depth_color_map=False)
    colors = {"cubo_rojo": (0, 0, 255), "huevo": (0, 255, 255), "tomate": (255, 0, 255),
              "cuchillo": (255, 255, 0), "plato": (0, 255, 0)}

    for camera, tag in [(StretchCameras.cam_d435i_rgb, "head"), (StretchCameras.cam_d405_rgb, "wrist")]:
        frame = data.get(camera)
        if frame is None:
            print(f"[det] {tag}: sin frame", flush=True)
            continue
        vis = frame.copy()
        H, W = frame.shape[:2]
        print(f"\n[det] === camara {tag} ({W}x{H}) ===", flush=True)
        for label in labels:
            d = det.detect(camera, label)
            if d is None:
                print(f"[det]   {label:10s}: sin pose", flush=True)
                continue
            x, y = int(round(d.centroid[0])), int(round(d.centroid[1]))
            inside = "DENTRO" if d.in_frame else "fuera "
            print(f"[det]   {label:10s}: pixel=({x:4d},{y:4d}) {inside} depth={d.depth:.2f}m", flush=True)
            if d.in_frame:
                col = colors.get(label, (255, 255, 255))
                cv2.circle(vis, (x, y), 8, col, 2)
                cv2.putText(vis, label, (x + 8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
        # Calibracion con el cubo rojo (color vs proyeccion)
        rc = red_cube_pixel_by_color(frame)
        dproj = det.detect(camera, "cubo_rojo")
        if rc is not None and dproj is not None and dproj.in_frame:
            err = np.hypot(rc[0] - dproj.centroid[0], rc[1] - dproj.centroid[1])
            cv2.drawMarker(vis, rc, (255, 255, 255), cv2.MARKER_CROSS, 12, 1)
            print(f"[det]   CALIBRACION cubo_rojo: color=({rc[0]},{rc[1]}) "
                  f"proyeccion=({dproj.centroid[0]:.0f},{dproj.centroid[1]:.0f}) "
                  f"error={err:.1f}px  {'OK' if err < 8 else 'REVISAR'}", flush=True)
        elif rc is None:
            print("[det]   (cubo rojo no visible por color en esta camara)", flush=True)
        cv2.imwrite(str(OUT / f"det_{tag}.png"), vis)
        print(f"[det]   guardado det_{tag}.png", flush=True)

    print("\n[det] ===== DONE =====", flush=True)
    sim.stop()


if __name__ == "__main__":
    main()
