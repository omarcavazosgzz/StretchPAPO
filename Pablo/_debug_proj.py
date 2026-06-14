"""Debug de la proyeccion del oraculo en la camara HEAD: compara la proyeccion
3D->2D (en frame NATIVO sin rotar) con la deteccion por color del cubo rojo, en
el frame nativo y en el mostrado (rot90), para derivar el mapeo correcto.
"""
import time
from pathlib import Path
import numpy as np
import cv2

OUT = Path(__file__).resolve().parent / "snaps"


def cube_color(frame_bgr):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (0, 120, 80), (10, 255, 255)) | cv2.inRange(hsv, (170, 120, 80), (180, 255, 255))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if cv2.contourArea(c) >= 20]
    if not cnts:
        return None
    M = cv2.moments(max(cnts, key=cv2.contourArea))
    return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))


def main():
    from scene import build_pablo_kitchen
    from detection import _CamModel
    from stretch_mujoco import StretchMujocoSimulator
    from stretch_mujoco.enums.stretch_cameras import StretchCameras
    import mujoco

    model, xml, info = build_pablo_kitchen()
    cam = StretchCameras.cam_d435i_rgb
    sim = StretchMujocoSimulator(model=model, cameras_to_use=[cam])
    sim.start(headless=True)
    time.sleep(1.5)
    sim.move_to("head_pan", -0.75)
    sim.move_to("head_tilt", -0.6)
    time.sleep(2.0)

    cm = _CamModel(cam, model)
    st = sim.pull_status()
    world = st.object_poses["cubo_rojo"][:3]
    cps = st.camera_poses["d435i_camera_rgb"]
    R = np.array(cps["xmat"]).reshape(3, 3)
    p = np.array(world) - np.array(cps["pos"])
    Xc, Yc, Zc = R.T @ p
    depth = -Zc
    u = cm.cx + cm.f * (Xc / depth)
    v = cm.cy - cm.f * (Yc / depth)
    print(f"[dbg] native W={cm.W} H={cm.H} f={cm.f:.1f} cx={cm.cx} cy={cm.cy}", flush=True)
    print(f"[dbg] cam frame Xc={Xc:.3f} Yc={Yc:.3f} Zc={Zc:.3f} depth={depth:.3f}", flush=True)
    print(f"[dbg] NATIVE projection u={u:.1f} v={v:.1f}", flush=True)

    data = sim.pull_camera_data()
    native = data.get_camera_data(cam, auto_rotate=False)   # sin rotar
    shown = data.get_camera_data(cam, auto_rotate=True)     # mostrado (rot90 -1)
    print(f"[dbg] native shape={native.shape}  shown shape={shown.shape}", flush=True)
    cn = cube_color(native)
    cs = cube_color(shown)
    print(f"[dbg] cubo por COLOR en native = {cn}", flush=True)
    print(f"[dbg] cubo por COLOR en shown  = {cs}", flush=True)
    print(f"[dbg] -> comparar: proyeccion native (u,v)=({u:.0f},{v:.0f}) vs color native {cn}", flush=True)

    # dibujar y guardar
    nv = native.copy()
    if 0 <= u < cm.W and 0 <= v < cm.H:
        cv2.circle(nv, (int(u), int(v)), 8, (0, 0, 255), 2)
    if cn:
        cv2.drawMarker(nv, cn, (255, 255, 255), cv2.MARKER_CROSS, 14, 2)
    cv2.imwrite(str(OUT / "dbg_native.png"), nv)
    print("[dbg] guardado dbg_native.png (circulo=proyeccion, cruz=color)", flush=True)
    sim.stop()


if __name__ == "__main__":
    main()
