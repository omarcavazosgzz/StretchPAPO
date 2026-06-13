"""
Navegacion de la base por odometria con monitoreo de LiDAR (para no chocar).

go_to_xy(): gira hacia el objetivo y avanza hasta un 'standoff', frenando si el
LiDAR detecta un obstaculo al frente.

El frame de base_x/base_y/base_theta (de controller.get_state) coincide con el
frame de get_object_pose, asi que podemos navegar hacia la posicion del objeto.
"""
import time
import numpy as np


# Calibrado empiricamente (test_nav_calib.py): en este toolkit +base_forward
# avanza alineado con theta, pero +base_counterclockwise gira CW (baja theta).
# Por eso el comando de giro lleva signo negativo.
BASE_YAW_SIGN = -1.0


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def front_min_lidar(ranges, front_index=0, half_cone=35):
    """Distancia minima en un cono frontal. front_index se calibra aparte."""
    if ranges is None:
        return np.inf
    n = len(ranges)
    idx = [(front_index + d) % n for d in range(-half_cone, half_cone + 1)]
    vals = np.asarray(ranges)[idx]
    vals = vals[np.isfinite(vals)]
    return float(vals.min()) if vals.size else np.inf


def go_to_xy(controller, target_xy, standoff=0.45, max_time=25.0,
             front_index=0, stop_dist=0.35, slow_dist=0.7, log=None):
    """Lleva la base a 'standoff' metros del objetivo (x,y) en frame mundo.

    Devuelve dict con resultado: reached, final_dist, min_lidar, collided.
    """
    t_end = time.time() + max_time
    min_lidar_seen = np.inf
    collided = False
    last_log = 0.0

    while time.time() < t_end:
        st = controller.get_state()
        bx, by, bt = st["base_x"], st["base_y"], st["base_theta"]
        dx, dy = target_xy[0] - bx, target_xy[1] - by
        dist = float(np.hypot(dx, dy))

        if dist <= standoff:
            controller.set_velocities({"base_forward": 0.0, "base_counterclockwise": 0.0})
            return {"reached": True, "final_dist": dist, "min_lidar": min_lidar_seen, "collided": collided}

        desired = np.arctan2(dy, dx)
        herr = wrap(desired - bt)

        ranges = controller.get_lidar_ranges()
        fmin = front_min_lidar(ranges, front_index=front_index)
        min_lidar_seen = min(min_lidar_seen, fmin)

        cmd = {}
        if abs(herr) > 0.2:
            # primero orientar
            cmd["base_counterclockwise"] = float(np.clip(BASE_YAW_SIGN * 1.2 * herr, -1, 1))
            cmd["base_forward"] = 0.0
        else:
            # avanzar con freno por obstaculo y por cercania
            if fmin < stop_dist:
                fwd = 0.0
                collided = True  # bloqueado por obstaculo frontal
            elif fmin < slow_dist:
                fwd = 0.25
            else:
                fwd = float(np.clip(0.9 * dist, 0.3, 1.0))
            cmd["base_forward"] = fwd
            cmd["base_counterclockwise"] = float(np.clip(BASE_YAW_SIGN * 1.0 * herr, -0.4, 0.4))
        controller.set_velocities(cmd)

        if log is not None and time.time() - last_log > 1.0:
            log(f"  nav: dist={dist:.2f} herr={herr:+.2f} fmin={fmin:.2f} cmd={ {k: round(v,2) for k,v in cmd.items()} }")
            last_log = time.time()

        time.sleep(1 / 30)

    controller.set_velocities({"base_forward": 0.0, "base_counterclockwise": 0.0})
    st = controller.get_state()
    dist = float(np.hypot(target_xy[0] - st["base_x"], target_xy[1] - st["base_y"]))
    return {"reached": dist <= standoff, "final_dist": dist, "min_lidar": min_lidar_seen, "collided": collided}
