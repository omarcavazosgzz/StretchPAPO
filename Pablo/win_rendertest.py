"""Prueba minima: verifica que MuJoCo puede renderizar offscreen (camara) en Windows.
No depende de RoboCasa ni de los assets de 5GB. Solo valida el backend de OpenGL.
"""
import os
import numpy as np
import mujoco

XML = """
<mujoco>
  <worldbody>
    <light pos="0 0 2"/>
    <geom name="floor" type="plane" size="2 2 .1" rgba=".8 .8 .8 1"/>
    <body pos="0 0 .3">
      <freejoint/>
      <geom name="cube" type="box" size=".1 .1 .1" rgba="1 0 0 1"/>
    </body>
    <camera name="topcam" pos="0 0 2" xyaxes="1 0 0 0 1 0"/>
  </worldbody>
</mujoco>
"""

def main():
    print("MUJOCO_GL =", os.environ.get("MUJOCO_GL", "(default)"))
    print("mujoco version:", mujoco.__version__)
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    with mujoco.Renderer(model, height=240, width=320) as r:
        r.update_scene(data, camera="topcam")
        img = r.render()
    print("render OK -> shape:", img.shape, "dtype:", img.dtype)
    print("mean RGB:", img.reshape(-1, 3).mean(axis=0))
    print("max  RGB:", img.reshape(-1, 3).max(axis=0))
    print("min  RGB:", img.reshape(-1, 3).min(axis=0))
    print("unique colors:", len(np.unique(img.reshape(-1, 3), axis=0)))
    red = (img[:, :, 0] > 120) & (img[:, :, 1] < 80) & (img[:, :, 2] < 80)
    print("pixeles rojos (cubo visible):", int(red.sum()))
    import imageio.v2 as imageio
    out = os.path.join(os.path.dirname(__file__), "win_rendertest.png")
    imageio.imwrite(out, img)
    print("imagen guardada en:", out)
    is_blank = len(np.unique(img.reshape(-1, 3), axis=0)) <= 2
    if is_blank:
        print("\nADVERTENCIA: imagen casi en blanco -> render vacio")
    else:
        print("\nOK: MuJoCo renderiza contenido (no esta en blanco) en Windows.")

if __name__ == "__main__":
    main()
