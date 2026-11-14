"""
Render a mesh along a 360 degree orbit (same normalization and field of view as render_mesh.py),
and save the camera of every frame so sketches can be rendered from exactly the same viewpoints.

    blender -b -P tools/render_turntable.py -- --mesh meshes/objaverse/bicycle.glb --out turntables/bicycle
Output: <out>/frame_XXX.png (RGBA) and <out>/cameras.json (fov, resolution, c2w per frame).
"""
import argparse
import json
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render_mesh as rm  # noqa: E402


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    p = argparse.ArgumentParser()
    p.add_argument("--mesh", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--frames", type=int, default=72, help="Frames per full turn")
    p.add_argument("--elevation", type=float, default=20.0, help="Camera elevation in degrees")
    p.add_argument("--resolution", type=int, default=320)
    p.add_argument("--fov", type=float, default=0.6911112070083618)
    p.add_argument("--margin", type=float, default=1.1)
    p.add_argument("--samples", type=int, default=32)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "gpu"])
    p.add_argument("--sun", type=float, default=2.0)
    p.add_argument("--ambient", type=float, default=1.0)
    return p.parse_args(argv)


def main():
    args = parse_args()
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    rm.reset_scene()
    meshes = rm.import_mesh(os.path.abspath(args.mesh))
    rm.normalize_to_unit_sphere(meshes)
    cam = rm.setup_render(args)
    scene = bpy.context.scene
    rm.set_png_output(scene)

    dist = args.margin / math.sin(0.5 * args.fov)
    el = math.radians(args.elevation)
    c2ws = []
    for i in range(args.frames):
        az = 2 * math.pi * i / args.frames
        eye = dist * np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
        c2w = rm.look_at_c2w(eye)
        c2ws.append(c2w.tolist())
        path = os.path.join(out, f"frame_{i:03d}.png")
        if not os.path.exists(path):
            cam.matrix_world = Matrix(c2w.tolist())
            bpy.context.view_layer.update()
            scene.render.filepath = path
            bpy.ops.render.render(write_still=True)
        print(f"[turntable] {i + 1}/{args.frames}", flush=True)
    with open(os.path.join(out, "cameras.json"), "w") as f:
        json.dump({"fov": args.fov, "resolution": args.resolution, "elevation": args.elevation,
                   "distance": dist, "c2w": c2ws}, f)
    print(f"[turntable] done: {out}")


if __name__ == "__main__":
    main()
