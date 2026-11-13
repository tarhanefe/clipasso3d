"""
Export a ground-truth mesh for the website: normalized exactly like the training data (unit sphere),
decimated to at most --max_faces triangles, textures downscaled, saved as one .glb.

    blender -b -P tools/export_gt_glb.py -- --mesh meshes/objaverse/bicycle.glb --out site/static/data/gt/bicycle.glb
"""
import argparse
import os
import sys

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render_mesh as rm  # noqa: E402


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    p = argparse.ArgumentParser()
    p.add_argument("--mesh", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max_faces", type=int, default=40000)
    p.add_argument("--texture_size", type=int, default=512)
    return p.parse_args(argv)


def main():
    args = parse_args()
    rm.reset_scene()
    meshes = rm.import_mesh(os.path.abspath(args.mesh))
    rm.normalize_to_unit_sphere(meshes)

    faces = sum(len(o.data.polygons) for o in meshes)
    if faces > args.max_faces:
        ratio = args.max_faces / faces
        for o in meshes:
            mod = o.modifiers.new("decimate", "DECIMATE")
            mod.ratio = ratio
    for img in bpy.data.images:
        if img.size[0] > args.texture_size or img.size[1] > args.texture_size:
            s = args.texture_size / max(img.size)
            img.scale(max(1, int(img.size[0] * s)), max(1, int(img.size[1] * s)))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.export_scene.gltf(filepath=os.path.abspath(args.out), export_format="GLB", use_selection=False,
                              export_apply=True, export_animations=False, export_image_format="JPEG")
    print(f"[gt] {args.out}: {faces} -> <= {args.max_faces} faces, {os.path.getsize(args.out) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
