"""
Render a mesh (.glb/.gltf/.obj/.fbx/.ply/.stl) into a NeRF-style dataset that the
training pipeline reads directly:

    <out>/transforms_{train,val,test}.json   camera_angle_x + per-frame c2w (Blender/OpenGL convention)
    <out>/rgb/{split}/r_{i}.png              RGBA, transparent background
    <out>/world_pos/{split}/r_{i}.exr        RGBA float: world xyz of the surface seen by each pixel, A = coverage
    <out>/surface_points.npy                 (N,3) points sampled on the normalized mesh surface (for 3D metrics)

The mesh is translated and scaled so that its bounding sphere is the unit sphere at the
origin, and cameras look at the origin from a distance that keeps the whole sphere in view.

Run with Blender (no GUI needed):
    blender -b -P tools/render_mesh.py -- --mesh assets/chair.glb --out data/chair
or with the `bpy` pip module:
    python tools/render_mesh.py --mesh assets/chair.glb --out data/chair
"""
import argparse
import json
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Matrix, Vector


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    p = argparse.ArgumentParser(description="Render a mesh into a NeRF-style dataset")
    p.add_argument("--mesh", required=True, help="Path to .glb/.gltf/.obj/.fbx/.ply/.stl")
    p.add_argument("--out", required=True, help="Output dataset folder (e.g. data/chair)")
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--fov", type=float, default=0.6911112070083618, help="Horizontal FOV in radians (same as the original datasets)")
    p.add_argument("--margin", type=float, default=1.1, help="Bounding sphere radius / visible radius; >1 keeps a border")
    p.add_argument("--n_train", type=int, default=100)
    p.add_argument("--n_val", type=int, default=30)
    p.add_argument("--n_test", type=int, default=30)
    p.add_argument("--elev_min", type=float, default=-15.0, help="Lowest camera elevation in degrees")
    p.add_argument("--elev_max", type=float, default=75.0, help="Highest camera elevation in degrees")
    p.add_argument("--samples", type=int, default=64, help="Cycles samples per pixel")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "gpu"])
    p.add_argument("--sun", type=float, default=2.0, help="Sun strength (0 = ambient light only)")
    p.add_argument("--ambient", type=float, default=1.0, help="Uniform white environment light strength")
    p.add_argument("--n_surface_points", type=int, default=200000)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


# ---------------------------------------------------------------- scene setup

def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_mesh(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=path)
    elif ext == ".obj":
        bpy.ops.wm.obj_import(filepath=path)
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=path)
    elif ext == ".ply":
        bpy.ops.wm.ply_import(filepath=path)
    elif ext == ".stl":
        bpy.ops.wm.stl_import(filepath=path)
    else:
        raise ValueError(f"Unsupported mesh format: {ext}")
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        raise RuntimeError(f"No mesh objects found in {path}")
    return meshes


def world_vertices(objs):
    """All evaluated vertices of `objs` in world space, (N,3)."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    out = []
    for o in objs:
        ev = o.evaluated_get(depsgraph)
        me = ev.to_mesh()
        v = np.empty(len(me.vertices) * 3, dtype=np.float64)
        me.vertices.foreach_get("co", v)
        v = v.reshape(-1, 3)
        M = np.array(ev.matrix_world)
        out.append(v @ M[:3, :3].T + M[:3, 3])
        ev.to_mesh_clear()
    return np.concatenate(out, 0)


def normalize_to_unit_sphere(objs):
    """Parent every root object to an empty that maps the bounding sphere to the unit sphere."""
    v = world_vertices(objs)
    center = 0.5 * (v.min(0) + v.max(0))
    radius = float(np.linalg.norm(v - center, axis=1).max())
    root = bpy.data.objects.new("normalize", None)
    bpy.context.scene.collection.objects.link(root)
    for o in bpy.context.scene.objects:
        if o.parent is None and o is not root:
            mw = o.matrix_world.copy()
            o.parent = root
            o.matrix_world = mw
    s = 1.0 / radius
    root.matrix_world = Matrix.Diagonal((s, s, s, 1.0)) @ Matrix.Translation(Vector(-center))
    bpy.context.view_layer.update()
    return center.tolist(), radius


def sample_surface(objs, n, rng):
    """Area-weighted uniform samples on the (normalized) mesh surface."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    tris = []
    for o in objs:
        ev = o.evaluated_get(depsgraph)
        me = ev.to_mesh()
        me.calc_loop_triangles()
        v = np.empty(len(me.vertices) * 3); me.vertices.foreach_get("co", v)
        t = np.empty(len(me.loop_triangles) * 3, dtype=np.int64); me.loop_triangles.foreach_get("vertices", t)
        M = np.array(ev.matrix_world)
        v = v.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]
        tris.append(v[t.reshape(-1, 3)])
        ev.to_mesh_clear()
    tris = np.concatenate(tris, 0)                                  # (T,3,3)
    area = 0.5 * np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1)
    idx = rng.choice(len(tris), size=n, p=area / area.sum())
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)
    w = np.stack([1 - s, s * (1 - r2), s * r2], 1)
    return (w[:, :, None] * tris[idx]).sum(1).astype(np.float32)


def setup_render(args):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = args.samples
    scene.cycles.use_denoising = True
    scene.render.resolution_x = scene.render.resolution_y = args.resolution
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.view_settings.view_transform = "Standard"   # faithful colors (AgX/Filmic desaturate)
    scene.view_layers[0].use_pass_position = True

    if args.device != "cpu":
        prefs = bpy.context.preferences.addons["cycles"].preferences
        for backend in ("OPTIX", "CUDA", "HIP", "METAL", "ONEAPI"):
            try:
                prefs.compute_device_type = backend
            except TypeError:
                continue
            prefs.get_devices()
            gpus = [d for d in prefs.devices if d.type != "CPU"]
            if gpus:
                for d in prefs.devices:
                    d.use = d.type != "CPU"
                scene.cycles.device = "GPU"
                print(f"[render_mesh] Cycles on {backend}: {[d.name for d in gpus]}")
                break
        else:
            if args.device == "gpu":
                raise RuntimeError("No GPU available for Cycles")
            print("[render_mesh] No GPU found, rendering on CPU")

    # Lighting: uniform white environment (keeps texture colors readable) + optional sun
    world = bpy.data.worlds.new("world")
    if not world.node_tree:
        world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (1, 1, 1, 1)
    bg.inputs["Strength"].default_value = args.ambient
    scene.world = world
    if args.sun > 0:
        sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
        sun.data.energy = args.sun
        sun.rotation_euler = (math.radians(40), 0, math.radians(30))
        scene.collection.objects.link(sun)

    cam = bpy.data.objects.new("camera", bpy.data.cameras.new("camera"))
    cam.data.sensor_fit = "HORIZONTAL"
    cam.data.angle_x = args.fov
    cam.data.clip_start, cam.data.clip_end = 0.01, 100.0
    scene.collection.objects.link(cam)
    scene.camera = cam
    return cam


# ---------------------------------------------------------------- cameras

def look_at_c2w(eye):
    """Camera-to-world, OpenGL/Blender convention: camera looks along -Z, +Y up, target = origin, world up = +Z."""
    eye = np.asarray(eye, dtype=np.float64)
    z = eye / np.linalg.norm(eye)
    up = np.array([0.0, 0.0, 1.0])
    if abs(z @ up) > 0.999:
        up = np.array([0.0, 1.0, 0.0])
    x = np.cross(up, z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = x, y, z, eye
    return c2w


def camera_positions(n, dist, elev_min, elev_max, rng=None):
    """Fibonacci spiral over the elevation band (rng=None), or seeded random positions in the band."""
    lo, hi = math.sin(math.radians(elev_min)), math.sin(math.radians(elev_max))
    if rng is None:
        k = np.arange(n) + 0.5
        sz = lo + (hi - lo) * k / n
        az = k * math.pi * (3 - math.sqrt(5))
    else:
        sz = rng.uniform(lo, hi, n)
        az = rng.uniform(0, 2 * math.pi, n)
    r = np.sqrt(1 - sz ** 2)
    return dist * np.stack([r * np.cos(az), r * np.sin(az), sz], 1)


# ---------------------------------------------------------------- output

def read_position(exr_path):
    """Read the Position pass from a multilayer EXR as (H,W,3) float32, top row first.
    Blender 5 writes one EXR part per pass, older versions one part with all channels."""
    import OpenImageIO as oiio
    inp = oiio.ImageInput.open(exr_path)
    if inp is None:
        raise RuntimeError(oiio.geterror())
    found, seen, sub = {}, [], 0
    while inp.seek_subimage(sub, 0):
        spec = inp.spec()
        names = list(spec.channelnames)
        seen += names
        idx = {c.split(".")[-1]: k for k, c in enumerate(names) if c.split(".")[-2:-1] == ["Position"]}
        if idx:
            pix = inp.read_image(sub, 0, 0, spec.nchannels, oiio.FLOAT).reshape(spec.height, spec.width, spec.nchannels)
            found = {a: pix[..., idx[a]] for a in "XYZ" if a in idx}
            break
        sub += 1
    inp.close()
    if len(found) != 3:
        raise RuntimeError(f"No Position pass in {exr_path}: {seen}")
    return np.stack([found["X"], found["Y"], found["Z"]], -1)


def write_world_pos(path, xyz, alpha):
    """Write (H,W,3) world positions and (H,W) coverage as a half-float RGBA EXR."""
    import OpenImageIO as oiio
    h, w = alpha.shape
    data = np.concatenate([xyz * (alpha[..., None] > 0), alpha[..., None]], -1).astype(np.float32)
    spec = oiio.ImageSpec(w, h, 4, oiio.HALF)
    spec.attribute("compression", "zip")
    out = oiio.ImageOutput.create(path)
    out.open(path, spec)
    out.write_image(data)
    out.close()


def set_output_format(scene, kind):
    """PNG RGBA 8-bit, or multilayer float EXR (Blender 5 moved multilayer to media_type)."""
    st = scene.render.image_settings
    if kind == "png":
        if hasattr(st, "media_type"):
            st.media_type = "IMAGE"
        st.file_format, st.color_mode, st.color_depth = "PNG", "RGBA", "8"
    else:
        if hasattr(st, "media_type"):
            st.media_type = "MULTI_LAYER_IMAGE"
        st.file_format, st.color_depth = "OPEN_EXR_MULTILAYER", "32"


def render_split(split, positions, cam, out_dir, args):
    scene = bpy.context.scene
    os.makedirs(os.path.join(out_dir, "rgb", split), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "world_pos", split), exist_ok=True)
    tmp_exr = os.path.join(out_dir, f".tmp_{split}.exr")
    frames = []
    for i, eye in enumerate(positions):
        c2w = look_at_c2w(eye)
        cam.matrix_world = Matrix(c2w.tolist())
        bpy.context.view_layer.update()
        stem = f"r_{i}"
        png = os.path.join(out_dir, "rgb", split, stem + ".png")
        wp = os.path.join(out_dir, "world_pos", split, stem + ".exr")
        if not (os.path.exists(png) and os.path.exists(wp)):   # resumable
            # write_still saves every enabled pass (incl. Position) into the multilayer EXR;
            # the PNG comes from the same render result with the view transform applied
            set_output_format(scene, "multilayer_exr")
            scene.render.filepath = tmp_exr
            bpy.ops.render.render(write_still=True)
            set_output_format(scene, "png")
            bpy.data.images["Render Result"].save_render(png + ".tmp.png", scene=scene)
            xyz = read_position(tmp_exr)
            alpha = read_alpha(png + ".tmp.png")
            write_world_pos(wp + ".tmp.exr", xyz, alpha)
            os.replace(wp + ".tmp.exr", wp)
            os.replace(png + ".tmp.png", png)
        frames.append({
            "file_path": f"{split}/{stem}",
            "frame_id": i,
            "camera_location": [float(x) for x in eye],
            "transform_matrix": c2w.tolist(),
        })
        print(f"[render_mesh] {split} {i + 1}/{len(positions)}", flush=True)
    if os.path.exists(tmp_exr):
        os.remove(tmp_exr)
    return frames


def read_alpha(png_path):
    import OpenImageIO as oiio
    inp = oiio.ImageInput.open(png_path)
    spec = inp.spec()
    pix = inp.read_image(0, 0, 0, spec.nchannels, oiio.FLOAT).reshape(spec.height, spec.width, spec.nchannels)
    inp.close()
    return pix[..., 3] if spec.nchannels == 4 else np.ones(pix.shape[:2], np.float32)


def main():
    args = parse_args()
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    reset_scene()
    meshes = import_mesh(os.path.abspath(args.mesh))
    center, radius = normalize_to_unit_sphere(meshes)
    cam = setup_render(args)

    pts_path = os.path.join(out_dir, "surface_points.npy")
    if not os.path.exists(pts_path):
        np.save(pts_path, sample_surface(meshes, args.n_surface_points, rng))

    dist = args.margin / math.sin(0.5 * args.fov)
    splits = {
        "train": camera_positions(args.n_train, dist, args.elev_min, args.elev_max),
        "val": camera_positions(args.n_val, dist, args.elev_min, args.elev_max, np.random.default_rng(args.seed + 1)),
        "test": camera_positions(args.n_test, dist, args.elev_min, args.elev_max, np.random.default_rng(args.seed + 2)),
    }
    for split, pos in splits.items():
        frames = render_split(split, pos, cam, out_dir, args)
        meta = {
            "metadata": {
                "blender_version": bpy.app.version_string,
                "engine": "CYCLES",
                "source_mesh": os.path.basename(args.mesh),
                "normalization": {"center": center, "radius": radius},
                "render_args": vars(args),
            },
            "camera_angle_x": args.fov,
            "object_position": [0.0, 0.0, 0.0],
            "frames": frames,
        }
        tmp = os.path.join(out_dir, f"transforms_{split}.json.tmp")
        with open(tmp, "w") as f:
            json.dump(meta, f, indent=2)
        os.replace(tmp, os.path.join(out_dir, f"transforms_{split}.json"))
    print(f"[render_mesh] done: {out_dir}")


if __name__ == "__main__":
    main()
