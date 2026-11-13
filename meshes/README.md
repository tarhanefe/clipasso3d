# Meshes → training data

Any 3D mesh can be turned into a training scene. The renderer places cameras around the
object, renders color images with a transparent background, and writes the camera poses in
the same NeRF/Blender format the training code reads (`data/<name>/transforms_*.json`).

```
meshes/
├── README.md              (this file)
├── my_objects/            your own meshes, one file per object
│   ├── chair.glb
│   └── lamp/              .obj: keep the .mtl and textures next to the .obj
│       ├── lamp.obj
│       ├── lamp.mtl
│       └── lamp_diffuse.png
├── objaverse_selection.json   the project's Objaverse objects (ids + attribution, committed)
└── objaverse/             downloaded by tools/fetch_objaverse.py (+ manifest.json)
```

Mesh files and rendered data are git-ignored (large, and licensed by their authors); only this
README and the selection list are in the repository.

## 1. Get a mesh

**Your own file.** Supported: `.glb` / `.gltf` (best: geometry, colors and textures in one
file), `.obj` (colors come from the `.mtl` file and the texture images it names, so copy the
whole folder), `.fbx`, `.ply` (vertex colors), `.stl` (no color). Size, position and up-axis do
not matter: every mesh is centered and scaled to the unit sphere, and glTF/OBJ are converted to
Blender's Z-up automatically.

`render_all.sh` (step 2) picks up mesh files directly inside the folder you give it, so for an
`.obj` in its own subfolder render it with `render_mesh.py` directly.

**From Objaverse (the project's object set).** The 20 objects used in the project (cars, trucks,
planes, chairs, mugs, ...; all CC-BY or CC0) are listed in `meshes/objaverse_selection.json`
together with their authors and sources. This downloads exactly them into `meshes/objaverse/`:

```bash
pip install objaverse
python tools/fetch_objaverse.py --out meshes/objaverse --selection meshes/objaverse_selection.json
```

**Picking new Objaverse objects.** Without `--selection`, one object per LVIS category is picked
automatically (filtered by license, 1k–60k faces and no animation, ranked by Sketchfab likes):

```bash
python tools/fetch_objaverse.py --out meshes/objaverse --categories teapot guitar          # only some
python tools/fetch_objaverse.py --out meshes/objaverse --swap "car_(automobile)" bicycle   # replace bad picks
python tools/fetch_objaverse.py --out meshes/objaverse --write_selection meshes/objaverse_selection.json
```

Look at the renders before training: some objects come with a ground plate, a shadow plane or
are unfinished. `--swap` replaces a pick with the next backup, and `--write_selection` saves the
current picks so others download the same set.

## 2. Render

Needs Blender (≥ 3.6; tested with 5.1). On macOS the app in `/Applications` is found
automatically; elsewhere set `BLENDER=/path/to/blender` (for the pods, see below).

```bash
# one mesh
blender -b -P tools/render_mesh.py -- --mesh meshes/my_objects/chair.glb --out data/chair

# every mesh in a folder, resumable
tools/render_all.sh meshes/objaverse data/objaverse
python tools/render_status.py data/objaverse        # x/y views done + ETA
```

Useful options of `render_mesh.py` (also accepted by `render_all.sh`):

| Option | Default | Meaning |
|---|---|---|
| `--resolution` | 512 | Image size (training resizes to `--width/--height`, 224) |
| `--n_train / --n_val / --n_test` | 100 / 30 / 30 | Number of views per split |
| `--elev_min / --elev_max` | -15 / 75 | Camera elevation band in degrees (train: even spiral, val/test: random) |
| `--margin` | 1.1 | Camera distance so the unit sphere fills 1/margin of the frame |
| `--fov` | 0.691 | Horizontal field of view (radians), same as the original datasets |
| `--samples` | 64 | Cycles samples per pixel |
| `--device` | auto | `gpu`, `cpu` or `auto` |
| `--sun`, `--ambient` | 2.0, 1.0 | Light strengths (white environment + one sun) |

Output for each object:

```
data/<name>/
├── transforms_{train,val,test}.json   camera_angle_x + camera-to-world per view (OpenGL/Blender convention)
├── rgb/{train,val,test}/r_<i>.png     RGBA, transparent background
├── world_pos/{train,val,test}/r_<i>.exr  world xyz seen by each pixel (A = coverage)
└── surface_points.npy                 200k points on the normalized surface, for 3D metrics
```

Re-running skips views that already exist, so an interrupted render continues where it stopped.

## 3. Check and train

```bash
pip install OpenEXR
python tools/check_poses.py data/objaverse/*       # must report ~0.01 px and no bad frames
python run.py --data_name objaverse/car_\(automobile\)
```

`check_poses.py` projects every pixel's world position with its camera pose and checks that it
lands back on the same pixel, so a wrong pose, axis convention or frame order is caught before
any training is spent on it.

## Running on the RunAI pods

Blender runs headless on the pods (Cycles uses the GPU). Install it once on `/scratch`:

```bash
cd /scratch/<user> && wget https://download.blender.org/release/Blender5.1/blender-5.1.2-linux-x64.tar.xz
tar xf blender-5.1.2-linux-x64.tar.xz && export BLENDER=/scratch/<user>/blender-5.1.2-linux-x64/blender
tmux new-session -d -s render "tools/render_all.sh meshes/objaverse data/objaverse 2>&1 | tee -a data/objaverse/logs/render_all.log"
```
