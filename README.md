# From Pixels to Wireframes: 3D Reconstruction via CLIP-Based Sketch Abstraction

<p align="center">
  <img src="assets/pipeline.png" width=700px />
</p>

This repository contains the complete implementation and supporting modules for the course project titled **“From Pixels to Wireframes: Reconstruction via CLIP-Based Sketch Abstraction”**, developed as part of the EPFL **CS503 — Visual Intelligence: Machines and Minds** course. The project explores a novel approach to abstracting 3D surfaces into wireframe-like representations by leveraging CLIP-based semantic guidance and Bézier curve parameterizations.

---

# Project Structure

```
clipasso3d/
├── CLIP_/                    
├── data/
├── notebooks/
├── source/
├── .gitignore
├── .gitmodules
├── README.md
└── requirements.txt
```

---

# Setup & Installation

**Clone the repository:**
   ```bash
   git clone --recurse-submodules https://github.com/tarhanefe/clipasso3d.git
   cd clipasso3d
   ```

**Create a new Conda environment with python version 3.10:**
   ```bash
   conda create -n 3dsketch python=3.10 -y
   conda activate 3dsketch
   ```
**Install Pytorch for Cuda version 12.1:**
   ```bash
   pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
   ```

**Install library requirements:**
   ```bash
   pip install -r requirements.txt
   ```
---
# How to Use ? 

## 1- Trial Notebook

Use ```WireframeTrial.ipynb```  notebook inside the notebooks folder to interactively examine the training process and obtained results.

## 2- Run file 

```bash
  cd clipasso3d
  python run.py \
  --data_name rose \
  --run_name default \
  --output_gif training_evolution.gif \
  --batch_size 1 \
  --epochs 10 \
  --inner_steps 30 \
  --learning_rate 0.005 \
  --n_curves 25 \
  --thickness 0.02 \
  --radius 0.8 \
  --length 0.02 \
  --overlap 0.6 \
  --clip_weight 1.0 \
  --clip_conv_loss 1.0 \
  --clip_fc_loss_weight 0.1 \
  --clip_conv_layer_weights 0 0 1.0 1.0 0.0 \
  --gif_fps 120 \
  --rotation_time 6.0 \
  --revolutions 3.0 \
  --output_semigif semihelical.gif
```
where 

### 🔧 Argument Descriptions

| Argument                     | Description                                                                 |
|-----------------------------|-----------------------------------------------------------------------------|
| `--data_name`               | Dataset folder inside `--data_root` (default `data/`), e.g. `rose` or `objaverse/teapot`. |
| `--run_name`                | Name of the configuration; outputs go to `runs/<data_name>/<run_name>/` (or `--out_dir`). |
| `--seed`                    | Random seed (curve initialization, view sampling, augmentations).           |
| `--output_gif`              | Training evolution GIF name inside the run folder.                          |
| `--batch_size`              | Number of images processed per training batch.                              |
| `--epochs`                  | Number of training epochs.                                                  |
| `--inner_steps`             | Inner optimization steps per batch.                                         |
| `--learning_rate`           | Learning rate for Adam optimizer.                                           |
| `--n_curves`                | Number of Bézier curves initialized in the scene.                           |
| `--thickness`               | Radius (thickness) of each rendered curve/sphere segment.                   |
| `--radius`                  | Distance from the scene center to place initial curve points.               |
| `--length`                  | Length of the each initial curve segment.                                   |
| `--overlap`                 | Degree of allowed overlap between nearby spherical gaussians on a curve.    |
| `--clip_weight`             | Total CLIP loss weight.                                                     |
| `--clip_conv_loss`          | Weight of the convolutional CLIP loss.                                      |
| `--clip_fc_loss_weight`     | Weight of CLIP’s final-layer (semantic) similarity loss.                    |
| `--clip_conv_layer_weights` | Weights for each CLIP convolutional layer (ViT-B/32 has 5 layers).          |
| `--gif_fps`                 | Frame rate (FPS) of generated GIFs.                                         |
| `--rotation_time`           | Duration (in seconds) of 3D camera rotation in the fly-around GIF.          |
| `--revolutions`             | Number of full revolutions in the semihelical fly-around.                   |
| `--output_semigif`          | Output path for the 3D rendered semihelical GIF animation.                  |

---

### Outputs, checkpoints and resuming

Each run writes to its own folder `runs/<data_name>/<run_name>/`: `config.json` (arguments and git
commit), `ckpt.pt` (saved every `--ckpt_every` batches), `log.jsonl` (one line per batch),
`frames/`, `curves.pt`, `metrics.json`, the two GIFs and finally `DONE`. Running the same command
again resumes from the checkpoint (bit-exact) or skips the run if it is finished.

`metrics.json` is computed on the held-out test views: `clip_similarity` (CLIP ViT-B/32 cosine
similarity between render and photo), `silhouette_precision` (fraction of stroke ink on the object)
and, for rendered meshes, `accuracy` / `completeness` / `chamfer` (distances between the curves and
the true surface, in units of the unit sphere).

Optional Weights & Biases logging: `--wandb [--wandb_project clipasso3d] [--wandb_entity ...]`.
A restarted run continues the same wandb run; `--wandb_mode offline` logs into the run folder for
machines without internet (upload later with `wandb sync runs/.../wandb/offline-run-*`).

### Many runs on many GPUs

```bash
# one job per dataset; arguments after -- go to run.py
python tools/sweep.py make --datasets data/objaverse/* --run_name baseline --out jobs/baseline.jsonl -- --epochs 10
# start one worker per GPU (any number of pods); workers share jobs through the file system
CUDA_VISIBLE_DEVICES=0 python tools/sweep.py run --jobs jobs/baseline.jsonl --worker 0 --nworkers 2
CUDA_VISIBLE_DEVICES=1 python tools/sweep.py run --jobs jobs/baseline.jsonl --worker 1 --nworkers 2
# progress: jobs done / total, running workers, failures, ETA, best results
python tools/status.py jobs/baseline.jsonl
```

Workers claim jobs atomically, refresh a heartbeat and take over jobs of workers that died; a job
that keeps failing is marked `FAILED`, and CUDA errors make the worker exit so a fresh process retries.

## 3- Train on your own meshes or Objaverse objects

Any `.glb` / `.obj` / `.fbx` / `.ply` / `.stl` mesh can be rendered into a training scene with Blender
(`tools/render_mesh.py`), and `tools/fetch_objaverse.py` downloads a set of simple CC-BY objects.
See [meshes/README.md](meshes/README.md) for where to put the files and how to render, check and train.

## Examples

#### The plant 


<table align="center">
  <tr>
    <td align="center"><img src="assets/tree.gif" width="250px"><br>Training of the Model</td>
    <td align="center"><img src="assets/tree.png" width="250px"><br>Model w/ 3D Plant Model</td>
    <td align="center"><img src="assets/tree_gt.png" width="250px"><br>Model w/ Ground Truth Plant</td>
  </tr>
</table>


#### The rose 

<table align="center">
  <tr>
    <td align="center"><img src="assets/rose_5.gif" width="200px"><br>Model w/ 5 Curves</td>
    <td align="center"><img src="assets/rose_10.gif" width="200px"><br>Model w/ 10 Curves</td>
    <td align="center"><img src="assets/rose_15.gif" width="200px"><br>Model w/ 15 Curves</td>
  </tr>
</table>
<table align="center">
  <tr>
    <td align="center"><img src="assets/rose_20.gif" width="200px"><br>Model w/ 20 Curves</td>
    <td align="center"><img src="assets/rose_25.gif" width="200px"><br>Model w/ 25 Curves</td>
    <td align="center"><img src="assets/rose_gt.gif" width="200px"><br>Ground Truth Scene</td>
  </tr>
</table>


#### The duck 

<table align="center">
  <tr>
    <td align="center"><img src="assets/duck_002.gif" width="200px"><br>Model w/ 0.02 Thickness</td>
    <td align="center"><img src="assets/duck_004.gif" width="200px"><br>Model w/ 0.04 Thickness</td>
    <td align="center"><img src="assets/duck_gt.gif" width="200px"><br>Ground Truth Scene</td>
  </tr>
</table>
</table>

