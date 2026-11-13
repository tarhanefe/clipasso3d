# run.py
import argparse
from pathlib import Path

from source.pipeline import run_pipeline


def parse_args():
    parser = argparse.ArgumentParser(description="3D Sketch Abstraction Training Pipeline")

    # Dataset and I/O
    parser.add_argument('--data_name', type=str, default='rose', help='Dataset folder inside --data_root (e.g. objaverse/teapot)')
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--run_name', type=str, default='default', help='Name of this configuration')
    parser.add_argument('--out_dir', type=str, default=None, help='Run folder (default: runs/<data_name>/<run_name>)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--limit_views', type=int, default=0, help='Use only the first N views per split (smoke tests)')
    parser.add_argument('--width', type=int, default=224, help='Image width')
    parser.add_argument('--height', type=int, default=224, help='Image height')
    parser.add_argument('--output_gif', type=str, default='training_evolution.gif', help='Training GIF name inside the run folder')

    # Optimization parameters
    parser.add_argument('--batch_size', type=int, default=8, help='Different training views per optimization step')
    parser.add_argument('--epochs', type=int, default=150, help='Passes over the training views (100 views / 8 = 13 steps each)')
    parser.add_argument('--inner_steps', type=int, default=1, help='Optimization steps per sampled batch of views')
    parser.add_argument('--learning_rate', type=float, default=0.005)
    parser.add_argument('--eval_interval', type=int, default=100, help='Validate every N batches (0 = once per epoch)')
    parser.add_argument('--early_stop_patience', type=int, default=0,
                        help='Stop when the val loss has not improved for this many validations (0 = run all epochs)')
    parser.add_argument('--early_stop_tol', type=float, default=0.005, help='Relative val-loss improvement that counts')
    parser.add_argument('--min_batches', type=int, default=500, help='Never stop early before this many batches')
    parser.add_argument('--ckpt_every', type=int, default=20, help='Checkpoint every N batches')
    parser.add_argument('--frame_every', type=int, default=10, help='Save a render|target frame every N batches')
    parser.add_argument('--no_compile', action='store_true', help='Skip torch.compile (CPU smoke tests)')
    parser.add_argument('--rasterizer', type=str, default='sparse', choices=['sparse', 'dense'],
                        help='dense: every sphere at every pixel; sparse: +-4 sigma window per sphere (needed for many curves)')

    # Curve parameters
    parser.add_argument('--n_curves', type=int, default=25)
    parser.add_argument('--thickness', type=float, default=0.008, help='Stroke radius (Gaussian sigma) in scene units; 0.008 ~ 0.75 px')
    parser.add_argument('--radius', type=float, default=0.8)
    parser.add_argument('--length', type=float, default=0.15, help='Length of each initial line')
    parser.add_argument('--overlap', type=float, default=0.6)
    parser.add_argument('--sample_spacing', type=float, default=0.5,
                        help='Spheres along each curve at most this many thicknesses apart, so strokes stay continuous '
                             'at any length (0 = use --samples_per_curve)')
    parser.add_argument('--samples_per_curve', type=int, default=0,
                        help='With --sample_spacing 0: fixed spheres per curve, or 0 = from length/thickness/overlap (original)')
    parser.add_argument('--init', type=str, default='hull', choices=['hull', 'sphere'],
                        help='hull: spread over the visual hull surface of the training masks; sphere: random short lines on a sphere of --radius')
    parser.add_argument('--hull_grid', type=int, default=64, help='Voxel grid resolution of the visual hull')
    parser.add_argument('--prune_every', type=int, default=0,
                        help='Every N batches delete curves off the object (outside the visual hull) or collapsed; 0 = off')
    parser.add_argument('--prune_start', type=int, default=0, help='First batch at which pruning may happen')
    parser.add_argument('--prune_min_on_frac', type=float, default=0.5,
                        help='Delete a curve if less than this fraction of its points lies inside the visual hull')
    parser.add_argument('--prune_min_length', type=float, default=0.005, help='Delete a curve shorter than this')
    parser.add_argument('--connect_weight', type=float, default=0.0,
                        help='Connectivity loss weight: every curve touches another and all form one piece (0 = off)')
    parser.add_argument('--connect_eps', type=float, default=0.0, help='Contact distance (0 = 2 x thickness)')
    parser.add_argument('--connect_start', type=int, default=0, help='First batch with the connectivity loss')
    parser.add_argument('--n_snapshots', type=int, default=100,
                        help='Control-point snapshots saved over training (curves_history.npz, for the 4D viewer)')
    parser.add_argument('--snapshot_every', type=int, default=0, help='Snapshot every N batches instead (0 = use --n_snapshots)')
    parser.add_argument('--style', type=str, default='black_on_white', choices=['black_on_white', 'white_on_black'],
                        help='Stroke/background colors (targets are composited on the same background)')

    # CLIP-related weights
    parser.add_argument('--clip_weight', type=float, default=0.0,
                        help='Extra semantic CLIPLoss with augmentations (0 = CLIPasso loss balance: conv layers + 0.1 x fc)')
    parser.add_argument('--clip_conv_loss', type=float, default=1.0)
    parser.add_argument('--clip_fc_loss_weight', type=float, default=0.1)
    parser.add_argument('--clip_conv_layer_weights', type=float, nargs=5, default=[0, 0, 1.0, 1.0, 0.0])

    # GIF rendering
    parser.add_argument('--gif_fps', type=int, default=120)
    parser.add_argument('--rotation_time', type=float, default=6.0)
    parser.add_argument('--revolutions', type=float, default=3.0)
    parser.add_argument('--output_semigif', type=str, default='semihelical.gif')

    # Weights & Biases (optional)
    parser.add_argument('--wandb', action='store_true', help='Log to Weights & Biases')
    parser.add_argument('--wandb_project', type=str, default='clipasso3d')
    parser.add_argument('--wandb_entity', type=str, default=None)
    parser.add_argument('--wandb_mode', type=str, default='online', choices=['online', 'offline'],
                        help='offline: log to the run folder, upload later with `wandb sync`')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_pipeline(args)