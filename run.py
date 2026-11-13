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
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--inner_steps', type=int, default=30)
    parser.add_argument('--learning_rate', type=float, default=0.005)
    parser.add_argument('--eval_interval', type=int, default=0, help='Validate every N batches (0 = once per epoch)')
    parser.add_argument('--ckpt_every', type=int, default=20, help='Checkpoint every N batches')
    parser.add_argument('--frame_every', type=int, default=1, help='Save a render|target frame every N batches')
    parser.add_argument('--no_compile', action='store_true', help='Skip torch.compile (CPU smoke tests)')

    # Curve parameters
    parser.add_argument('--n_curves', type=int, default=25)
    parser.add_argument('--thickness', type=float, default=0.02)
    parser.add_argument('--radius', type=float, default=0.8)
    parser.add_argument('--length', type=float, default=0.02)
    parser.add_argument('--overlap', type=float, default=0.6)

    # CLIP-related weights
    parser.add_argument('--clip_weight', type=float, default=1.0)
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