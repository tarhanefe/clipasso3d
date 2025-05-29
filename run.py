# run.py
import argparse
from pathlib import Path

from source.pipeline import run_pipeline  # You will need to move the large code into a function in source/pipeline.py


def parse_args():
    parser = argparse.ArgumentParser(description="3D Sketch Abstraction Training Pipeline")

    # Dataset and I/O
    parser.add_argument('--data_name', type=str, default='rose', help='Dataset name')
    parser.add_argument('--width', type=int, default=224, help='Image width')
    parser.add_argument('--height', type=int, default=224, help='Image height')
    parser.add_argument('--save_dir', type=str, default='training_frames', help='Directory to save training outputs')
    parser.add_argument('--output_gif', type=str, default='training_evolution.gif', help='Output GIF path')

    # Optimization parameters
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--inner_steps', type=int, default=30)
    parser.add_argument('--learning_rate', type=float, default=0.005)

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

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_pipeline(args)