import torch
import math
import numpy as np
from pathlib import Path
from PIL import Image as PILImage
import matplotlib.pyplot as plt
from types import SimpleNamespace
from source.cliploss import Loss
from source.beziercurve import CurveSet
from source.diffrasterizer import rasterize_spheres
from source.imagesampler import ImageSampler
from source.initalizer import random_short_lines
from source.trainer import Trainer


def run_pipeline(args):
    torch.random.manual_seed(0)
    torch.cuda.empty_cache()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)

    base = Path('data') / args.data_name
    train_sampler = ImageSampler(str(base / 'transforms_train.json'), str(base / 'rgb'), args.width, args.height, device)
    val_sampler   = ImageSampler(str(base / 'transforms_val.json'), str(base / 'rgb'), args.width, args.height, device)
    test_sampler  = ImageSampler(str(base / 'transforms_test.json'), str(base / 'rgb'), args.width, args.height, device)

    criterion = Loss(SimpleNamespace(
        device=device,
        percep_loss='none',
        train_with_clip=True,
        clip_weight=args.clip_weight,
        start_clip=0,
        clip_conv_loss=args.clip_conv_loss,
        clip_fc_loss_weight=args.clip_fc_loss_weight,
        clip_text_guide=0.0,
        num_aug_clip=4,
        augemntations=['affine'],
        include_target_in_aug=False,
        augment_both=False,
        clip_model_name='ViT-B/32',
        clip_conv_loss_type='L2',
        clip_conv_layer_weights=args.clip_conv_layer_weights
    )).to(device)

    rasterizer = torch.compile(rasterize_spheres)
    center_t = torch.tensor(train_sampler.scene_center, device=device, dtype=torch.float32)
    curves = random_short_lines(center_t, args.n_curves, args.radius, args.length, device)

    curve_set = CurveSet(curves, args.thickness, args.overlap, arc_samples=300, device=device).to(device)
    optimizer = torch.optim.Adam(curve_set.parameters(), lr=args.learning_rate)

    trainer = Trainer(train_sampler, val_sampler, test_sampler, curve_set, rasterizer, criterion, optimizer,
                      batch_size=args.batch_size, inner_steps=args.inner_steps, epochs=args.epochs,
                      width=args.width, height=args.height, save_dir=str(save_dir),
                      eval_interval=2, device=device, display_plots=True)

    means, thicknesses = trainer.train()

    torch.save({'means': means, 'thicknesses': thicknesses, 'scene_center': trainer.train_sampler.scene_center, 'K': trainer.train_sampler.K}, 'tensors.pt')

    render_semihelical_gif(means, args.thickness, torch.tensor(trainer.train_sampler.scene_center), 15,
                           trainer.train_sampler.K, args.width, args.height,
                           args.gif_fps, args.rotation_time, args.revolutions,
                           output_path=args.output_semigif)

    frame_paths = sorted(save_dir.glob("batch_*.png"))
    frames = [PILImage.open(p) for p in frame_paths]
    frames[0].save(args.output_gif, save_all=True, append_images=frames[1:], duration=50, loop=0)


def look_at(eye, center, up):
    z = (eye - center)
    z = z / z.norm(dim=0, keepdim=True)
    x = torch.cross(up, z)
    x = x / x.norm(dim=0, keepdim=True)
    y = torch.cross(z, x)
    R = torch.stack([x, y, z], dim=0)
    T = -R @ eye.view(3, 1)
    view = torch.eye(4, device=eye.device, dtype=eye.dtype)
    view[:3, :3] = R
    view[:3, 3] = T.squeeze()
    return view


def render_semihelical_gif(means, radii, scene_center, scene_radius, K, width, height, fps, rotation_time, revolutions, up=torch.tensor([0., 0, 1]), output_path="semihelical.gif"):
    device = means.device
    center = scene_center.to(device=device, dtype=means.dtype)
    up_vec = up.to(device=device, dtype=means.dtype)
    n_frames = int(round(fps * rotation_time))
    duration_ms = int(round(1000.0 / fps))
    frames = []
    for i in range(n_frames):
        t = i / (n_frames - 1)
        phi = t * (math.pi / 2)
        theta = t * (2 * math.pi * revolutions)
        x = center[0] + scene_radius * math.cos(phi) * math.cos(theta)
        y = center[1] + scene_radius * math.cos(phi) * math.sin(theta)
        z = center[2] + scene_radius * math.sin(phi)
        cam_pos = torch.tensor([x, y, z], device=device, dtype=means.dtype)
        V = look_at(cam_pos, center, up_vec)
        alpha = rasterize_spheres(means, radii, V, K, width, height)
        a_np = (alpha[..., 0].detach().cpu().numpy() * 255).astype(np.uint8)
        rgb = np.stack([a_np] * 3, axis=-1)
        frames.append(PILImage.fromarray(rgb))
    frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0)
    print(f"Saved {output_path}: {n_frames} frames, {fps} FPS, {revolutions} revolutions.")