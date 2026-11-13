import json
import os
import random
import subprocess
import uuid
import torch
import math
import numpy as np
from pathlib import Path
from PIL import Image as PILImage
from types import SimpleNamespace
from source.cliploss import Loss
from source.beziercurve import CurveSet
from source.diffrasterizer import rasterize_spheres, rasterize_spheres_sparse
from source.imagesampler import ImageSampler
from source.initalizer import random_short_lines, hull_short_lines
from source.metrics import evaluate_views, chamfer_to_surface
from source.trainer import Trainer
from source.connectivity import ConnectivityLoss, connectivity_stats, weld


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def write_json(path, obj):
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def limit_views(sampler, n):
    """Keep only the first `n` views (quick smoke tests)."""
    if n and n < len(sampler):
        sampler.images, sampler.masks = sampler.images[:n], sampler.masks[:n]
        sampler.c2w_all, sampler.w2c_all = sampler.c2w_all[:n], sampler.w2c_all[:n]
    return sampler


def init_wandb(args, out_dir):
    """Start or resume (after a pod restart) the wandb run that belongs to `out_dir`."""
    import wandb

    id_path = out_dir / "wandb_id.txt"
    if id_path.exists():
        run_id = id_path.read_text().strip()
    else:
        run_id = uuid.uuid4().hex[:8]
        id_path.write_text(run_id)
    return wandb.init(
        project=args.wandb_project, entity=args.wandb_entity, mode=args.wandb_mode,
        id=run_id, resume="allow", name=f"{args.data_name}/{args.run_name}",
        group=args.run_name, config=vars(args), dir=str(out_dir),
    )


def run_pipeline(args):
    out_dir = Path(args.out_dir or Path("runs") / args.data_name / args.run_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "DONE").exists():
        print(f"{out_dir} is already finished, skipping")
        return
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(exist_ok=True)

    seed_everything(args.seed)
    torch.cuda.empty_cache()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    write_json(out_dir / "config.json", {**vars(args), "git_commit": git_commit(), "device": device})

    # black_on_white: strokes are drawn dark on white and the targets are composited on white
    invert = args.style == 'black_on_white'
    bg = (255, 255, 255) if invert else (0, 0, 0)
    base = Path(args.data_root) / args.data_name
    train_sampler = ImageSampler(str(base / 'transforms_train.json'), str(base / 'rgb'), args.width, args.height, device, bg)
    val_sampler   = ImageSampler(str(base / 'transforms_val.json'), str(base / 'rgb'), args.width, args.height, device, bg)
    test_sampler  = ImageSampler(str(base / 'transforms_test.json'), str(base / 'rgb'), args.width, args.height, device, bg)
    for s in (train_sampler, val_sampler, test_sampler):
        limit_views(s, args.limit_views)

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
        clip_conv_layer_weights=args.clip_conv_layer_weights,
        aug_fill=1.0 if invert else 0.0,
    )).to(device)

    if args.rasterizer == 'sparse':      # windowed splatting: scales to thousands of curves
        rasterizer = rasterize_spheres_sparse
    else:
        rasterizer = rasterize_spheres if args.no_compile else torch.compile(rasterize_spheres)
    center_t = torch.tensor(train_sampler.scene_center, device=device, dtype=torch.float32)
    if args.init == 'hull':
        curves = hull_short_lines(train_sampler, args.n_curves, args.length, args.hull_grid, device)
    else:
        curves = random_short_lines(center_t, args.n_curves, args.radius, args.length, device)

    curve_set = CurveSet(curves, args.thickness, args.overlap, arc_samples=300, device=device,
                         samples_per_curve=args.samples_per_curve, sample_spacing=args.sample_spacing).to(device)
    optimizer = torch.optim.Adam(curve_set.parameters(), lr=args.learning_rate)

    contact_eps = args.connect_eps or 2 * args.thickness
    connect_loss = ConnectivityLoss(args.connect_weight, contact_eps) if args.connect_weight > 0 else None
    wandb_run = init_wandb(args, out_dir) if args.wandb else None
    trainer = Trainer(train_sampler, val_sampler, test_sampler, curve_set, rasterizer, criterion, optimizer,
                      batch_size=args.batch_size, inner_steps=args.inner_steps, epochs=args.epochs,
                      width=args.width, height=args.height, save_dir=str(frames_dir),
                      eval_interval=args.eval_interval or math.ceil(len(train_sampler) / args.batch_size),
                      device=device, display_plots=False,
                      ckpt_path=str(out_dir / "ckpt.pt"), ckpt_every=args.ckpt_every,
                      frame_every=args.frame_every, log_path=str(out_dir / "log.jsonl"),
                      wandb_run=wandb_run, invert=invert,
                      prune_every=args.prune_every, prune_start=args.prune_start,
                      prune_min_on_frac=args.prune_min_on_frac, prune_min_length=args.prune_min_length,
                      n_snapshots=args.n_snapshots, connect_loss=connect_loss, connect_start=args.connect_start,
                      early_stop_patience=args.early_stop_patience, early_stop_tol=args.early_stop_tol,
                      min_batches=args.min_batches, snapshot_every=args.snapshot_every,
                      prune_mode=args.prune_mode, respawn_length=args.length, prune_stop=args.prune_stop)

    means, thicknesses = trainer.train()
    weld_info = {}
    if connect_loss is not None:
        # guarantee one buildable piece: translate leftover separate clusters onto the main one
        n_moved, dist = weld(curve_set, contact_eps)
        weld_info = {"weld_clusters_moved": n_moved, "weld_total_shift": dist}
        print(f"Weld: moved {n_moved} clusters by {dist:.4f} in total")
        trainer.snapshot(trainer.total_batches + 1)
        with torch.no_grad():
            means, thicknesses = curve_set()
    trainer.save_snapshots(str(out_dir / "curves_history.npz"))

    # -- final curves --
    torch.save({
        'control_points': [c.P.detach().cpu() for c in curve_set.curves],
        'curve_ids': list(curve_set.ids),
        'means': means.cpu(), 'thicknesses': thicknesses.cpu(),
        'scene_center': train_sampler.scene_center, 'K': train_sampler.K.cpu(),
    }, out_dir / 'curves.pt.tmp')
    os.replace(out_dir / 'curves.pt.tmp', out_dir / 'curves.pt')

    # -- metrics on the held-out test views (and on the true surface when the dataset has it) --
    clip_model = criterion.loss_mapper["clip"].model
    metrics = {"test_loss": trainer.test_loss, "n_curves_init": args.n_curves,
               "batches_trained": trainer.stopped_at or trainer.total_batches, "best_val_loss": trainer.best_val if trainer.best_val != float("inf") else None,
               "n_curves_final": len(curve_set.curves),
               **evaluate_views(curve_set, rasterizer, test_sampler, clip_model, args.width, args.height, invert)}
    metrics.update(connectivity_stats(curve_set, contact_eps))
    metrics.update(weld_info)
    surface = base / "surface_points.npy"
    if surface.exists():
        metrics.update(chamfer_to_surface(means, torch.from_numpy(np.load(surface))))
    write_json(out_dir / "metrics.json", metrics)
    print("Metrics:", json.dumps(metrics))

    # -- GIFs --
    # orbit at the mean training-camera distance, so normalized (Objaverse) and old scenes both fit
    cam_pos = trainer.train_sampler.c2w_all[:, :3, 3].cpu()
    scene_center = torch.tensor(trainer.train_sampler.scene_center, dtype=cam_pos.dtype)
    orbit_radius = (cam_pos - scene_center).norm(dim=1).mean().item()
    semigif = out_dir / args.output_semigif
    render_semihelical_gif(means, args.thickness, scene_center, orbit_radius,
                           trainer.train_sampler.K, args.width, args.height,
                           args.gif_fps, args.rotation_time, args.revolutions,
                           output_path=str(semigif), invert=invert,
                           rasterize_fn=rasterize_spheres_sparse if args.rasterizer == 'sparse' else rasterize_spheres)

    train_gif = out_dir / args.output_gif
    frame_paths = sorted(frames_dir.glob("batch_*.png"))
    if frame_paths:
        frames = [PILImage.open(p) for p in frame_paths]
        frames[0].save(train_gif, save_all=True, append_images=frames[1:], duration=50, loop=0)

    if wandb_run is not None:
        import wandb
        wandb_run.summary.update(metrics)
        media = {"semihelical": wandb.Video(str(semigif), format="gif")}
        if train_gif.exists():
            media["training_evolution"] = wandb.Video(str(train_gif), format="gif")
        wandb_run.log(media)
        wandb_run.finish()

    (out_dir / "DONE").touch()


def look_at(eye, center, up):
    z = (eye - center)
    z = z / z.norm(dim=0, keepdim=True)
    x = torch.cross(up, z, dim=0)
    if x.norm() < 1e-6:            # looking straight along `up` (top of the semihelix)
        x = torch.cross(torch.tensor([0., 1., 0.], device=eye.device, dtype=eye.dtype), z, dim=0)
    x = x / x.norm(dim=0, keepdim=True)
    y = torch.cross(z, x, dim=0)
    R = torch.stack([x, y, z], dim=0)
    T = -R @ eye.view(3, 1)
    view = torch.eye(4, device=eye.device, dtype=eye.dtype)
    view[:3, :3] = R
    view[:3, 3] = T.squeeze()
    return view


def render_semihelical_gif(means, radii, scene_center, scene_radius, K, width, height, fps, rotation_time, revolutions, up=torch.tensor([0., 0, 1]), output_path="semihelical.gif", invert=False, rasterize_fn=rasterize_spheres):
    device = means.device
    center = scene_center.to(device=device, dtype=means.dtype)
    up_vec = up.to(device=device, dtype=means.dtype)
    K = K.to(device)
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
        with torch.no_grad():
            alpha = rasterize_fn(means, radii, V, K, width, height)
        a = alpha[..., 0].detach().cpu().numpy()
        a_np = ((1 - a if invert else a) * 255).astype(np.uint8)
        rgb = np.stack([a_np] * 3, axis=-1)
        frames.append(PILImage.fromarray(rgb))
    frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0)
    print(f"Saved {output_path}: {n_frames} frames, {fps} FPS, {revolutions} revolutions.")
