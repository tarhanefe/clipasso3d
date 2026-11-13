"""
Check how the curves of a run line up with the object: strokes drawn in red over each view's own
target image, plus the fraction of stroke ink on the object mask per view.

    python tools/overlay.py runs/abl_L20_normal/objaverse/bicycle --split train --views 0 25 50 75
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from source.beziercurve import CurveSet
from source.diffrasterizer import rasterize_spheres_sparse
from source.imagesampler import ImageSampler
from source.metrics import silhouette_precision


def load_curves(run_dir, cfg):
    """Latest curves of a run: the final curves.pt, else the training checkpoint."""
    final = os.path.join(run_dir, "curves.pt")
    if os.path.exists(final):
        pts = torch.load(final, map_location="cpu", weights_only=False)["control_points"]
    else:
        state = torch.load(os.path.join(run_dir, "ckpt.pt"), map_location="cpu", weights_only=False)["curve_set"]
        n = len({k.split('.')[1] for k in state if k.startswith('curves.')})
        pts = [state[f"curves.{i}.P"] for i in range(n)]
    return CurveSet(pts, cfg["thickness"], cfg["overlap"], device="cpu",
                    samples_per_curve=cfg.get("samples_per_curve", 0), sample_spacing=cfg.get("sample_spacing", 0))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--split", default="train")
    p.add_argument("--views", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--out", default=None, help="Output image (default <run_dir>/overlay_<split>.png)")
    args = p.parse_args()

    cfg = json.load(open(os.path.join(args.run_dir, "config.json")))
    base = os.path.join(cfg["data_root"], cfg["data_name"])
    W, H = cfg["width"], cfg["height"]
    sampler = ImageSampler(os.path.join(base, f"transforms_{args.split}.json"), os.path.join(base, "rgb"),
                           W, H, "cpu", (255, 255, 255))
    curves = load_curves(args.run_dir, cfg)
    with torch.no_grad():
        means, th = curves()
    tiles, precs = [], []
    for v in args.views:
        rgb, K, w2c = sampler.sample(v)
        a = rasterize_spheres_sparse(means, th, w2c, K, W, H)[..., 0]
        precs.append(silhouette_precision(a[None, None], sampler.masks[v]).item())
        img = rgb[0].permute(1, 2, 0).numpy()
        red = np.array([0.9, 0.05, 0.05])
        over = img * (1 - a.numpy()[..., None]) + red * a.numpy()[..., None]
        tiles.append((np.clip(over, 0, 1) * 255).astype(np.uint8))
    out = args.out or os.path.join(args.run_dir, f"overlay_{args.split}.png")
    Image.fromarray(np.concatenate(tiles, axis=1)).save(out)
    print(f"{out}: {len(curves.curves)} curves; ink on object per view: "
          + " ".join(f"{v}:{q:.2f}" for v, q in zip(args.views, precs)))


if __name__ == "__main__":
    main()
