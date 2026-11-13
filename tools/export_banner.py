"""
Banner for the website: the ground-truth object and the same view of sketches with more and more lines,
as SVG files (vector strokes) plus one PNG of the object.

    python tools/export_banner.py --object bicycle --turntable turntables/bicycle --frame 9 \\
        --runs runs/abl_L5_normal runs/abl_L20_normal runs/abl_L50_normal runs/abl_L200_normal \\
        --out ../CLIPasso3DWebsite/static/banner
Writes gt.png and sketch_1.svg ... sketch_K.svg (in the order of --runs).
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.overlay import load_curves  # noqa: E402


def project(points, c2w, fov, size):
    """World points (N,3) -> pixel coordinates (N,2) in a size x size image (OpenGL camera)."""
    w2c = np.linalg.inv(c2w)
    cam = points @ w2c[:3, :3].T + w2c[:3, 3]
    f = 0.5 * size / math.tan(0.5 * fov)
    z = -cam[:, 2]
    return np.stack([size / 2 + f * cam[:, 0] / z, size / 2 - f * cam[:, 1] / z], 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--object", required=True)
    p.add_argument("--turntable", required=True, help="Folder from tools/render_turntable.py")
    p.add_argument("--frame", type=int, default=9, help="Turntable frame (camera) to use")
    p.add_argument("--runs", nargs="+", required=True, help="Run folders without /objaverse/<object>")
    p.add_argument("--out", required=True)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--zoom", type=float, default=1.3)
    p.add_argument("--stroke", type=float, default=6.0, help="Stroke width in pixels of a --size image")
    args = p.parse_args()

    cams = json.load(open(os.path.join(args.turntable, "cameras.json")))
    c2w = np.array(cams["c2w"][args.frame])
    S = args.size
    crop = S / args.zoom
    x0 = (S - crop) / 2
    os.makedirs(args.out, exist_ok=True)

    gt = Image.open(os.path.join(args.turntable, f"frame_{args.frame:03d}.png")).convert("RGBA")
    w = gt.size[0]
    c = w / args.zoom
    gt = gt.crop(((w - c) / 2, (w - c) / 2, (w + c) / 2, (w + c) / 2)).resize((S, S), Image.LANCZOS)
    gt.save(os.path.join(args.out, "gt.png"))

    t = torch.linspace(0, 1, 64)
    for k, run in enumerate(args.runs, 1):
        rd = os.path.join(run, "objaverse", args.object)
        cfg = json.load(open(os.path.join(rd, "config.json")))
        curves = load_curves(rd, cfg)
        with torch.no_grad():
            pts = curves.points_at(t).numpy()                       # (C,64,3)
        paths = []
        for cpts in pts:
            uv = project(cpts, c2w, cams["fov"], S)
            d = "M" + " L".join(f"{u - x0:.1f},{v - x0:.1f}" for u, v in uv)
            paths.append(f'<path d="{d}"/>')
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {crop:.1f} {crop:.1f}">'
               f'<g fill="none" stroke="#111" stroke-width="{args.stroke / args.zoom:.2f}" stroke-linecap="round" '
               f'stroke-linejoin="round">{"".join(paths)}</g></svg>')
        path = os.path.join(args.out, f"sketch_{k}.svg")
        with open(path, "w") as fh:
            fh.write(svg)
        print(f"{path}: {len(pts)} curves from {rd}")


if __name__ == "__main__":
    main()
