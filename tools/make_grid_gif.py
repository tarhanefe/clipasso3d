"""
Animated grid of one object's ablation: rows = abstraction levels, columns = line counts, all
sketches rotating in sync with the ground-truth object (rendered by tools/render_turntable.py).

    python tools/make_grid_gif.py --object bicycle --turntable turntables/bicycle --out figures/bicycle_grid.gif
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from source.diffrasterizer import rasterize_spheres_sparse  # noqa: E402
from tools.overlay import load_curves  # noqa: E402

LEVELS = ["too_geometric", "middle_geometric", "normal", "middle_abstract", "too_abstract"]
LINES = [5, 10, 20, 50, 100, 200]


def font(size):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/System/Library/Fonts/Helvetica.ttc"):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def intrinsics(fov, res):
    f = 0.5 * res / math.tan(0.5 * fov)
    return torch.tensor([[f, 0, res / 2], [0, f, res / 2], [0, 0, 1]], dtype=torch.float32)


def zoom_crop(img, zoom):
    """Central 1/zoom of the image (same crop for sketches and ground truth)."""
    if zoom <= 1:
        return img
    w, h = img.size
    cw, ch = w / zoom, h / zoom
    return img.crop(((w - cw) / 2, (h - ch) / 2, (w + cw) / 2, (h + ch) / 2))


@torch.no_grad()
def render_orbit(run_dir, c2ws, K, res, cell, zoom=1.0):
    """Sketch of one run from every turntable camera, black on white, as (frames, cell, cell) uint8."""
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    curves = load_curves(run_dir, cfg)
    means, th = curves()
    out = []
    for c2w in c2ws:
        a = rasterize_spheres_sparse(means, th, torch.linalg.inv(c2w), K, res, res)[..., 0]
        img = zoom_crop(Image.fromarray(((1 - a.numpy()) * 255).astype(np.uint8)), zoom).resize((cell, cell), Image.LANCZOS)
        out.append(np.asarray(img))
    return np.stack(out), len(curves.curves)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--object", required=True, help="Dataset name under data/objaverse, e.g. bicycle")
    p.add_argument("--turntable", required=True, help="Folder from tools/render_turntable.py")
    p.add_argument("--runs_root", default="runs")
    p.add_argument("--prefix", default="abl", help="Run names are <prefix>_L<lines>_<level>")
    p.add_argument("--levels", nargs="+", default=LEVELS)
    p.add_argument("--lines", type=int, nargs="+", default=LINES)
    p.add_argument("--cell", type=int, default=150)
    p.add_argument("--render_res", type=int, default=224, help="Sketch render resolution (training resolution)")
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--zoom", type=float, default=1.3, help="Crop the central 1/zoom of every view (object larger)")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    cams = json.load(open(os.path.join(args.turntable, "cameras.json")))
    c2ws = [torch.tensor(c, dtype=torch.float32) for c in cams["c2w"]]
    K = intrinsics(cams["fov"], args.render_res)
    n_frames = len(c2ws)
    C = args.cell

    # layout: [ground truth] [row labels] [grid with column headers]
    gt_size = 2 * C
    pad, label_w, title_h, head_h = 16, 170, 46, 30
    grid_x = pad + gt_size + pad + label_w
    grid_y = title_h + head_h
    W = grid_x + len(args.lines) * C + pad
    H = grid_y + len(args.levels) * C + pad
    f_title, f_head, f_small = font(24), font(17), font(13)

    sketches, counts = {}, {}
    for level in args.levels:
        for n in args.lines:
            rd = os.path.join(args.runs_root, f"{args.prefix}_L{n}_{level}", "objaverse", args.object)
            if os.path.exists(os.path.join(rd, "curves.pt")) or os.path.exists(os.path.join(rd, "ckpt.pt")):
                sketches[level, n], counts[level, n] = render_orbit(rd, c2ws, K, args.render_res, C, args.zoom)
                print(f"rendered {level} L{n}", flush=True)
            else:
                print(f"missing {rd}", flush=True)

    base = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(base)
    name = args.object.replace("_", " ").split(" (")[0]
    d.text((pad, 10), f"{name}: number of lines x abstraction", fill="black", font=f_title)
    d.text((pad, grid_y + (len(args.levels) * C - gt_size) // 2 - 22), "ground truth", fill="black", font=f_head)
    for j, n in enumerate(args.lines):
        d.text((grid_x + j * C + C // 2, title_h + head_h // 2), f"{n} lines", fill="black", font=f_head, anchor="mm")
    for i, level in enumerate(args.levels):
        d.text((pad + gt_size + pad, grid_y + i * C + C // 2), level.replace("_", " "), fill="black",
               font=f_head, anchor="lm")
    for i in range(len(args.levels) + 1):
        d.line([(grid_x, grid_y + i * C), (grid_x + len(args.lines) * C, grid_y + i * C)], fill=(225, 225, 225))
    for j in range(len(args.lines) + 1):
        d.line([(grid_x + j * C, grid_y), (grid_x + j * C, grid_y + len(args.levels) * C)], fill=(225, 225, 225))

    frames = []
    for t in range(n_frames):
        im = base.copy()
        gt = Image.open(os.path.join(args.turntable, f"frame_{t:03d}.png")).convert("RGBA")
        bg = Image.new("RGBA", gt.size, (255, 255, 255, 255))
        gt = zoom_crop(Image.alpha_composite(bg, gt).convert("RGB"), args.zoom).resize((gt_size, gt_size), Image.LANCZOS)
        im.paste(gt, (pad, grid_y + (len(args.levels) * C - gt_size) // 2))
        dd = ImageDraw.Draw(im)
        for i, level in enumerate(args.levels):
            for j, n in enumerate(args.lines):
                x, y = grid_x + j * C, grid_y + i * C
                if (level, n) in sketches:
                    im.paste(Image.fromarray(sketches[level, n][t]).convert("RGB").crop((1, 1, C - 1, C - 1)), (x + 1, y + 1))
                    if counts[level, n] != n:      # some curves were pruned
                        dd.text((x + C - 4, y + C - 4), f"{counts[level, n]} kept", fill=(140, 140, 140),
                                font=f_small, anchor="rd")
                else:
                    dd.text((x + C // 2, y + C // 2), "training...", fill=(160, 160, 160), font=f_small, anchor="mm")
        frames.append(im)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    pal = [f.quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for f in frames]
    pal[0].save(args.out + ".tmp.gif", save_all=True, append_images=pal[1:], duration=int(1000 / args.fps), loop=0,
                optimize=True)
    os.replace(args.out + ".tmp.gif", args.out)
    print(f"saved {args.out}: {n_frames} frames, {W}x{H}, {len(sketches)}/{len(args.levels) * len(args.lines)} runs")


if __name__ == "__main__":
    main()
