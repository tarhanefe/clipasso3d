"""
Check that a dataset's camera poses agree with its images.

Every pixel of world_pos/{split}/r_i.exr stores the 3D point the pixel sees. Projecting
those points with the pose and intrinsics from transforms_{split}.json must land them
back on the same pixel (OpenGL camera, pixel centers at +0.5). Frames whose median
reprojection error is above --tol pixels are reported, and the script exits non-zero.

    python tools/check_poses.py data/rose data/duck
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import OpenEXR
from PIL import Image


def load_world_pos(path):
    f = OpenEXR.File(path)
    ch = f.channels()
    if "RGBA" in ch:
        return ch["RGBA"].pixels.astype(np.float64)
    if "RGB" in ch:
        p = ch["RGB"].pixels.astype(np.float64)
        return np.concatenate([p, np.ones(p.shape[:2] + (1,))], -1)
    raise ValueError(f"{path}: expected RGB(A) channels, got {list(ch)}")


def interior_mask(m):
    """Drop pixels next to the silhouette, whose positions are blended with the background."""
    e = m.copy()
    e[1:] &= m[:-1]; e[:-1] &= m[1:]
    e[:, 1:] &= m[:, :-1]; e[:, :-1] &= m[:, 1:]
    return e


def frame_error(c2w, pos, rgba, focal):
    h, w = pos.shape[:2]
    cover = pos[..., 3] > 0.999
    hit = np.abs(pos[..., :3]).sum(-1) > 0
    if rgba.shape[-1] == 4 and rgba[..., 3].min() < 255:
        fg = rgba[..., 3] > 250
    else:                                   # old datasets: opaque black background
        fg = rgba[..., :3].astype(int).sum(-1) > 30
    ii, jj = np.nonzero(interior_mask(interior_mask(cover & hit & fg)))
    if len(ii) == 0:
        return float("nan"), 0
    X = pos[ii, jj, :3]
    w2c = np.linalg.inv(c2w)
    cam = X @ w2c[:3, :3].T + w2c[:3, 3]
    front = cam[:, 2] < 0
    if front.mean() < 0.5:                  # most of the surface behind the camera: pose is certainly wrong
        return float("inf"), len(ii)
    cam, ii, jj = cam[front], ii[front], jj[front]
    u = w / 2 + focal * cam[:, 0] / -cam[:, 2]
    v = h / 2 - focal * cam[:, 1] / -cam[:, 2]
    return float(np.median(np.hypot(u - (jj + 0.5), v - (ii + 0.5)))), len(ii)


def check_dataset(root, tol):
    bad = []
    for split in ("train", "val", "test"):
        jpath = os.path.join(root, f"transforms_{split}.json")
        if not os.path.exists(jpath):
            continue
        meta = json.load(open(jpath))
        errs = []
        for fr in meta["frames"]:
            rel = fr["file_path"].lstrip("./")
            sub, stem = os.path.split(rel)
            pos = load_world_pos(os.path.join(root, "world_pos", sub, stem + ".exr"))
            rgba = np.array(Image.open(os.path.join(root, "rgb", sub, stem + ".png")).convert("RGBA"))
            if rgba.shape[:2] != pos.shape[:2]:
                raise ValueError(f"{rel}: image {rgba.shape[:2]} vs world_pos {pos.shape[:2]}")
            focal = 0.5 * pos.shape[1] / math.tan(0.5 * meta["camera_angle_x"])
            e, n = frame_error(np.array(fr["transform_matrix"], dtype=np.float64), pos, rgba, focal)
            errs.append(e)
            if not e <= tol:
                bad.append((split, rel, e))
        errs = np.array(errs)
        print(f"{root} {split:5s} frames={len(errs):3d}  median_err={np.nanmedian(errs):7.3f}px  "
              f"bad(>{tol}px)={int((~(errs <= tol)).sum())}")
    for split, rel, e in bad[:20]:
        print(f"    BAD {rel}: {e:.2f}px")
    if len(bad) > 20:
        print(f"    ... {len(bad) - 20} more")
    return bad


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("datasets", nargs="+", help="Dataset folders (containing transforms_*.json)")
    p.add_argument("--tol", type=float, default=1.0, help="Max median reprojection error in pixels")
    args = p.parse_args()
    n_bad = sum(len(check_dataset(d.rstrip("/"), args.tol)) for d in args.datasets)
    sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()
