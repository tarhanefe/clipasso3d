import torch
import random
import os 
from PIL import Image
import torchvision.transforms as T
from source.utils import load_scene

# --- Initialize curves as short lines at random locations on a sphere ---
def random_short_lines(center: torch.Tensor,
                       n_curves: int,
                       radius: float,
                       length: float,
                       device: str):
    """
    center: (3,) origin of the sphere
    n_curves: how many line‐curves to make
    radius:  how far from center to place each line
    length:  total length of each line segment
    """
    lines = []
    for _ in range(n_curves):
        # 1) Pick a random point ON the sphere
        dir_loc = torch.randn(3, device=device)
        dir_loc = dir_loc / dir_loc.norm()
        loc = center + radius * dir_loc       # (3,)

        # 2) Pick a random line direction orthogonal to dir_loc
        dir_line = torch.randn(3, device=device)
        dir_line = dir_line - (dir_line @ dir_loc) * dir_loc
        dir_line = dir_line / dir_line.norm()

        # 3) Build 4 control points along that line of total length `length`
        offsets = torch.tensor([-0.5, -0.1667, 0.1667, 0.5],
                               device=device) * length  # (4,)
        ctrl_pts = loc[None] + offsets[:, None] * dir_line[None]  # (4,3)

        lines.append(ctrl_pts)
    return lines

def initialize_saliency_curves(
    transforms_json: str,
    image_dir: str,
    width: int,
    height: int,
    saliency_model,
    num_points: int,
    min_distance: float,
    high_threshold: float,
    radius: float,
    length: float,
    n_curves: int,
    device: str = "cuda",
):
    """
    1) Load scene and compute its center.
    2) Load & resize all images.
    3) Compute DINO saliency maps and pick top 2D seeds per view.
    4) Project all 2D seeds onto a sphere at scene_center.
    5) Sample up to n_curves of those 3D seeds.
    6) Around each seed, build a short line of total length `length`.

    Returns:
        ctrl_pts: Tensor of shape (n_actual, 4, 3) of control points.
    """
    # 1. Load scene
    file_paths, c2w_all, K = load_scene(transforms_json, width, height, device)
    c2w_all = c2w_all.to(device)
    K = K.to(device)

    # compute scene_center by least‐squares ray intersection
    P = c2w_all[:, :3, 3]                        # (N,3)
    N_vec = -c2w_all[:, :3, 2]                  # (N,3)
    I = torch.eye(3, device=device)
    A = torch.zeros((3, 3), device=device)
    b = torch.zeros((3,), device=device)
    for i in range(P.shape[0]):
        n_i = N_vec[i].unsqueeze(1)              # (3,1)
        M_i = I - n_i @ n_i.T
        A += M_i
        b += M_i @ P[i]
    center = torch.linalg.solve(A, b)           # (3,) on device

    # 2. Load & resize images
    to_tensor = T.ToTensor()
    images = []
    for fp in file_paths:
        clean = fp.lstrip("./")
        rel, stem = os.path.split(clean)
        for ext in (".png", ".jpg", ".jpeg"):
            cand = os.path.join(image_dir, rel, stem + ext)
            if os.path.exists(cand):
                img = Image.open(cand).convert("RGB")
                break
        else:
            raise FileNotFoundError(f"No image for {stem}")
        t = to_tensor(img).unsqueeze(0).to(device)
        t = torch.nn.functional.interpolate(t, size=(height, width), mode="bilinear")
        images.append(t)

    # 3. Compute saliency maps & pick 2D seeds
    sal_maps = []
    seeds_2d = []
    with torch.no_grad():
        for img in images:
            sal = saliency_model.compute(img, output_size=(height, width))  # (heads,H,W)
            sal_map = sal.mean(0)                                           # (H,W)
            sal_maps.append(sal_map.cpu())
            flat = sal_map.flatten()
            valid = (flat >= high_threshold).nonzero(as_tuple=True)[0]
            sorted_idx = valid[torch.argsort(flat[valid], descending=True)].tolist()
            H, W = sal_map.shape
            sel = []
            for ind in sorted_idx:
                y, x = divmod(ind, W)
                if not any((x - x0)**2 + (y - y0)**2 < min_distance**2 for x0, y0 in sel):
                    sel.append((x, y))
                    if len(sel) >= num_points:
                        break
            seeds_2d.append(sel)

    # 4. Project all 2D seeds to sphere
    K_inv = torch.linalg.inv(K)
    all_seeds3d = []
    for vidx, pts in enumerate(seeds_2d):
        c2w = torch.linalg.inv(c2w_all[vidx])
        R = c2w[:3, :3]
        for (u, v) in pts:
            uv1 = torch.tensor([u, v, 1.0], device=device)
            d_cam = -(K_inv @ uv1)
            d_cam = d_cam / d_cam.norm()
            d_world = (R @ d_cam)
            d_world = d_world / d_world.norm()
            P3 = center + radius * d_world
            all_seeds3d.append(P3)

    if not all_seeds3d:
        raise ValueError("No salient 3D seeds found.")

    # 5. Sample up to n_curves seeds
    M = len(all_seeds3d)
    if M >= n_curves:
        chosen = random.sample(all_seeds3d, n_curves)
    else:
        chosen = all_seeds3d

    # 6. Build short line-curves around each seed
    lines = []
    for loc in chosen:
        # orthonormal direction
        dir_loc = (loc - center)
        dir_loc = dir_loc / dir_loc.norm()
        v = torch.randn(3, device=device)
        dir_line = v - (v @ dir_loc) * dir_loc
        dir_line = dir_line / dir_line.norm()
        # four control points
        offs = torch.tensor([-0.5, -0.1667, 0.1667, 0.5], device=device) * length
        ctrl = loc[None] + offs[:, None] * dir_line[None]  # (4,3)
        lines.append(ctrl)

    return torch.stack(lines, dim=0)  # (n_actual, 4, 3)