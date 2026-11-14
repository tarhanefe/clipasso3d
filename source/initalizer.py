import torch


# --- Initialize curves on the visual hull carved from the training silhouettes ---
@torch.no_grad()
def visual_hull(sampler, grid: int = 64, min_frac: float = 0.95):
    """
    Carve a voxel grid with the object masks of all views in `sampler`.

    The grid covers the cube that the cameras see around the scene center. A voxel is kept
    if it projects inside the mask in at least `min_frac` of the views that see it.
    Returns (occupied voxel centers (M,3), surface voxel centers (S,3)).
    """
    device = sampler.K.device
    K = sampler.K
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    masks = torch.cat(sampler.masks)[:, 0]                              # (V,H,W)
    V, H, W = masks.shape
    center = torch.as_tensor(sampler.scene_center, device=device, dtype=torch.float32)
    dist = (sampler.c2w_all[:, :3, 3] - center).norm(dim=1).mean()
    extent = dist * (W / 2) / fx                                         # half-width visible at the center

    lin = torch.linspace(-1, 1, grid, device=device) * extent
    X = torch.stack(torch.meshgrid(lin, lin, lin, indexing='ij'), -1).reshape(-1, 3) + center
    seen = torch.zeros(len(X), device=device)
    inside = torch.zeros(len(X), device=device)
    for v in range(V):
        w2c = sampler.w2c_all[v]
        cam = X @ w2c[:3, :3].T + w2c[:3, 3]
        depth = -cam[:, 2]
        u = cx + fx * cam[:, 0] / depth.clamp_min(1e-6)
        r = cy - fy * cam[:, 1] / depth.clamp_min(1e-6)
        ok = (depth > 0) & (u >= 0) & (u < W) & (r >= 0) & (r < H)
        ui, ri = u.long().clamp(0, W - 1), r.long().clamp(0, H - 1)
        seen += ok.float()
        inside += (ok & (masks[v, ri, ui] > 0.5)).float()
    occ = (seen > 0) & (inside >= min_frac * seen)
    occ3 = occ.reshape(grid, grid, grid)

    # surface voxels: occupied with at least one empty 6-neighbour
    pad = torch.nn.functional.pad(occ3.float()[None, None], (1, 1, 1, 1, 1, 1))[0, 0] > 0
    nb_empty = torch.zeros_like(occ3)
    for d in range(3):
        for s in (-1, 1):
            nb_empty |= ~pad.roll(s, d)[1:-1, 1:-1, 1:-1]
    surf = occ3 & nb_empty
    return X[occ], X[surf.reshape(-1)]


def farthest_points(pts: torch.Tensor, n: int):
    """Greedy farthest point sampling: `n` well-spread points of `pts` (random first point)."""
    n = min(n, len(pts))
    idx = torch.empty(n, dtype=torch.long, device=pts.device)
    idx[0] = torch.randint(len(pts), (1,), device=pts.device)
    d = (pts - pts[idx[0]]).norm(dim=1)
    for i in range(1, n):
        idx[i] = d.argmax()
        d = torch.minimum(d, (pts - pts[idx[i]]).norm(dim=1))
    return pts[idx]


def hull_short_lines(sampler, n_curves: int, length: float, grid: int = 64, device: str = "cuda"):
    """
    Short line-curves spread evenly (farthest point sampling) over the visual hull surface.
    Each line lies in the plane orthogonal to the direction from the scene center, i.e. roughly
    tangent to the object.
    """
    center = torch.as_tensor(sampler.scene_center, device=device, dtype=torch.float32)
    occupied, surface = visual_hull(sampler, grid)
    pts = surface if len(surface) >= n_curves else occupied
    if len(pts) == 0:
        raise RuntimeError("The visual hull is empty: check the object masks (alpha channel) of the training views")
    print(f"Visual hull: {len(occupied)} voxels, {len(surface)} on the surface")
    return [line_at(loc, center, length) for loc in farthest_points(pts.to(device), n_curves)]


def line_at(loc: torch.Tensor, center: torch.Tensor, length: float):
    """4 control points of a short straight line through `loc`, orthogonal to the direction from `center`."""
    dir_loc = loc - center
    dir_loc = dir_loc / dir_loc.norm().clamp_min(1e-8)
    v = torch.randn(3, device=loc.device)
    dir_line = v - (v @ dir_loc) * dir_loc
    dir_line = dir_line / dir_line.norm().clamp_min(1e-8)
    offs = torch.tensor([-0.5, -0.1667, 0.1667, 0.5], device=loc.device) * length
    return loc[None] + offs[:, None] * dir_line[None]
