import math

import torch


def rasterize_spheres(
    means: torch.Tensor,
    radii_world: float | torch.Tensor,
    viewmat: torch.Tensor,
    K: torch.Tensor,
    width: int,
    height: int,
    near: float = 1e-3,
    n_sigma: float = 4.0,
) -> torch.Tensor:
    """
    Differentiable splatting of isotropic Gaussians: alpha = 1 - prod_n (1 - w_n), returned as (H,W,1).

    Camera convention is OpenGL/Blender (as in NeRF transforms.json): the camera looks along -Z
    with +Y up, so a point in front of the camera has z_cam < 0. Pixel (row i, col j) has its
    center at (j + 0.5, i + 0.5), row 0 is the top of the image. Spheres behind the camera (or
    closer than `near`) are not drawn. Each Gaussian is only evaluated in a window of +-n_sigma
    pixel-space standard deviations around its center (the tails beyond 4 sigma are below 3e-4),
    so memory and time scale with the number of spheres, not with spheres x pixels.
    """
    device, dtype = means.device, means.dtype
    N = means.shape[0]

    if isinstance(radii_world, (float, int)):
        R_world = torch.full((N,), float(radii_world), device=device, dtype=dtype)
    else:
        R_world = radii_world.to(device=device, dtype=dtype).expand(N) if radii_world.ndim == 0 \
            else radii_world.to(device=device, dtype=dtype)

    # -- camera space and projection --
    cam = means @ viewmat[:3, :3].T + viewmat[:3, 3]
    depth = -cam[:, 2]
    visible = depth > near
    depth = torch.where(visible, depth, torch.ones_like(depth))
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    r_px = fx * R_world / depth
    mu_x = cx + fx * cam[:, 0] / depth
    mu_y = cy - fy * cam[:, 1] / depth

    # -- window of pixels around each center --
    rad = int(math.ceil(n_sigma * r_px[visible].max().item())) if visible.any() else 1
    rad = max(1, min(rad, max(width, height)))
    offs = torch.arange(-rad, rad + 1, device=device)
    bx = torch.floor(mu_x).long().detach()
    by = torch.floor(mu_y).long().detach()
    px = bx[:, None, None] + offs[None, None, :]                    # (N,1,k)
    py = by[:, None, None] + offs[None, :, None]                    # (N,k,1)
    dx = (px.to(dtype) + 0.5) - mu_x[:, None, None]
    dy = (py.to(dtype) + 0.5) - mu_y[:, None, None]
    w = torch.exp(-0.5 * (dx * dx + dy * dy) / (r_px[:, None, None] ** 2 + 1e-8))
    w = w * visible.to(dtype)[:, None, None]

    px, py = torch.broadcast_tensors(px, py)
    valid = (px >= 0) & (px < width) & (py >= 0) & (py < height) & visible[:, None, None]
    log_t = torch.log(1 - w + 1e-10)                               # transmittance of each splat
    idx = (py * width + px)[valid]
    acc = torch.zeros(height * width, device=device, dtype=dtype).index_add(0, idx, log_t[valid])
    alpha = 1 - torch.exp(acc)
    return alpha.view(height, width, 1)
