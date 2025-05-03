import math
import torch
import torch.nn as nn
import json


def rasterize_spheres(
    means: torch.Tensor,
    radii_world: float | torch.Tensor,
    viewmat: torch.Tensor,
    K: torch.Tensor,
    width: int,
    height: int,
) -> torch.Tensor:
    """
    Differentiable rasterizer for spheres of given world-space radius,
    rendered as Gaussian splats whose pixel-size varies with depth.

    Args:
        means:       (N,3) tensor of world-space sphere centers.
        radii_world: float or (N,) tensor of world-space sphere radii.
        viewmat:     (4,4) world->camera matrix.
        K:           (3,3) intrinsics matrix.
        width:       image width in pixels.
        height:      image height in pixels.

    Returns:
        (H,W,1) float32 tensor in [0,1]: black Gaussian spheres on white.
    """
    device, dtype = means.device, means.dtype
    N = means.shape[0]

    # -- ensure radii_world is a (N,) tensor on the right device/dtype
    if isinstance(radii_world, (float, int)):
        R_world = torch.full((N,), float(radii_world), device=device, dtype=dtype)
    else:
        R_world = radii_world.to(device=device, dtype=dtype)
        if R_world.ndim == 0:
            R_world = R_world.expand(N)
        elif R_world.numel() != N:
            raise ValueError(f"radii_world must be length-1 or length-{N}, got {R_world.shape}")

    # -- 1. World → camera space --
    ones   = torch.ones((N,1), device=device, dtype=dtype)
    homog  = torch.cat([means, ones], dim=1)                    # (N,4)
    cam_h  = (viewmat @ homog.T).T                             # (N,4)
    cam_pts= cam_h[:, :3] / cam_h[:, 3:4]                       # (N,3)
    z_cam  = cam_pts[:, 2]                                      # (N,)

    # -- 2. Compute per-sphere pixel radii: r_px = f * R_world / Z_cam --
    f      = K[0, 0]                                           # focal length in px
    r_px   = f * R_world / z_cam                               # (N,)

    # -- 3. Project to image plane --
    proj    = (K @ cam_pts.T).T                                # (N,3)
    means2D = proj[:, :2] / proj[:, 2:3]                       # (N,2)

    # -- 4. Build pixel grid --
    ys = torch.arange(height, device=device, dtype=dtype)
    xs = torch.arange(width,  device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')              # (H,W)
    pix     = torch.stack([xx, yy], dim=-1).unsqueeze(2)        # (H,W,1,2)

    # -- 5. Build per-sphere inverse covariance --
    inv_var = 1.0 / (r_px**2)                                   # (N,)
    inv_cov = inv_var.view(1,1,N,1,1) * torch.eye(2, device=device, dtype=dtype).view(1,1,1,2,2)

    # -- 6. Mahalanobis distance --
    means2D_exp = means2D.view(1,1,N,2)                         # (1,1,N,2)
    diff        = pix - means2D_exp                             # (H,W,N,2)
    diff_u      = diff.unsqueeze(-1)                            # (H,W,N,2,1)
    m2          = (diff_u.transpose(-2,-1) @ inv_cov @ diff_u)  # (H,W,N,1,1)
    m2          = m2.squeeze(-1).squeeze(-1)                    # (H,W,N)

    # -- 7. Gaussian weights & composite alpha --
    w     = torch.exp(-0.5 * m2)                                # (H,W,N)
    alpha = 1 - torch.prod(1 - w + 1e-10, dim=2)                # (H,W)

    # -- 8. Black spheres on white bg --
    img   = alpha                                               # white minus alpha
    return img.unsqueeze(-1)                                    # (H,W,1)