import torch
import math

def rasterize_spheres(
    means: torch.Tensor,
    radii_world: float | torch.Tensor,
    viewmat: torch.Tensor,
    K: torch.Tensor,
    width: int,
    height: int,
) -> torch.Tensor:
    """
    Efficient differentiable rasterizer for spheres using Gaussian splatting.
    Optimized for memory by removing intermediate large tensors.
    """
    device, dtype = means.device, means.dtype
    N = means.shape[0]

    # -- Radii tensor setup --
    if isinstance(radii_world, (float, int)):
        R_world = torch.full((N,), float(radii_world), device=device, dtype=dtype)
    else:
        R_world = radii_world.to(device=device, dtype=dtype)
        if R_world.ndim == 0:
            R_world = R_world.expand(N)
        elif R_world.numel() != N:
            raise ValueError(f"radii_world must be length-1 or length-{N}, got {R_world.shape}")

    # -- 1. Transform to camera space --
    ones = torch.ones((N,1), device=device, dtype=dtype)
    homog = torch.cat([means, ones], dim=1)
    cam_h = (viewmat @ homog.T).T
    cam_pts = cam_h[:, :3] / cam_h[:, 3:4]
    z_cam = cam_pts[:, 2]

    # -- 2. Project to image plane --
    f = K[0, 0]
    r_px = f * R_world / z_cam  # pixel-space radii
    proj = (K @ cam_pts.T).T
    means2D = proj[:, :2] / proj[:, 2:3]  # (N,2)

    # -- 3. Build pixel grid --
    ys = torch.arange(height, device=device, dtype=dtype)
    xs = torch.arange(width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')  # (H,W)
    pix_x = xx.unsqueeze(-1).unsqueeze(-1)  # (H,W,1,1)
    pix_y = yy.unsqueeze(-1).unsqueeze(-1)  # (H,W,1,1)

    mu_x = means2D[:, 0].view(1, 1, -1, 1)  # (1,1,N,1)
    mu_y = means2D[:, 1].view(1, 1, -1, 1)  # (1,1,N,1)

    dx = pix_x - mu_x  # (H,W,N,1)
    dy = pix_y - mu_y  # (H,W,N,1)

    inv_var = 1.0 / (r_px**2 + 1e-8)  # avoid divide-by-zero
    inv_cov_xx = inv_var.view(1,1,-1,1)
    inv_cov_yy = inv_var.view(1,1,-1,1)

    m2 = (dx * dx * inv_cov_xx) + (dy * dy * inv_cov_yy)  # (H,W,N,1)
    m2 = m2.squeeze(-1)

    # -- 4. Composite Gaussian spheres into image --
    w = torch.exp(-0.5 * m2)
    alpha = 1 - torch.prod(1 - w + 1e-10, dim=2)  # (H,W)

    return alpha.unsqueeze(-1)  # (H,W,1)