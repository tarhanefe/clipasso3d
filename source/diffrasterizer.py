import torch
import math
from einops import rearrange, repeat

def rasterize_spheres_batched(
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

    Args:
        means: (B,N,3) tensor of sphere centers in world space.
        radii_world: (B,N,) tensor of sphere radii in world
        viewmat: (B,4,4) camera-to-world matrix.
        K: (B,3,3) camera intrinsics matrix.
        width: image width.
        height: image height.
    Returns:
        alpha: (B,H,W,1) tensor of alpha values for each pixel.
    """
    device, dtype = means.device, means.dtype
    B, N, C = means.shape

    # -- Radii tensor setup --
    if isinstance(radii_world, (float, int)):
        R_world = torch.full((N,), float(radii_world), device=device, dtype=dtype)
    else:
        R_world = radii_world.to(device=device, dtype=dtype)
        if R_world.ndim == 0:
            R_world = R_world.expand((B,N))
        elif R_world.numel() != N*B:
            raise ValueError(f"radii_world must be length-1 or length-{N}, got {R_world.shape}")

    # -- 1. Transform to camera space --
    ones = torch.ones((B,N,1), device=device, dtype=dtype)
    homog = torch.cat([means, ones], dim=-1)
    # add a 1 dimension to viewmat to make it (B,1,4,4)
    viewmat = viewmat.unsqueeze(1)  # (B,1,4,4)

    cam_h = (viewmat @ homog.unsqueeze(-1)).squeeze(-1)  # (B,N,4)
    cam_pts = cam_h[:, :, :3] / cam_h[:, :, 3:4]
    z_cam = cam_pts[:, :, 2]

    # -- 2. Project to image plane --
    f = K[0, 0]
    K = repeat(K, "c1 c2 -> b 1 c1 c2", b=B)  # (1,3,3)
    r_px = f * R_world / z_cam  # pixel-space radii
    proj = (K @ cam_pts.unsqueeze(-1)).squeeze(-1)  # (B,N,3)
    means2D = proj[:, :, :2] / proj[:, :, 2:3]  # (B, N,2)

    # -- 3. Build pixel grid --
    ys = torch.arange(height, device=device, dtype=dtype)
    xs = torch.arange(width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')  # (H,W)
    pix_x = xx.unsqueeze(-1).unsqueeze(-1).unsqueeze(0)  # (1,H,W,1,1)
    pix_y = yy.unsqueeze(-1).unsqueeze(-1).unsqueeze(0)  # (1,H,W,1,1)

    mu_x = rearrange(means2D[:, :, 0], "b n -> b 1 1 n 1", b=B, n=N)  # (1,1,B,N,1)
    mu_y = rearrange(means2D[:, :, 1], "b n -> b 1 1 n 1", b=B, n=N)  # (1,1,B,N,1)

    dx = pix_x - mu_x  # (B,H,W,N,1)
    dy = pix_y - mu_y  # (B,H,W,N,1)

    inv_var = 1.0 / (r_px**2 + 1e-8)  # avoid divide-by-zero
    inv_var = rearrange(inv_var, "b n -> b 1 1 n 1", b=B, n=N)  # (B,1,1,N,1)

    m2 = (dx * dx * inv_var) + (dy * dy * inv_var)  # (B,H,W,N,1)
    m2 = m2.squeeze(-1)

    # -- 4. Composite Gaussian spheres into image --
    w = torch.exp(-0.5 * m2)
    alpha = 1 - torch.prod(1 - w + 1e-10, dim=-1)  # (H,W)

    return alpha.unsqueeze(-1)  # (B,H,W,1)

def rasterize_spheres(
    means: torch.Tensor,
    radii_world: float | torch.Tensor,
    viewmat: torch.Tensor,
    K: torch.Tensor,
    width: int,
    height: int,
) -> torch.Tensor:
    """
    Rasterize spheres into a single image using Gaussian splatting.
    Args:
        means: (N,3) tensor of sphere centers in world space.
        radii_world: (N,) tensor of sphere radii in world space.
        viewmat: (4,4) camera-to-world matrix.
        K: (3,3) camera intrinsics matrix.
        width: image width.
        height: image height.
    Returns:
        alpha: (H,W,1) tensor of alpha values for each pixel.
    """
    return rasterize_spheres_batched(
        means.unsqueeze(0),
        radii_world.unsqueeze(0),
        viewmat.unsqueeze(0),
        K.unsqueeze(0),
        width,
        height,
    ).squeeze(0)