import torch
import math

def rasterize_spheres(
    means: torch.Tensor,
    radii_world: float | torch.Tensor,
    viewmat: torch.Tensor,
    K: torch.Tensor,
    width: int,
    height: int,
    near: float = 1e-3,
) -> torch.Tensor:
    """
    Efficient differentiable rasterizer for spheres using Gaussian splatting.
    Optimized for memory by removing intermediate large tensors.

    Camera convention is OpenGL/Blender (as in NeRF transforms.json): the camera looks
    along -Z with +Y up, so a point in front of the camera has z_cam < 0. Pixel (row i,
    col j) has its center at (j + 0.5, i + 0.5), row 0 is the top of the image.
    Spheres behind the camera (or closer than `near`) are not drawn.
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
    depth = -cam_pts[:, 2]                          # > 0 in front of the camera
    visible = depth > near
    depth = torch.where(visible, depth, torch.ones_like(depth))

    # -- 2. Project to image plane (OpenGL camera -> image with rows going down) --
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    r_px = fx * R_world / depth  # pixel-space radii
    mu_x = (cx + fx * cam_pts[:, 0] / depth).view(1, 1, -1)  # (1,1,N)
    mu_y = (cy - fy * cam_pts[:, 1] / depth).view(1, 1, -1)  # (1,1,N)

    # -- 3. Build pixel grid (pixel centers) --
    xs = torch.arange(width, device=device, dtype=dtype) + 0.5
    ys = torch.arange(height, device=device, dtype=dtype) + 0.5
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    pix_x = xx.unsqueeze(-1)  # (H,W,1)
    pix_y = yy.unsqueeze(-1)  # (H,W,1)

    dx = pix_x - mu_x  # (H,W,N)
    dy = pix_y - mu_y  # (H,W,N)

    inv_var = (1.0 / (r_px**2 + 1e-8)).view(1, 1, -1)  # avoid divide-by-zero
    m2 = (dx * dx + dy * dy) * inv_var  # (H,W,N)

    # -- 4. Composite Gaussian spheres into image --
    w = torch.exp(-0.5 * m2) * visible.to(dtype).view(1, 1, -1)
    alpha = 1 - torch.prod(1 - w + 1e-10, dim=2)  # (H,W)

    return alpha.unsqueeze(-1) # (H,W,1)
