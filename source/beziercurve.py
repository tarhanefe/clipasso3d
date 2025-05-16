import math
import torch
import torch.nn as nn

class BezierCurve(nn.Module):
    """
    Differentiable Bézier curve sampler producing Gaussian centers (means)
    and uniform thickness for each sample.
    """
    def __init__(
        self,
        init_control_points: torch.Tensor,
        thickness: float,
        overlap: float = 0.95,
        arc_samples: int = 200,
        device: str = 'cuda'
    ):
        super().__init__()
        # Control points as learnable parameters (N x 3)
        assert init_control_points.ndim == 2 and init_control_points.size(1) == 3
        self.P = nn.Parameter(init_control_points.to(device=device, dtype=torch.float32))
        self.thickness = float(thickness)
        self.overlap = float(overlap)
        self.arc_samples = int(arc_samples)
        self.device = device

        # Precompute dense parameter values t, and Bernstein coefficients
        t_dense = torch.linspace(0.0, 1.0, steps=self.arc_samples,
                                 device=device, dtype=torch.float32)
        self.register_buffer('t_dense', t_dense)

        N = self.P.size(0)
        binoms = torch.tensor([math.comb(N-1, k) for k in range(N)],
                              dtype=torch.float32, device=device)
        idxs = torch.arange(N, dtype=torch.float32, device=device)
        deg = torch.tensor(N-1, dtype=torch.float32, device=device)
        self.register_buffer('binoms', binoms)
        self.register_buffer('idxs', idxs)
        self.register_buffer('deg', deg)

    def _bezier_pts(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute Bézier points at parameters t (K x 1).
        Returns tensor of shape (K, 3).
        """
        # t: (K,1)
        t_pow = t ** self.idxs      # (K, N)
        one_pow = (1 - t) ** (self.deg - self.idxs)
        coefs = self.binoms * t_pow * one_pow  # (K, N)
        return coefs @ self.P                     # (K,3)

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample along the Bézier curve so that spheres of radius `thickness`
        overlap by `overlap` fraction.

        Returns:
            means: Tensor (M,3) of 3D sample points along curve.
            thicknesses: Tensor (M,) of uniform thickness values.
        """
        # 1. Sample dense points to approximate arc length
        pts = self._bezier_pts(self.t_dense.unsqueeze(1))   # (S,3)
        deltas = pts[1:] - pts[:-1]                         # (S-1,3)
        dists = deltas.norm(dim=1)                          # (S-1,)
        cumlen = torch.cat([torch.zeros(1, device=pts.device),
                            dists.cumsum(0)], dim=0)      # (S,)
        total_len = cumlen[-1].item()

        # 2. Determine number of samples M based on thickness and overlap
        diameter = 2.0 * self.thickness
        separation = diameter * (1.0 - self.overlap)
        M = max(1, int(math.ceil(total_len / separation)))

        # 3. Invert arc-length to get uniform spacing in parameter t
        s_des = torch.linspace(0.0, total_len, steps=M,
                               device=pts.device)
        idx = torch.bucketize(s_des, cumlen).clamp(1, self.arc_samples-1)
        c0, c1 = cumlen[idx-1], cumlen[idx]
        t0, t1 = self.t_dense[idx-1], self.t_dense[idx]
        t_samp = torch.lerp(t0, t1, (s_des - c0) / (c1 - c0))

        # 4. Compute final sample means and thickness vector
        means = self._bezier_pts(t_samp.unsqueeze(1))      # (M,3)
        thicknesses = torch.full((means.size(0),),
                                 self.thickness,
                                 dtype=means.dtype,
                                 device=means.device)
        return means, thicknesses
    

class CurveSet(nn.Module):
    def __init__(self, init_pts_list, thickness=0.02, overlap=0.8, arc_samples=300, device='cuda'):
        super().__init__()
        self.curves = nn.ModuleList([
            BezierCurve(pts.to(device), thickness, overlap, arc_samples, device=device)
            for pts in init_pts_list
        ])
        self.device = device

    def forward(self):
        means_list, th_list = [], []
        for curve in self.curves:
            m, t = curve()   # (M_i, 3), (M_i,)
            means_list.append(m)
            th_list.append(t)
        means = torch.cat(means_list, dim=0)       # (N,3)
        thicknesses = torch.cat(th_list, dim=0)    # (N,)
        return means, thicknesses

    def remove_curve(self, idx):
        """Remove the curve at index `idx` (0-based)."""
        if idx < 0 or idx >= len(self.curves):
            raise IndexError(f"No curve at index {idx}")
        del self.curves[idx]
