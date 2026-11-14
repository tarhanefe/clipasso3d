import torch
import torch.nn as nn

BINOM = (1.0, 3.0, 3.0, 1.0)


def bernstein(t: torch.Tensor) -> torch.Tensor:
    """Cubic Bernstein basis at parameters t (..., 1) -> (..., 4)."""
    k = torch.arange(4, device=t.device, dtype=t.dtype)
    return torch.tensor(BINOM, device=t.device, dtype=t.dtype) * t ** k * (1 - t) ** (3 - k)


class BezierCurve(nn.Module):
    """One cubic Bézier curve; its 4 control points (4,3) are the learnable parameters."""
    def __init__(self, control_points: torch.Tensor):
        super().__init__()
        assert control_points.shape == (4, 3)
        self.P = nn.Parameter(control_points.float())


class CurveSet(nn.Module):
    """
    3D cubic Bézier curves rendered as chains of spherical Gaussians of radius `thickness`.

    Every curve is sampled uniformly in arc length with as many Gaussians as its length needs
    for a spacing of at most `sample_spacing` x thickness (0.5 keeps strokes of any length
    continuous), all curves at once.
    """
    def __init__(self, control_points, thickness, sample_spacing=0.5, arc_samples=300, max_samples=4096,
                 device="cuda"):
        super().__init__()
        self.curves = nn.ModuleList([BezierCurve(p.to(device)) for p in control_points])
        self.thickness = float(thickness)
        self.sample_spacing = float(sample_spacing)
        self.max_samples = int(max_samples)
        self.register_buffer("t_dense", torch.linspace(0.0, 1.0, arc_samples, device=device), persistent=False)
        # persistent id of each curve (a respawned curve gets a new one), used by the training snapshots
        self.ids = list(range(len(self.curves)))

    def control_points(self):
        """(C,4,3) control points of all curves."""
        return torch.stack([c.P for c in self.curves])

    def points_at(self, t: torch.Tensor):
        """Points of every curve at parameters t (T,): (C,T,3), differentiable."""
        return torch.einsum("tn,cnk->ctk", bernstein(t.view(-1, 1)), self.control_points())

    def _arc_samples(self):
        """Points uniformly spaced in arc length on every curve: (points (C,S_max,3), valid mask (C,S_max))."""
        P = self.control_points()                                      # (C,4,3)
        C = P.size(0)
        pts = torch.einsum("dn,cnk->cdk", bernstein(self.t_dense.unsqueeze(1)), P)   # (C,D,3)
        seg = (pts[:, 1:] - pts[:, :-1]).norm(dim=-1)                  # (C,D-1)
        cumlen = torch.cat([seg.new_zeros(C, 1), seg.cumsum(1)], 1)    # (C,D)
        L = cumlen[:, -1]
        S_i = (torch.ceil(L.detach() / (self.sample_spacing * self.thickness)).long() + 1).clamp(2, self.max_samples)
        S = int(S_i.max())
        k = torch.arange(S, device=P.device)
        valid = k[None, :] < S_i[:, None]                              # (C,S)
        frac = torch.minimum(k[None, :], S_i[:, None] - 1).to(P.dtype) / (S_i[:, None] - 1).to(P.dtype)
        s_des = frac * L[:, None]                                      # (C,S)
        idx = torch.searchsorted(cumlen.detach().contiguous(), s_des.detach().contiguous())
        idx = idx.clamp(1, cumlen.size(1) - 1)
        c0, c1 = cumlen.gather(1, idx - 1), cumlen.gather(1, idx)
        t0, t1 = self.t_dense[idx - 1], self.t_dense[idx]
        t_samp = (t0 + (t1 - t0) * ((s_des - c0) / (c1 - c0).clamp_min(1e-12))).unsqueeze(-1)  # (C,S,1)
        return torch.einsum("csn,cnk->csk", bernstein(t_samp), P), valid

    def forward(self):
        """Gaussian centers (M,3) and radii (M,) of all curves."""
        pts, valid = self._arc_samples()
        means = pts[valid]
        return means, means.new_full((means.size(0),), self.thickness)

    def curve_samples(self):
        """The Gaussian centers of each curve, as a list of (S_i,3) tensors."""
        pts, valid = self._arc_samples()
        return [p[v] for p, v in zip(pts, valid)]
