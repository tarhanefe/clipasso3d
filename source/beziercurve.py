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
    """
    A set of Bézier curves rendered as chains of spheres.

    samples_per_curve = 0: each curve picks its own number of spheres from its length,
        thickness and overlap (original behaviour; the total changes every step).
    samples_per_curve = S > 0: every curve gets exactly S spheres spaced uniformly in arc
        length, computed for all curves at once (fixed shape, much faster with torch.compile).
        Long curves get sparse and look dotted when S * spacing is too small.
    sample_spacing = k > 0 (takes precedence): every curve gets as many spheres as its length
        needs for a spacing of at most k * thickness (the Gaussian sigma), computed for all
        curves at once. k = 0.5 gives continuous strokes of any length.
    """
    def __init__(self, init_pts_list, thickness=0.02, overlap=0.8, arc_samples=300, device='cuda',
                 samples_per_curve: int = 0, sample_spacing: float = 0.0, max_samples: int = 4096):
        super().__init__()
        self.curves = nn.ModuleList([
            BezierCurve(pts.to(device), thickness, overlap, arc_samples, device=device)
            for pts in init_pts_list
        ])
        self.device = device
        self.thickness = float(thickness)
        self.samples_per_curve = int(samples_per_curve)
        self.sample_spacing = float(sample_spacing)
        self.max_samples = int(max_samples)
        # persistent id of each curve (kept through pruning, used by training snapshots)
        self.ids = list(range(len(self.curves)))

    def control_points(self):
        """(C,N,3) control points of all curves."""
        return torch.stack([c.P for c in self.curves])

    def points_at(self, t: torch.Tensor):
        """Points of every curve at parameters t (T,): (C,T,3), differentiable."""
        ref = self.curves[0]
        t = t.view(-1, 1)
        coefs = ref.binoms * t ** ref.idxs * (1 - t) ** (ref.deg - ref.idxs)   # (T,N)
        return torch.einsum('tn,cnk->ctk', coefs, self.control_points())

    def _arc_samples(self, n_per_curve=None):
        """
        Points uniformly spaced in arc length on every curve, all curves at once.
        n_per_curve: fixed count, or None for the adaptive count from `sample_spacing`.
        Returns (points (C,S_max,3), valid mask (C,S_max)).
        """
        P = torch.stack([c.P for c in self.curves])                    # (C,N,3)
        ref = self.curves[0]
        C = P.size(0)
        t = ref.t_dense.unsqueeze(1)                                   # (D,1)
        coefs = ref.binoms * t ** ref.idxs * (1 - t) ** (ref.deg - ref.idxs)   # (D,N)
        pts = torch.einsum('dn,cnk->cdk', coefs, P)                    # (C,D,3)
        seg = (pts[:, 1:] - pts[:, :-1]).norm(dim=-1)                  # (C,D-1)
        cumlen = torch.cat([seg.new_zeros(C, 1), seg.cumsum(1)], 1)    # (C,D)
        L = cumlen[:, -1]
        if n_per_curve is None:
            sep = self.sample_spacing * self.thickness
            S_i = (torch.ceil(L.detach() / sep).long() + 1).clamp(2, self.max_samples)
        else:
            S_i = torch.full((C,), int(n_per_curve), device=P.device, dtype=torch.long)
        S = int(S_i.max())
        k = torch.arange(S, device=P.device)
        valid = k[None, :] < S_i[:, None]                              # (C,S)
        frac = torch.minimum(k[None, :], S_i[:, None] - 1).to(P.dtype) / (S_i[:, None] - 1).to(P.dtype)
        s_des = frac * L[:, None]                                      # (C,S)
        idx = torch.searchsorted(cumlen.detach().contiguous(), s_des.detach().contiguous())
        idx = idx.clamp(1, cumlen.size(1) - 1)
        c0, c1 = cumlen.gather(1, idx - 1), cumlen.gather(1, idx)
        t0, t1 = ref.t_dense[idx - 1], ref.t_dense[idx]
        t_samp = (t0 + (t1 - t0) * ((s_des - c0) / (c1 - c0).clamp_min(1e-12))).unsqueeze(-1)  # (C,S,1)
        coefs = ref.binoms * t_samp ** ref.idxs * (1 - t_samp) ** (ref.deg - ref.idxs)          # (C,S,N)
        return torch.einsum('csn,cnk->csk', coefs, P), valid

    def _forward_adaptive(self):
        pts, valid = self._arc_samples()
        means = pts[valid]                                             # (M,3)
        return means, means.new_full((means.size(0),), self.thickness)

    def _forward_fixed(self):
        P = torch.stack([c.P for c in self.curves])                    # (C,N,3)
        ref = self.curves[0]
        S, C = self.samples_per_curve, P.size(0)
        # dense points to approximate arc length, all curves at once
        t = ref.t_dense.unsqueeze(1)                                   # (D,1)
        coefs = ref.binoms * t ** ref.idxs * (1 - t) ** (ref.deg - ref.idxs)   # (D,N)
        pts = torch.einsum('dn,cnk->cdk', coefs, P)                    # (C,D,3)
        seg = (pts[:, 1:] - pts[:, :-1]).norm(dim=-1)                  # (C,D-1)
        cumlen = torch.cat([seg.new_zeros(C, 1), seg.cumsum(1)], 1)    # (C,D)
        # invert arc length: S targets uniformly spaced along each curve
        s_des = torch.linspace(0, 1, S, device=P.device).unsqueeze(0) * cumlen[:, -1:]   # (C,S)
        idx = torch.searchsorted(cumlen.detach().contiguous(), s_des.detach().contiguous())
        idx = idx.clamp(1, cumlen.size(1) - 1)
        c0, c1 = cumlen.gather(1, idx - 1), cumlen.gather(1, idx)
        t0, t1 = ref.t_dense[idx - 1], ref.t_dense[idx]
        t_samp = (t0 + (t1 - t0) * ((s_des - c0) / (c1 - c0).clamp_min(1e-12))).unsqueeze(-1)  # (C,S,1)
        coefs = ref.binoms * t_samp ** ref.idxs * (1 - t_samp) ** (ref.deg - ref.idxs)          # (C,S,N)
        means = torch.einsum('csn,cnk->csk', coefs, P).reshape(-1, 3)  # (C*S,3)
        thicknesses = means.new_full((means.size(0),), self.thickness)
        return means, thicknesses

    def forward(self):
        if self.sample_spacing > 0:
            return self._forward_adaptive()
        if self.samples_per_curve > 0:
            return self._forward_fixed()
        means_list, th_list = [], []
        for curve in self.curves:
            m, t = curve()   # (M_i, 3), (M_i,)
            means_list.append(m)
            th_list.append(t)
        means = torch.cat(means_list, dim=0)       # (N,3)
        thicknesses = torch.cat(th_list, dim=0)    # (N,)
        return means, thicknesses

    def curve_samples(self):
        """Sample points of each curve as a list of (S_i,3) tensors (what forward() renders)."""
        if self.sample_spacing > 0:
            pts, valid = self._arc_samples()
            return [p[v] for p, v in zip(pts, valid)]
        if self.samples_per_curve > 0:
            means, _ = self._forward_fixed()
            return list(means.view(len(self.curves), self.samples_per_curve, 3))
        return [curve()[0] for curve in self.curves]

    def keep_curves(self, keep):
        """Keep only the curves whose entry in `keep` (list of bools) is True."""
        self.curves = nn.ModuleList([c for c, k in zip(self.curves, keep) if k])
        self.ids = [i for i, k in zip(self.ids, keep) if k]

    def load_state_dict(self, state_dict, strict=True, assign=False):
        # a checkpoint taken after pruning has fewer curves than the freshly initialized set
        n = len({k.split('.')[1] for k in state_dict if k.startswith('curves.')})
        if n < len(self.curves):
            self.curves = nn.ModuleList(list(self.curves)[:n])
            self.ids = self.ids[:n]          # the trainer restores the real ids from its checkpoint
        return super().load_state_dict(state_dict, strict, assign)

    def remove_curve(self, idx):
        """Remove the curve at index `idx` (0-based)."""
        if idx < 0 or idx >= len(self.curves):
            raise IndexError(f"No curve at index {idx}")
        del self.curves[idx]
