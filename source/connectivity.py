import torch


def pairwise_curve_distance(pts: torch.Tensor, chunk: int = 64):
    """
    Minimum distance between every pair of curves, from points sampled on them.
    pts: (C,T,3). Returns (C,C) with +inf on the diagonal; differentiable.
    """
    C, T, _ = pts.shape
    flat = pts.reshape(C * T, 3)
    rows = []
    for i in range(0, C, chunk):
        a = pts[i:i + chunk].reshape(-1, 3)                                # (b*T,3)
        d = torch.cdist(a, flat).view(-1, T, C, T).amin(dim=(1, 3))       # (b,C)
        rows.append(d)
    D = torch.cat(rows)
    return D + torch.diag(torch.full((C,), float('inf'), device=D.device, dtype=D.dtype))


def components(D: torch.Tensor, eps: float):
    """Connected components of the contact graph (curves i, j touch if D_ij < eps). Returns labels (C,)."""
    C = D.shape[0]
    adj = (D.detach() < eps).cpu()
    parent = list(range(C))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in adj.nonzero().tolist():
        if i < j:
            parent[find(i)] = find(j)
    roots = [find(i) for i in range(C)]
    remap = {r: k for k, r in enumerate(dict.fromkeys(roots))}
    return torch.tensor([remap[r] for r in roots])


class ConnectivityLoss:
    """
    Pushes the curves to form one physically connected structure (a wire sculpture that can be
    built): every curve must touch another curve (distance < eps, e.g. two stroke radii), and
    every separate cluster is pulled towards the largest one.

        loss = weight * ( mean_i relu(d_i - margin*eps)  +  mean_c relu(D(c, main) - margin*eps) )

    d_i is the distance from curve i to its nearest other curve, D(c, main) the distance between
    cluster c and the largest cluster. Distances use `n_points` points per curve (incl. endpoints).
    Curves are pulled to margin*eps (< eps) so contacts do not flicker at the threshold.
    """
    def __init__(self, weight: float, eps: float, n_points: int = 8, margin: float = 0.5):
        self.weight = weight
        self.eps = eps
        self.n_points = n_points
        self.target = margin * eps

    def __call__(self, curve_set):
        if len(curve_set.curves) < 2:
            return curve_set.curves[0].P.sum() * 0
        t = torch.linspace(0, 1, self.n_points, device=curve_set.curves[0].P.device)
        D = pairwise_curve_distance(curve_set.points_at(t))
        floating = torch.relu(D.amin(dim=1) - self.target).mean()
        labels = components(D, self.eps).to(D.device)
        sizes = torch.bincount(labels)
        main = int(sizes.argmax())
        others = [c for c in range(len(sizes)) if c != main]
        if others:
            in_main = labels == main
            gaps = torch.stack([D[labels == c][:, in_main].min() for c in others])
            joining = torch.relu(gaps - self.target).mean()
        else:
            joining = D.new_zeros(())
        return self.weight * (floating + joining)


@torch.no_grad()
def connectivity_stats(curve_set, eps: float, n_points: int = 32):
    """How buildable the result is: number of separate pieces, curves touching nothing, main piece share."""
    C = len(curve_set.curves)
    if C < 2:
        return {"n_components": C, "n_isolated": C, "frac_in_main": 1.0}
    t = torch.linspace(0, 1, n_points, device=curve_set.curves[0].P.device)
    D = pairwise_curve_distance(curve_set.points_at(t))
    labels = components(D, eps)
    sizes = torch.bincount(labels)
    return {
        "n_components": int(len(sizes)),
        "n_isolated": int((D.amin(dim=1) >= eps).sum()),
        "frac_in_main": float(sizes.max()) / C,
    }


@torch.no_grad()
def weld(curve_set, eps: float, n_points: int = 32, max_iters: int = 10000):
    """
    Make the curves one connected piece: repeatedly take the separate cluster closest to the
    largest one and translate it rigidly (shape unchanged) until it touches (gap 0.5*eps).
    Returns (number of clusters moved, total translation distance).
    """
    t = torch.linspace(0, 1, n_points, device=curve_set.curves[0].P.device)
    moved, total = 0, 0.0
    for _ in range(max_iters):
        if len(curve_set.curves) < 2:
            break
        pts = curve_set.points_at(t)                                     # (C,T,3)
        D = pairwise_curve_distance(pts)
        labels = components(D, eps).to(D.device)
        sizes = torch.bincount(labels)
        if len(sizes) == 1:
            break
        main = int(sizes.argmax())
        in_main = labels == main
        sub = D[~in_main][:, in_main]
        r, c = divmod(int(sub.argmin()), sub.shape[1])
        i, j = int((~in_main).nonzero()[r]), int(in_main.nonzero()[c])  # closest pair of curves
        dd = torch.cdist(pts[i], pts[j])
        a, b = divmod(int(dd.argmin()), n_points)
        gap = pts[j, b] - pts[i, a]
        dist = gap.norm()
        shift = gap * (1 - 0.5 * eps / dist.clamp_min(1e-12))
        for k in (labels == labels[i]).nonzero().flatten().tolist():
            curve_set.curves[k].P.add_(shift)
        moved += 1
        total += float(shift.norm())
    return moved, total
