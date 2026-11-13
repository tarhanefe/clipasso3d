import torch
import torch.nn.functional as F


@torch.no_grad()
def curve_object_scores(curve_set, sampler, dilate: int = 2, view_frac: float = 0.9):
    """
    How much of each curve lies on the object, judged from the training silhouettes.

    Every sample point of every curve is projected into all views of `sampler`. A point is
    "on the object" if it falls inside the (dilated) object mask in at least `view_frac` of
    the views that see it, i.e. it lies inside the visual hull. Returns per curve:
      on_frac  fraction of the curve's points that are on the object      (C,)
      length   arc length of the curve                                     (C,)
    """
    pts = curve_set.curve_samples()                                   # list of (S_i,3)
    counts = torch.tensor([len(p) for p in pts], device=pts[0].device)
    X = torch.cat(pts)                                                # (P,3)
    masks = torch.cat(sampler.masks)                                  # (V,1,H,W)
    if dilate > 0:                                                    # keep thin parts (spokes, wires)
        masks = F.max_pool2d(masks, 2 * dilate + 1, stride=1, padding=dilate)
    masks = masks[:, 0] > 0.5
    V, H, W = masks.shape
    K = sampler.K
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    seen = torch.zeros(len(X), device=X.device)
    inside = torch.zeros(len(X), device=X.device)
    for v in range(V):
        w2c = sampler.w2c_all[v]
        cam = X @ w2c[:3, :3].T + w2c[:3, 3]
        depth = -cam[:, 2]
        u = cx + fx * cam[:, 0] / depth.clamp_min(1e-6)
        r = cy - fy * cam[:, 1] / depth.clamp_min(1e-6)
        ok = (depth > 0) & (u >= 0) & (u < W) & (r >= 0) & (r < H)
        hit = masks[v, r.long().clamp(0, H - 1), u.long().clamp(0, W - 1)]
        seen += ok.float()
        inside += (ok & hit).float()
    on_point = ((seen > 0) & (inside >= view_frac * seen)).float()

    curve_id = torch.repeat_interleave(torch.arange(len(pts), device=X.device), counts)
    on_frac = torch.zeros(len(pts), device=X.device).index_add(0, curve_id, on_point) / counts
    length = torch.stack([(p[1:] - p[:-1]).norm(dim=1).sum() if len(p) > 1 else p.new_zeros(())
                          for p in pts])
    return on_frac, length


def rebuild_optimizer(optimizer, params):
    """Same optimizer type and settings over `params`, keeping the state of params it already had."""
    new = optimizer.__class__(params, **optimizer.defaults)
    for p in params:
        if p in optimizer.state:
            new.state[p] = optimizer.state[p]
    return new


@torch.no_grad()
def prune_curves(curve_set, optimizer, sampler, min_on_frac=0.5, min_length=0.005, min_curves=1):
    """
    Delete curves that lie off the object (outside the visual hull) or have collapsed.
    Returns (new optimizer, number of curves removed, scores dict).
    """
    on_frac, length = curve_object_scores(curve_set, sampler)
    keep = (on_frac >= min_on_frac) & (length >= min_length)
    if keep.sum() < min_curves:                       # never delete everything: keep the best ones
        keep = torch.zeros_like(keep)
        keep[torch.argsort(on_frac, descending=True)[:min_curves]] = True
    removed = int((~keep).sum())
    if removed:
        curve_set.keep_curves(keep.tolist())
        optimizer = rebuild_optimizer(optimizer, list(curve_set.parameters()))
    return optimizer, removed, {"on_frac_mean": on_frac.mean().item(), "n_off_object": int((on_frac < min_on_frac).sum()),
                                "n_collapsed": int((length < min_length).sum())}
