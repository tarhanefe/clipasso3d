import torch
import torch.nn.functional as F

# CLIP image normalization (same constants as cliploss.py)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


@torch.no_grad()
def render_views(curve_set, rasterizer, sampler, width, height):
    """Render every view of `sampler`; returns (V,3,H,W) sketches in [0,1]."""
    means, thicknesses = curve_set()
    out = []
    for idx in range(len(sampler)):
        _, K, w2c = sampler.sample(idx)
        a = rasterizer(means, thicknesses, w2c, K, width, height)
        out.append(a.permute(2, 0, 1).repeat(3, 1, 1))
    return torch.stack(out)


@torch.no_grad()
def clip_similarity(clip_model, sketches, targets, batch=32):
    """Cosine similarity of CLIP image embeddings, per view. sketches/targets: (V,3,H,W) in [0,1]."""
    device = next(clip_model.parameters()).device
    mean = torch.tensor(CLIP_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD, device=device).view(1, 3, 1, 1)

    def embed(x):
        x = F.interpolate(x.to(device), size=(224, 224), mode="bilinear", align_corners=False, antialias=True)
        return F.normalize(clip_model.encode_image((x - mean) / std).float(), dim=-1)

    sims = []
    for i in range(0, len(sketches), batch):
        sims.append((embed(sketches[i:i + batch]) * embed(targets[i:i + batch])).sum(-1))
    return torch.cat(sims)


@torch.no_grad()
def silhouette_precision(sketches, masks):
    """Fraction of stroke ink that falls on the object, per view. masks: (V,1,H,W) in [0,1]."""
    ink = sketches[:, :1]
    return (ink * masks).flatten(1).sum(1) / ink.flatten(1).sum(1).clamp_min(1e-8)


@torch.no_grad()
def chamfer_to_surface(curve_points, surface_points, max_surface=20000, chunk=4096, seed=0):
    """
    Distances between curve samples and points on the true surface (same normalized frame):
      accuracy     mean distance from each curve point to the surface      (are strokes on the object?)
      completeness mean distance from each surface point to the curves     (is the whole object covered?)
    """
    device = curve_points.device
    surf = surface_points.to(device=device, dtype=curve_points.dtype)
    if len(surf) > max_surface:
        g = torch.Generator().manual_seed(seed)
        surf = surf[torch.randperm(len(surf), generator=g)[:max_surface].to(device)]

    def nn_dist(a, b):
        return torch.cat([torch.cdist(a[i:i + chunk], b).min(1).values for i in range(0, len(a), chunk)])

    acc = nn_dist(curve_points, surf)
    comp = nn_dist(surf, curve_points)
    return {
        "accuracy": acc.mean().item(),
        "completeness": comp.mean().item(),
        "chamfer": 0.5 * (acc.mean() + comp.mean()).item(),
    }


@torch.no_grad()
def evaluate_views(curve_set, rasterizer, sampler, clip_model, width, height):
    """CLIP similarity and silhouette precision on all views of `sampler` (mean over views)."""
    sketches = render_views(curve_set, rasterizer, sampler, width, height)
    targets = torch.cat(sampler.images)
    masks = torch.cat(sampler.masks)
    sim = clip_similarity(clip_model, sketches, targets)
    prec = silhouette_precision(sketches, masks)
    return {
        "clip_similarity": sim.mean().item(),
        "silhouette_precision": prec.mean().item(),
        "n_views": len(sampler),
    }
