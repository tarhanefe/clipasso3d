import clip
import torch
import torch.nn as nn
from torchvision import transforms

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
GEOMETRIC_LAYERS = (2, 3)       # intermediate ViT-B/32 blocks compared by the geometric loss (CLIPasso)


class CLIPassoLoss(nn.Module):
    """
    CLIPasso's loss (Vinker et al. 2022) between a rendered sketch and the photo of the same view:

        loss = w_g * sum_{l in 2,3} mean((f_l(S) - f_l(I))^2)  +  w_s * (1 - cos(e(S), e(I)))

    f_l are intermediate activations and e the final embedding of a frozen CLIP ViT-B/32. During
    training both terms are averaged over the clean pair and `num_augs` random perspective + crop
    augmentations applied identically to sketch and photo; evaluation uses the clean pair only.
    """
    def __init__(self, device, geometric_weight=1.0, semantic_weight=0.1, num_augs=4, aug_fill=1.0):
        super().__init__()
        self.model, preprocess = clip.load("ViT-B/32", device, jit=False)
        self.model.eval()
        self.geometric_weight = geometric_weight
        self.semantic_weight = semantic_weight
        self.num_augs = num_augs
        self.featuremaps = {}
        for i in GEOMETRIC_LAYERS:
            self.model.visual.transformer.resblocks[i].register_forward_hook(self._hook(i))
        self.normalize = transforms.Compose([preprocess.transforms[0], preprocess.transforms[1],   # resize, crop
                                             transforms.Normalize(CLIP_MEAN, CLIP_STD)])
        self.augment = transforms.Compose([
            transforms.RandomPerspective(fill=aug_fill, p=1.0, distortion_scale=0.5),
            transforms.RandomResizedCrop(224, scale=(0.8, 0.8), ratio=(1.0, 1.0)),
            transforms.Normalize(CLIP_MEAN, CLIP_STD),
        ])

    def _hook(self, i):
        def hook(module, inputs, output):
            self.featuremaps[i] = output.permute(1, 0, 2)          # LND -> NLD
        return hook

    def encode(self, x):
        self.featuremaps = {}
        fc = self.model.encode_image(x).float()
        return fc, [self.featuremaps[i] for i in GEOMETRIC_LAYERS]

    def forward(self, sketch, target, mode="train"):
        """sketch, target: (1,3,H,W) in [0,1]."""
        xs, ys = [self.normalize(sketch)], [self.normalize(target)]
        if mode == "train":
            for _ in range(self.num_augs):
                pair = self.augment(torch.cat([sketch, target]))
                xs.append(pair[0:1])
                ys.append(pair[1:2])
        x_fc, x_maps = self.encode(torch.cat(xs))
        y_fc, y_maps = self.encode(torch.cat(ys))
        loss = sum(self.geometric_weight * torch.square(a - b).mean() for a, b in zip(x_maps, y_maps))
        if self.semantic_weight:
            loss = loss + self.semantic_weight * (1 - torch.cosine_similarity(x_fc, y_fc, dim=1)).mean()
        return loss
