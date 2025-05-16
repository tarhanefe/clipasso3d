import torch
import torch.nn.functional as F

class DinoSaliency:
    def __init__(self, preprocess_shape=(224, 224), device="cuda"):
        """
        Args:
            preprocess_shape: (H, W) to resize *any* input tensor to before DINO
            device: "cuda" or "cpu"
        """
        self.device = device
        self.preprocess_shape = preprocess_shape

        # ImageNet normalization (broadcastable over B×3×H×W)
        mean = torch.tensor([0.485, 0.456, 0.406], device=device)
        std  = torch.tensor([0.229, 0.224, 0.225], device=device)
        # keep as buffers
        self.register_buffer = lambda n, t: setattr(self, n, t.view(1,3,1,1))
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

        # Load DINO ViT-S/8 once
        self.model = torch.hub.load(
            'facebookresearch/dino:main', 'dino_vits8'
        ).eval().to(self.device)

    def compute(
        self,
        img_tensor: torch.Tensor,
        output_size=(224, 224),
        threshold: float = 0.6
    ) -> torch.Tensor:
        """
        Args:
            img_tensor: Tensor in [0,1], shape (3,H,W) or (1,3,H,W)
            output_size: (H_out, W_out) for the returned saliency map
            threshold: fraction of attention mass to keep per head

        Returns:
            saliency_map: Tensor (heads, H_out, W_out), values 0 or 1
        """
        # 1) Ensure batch dim
        if img_tensor.dim() == 3:
            img = img_tensor.unsqueeze(0)
        else:
            img = img_tensor
        img = img.to(self.device).float()

        # 2) Resize to DINO’s expected input size
        img = F.interpolate(
            img,
            size=self.preprocess_shape,
            mode='bilinear',
            align_corners=False
        )

        # 3) Normalize
        img = (img - self.mean) / self.std

        # 4) Get self‐attention
        with torch.no_grad():
            attn = self.model.get_last_selfattention(img)[0]  # (heads, tokens, tokens)

        heads, _, _ = attn.shape
        # 5) Extract [CLS]→patch attention
        cls_attn = attn[:, 0, 1:].reshape(heads, -1)  # (heads, num_patches)

        # 6) Normalize & threshold top mass
        vals, idxs = torch.sort(cls_attn)                       # ascending
        vals = vals / vals.sum(dim=1, keepdim=True)             # sum to 1
        cums = vals.cumsum(dim=1)                               # cumulative
        mask = cums > (1 - threshold)                           # top `threshold`
        reorder = torch.argsort(idxs)
        for h in range(heads):
            mask[h] = mask[h][reorder[h]]                       # undo sort

        # 7) Reshape to feature grid
        ph, pw = self.preprocess_shape[0] // 8, self.preprocess_shape[1] // 8
        mask = mask.reshape(heads, ph, pw).float()              # (heads, ph, pw)

        # 8) Upsample to requested output_size
        saliency = F.interpolate(
            mask.unsqueeze(0),
            size=output_size,
            mode='bilinear',
            align_corners=False
        )[0].cpu()                                              # (heads, H_out, W_out)

        return saliency
