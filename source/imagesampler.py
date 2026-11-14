import os

import torch
import torchvision.transforms as T

from source.utils import load_scene, load_rgb, load_mask


class ImageSampler:
    """The views of one split: images composited on `background`, object masks, intrinsics and poses."""
    def __init__(
        self,
        transforms_json: str,
        image_dir: str,
        width: int,
        height: int,
        device: str = "cuda",
        background=(255, 255, 255),
    ):
        # Load scene definitions (paths, c2w, intrinsics)
        file_paths, c2w_all, K = load_scene(transforms_json, width, height, device)
        self.device = device
        self.K = K.to(device)

        # camera-to-world and its inverse
        self.c2w_all = c2w_all.to(device)  # (N,4,4)
        self.w2c_all = torch.linalg.inv(self.c2w_all)

        # scene center: the point closest to all viewing axes (least squares)
        P = self.c2w_all[:, :3, 3]
        N = -self.c2w_all[:, :3, 2]
        I = torch.eye(3, device=P.device, dtype=P.dtype)
        A = torch.zeros((3, 3), device=P.device, dtype=P.dtype)
        b = torch.zeros((3,), device=P.device, dtype=P.dtype)
        for i in range(P.shape[0]):
            n_i = N[i].unsqueeze(1)
            M_i = I - n_i @ n_i.T
            A += M_i
            b += M_i @ P[i]
        self.scene_center = torch.linalg.solve(A, b).cpu().numpy()

        # load and resize images
        to_tensor = T.ToTensor()
        self.images = []
        self.masks = []    # (1,1,H,W) object masks (visual hull, respawning, metrics)
        for fp in file_paths:
            cleaned = fp.lstrip("./")
            rel_dir, stem = os.path.split(cleaned)
            base_dir = os.path.join(image_dir, rel_dir)
            # find file
            for ext in [".png", ".jpg", ".jpeg"]:
                candidate = os.path.join(base_dir, stem + ext)
                if os.path.exists(candidate):
                    img = load_rgb(candidate, width, height, background)
                    break
            else:
                raise FileNotFoundError(f"Missing image for {stem} in {base_dir}")

            t = to_tensor(img).unsqueeze(0).to(device)
            self.images.append(t)
            self.masks.append(torch.from_numpy(load_mask(candidate, width, height))[None, None].to(device))

    def sample(self, idx: int):
        """
        Return a single sample by index:
          - rgb: (1,3,H,W)
          - K:  (3,3)
          - w2c: (4,4)
        """
        return self.images[idx], self.K, self.w2c_all[idx]

    def __len__(self):
        return len(self.images)
