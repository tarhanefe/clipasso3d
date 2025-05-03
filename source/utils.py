import json
import math
import torch


def load_scene(
    json_path: str,
    width: int,
    height: int,
    device: str = "cpu",
) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    """
    Load camera intrinsics and camera-to-world extrinsics from a NeRF-style transforms JSON.

    Args:
        json_path: Path to the JSON file containing "camera_angle_x" and "frames".
        width: Render width in pixels.
        height: Render height in pixels.
        device: Torch device string (e.g. 'cpu' or 'cuda').

    Returns:
        file_paths: List of image file paths (as given in JSON).
        c2w_mats:   Tensor of shape (N,4,4), camera-to-world transforms.
        K:          Tensor of shape (3,3), camera intrinsics.
    """
    # 1. Read JSON
    with open(json_path, "r") as f:
        meta = json.load(f)

    # 2. Build intrinsics K from horizontal FOV
    camera_angle_x = meta["camera_angle_x"]
    focal = 0.5 * width / math.tan(0.5 * camera_angle_x)
    K = torch.tensor(
        [
            [focal,    0.0, width * 0.5],
            [   0.0, focal, height * 0.5],
            [   0.0,    0.0,        1.0],
        ],
        dtype=torch.float32,
        device=device,
    )

    # 3. Collect file paths and camera-to-world matrices
    file_paths: list[str] = []
    c2w_mats: list[torch.Tensor] = []

    for frame in meta["frames"]:
        file_paths.append(frame["file_path"])
        mat = torch.tensor(frame["transform_matrix"], dtype=torch.float32, device=device)
        c2w_mats.append(mat)

    # 4. Stack into (N,4,4)
    c2w_mats = torch.stack(c2w_mats, dim=0)

    return file_paths, c2w_mats, K