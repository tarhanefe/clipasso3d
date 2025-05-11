import torch

# --- Initialize curves as short lines at random locations on a sphere ---
def random_short_lines(center: torch.Tensor,
                       n_curves: int,
                       radius: float,
                       length: float,
                       device: str):
    """
    center: (3,) origin of the sphere
    n_curves: how many line‐curves to make
    radius:  how far from center to place each line
    length:  total length of each line segment
    """
    lines = []
    for _ in range(n_curves):
        # 1) Pick a random point ON the sphere
        dir_loc = torch.randn(3, device=device)
        dir_loc = dir_loc / dir_loc.norm()
        loc = center + radius * dir_loc       # (3,)

        # 2) Pick a random line direction orthogonal to dir_loc
        dir_line = torch.randn(3, device=device)
        dir_line = dir_line - (dir_line @ dir_loc) * dir_loc
        dir_line = dir_line / dir_line.norm()

        # 3) Build 4 control points along that line of total length `length`
        offsets = torch.tensor([-0.5, -0.1667, 0.1667, 0.5],
                               device=device) * length  # (4,)
        ctrl_pts = loc[None] + offsets[:, None] * dir_line[None]  # (4,3)

        lines.append(ctrl_pts)
    return lines