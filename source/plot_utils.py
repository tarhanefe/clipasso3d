from source.beziercurve import BezierCurve, CurveSet
import numpy as np
import torch
import plotly.graph_objects as go

def plot_spheres(means: np.ndarray, radii: np.ndarray, resolution: int = 20):
    """
    Plot 3D spheres centered at `means` with given `radii`.

    Args:
        means      (N,3) numpy array of sphere centers
        radii      (N,) numpy array of sphere radii
        resolution number of subdivisions in θ,ϕ (higher → smoother)
    """
    # parameterize a unit sphere
    u = np.linspace(0, 2*np.pi, resolution)
    v = np.linspace(0, np.pi,   resolution)
    uu, vv = np.meshgrid(u, v)

    fig = go.Figure()

    for (x0, y0, z0), r in zip(means, radii):
        # sphere surface at center (x0,y0,z0)
        x = x0 + r * np.cos(uu) * np.sin(vv)
        y = y0 + r * np.sin(uu) * np.sin(vv)
        z = z0 + r * np.cos(vv)

        fig.add_trace(go.Surface(
            x=x, y=y, z=z,
            showscale=False,
            opacity=0.6,
            lighting=dict(ambient=0.5, diffuse=0.5, roughness=0.9),
            hoverinfo='skip'
        ))

    # make axes equal and add some padding
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X', backgroundcolor="rgb(230, 230,230)"),
            yaxis=dict(title='Y', backgroundcolor="rgb(230, 230,230)"),
            zaxis=dict(title='Z', backgroundcolor="rgb(230, 230,230)"),
            aspectmode='data'
        ),
        width=800, height=800,
        title="3D Spheres at Bézier Sample Locations"
    )

    fig.show()

def plot_curve_set(
        curve_set: CurveSet,
        c2w: torch.Tensor = None,
        resolution: int = 100,
        plot_control_points: bool = False,
    ):
    """
    Plot the Bézier curve defined by the control points in `bezier`.

    Args:
        bezier     BezierCurve object
        c2w        (4,4) camera-to-world matrix
        K          (3,3) camera intrinsic matrix
        resolution number of subdivisions in t (higher → smoother)
    """

    fig = go.Figure()

    for bezier in curve_set.curves:
        # compute the Bézier curve points
        t = torch.linspace(0.0, 1.0, steps=resolution, device=bezier.device)
        bezier_pts = bezier._bezier_pts(t.unsqueeze(1)).detach().cpu().numpy() # (K,3)

        # don't label the traces

        fig.add_trace(go.Scatter3d(
            x=bezier_pts[:, 0], y=bezier_pts[:, 1], z=bezier_pts[:, 2],
            mode='lines', line=dict(color='black', width=5),
            showlegend=False,
            hoverinfo='skip'
        ))

        if plot_control_points:
            # plot the control points
            control_pts = bezier.P.cpu().detach().numpy()
            fig.add_trace(go.Scatter3d(
                x=control_pts[:, 0], y=control_pts[:, 1], z=control_pts[:, 2],
                mode='markers', marker=dict(color='red', size=5),
                name='Control Points'
            ))

    # make axes equal and add some padding
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X', backgroundcolor="rgb(230, 230,230)", showspikes=False, showbackground=False, showgrid=False, visible=False),
            yaxis=dict(title='Y', backgroundcolor="rgb(230, 230,230)", showspikes=False, showbackground=False, showgrid=False, visible=False),
            zaxis=dict(title='Z', backgroundcolor="rgb(230, 230,230)", showspikes=False, showbackground=False, showgrid=False, visible=False),
            aspectmode='data'
        ),
        width=800, height=800,
        showlegend=False,
    )

    if c2w is not None:
        # set the camera position
        position = c2w[:3, 3].cpu().numpy()
        lookat = c2w[:3, 3].cpu().numpy() - c2w[:3, 2].cpu().numpy()
        up = c2w[:3, 1].cpu().numpy()
        fig.update_layout(
            scene_camera=dict(
                eye=dict(x=position[0], y=position[1], z=position[2]),
                center=dict(x=lookat[0], y=lookat[1], z=lookat[2]),
                up=dict(x=up[0], y=up[1], z=up[2])
            )
        )

    # show the plot
    fig.show()

    return fig
