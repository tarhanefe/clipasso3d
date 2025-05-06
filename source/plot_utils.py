import numpy as np
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
