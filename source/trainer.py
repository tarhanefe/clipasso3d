import torch
import math
import random
from pathlib import Path
import matplotlib.pyplot as plt
from IPython.display import clear_output, display

class Trainer:
    def __init__(
        self,
        sampler,
        curve_set,
        rasterizer,
        criterion,
        optimizer,
        batch_size,
        inner_steps,
        epochs,
        width,
        height,
        save_dir,
        display_plots: bool = True,
    ):
        self.sampler = sampler
        self.curve_set = curve_set
        self.rasterizer = rasterizer
        self.criterion = criterion
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.inner_steps = inner_steps
        self.epochs = epochs
        self.width = width
        self.height = height
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.display_plots = display_plots

        # Precompute training schedule
        self.N_views = len(self.sampler.images)
        self.updates_per_epoch = math.ceil(self.N_views / self.batch_size)
        self.total_batches = self.updates_per_epoch * self.epochs

    def train(self):
        # Setup display if enabled
        if self.display_plots:
            plt.ioff()
            self.fig, (self.ax_render, self.ax_target) = plt.subplots(1, 2, figsize=(12, 6))
            # Placeholder for render
            init_render = torch.zeros((self.height, self.width, 3), dtype=torch.float32).numpy()
            self.im_render = self.ax_render.imshow(init_render, vmin=0, vmax=1)
            self.ax_render.set_title("Render")
            self.ax_render.axis('off')
            # Placeholder for target (first sample)
            first_target = self.sampler.images[0][0].permute(1, 2, 0).cpu().numpy()
            self.im_target = self.ax_target.imshow(first_target, vmin=0, vmax=1)
            self.ax_target.set_title("Target")
            self.ax_target.axis('off')

        # Training loop
        means, thicknesses = None, None
        for batch_idx in range(self.total_batches):
            # Sample batch of views
            batch_data = []
            for _ in range(self.batch_size):
                v_idx = random.randrange(self.N_views)
                batch_data.append((
                    self.sampler.images[v_idx],  # target_rgb
                    self.sampler.K,              # camera intrinsics
                    self.sampler.w2c_all[v_idx]  # world-to-camera
                ))

            # Inner optimization steps
            for inner in range(self.inner_steps):
                self.optimizer.zero_grad()
                losses = []
                for target_rgb, K, w2c in batch_data:
                    means, thicknesses = self.curve_set()
                    img = self.rasterizer(means, thicknesses, w2c, K, self.width, self.height)
                    img = img.permute(2, 0, 1).unsqueeze(0).repeat(1, 3, 1, 1)
                    iteration = batch_idx * self.inner_steps + inner
                    losses.append(self.criterion(img, target_rgb, iteration))
                loss = torch.stack(losses).mean()
                loss.backward()
                self.optimizer.step()

            # Update visualization
            if self.display_plots:
                render_np = img[0].permute(1, 2, 0).detach().cpu().numpy()
                target_np = batch_data[0][0][0].permute(1, 2, 0).cpu().numpy()
                self.im_render.set_data(render_np)
                self.im_target.set_data(target_np)
                self.ax_render.set_title(
                    f"Render (Batch {batch_idx}/{self.total_batches})\nLoss {loss.item():.4f}"
                )
                self.ax_target.set_title("Target")
                self.fig.canvas.draw()
                self.fig.savefig(self.save_dir / f"batch_{batch_idx:04d}.png", bbox_inches='tight', pad_inches=0)
                clear_output(wait=True)
                display(self.fig)
            else:
                # Always save output even if not displaying
                if hasattr(self, 'fig'):
                    self.fig.savefig(self.save_dir / f"batch_{batch_idx:04d}.png", bbox_inches='tight', pad_inches=0)

        # Final show if enabled
        if self.display_plots:
            plt.show()

        # Return the last computed means and thicknesses
        return means, thicknesses