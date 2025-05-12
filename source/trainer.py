# import torch
# import math
# import random
# from pathlib import Path
# import matplotlib.pyplot as plt
# from IPython.display import clear_output, display

# class Trainer:
#     def __init__(
#         self,
#         sampler,
#         curve_set,
#         rasterizer,
#         criterion,
#         optimizer,
#         batch_size,
#         inner_steps,
#         epochs,
#         width,
#         height,
#         save_dir,
#         display_plots: bool = True,
#     ):
#         self.sampler = sampler
#         self.curve_set = curve_set
#         self.rasterizer = rasterizer
#         self.criterion = criterion
#         self.optimizer = optimizer
#         self.batch_size = batch_size
#         self.inner_steps = inner_steps
#         self.epochs = epochs
#         self.width = width
#         self.height = height
#         self.save_dir = Path(save_dir)
#         self.save_dir.mkdir(parents=True, exist_ok=True)
#         self.display_plots = display_plots

#         # Precompute training schedule
#         self.N_views = len(self.sampler.images)
#         self.updates_per_epoch = math.ceil(self.N_views / self.batch_size)
#         self.total_batches = self.updates_per_epoch * self.epochs
#         self.means, self.thicknesses = None, None
#     def train(self):
#         # Setup display if enabled
#         if self.display_plots:
#             plt.ioff()
#             self.fig, (self.ax_render, self.ax_target) = plt.subplots(1, 2, figsize=(12, 6))
#             # Placeholder for render
#             init_render = torch.zeros((self.height, self.width, 3), dtype=torch.float32).numpy()
#             self.im_render = self.ax_render.imshow(init_render, vmin=0, vmax=1)
#             self.ax_render.set_title("Render")
#             self.ax_render.axis('off')
#             # Placeholder for target (first sample)
#             first_target = self.sampler.images[0][0].permute(1, 2, 0).cpu().numpy()
#             self.im_target = self.ax_target.imshow(first_target, vmin=0, vmax=1)
#             self.ax_target.set_title("Target")
#             self.ax_target.axis('off')

#         # Training loop
#         for batch_idx in range(self.total_batches):
#             # Sample batch of views
#             batch_data = []
#             for _ in range(self.batch_size):
#                 v_idx = random.randrange(self.N_views)
#                 batch_data.append((
#                     self.sampler.images[v_idx],  # target_rgb
#                     self.sampler.K,              # camera intrinsics
#                     self.sampler.w2c_all[v_idx]  # world-to-camera
#                 ))

#             # Inner optimization steps
#             for inner in range(self.inner_steps):
#                 self.optimizer.zero_grad()
#                 losses = []
#                 for target_rgb, K, w2c in batch_data:
#                     self.means, self.thicknesses = self.curve_set()
#                     img = self.rasterizer(self.means, self.thicknesses, w2c, K, self.width, self.height)
#                     img = img.permute(2, 0, 1).unsqueeze(0).repeat(1, 3, 1, 1)
#                     iteration = batch_idx * self.inner_steps + inner
#                     losses.append(self.criterion(img, target_rgb, iteration))
#                 loss = torch.stack(losses).mean()
#                 loss.backward()
#                 self.optimizer.step()

#             # Update visualization
#             if self.display_plots:
#                 render_np = img[0].permute(1, 2, 0).detach().cpu().numpy()
#                 target_np = batch_data[0][0][0].permute(1, 2, 0).cpu().numpy()
#                 self.im_render.set_data(render_np)
#                 self.im_target.set_data(target_np)
#                 self.ax_render.set_title(
#                     f"Render (Batch {batch_idx}/{self.total_batches})\nLoss {loss.item():.4f}"
#                 )
#                 self.ax_target.set_title("Target")
#                 self.fig.canvas.draw()
#                 self.fig.savefig(self.save_dir / f"batch_{batch_idx:04d}.png", bbox_inches='tight', pad_inches=0)
#                 clear_output(wait=True)
#                 display(self.fig)
#             else:
#                 # Always save output even if not displaying
#                 if hasattr(self, 'fig'):
#                     self.fig.savefig(self.save_dir / f"batch_{batch_idx:04d}.png", bbox_inches='tight', pad_inches=0)

#         # Final show if enabled
#         if self.display_plots:
#             plt.show()

#         # Return the last computed means and thicknesses
#         return self.means, self.thicknesses
import math
import os
import sys
import random
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


sys.path.append("..")
from source.utils import load_scene
from source.imagesampler import ImageSampler

class Trainer:
    def __init__(
        self,
        train_sampler: ImageSampler,
        val_sampler: ImageSampler,
        test_sampler: ImageSampler,
        curve_set,
        rasterizer,
        criterion,
        optimizer,
        batch_size: int,
        inner_steps: int,
        epochs: int,
        width: int,
        height: int,
        save_dir: str,
        eval_interval: int = 100,
        device: str = "cuda",
        display_plots: bool = True,
    ):
        self.train_sampler = train_sampler
        self.val_sampler = val_sampler
        self.test_sampler = test_sampler
        self.curve_set = curve_set
        self.rasterizer = rasterizer
        self.criterion = criterion
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.inner_steps = inner_steps
        self.epochs = epochs
        self.width = width
        self.height = height
        self.device = device
        self.save_dir = os.path.expanduser(save_dir)
        os.makedirs(self.save_dir, exist_ok=True)
        self.eval_interval = eval_interval
        self.display_plots = display_plots

        # schedule
        self.N_train = len(train_sampler)
        self.updates_per_epoch = math.ceil(self.N_train / batch_size)
        self.total_batches = self.updates_per_epoch * epochs
        self.means = None
        self.thicknesses = None

    def train(self):
        # optional real-time display
        if self.display_plots:
            plt.ion()
            fig, (ax_r, ax_t) = plt.subplots(1, 2, figsize=(12, 6))
            self.im_r = ax_r.imshow(np.zeros((self.height, self.width, 3)), vmin=0, vmax=1)
            ax_r.axis('off'); ax_r.set_title('Render')
            first,_,_ = self.train_sampler.sample(0)
            self.im_t = ax_t.imshow(first[0].permute(1,2,0).cpu().numpy(), vmin=0, vmax=1)
            ax_t.axis('off'); ax_t.set_title('Target')
            fig.tight_layout()

        batch_idx = 0
        for epoch in range(self.epochs):
            for _ in range(self.updates_per_epoch):
                # form a batch
                batch = [self.train_sampler.sample(random.randrange(self.N_train))
                         for _ in range(self.batch_size)]
                # inner loop
                for _ in range(self.inner_steps):
                    self.optimizer.zero_grad()
                    losses = []
                    for rgb, K, w2c in batch:
                        self.means, self.thicknesses = self.curve_set()
                        render = self.rasterizer(self.means, self.thicknesses, w2c, K, self.width, self.height)
                        render = render.permute(2,0,1).unsqueeze(0).repeat(1,3,1,1)
                        iteration = batch_idx * self.inner_steps
                        losses.append(self.criterion(render, rgb, iteration))
                    loss = torch.stack(losses).mean()
                    loss.backward()
                    self.optimizer.step()

                # periodic evaluation
                if batch_idx % self.eval_interval == 0:
                    val_loss = self.evaluate(self.val_sampler)
                    print(f"Batch {batch_idx}/{self.total_batches}: train_loss={loss.item():.4f}, val_loss={val_loss:.4f}")

                # update display or save
                if self.display_plots:
                    self.im_r.set_data(render[0].permute(1,2,0).detach().cpu().numpy())
                    self.im_t.set_data(batch[0][0][0].permute(1,2,0).cpu().numpy())
                    plt.draw(); plt.pause(0.001)
                else:
                    plt.savefig(os.path.join(self.save_dir, f"batch_{batch_idx:04d}.png"))

                batch_idx += 1

        # final evaluation on test set
        test_loss = self.evaluate(self.test_sampler)
        print(f"Training completed. Test loss: {test_loss:.4f}")
        return self.means, self.thicknesses

    def evaluate(self, sampler: ImageSampler):
        self.curve_set.eval()
        total = 0.0
        with torch.no_grad():
            for idx in range(len(sampler)):
                rgb, K, w2c = sampler.sample(idx)
                means, thicknesses = self.curve_set()
                render = self.rasterizer(means, thicknesses, w2c, K, self.width, self.height)
                render = render.permute(2,0,1).unsqueeze(0).repeat(1,3,1,1)
                loss = self.criterion(render, rgb, -1)
                total += loss.item()
        self.curve_set.train()
        return total / len(sampler)
