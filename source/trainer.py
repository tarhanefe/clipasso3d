import math
import os
import sys
import random
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from IPython.display import clear_output, display

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

        # training schedule
        self.N_train = len(train_sampler) -1
        self.updates_per_epoch = math.ceil(self.N_train / self.batch_size)
        self.total_batches = self.updates_per_epoch * self.epochs

        # metrics tracking
        self.batch_indices = []
        self.train_losses = []
        self.val_losses = []

        self.means = None
        self.thicknesses = None

    def train(self):
        # setup real-time display
        if self.display_plots:
            plt.ion()
            self.fig = plt.figure(figsize=(8, 6))
            gs = GridSpec(2, 2, figure=self.fig, height_ratios=[3, 1])
            self.ax_r = self.fig.add_subplot(gs[0, 0])
            self.ax_t = self.fig.add_subplot(gs[0, 1])
            self.ax_m = self.fig.add_subplot(gs[1, :])

            # init render image
            blank = np.zeros((self.height, self.width, 3))
            self.im_r = self.ax_r.imshow(blank, vmin=0, vmax=1)
            self.ax_r.axis('off'); self.ax_r.set_title('Render')

            # init target image
            first, _, _ = self.train_sampler.sample(0)
            target_img = first[0].permute(1, 2, 0).cpu().numpy()
            self.im_t = self.ax_t.imshow(target_img, vmin=0, vmax=1)
            self.ax_t.axis('off'); self.ax_t.set_title('Target')

            # metrics subplot
            self.ax_m.set_title('Loss Curves')
            self.ax_m.set_xlabel('Batch')
            self.ax_m.set_ylabel('Loss')
            self.ax_m.grid(True)
            plt.show(block=False)

        batch_idx = 0
        for epoch in range(self.epochs):
            for _ in range(self.updates_per_epoch):
                # sample batch
                batch = [self.train_sampler.sample(random.randrange(self.N_train))
                         for _ in range(self.batch_size)]

                # inner optimization
                for _ in range(self.inner_steps):
                    self.optimizer.zero_grad()
                    losses = []
                    for rgb, K, w2c in batch:
                        self.means, self.thicknesses = self.curve_set()
                        render = self.rasterizer(
                            self.means, self.thicknesses, w2c, K,
                            self.width, self.height
                        )
                        render = render.permute(2, 0, 1).unsqueeze(0).repeat(1, 3, 1, 1)
                        it = batch_idx * self.inner_steps
                        losses.append(self.criterion(render, rgb, it))
                    train_loss = torch.stack(losses).mean().item()
                    train_loss_tensor = torch.stack(losses).mean()
                    train_loss_tensor.backward()
                    self.optimizer.step()

                # record training loss
                self.batch_indices.append(batch_idx)
                self.train_losses.append(train_loss)

                # periodic validation
                val_loss = None
                if batch_idx % self.eval_interval == 0:
                    val_loss = self.evaluate(self.val_sampler)
                    self.val_losses.append(val_loss)
                    status = f"Batch {batch_idx}/{self.total_batches}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
                else:
                    status = f"Batch {batch_idx}/{self.total_batches}: train_loss={train_loss:.4f}"

                # update display or save
                if self.display_plots:
                    # update images
                    self.im_r.set_data(render[0].permute(1, 2, 0).detach().cpu().numpy())
                    self.im_t.set_data(batch[0][0][0].permute(1, 2, 0).cpu().numpy())

                    # update metrics plot
                    self.ax_m.clear()
                    self.ax_m.plot(self.batch_indices, self.train_losses, label='Train')
                    if self.val_losses:
                        eval_x = self.batch_indices[::self.eval_interval]
                        self.ax_m.plot(eval_x, self.val_losses, label='Val')
                    self.ax_m.set_title(status)
                    self.ax_m.set_xlabel('Batch')
                    self.ax_m.set_ylabel('Loss')
                    self.ax_m.legend()
                    self.ax_m.grid(True)

                    # redraw and display
                    self.fig.canvas.draw()
                    self.fig.canvas.flush_events()
                    clear_output(wait=True)
                    display(self.fig)
                else:
                    self.fig.savefig(os.path.join(self.save_dir, f"batch_{batch_idx:04d}.png"))

                batch_idx += 1

        # final test evaluation
        test_loss = self.evaluate(self.test_sampler)
        print(f"Training completed. Test loss: {test_loss:.4f}")
        return self.means, self.thicknesses

    def evaluate(self, sampler: ImageSampler) -> float:
        self.curve_set.eval()
        total_loss = 0.0
        count = len(sampler) - 1
        with torch.no_grad():
            for idx in range(count):
                rgb, K, w2c = sampler.sample(idx)
                means, thicknesses = self.curve_set()
                render = self.rasterizer(means, thicknesses, w2c, K, self.width, self.height)
                render = render.permute(2, 0, 1).unsqueeze(0).repeat(1, 3, 1, 1)
                loss = self.criterion(render, rgb, -1)
                total_loss += loss.item()
        self.curve_set.train()
        return total_loss / count
