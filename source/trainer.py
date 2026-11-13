import json
import math
import os
import sys
import random
import time
import torch
import numpy as np
from PIL import Image

sys.path.append("..")
from source.imagesampler import ImageSampler
from source.pruning import prune_curves, rebuild_optimizer

class Trainer:
    """
    Multi-view optimization of the curve set.

    Restart-safe: with `ckpt_path` set, the curves, optimizer, losses and RNG states are
    saved atomically every `ckpt_every` batches and training resumes from that file. Each
    batch appends a line to `log_path` (JSONL) that tools/status.py reads for progress/ETA.
    `display_plots=True` keeps the live matplotlib view for notebooks. With `wandb_run`
    (an active wandb run, see pipeline.py) the same entries and the frames go to Weights & Biases.
    """
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
        ckpt_path: str = None,
        ckpt_every: int = 20,
        frame_every: int = 1,
        log_path: str = None,
        wandb_run=None,
        invert: bool = False,
        prune_every: int = 0,
        prune_start: int = 0,
        prune_min_on_frac: float = 0.5,
        prune_min_length: float = 0.005,
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
        self.save_dir = os.path.expanduser(save_dir) if save_dir else None
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
        self.eval_interval = eval_interval
        self.display_plots = display_plots
        self.ckpt_path = ckpt_path
        self.ckpt_every = ckpt_every
        self.frame_every = frame_every
        self.log_path = log_path
        self.wandb_run = wandb_run
        self.invert = invert    # True: black strokes on white (targets must be on white too)
        # pruning: every `prune_every` batches from `prune_start` on, delete curves that lie
        # outside the visual hull of the training masks or have collapsed (0 = off)
        self.prune_every = prune_every
        self.prune_start = prune_start
        self.prune_min_on_frac = prune_min_on_frac
        self.prune_min_length = prune_min_length

        # training schedule
        self.N_train = len(train_sampler)
        self.updates_per_epoch = math.ceil(self.N_train / self.batch_size)
        self.total_batches = self.updates_per_epoch * self.epochs

        # metrics tracking
        self.batch_indices = []
        self.train_losses = []
        self.val_batches = []
        self.val_losses = []
        self.start_batch = 0

        self.means = None
        self.thicknesses = None

    # ------------------------------------------------------------ checkpoints

    def state_dict(self, next_batch):
        state = {
            "next_batch": next_batch,
            "curve_set": self.curve_set.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "batch_indices": self.batch_indices,
            "train_losses": self.train_losses,
            "val_batches": self.val_batches,
            "val_losses": self.val_losses,
            "rng_python": random.getstate(),
            "rng_numpy": np.random.get_state(),
            "rng_torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["rng_cuda"] = torch.cuda.get_rng_state_all()
        return state

    def save_checkpoint(self, next_batch):
        if not self.ckpt_path:
            return
        tmp = self.ckpt_path + ".tmp"
        torch.save(self.state_dict(next_batch), tmp)
        os.replace(tmp, self.ckpt_path)

    def resume(self):
        """Load `ckpt_path` if it exists; returns the batch index to continue from."""
        if not (self.ckpt_path and os.path.exists(self.ckpt_path)):
            return 0
        state = torch.load(self.ckpt_path, map_location=self.device, weights_only=False)
        self.curve_set.load_state_dict(state["curve_set"])      # also drops curves pruned before
        self.optimizer = rebuild_optimizer(self.optimizer, list(self.curve_set.parameters()))
        self.optimizer.load_state_dict(state["optimizer"])
        self.batch_indices = state["batch_indices"]
        self.train_losses = state["train_losses"]
        self.val_batches = state["val_batches"]
        self.val_losses = state["val_losses"]
        random.setstate(state["rng_python"])
        np.random.set_state(state["rng_numpy"])
        torch.set_rng_state(state["rng_torch"].cpu())
        if "rng_cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in state["rng_cuda"]])
        print(f"Resumed from {self.ckpt_path} at batch {state['next_batch']}/{self.total_batches}")
        return state["next_batch"]

    def log(self, **entry):
        if self.wandb_run is not None:
            self.wandb_run.log({k: v for k, v in entry.items() if v is not None and k not in ("batch", "total")},
                               step=entry["batch"])
        if not self.log_path:
            return
        entry["time"] = time.time()
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def to_image(self, alpha):
        """(H,W,1) stroke coverage -> (1,3,H,W) sketch image."""
        img = alpha.permute(2, 0, 1).unsqueeze(0).repeat(1, 3, 1, 1)
        return 1 - img if self.invert else img

    # ------------------------------------------------------------ display / frames

    def _init_display(self):
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

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

    def _update_display(self, render, target, status):
        from IPython.display import clear_output, display

        # update images
        self.im_r.set_data(render[0].permute(1, 2, 0).detach().cpu().numpy())
        self.im_t.set_data(target[0].permute(1, 2, 0).cpu().numpy())

        # update metrics plot
        self.ax_m.clear()
        self.ax_m.plot(self.batch_indices, self.train_losses, label='Train')
        if self.val_losses:
            self.ax_m.plot(self.val_batches, self.val_losses, label='Val')
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

    def _save_frame(self, batch_idx, render, target):
        path = os.path.join(self.save_dir, f"batch_{batch_idx:04d}.png")
        if self.display_plots:
            self.fig.savefig(path)
            return
        pair = torch.cat([render[0].detach(), target[0]], dim=2).clamp(0, 1)   # render | target
        arr = (pair.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(arr).save(path + ".tmp.png")
        os.replace(path + ".tmp.png", path)
        if self.wandb_run is not None:
            import wandb
            self.wandb_run.log({"render|target": wandb.Image(arr)}, step=batch_idx + 1)

    # ------------------------------------------------------------ training

    def train(self):
        if self.display_plots:
            self._init_display()

        self.start_batch = self.resume()
        for batch_idx in range(self.start_batch, self.total_batches):
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
                    render = self.to_image(render)
                    it = batch_idx * self.inner_steps
                    losses.append(self.criterion(render, rgb, it))
                train_loss_tensor = torch.stack(losses).mean()
                train_loss = train_loss_tensor.item()
                train_loss_tensor.backward()
                self.optimizer.step()

            # record training loss
            self.batch_indices.append(batch_idx)
            self.train_losses.append(train_loss)

            # periodic pruning of curves off the object
            n_pruned = None
            if (self.prune_every and batch_idx >= self.prune_start
                    and (batch_idx - self.prune_start) % self.prune_every == 0):
                self.optimizer, n_pruned, info = prune_curves(
                    self.curve_set, self.optimizer, self.train_sampler,
                    self.prune_min_on_frac, self.prune_min_length)
                print(f"Pruning at batch {batch_idx}: removed {n_pruned}, {len(self.curve_set.curves)} curves left "
                      f"({info['n_off_object']} off the object, {info['n_collapsed']} collapsed)", flush=True)

            # periodic validation
            val_loss = None
            if batch_idx % self.eval_interval == 0 or batch_idx == self.total_batches - 1:
                val_loss = self.evaluate(self.val_sampler)
                self.val_batches.append(batch_idx)
                self.val_losses.append(val_loss)
                status = f"Batch {batch_idx}/{self.total_batches}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
            else:
                status = f"Batch {batch_idx}/{self.total_batches}: train_loss={train_loss:.4f}"
            print(status, flush=True)

            # update display or save
            if self.display_plots:
                self._update_display(render, batch[0][0], status)
            if self.save_dir and batch_idx % self.frame_every == 0:
                self._save_frame(batch_idx, render, batch[0][0])

            self.log(batch=batch_idx + 1, total=self.total_batches, train_loss=train_loss, val_loss=val_loss,
                     n_curves=len(self.curve_set.curves), n_pruned=n_pruned)
            if (batch_idx + 1) % self.ckpt_every == 0 or batch_idx == self.total_batches - 1:
                self.save_checkpoint(batch_idx + 1)

        # final test evaluation
        test_loss = self.evaluate(self.test_sampler)
        print(f"Training completed. Test loss: {test_loss:.4f}")
        self.test_loss = test_loss
        with torch.no_grad():
            self.means, self.thicknesses = self.curve_set()
        return self.means, self.thicknesses

    def evaluate(self, sampler: ImageSampler) -> float:
        self.curve_set.eval()
        total_loss = 0.0
        count = len(sampler)
        with torch.no_grad():
            means, thicknesses = self.curve_set()
            for idx in range(count):
                rgb, K, w2c = sampler.sample(idx)
                render = self.rasterizer(means, thicknesses, w2c, K, self.width, self.height)
                render = self.to_image(render)
                loss = self.criterion(render, rgb, -1, mode="eval")
                total_loss += loss.item()
        self.curve_set.train()
        return total_loss / count
