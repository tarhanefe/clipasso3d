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
        n_snapshots: int = 100,
        connect_loss=None,
        connect_start: int = 0,
        early_stop_patience: int = 0,
        early_stop_tol: float = 0.005,
        min_batches: int = 0,
        snapshot_every: int = 0,
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
        # snapshots of all control points during training (for the interactive 4D viewer)
        self.n_snapshots = n_snapshots
        self.snapshots = []          # list of (batch, curve ids, control points float16 (C,N,3))
        # optional extra loss on the curve set (e.g. connectivity), added from batch `connect_start`
        self.connect_loss = connect_loss
        self.connect_start = connect_start
        # early stopping: stop when the validation loss has not improved by more than
        # `early_stop_tol` (relative) for `early_stop_patience` validations (0 = off)
        self.early_stop_patience = early_stop_patience
        self.early_stop_tol = early_stop_tol
        self.min_batches = min_batches
        self.best_val = float("inf")
        self.bad_evals = 0
        self.stopped_at = None

        # training schedule
        self.N_train = len(train_sampler)
        self.updates_per_epoch = math.ceil(self.N_train / self.batch_size)
        self.total_batches = self.updates_per_epoch * self.epochs
        # fixed interval (useful when early stopping makes the length unknown), else ~n_snapshots in total
        self.snapshot_every = snapshot_every or (max(1, self.total_batches // self.n_snapshots) if self.n_snapshots else 0)

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
            "curve_ids": self.curve_set.ids,
            "best_val": self.best_val,
            "bad_evals": self.bad_evals,
            "stopped_at": self.stopped_at,
            "snapshots": self.snapshots,
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
        self.curve_set.ids = list(state.get("curve_ids", self.curve_set.ids))
        self.snapshots = state.get("snapshots", [])
        self.best_val = state.get("best_val", float("inf"))
        self.bad_evals = state.get("bad_evals", 0)
        self.stopped_at = state.get("stopped_at")
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

    @torch.no_grad()
    def snapshot(self, batch):
        if self.snapshots and self.snapshots[-1][0] == batch:
            return
        P = self.curve_set.control_points().detach().cpu().to(torch.float16).numpy()
        self.snapshots.append((batch, list(self.curve_set.ids), P))

    def save_snapshots(self, path):
        """Training history as npz: batches (T,), offsets (T+1,), ids (sum C,), P (sum C,N,3) float16."""
        if not self.snapshots:
            return
        batches = np.array([b for b, _, _ in self.snapshots], dtype=np.int32)
        counts = [len(ids) for _, ids, _ in self.snapshots]
        offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        ids = np.concatenate([np.asarray(i, dtype=np.int32) for _, i, _ in self.snapshots])
        P = np.concatenate([p for _, _, p in self.snapshots])
        tmp = path + ".tmp.npz"
        np.savez_compressed(tmp, batches=batches, offsets=offsets, ids=ids, P=P)
        os.replace(tmp, path)

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
        if self.wandb_run is not None and (batch_idx // self.frame_every) % 10 == 0:   # every 10th frame
            import wandb
            self.wandb_run.log({"render|target": wandb.Image(arr)}, step=batch_idx + 1)

    # ------------------------------------------------------------ training

    def train(self):
        if self.display_plots:
            self._init_display()

        self.start_batch = self.resume()
        if self.snapshot_every and not self.snapshots:
            self.snapshot(0)                                    # initialization
        if self.stopped_at is not None:                         # resumed after an early stop
            print(f"Early-stopped at batch {self.stopped_at} (resumed run)")
        for batch_idx in range(self.start_batch, self.total_batches if self.stopped_at is None else 0):
            # sample batch
            # `batch_size` different views (with replacement only if there are fewer views than that)
            if self.batch_size <= self.N_train:
                views = random.sample(range(self.N_train), self.batch_size)
            else:
                views = [random.randrange(self.N_train) for _ in range(self.batch_size)]
            batch = [self.train_sampler.sample(i) for i in views]

            # inner optimization
            for _ in range(self.inner_steps):
                self.optimizer.zero_grad()
                losses = []
                for v, (rgb, K, w2c) in enumerate(batch):
                    self.means, self.thicknesses = self.curve_set()
                    render = self.rasterizer(
                        self.means, self.thicknesses, w2c, K,
                        self.width, self.height
                    )
                    render = self.to_image(render)
                    if v == 0:
                        shown = render          # frames show view 0 next to its own target
                    it = batch_idx * self.inner_steps
                    losses.append(self.criterion(render, rgb, it))
                train_loss_tensor = torch.stack(losses).mean()
                connect = None
                if self.connect_loss is not None and batch_idx >= self.connect_start:
                    connect = self.connect_loss(self.curve_set)
                    train_loss_tensor = train_loss_tensor + connect
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
                self._update_display(shown, batch[0][0], status)
            if self.save_dir and batch_idx % self.frame_every == 0:
                self._save_frame(batch_idx, shown, batch[0][0])

            # early stopping on the (deterministic) validation loss
            stop = False
            if val_loss is not None and self.early_stop_patience:
                if val_loss < self.best_val * (1 - self.early_stop_tol):
                    self.best_val, self.bad_evals = val_loss, 0
                else:
                    self.bad_evals += 1
                stop = self.bad_evals >= self.early_stop_patience and batch_idx + 1 >= self.min_batches
                if stop:
                    self.stopped_at = batch_idx + 1
                    print(f"Early stop at batch {batch_idx + 1}: val loss flat for {self.bad_evals} validations "
                          f"(best {self.best_val:.5f})", flush=True)

            if self.snapshot_every and ((batch_idx + 1) % self.snapshot_every == 0
                                        or batch_idx == self.total_batches - 1 or stop):
                self.snapshot(batch_idx + 1)
            self.log(batch=batch_idx + 1, total=self.total_batches, train_loss=train_loss, val_loss=val_loss,
                     n_curves=len(self.curve_set.curves), n_pruned=n_pruned,
                     connect_loss=None if connect is None else connect.item())
            if (batch_idx + 1) % self.ckpt_every == 0 or batch_idx == self.total_batches - 1 or stop:
                self.save_checkpoint(batch_idx + 1)
            if stop:
                break

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
