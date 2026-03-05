# calibration.py
import torch
import argparse
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader, Subset

import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from diffusers.models import AutoencoderKL
from PIL import Image
import random
import glob
import os
import random
import time
from typing import Optional, List, Tuple

def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

def dataloader_builder(
    dataset_or_loader,
    batchsize: int = 32,
    nsamples: int = 1024,
    same_subset: bool = True,
    base_seed: int = 42,
    shuffle: bool = False,

    # NEW: stage-balancing at the SUBSET level
    stages: Optional[List[Tuple[int,int]]] = None,   # inclusive (lo, hi)
    per_stage_rep: str = "uniform",
    plan_seed: Optional[int] = 42,
) -> DataLoader:
    """
    Build a re-iterable calibration DataLoader over a subset of size `nsamples`.

    - If same_subset=True: deterministic subset via `base_seed`.
    - If same_subset=False: fresh random subset each call.

    NEW:
    - If `stages` is provided, we create a **balanced plan over the chosen subset indices**
      and install it into the dataset via `set_fixed_t_plan_for_subset`.
    """
    # Extract dataset
    dataset = dataset_or_loader.dataset if isinstance(dataset_or_loader, DataLoader) else dataset_or_loader

    # Resolve seed for picking subset indices
    if same_subset:
        seed = int(base_seed if base_seed is not None else 0)
        rng = random.Random(seed)
    else:
        seed = int(time.time() * 1e6) & 0xFFFFFFFF
        rng = random.Random(seed)

    # Pick subset indices
    n_total = len(dataset)
    k = min(int(nsamples), n_total)
    indices = list(range(n_total))
    rng.shuffle(indices)
    indices = indices[:k]

    # If stages provided and dataset supports the override, install a subset-level plan
    if (stages is not None) and hasattr(dataset, "set_fixed_t_plan_for_subset"):
        dataset.set_fixed_t_plan_for_subset(
            subset_indices=indices,
            stages=stages,
            per_stage_rep=per_stage_rep,
            plan_seed=plan_seed,
        )

    subset = Subset(dataset, indices)
    # Keep shuffle=False for stability unless you explicitly want batch-level shuffle
    return DataLoader(subset, batch_size=batchsize, shuffle=shuffle)

def get_alpha_bar(diffusion_steps=1000, schedule='linear'):
    scale = 1000 / diffusion_steps
    if schedule == 'linear':
        beta_start = 0.0001 * scale
        beta_end = 0.02 * scale
        betas = torch.linspace(beta_start, beta_end, diffusion_steps)
        alphas = 1 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        one_tensor = torch.tensor([1.0], dtype=alpha_bars.dtype, device=alpha_bars.device)
        alpha_bars = torch.cat((one_tensor, alpha_bars), dim=0)
    elif schedule == 'scaled_linear':
        beta_start = 0.00085 * scale
        beta_end = 0.012 * scale
        betas = torch.linspace(beta_start**0.5, beta_end**0.5, diffusion_steps, dtype=torch.float32) ** 2
        alphas = 1 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        one_tensor = torch.tensor([1.0], dtype=alpha_bars.dtype, device=alpha_bars.device)
        alpha_bars = torch.cat((one_tensor, alpha_bars), dim=0)
    elif schedule == 'squaredcos_cap_v2':
        t = torch.linspace(0, diffusion_steps, diffusion_steps + 1)
        alpha_bars = torch.cos(((t / diffusion_steps) + 0.008) / 1.008 * torch.pi / 2) ** 2
    else:
        raise NotImplementedError

    return alpha_bars

class ImageDiTDataset(Dataset):
    """
    ImageNet -> VAE latents -> add noise at timestep t.

    Two ways to control t:
      (A) Default: random in [step_start, step_end) (deterministic if base_seed set)
      (B) Subset-level fixed plan: call set_fixed_t_plan_for_subset(indices, stages, ...)
          to assign a balanced stage-aware t per *selected* index only.

    Notes:
      - Stage bounds (lo, hi) are INCLUSIVE.
      - When both a per-index override and a global plan exist, per-index override wins.
    """

    def __init__(
        self,
        image_dir: str,
        vae,
        image_size: int = 256,
        num_classes: int = 1000,
        diffusion_steps: int = 1000,
        step_start: int = 0,
        step_end: int = 1000,       # exclusive
        device: str = 'cuda',

        # Legacy/random controls
        base_seed: Optional[int] = 42,   # None -> fresh randomness every epoch
    ):
        self.dataset = datasets.ImageFolder(
            image_dir,
            transform=transforms.Compose([
                transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
            ])
        )
        self.num_classes = int(num_classes)
        self.step_start = int(step_start)
        self.step_end   = int(step_end)  # exclusive
        if self.step_end <= self.step_start:
            raise ValueError(f"step_end ({self.step_end}) must be > step_start ({self.step_start}).")

        self.device = device
        self.vae = vae.to(device)

        # Precompute alpha_bar (len = diffusion_steps+1)
        self.diffusion_steps = int(diffusion_steps)
        self.alpha_bar = get_alpha_bar(diffusion_steps=self.diffusion_steps)

        # RNG for legacy random t
        self._gen = None
        if base_seed is not None:
            self._gen = torch.Generator()
            self._gen.manual_seed(int(base_seed))

        # ---- t control storage ----
        # Global plan (None by default; not used in this version)
        self._t_values_global: Optional[List[int]] = None
        # Per-index overrides installed by dataloader_builder for a subset
        self._t_by_index: Dict[int, int] = {}

    def __len__(self):
        return len(self.dataset)

    def _randint(self, low: int, high: int) -> int:
        if self._gen is None:
            return int(torch.randint(low, high, (1,)).item())
        return int(torch.randint(low, high, (1,), generator=self._gen).item())

    def _t_for_index(self, idx: int) -> int:
        # 1) Per-index override (subset plan) has highest priority
        if idx in self._t_by_index:
            return int(self._t_by_index[idx])

        # 2) Global plan (not used by builder; kept for completeness)
        if self._t_values_global is not None:
            return int(self._t_values_global[idx])

        # 3) Legacy random in [step_start, step_end)
        return self._randint(self.step_start, self.step_end)

    def __getitem__(self, idx: int):
        img, class_idx = self.dataset[idx]
        img = img.unsqueeze(0).to(self.device)

        with torch.no_grad():
            latents = self.vae.encode(img).latent_dist.sample()
            latents = latents * 0.18215
        latents = latents.squeeze(0)  # [C,H,W]

        t_val = self._t_for_index(idx)
        alpha_bar_t = self.alpha_bar[t_val]
        alpha_bar_t = torch.as_tensor([alpha_bar_t], device=(self.device if latents.is_cuda else 'cpu'))

        eps = torch.randn_like(latents)
        x_t = alpha_bar_t.sqrt() * latents + (1 - alpha_bar_t).sqrt() * eps

        t = torch.tensor([t_val])
        if class_idx < self.num_classes:
            y = torch.tensor([class_idx])
        else:
            y = torch.randint(0, self.num_classes, (1,))

        return {'x': x_t, 't': t, 'y': y}

    # ---------- NEW: install a balanced per-index plan for a subset ----------
    def set_fixed_t_plan_for_subset(
        self,
        subset_indices: List[int],
        stages: List[Tuple[int,int]],           # inclusive lo, hi
        per_stage_rep: str = "uniform",         # "uniform" | "midpoint"
        plan_seed: Optional[int] = 42,
    ):
        """
        Assign a balanced stage-aware t to the *given indices only*.
        After calling this, __getitem__(i) for i in subset_indices will use the fixed t.

        Balanced quotas: len(subset) // S  (+1 for first `rem` stages).
        """
        assert len(stages) > 0, "stages must be non-empty"
        # Clamp & normalize stages
        max_t = self.diffusion_steps - 1
        clamped = []
        for lo, hi in stages:
            lo = max(0, min(max_t, int(lo)))
            hi = max(0, min(max_t, int(hi)))
            if lo > hi:
                lo, hi = hi, lo
            clamped.append((lo, hi))
        stages = clamped

        K = len(subset_indices)
        S = len(stages)

        base = K // S
        rem  = K - base * S
        rng  = random.Random(plan_seed) if plan_seed is not None else random.Random()

        order = list(range(S))
        rng.shuffle(order)
        extra = set(order[:rem])

        # Build the t list for exactly these K slots
        t_list: List[int] = []
        for s, (lo, hi) in enumerate(stages):
            quota = base + (1 if s in extra else 0)
            if quota <= 0:
                continue
            if per_stage_rep == "midpoint":
                t_mid = (lo + hi) // 2
                t_list.extend([int(t_mid)] * quota)
            else:
                t_list.extend([rng.randint(lo, hi) for _ in range(quota)])

        # Shuffle to avoid stage blocks
        rng.shuffle(t_list)
        assert len(t_list) == K, f"internal plan size mismatch: {len(t_list)} vs {K}"

        # Assign per-index override
        for idx, t in zip(subset_indices, t_list):
            self._t_by_index[int(idx)] = int(t)


class ImageDiTCleanDataset(Dataset):
    """
    ImageNet -> VAE latents (clean x0). Use with diffusion.training_losses (which adds noise).
    """
    def __init__(
        self,
        image_dir: str,
        vae,
        image_size: int = 256,
        num_classes: int = 1000,
        device: str = 'cuda',
        base_seed: Optional[int] = 42,
        emit_t: bool = False,  # keep False; diff_pruning_dit will sample t if missing
    ):
        self.dataset = datasets.ImageFolder(
            image_dir,
            transform=transforms.Compose([
                transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
            ])
        )
        self.num_classes = int(num_classes)
        self.device = device
        self.vae = vae.to(device)
        self.emit_t = bool(emit_t)

        self._gen = None
        if base_seed is not None:
            self._gen = torch.Generator()
            self._gen.manual_seed(int(base_seed))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        img, class_idx = self.dataset[idx]
        img = img.unsqueeze(0).to(self.device)
        with torch.no_grad():
            latents = self.vae.encode(img).latent_dist.sample()
            latents = latents * 0.18215  # SD scaling
        x0 = latents.squeeze(0)

        if class_idx < self.num_classes:
            y = torch.tensor([class_idx], device=self.device)
        else:
            y = torch.randint(0, self.num_classes, (1,), device=self.device)

        batch = {'x0': x0, 'y': y}
        if self.emit_t:
            # optional: provide a t; your diff_pruning_dit samples if missing
            t = torch.randint(0, 1000, (1,), device=self.device)
            batch['t'] = t
        return batch

def get_SNR(diffusion_steps=1000, schedule='linear'):
    alpha_bars = get_alpha_bar(diffusion_steps=diffusion_steps, schedule=schedule)
    snr = alpha_bars / (1 - alpha_bars)
    log_snr = torch.log(snr)
    plt.figure(figsize=(8, 6))
    plt.xticks(fontsize=28)  # Increase fontsize of x-axis ticks
    plt.yticks(fontsize=28)  # Increase fontsize of y-axis ticks
    plt.plot(range(diffusion_steps + 1), log_snr.numpy(), marker='o', linestyle='--', color='blue', label=schedule)
    plt.xlabel("Sampling Step", fontsize=28)
    plt.ylabel("lnSNR", fontsize=28)
    # plt.title("lnSNR across Diffusion Steps for Linear Schedules", fontsize=14)
    plt.grid(False)
    plt.legend(fontsize=35)
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.savefig(f"analysis/SNR_{schedule}.png", dpi=300)
    plt.show()
    return log_snr

def main(args):
    get_SNR(diffusion_steps=1000, schedule=args.schedule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule", type=str, choices=['linear', 'scaled_linear', 'squaredcos_cap_v2'], default="linear")
    parser.add_argument("--strategy", type=str, choices=["max", "owl"], default="max")
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--target-sparsity", type=float, default=0.5)
    parser.add_argument("--scale", type=float, default=3)
    parser.add_argument("--M", type=float, default=0.55)

    args = parser.parse_args()
    main(args)
