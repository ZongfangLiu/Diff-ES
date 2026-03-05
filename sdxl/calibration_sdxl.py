# calibration_sdxl.py
import torch
import argparse
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.datasets import CocoCaptions
from torchvision import transforms
from typing import Optional, List, Tuple, Dict


def get_alpha_bar(diffusion_steps=1000, schedule='linear'):
    scale = 1000 / diffusion_steps
    if schedule == 'linear':
        beta_start = 0.0001 * scale
        beta_end = 0.02 * scale
        betas = torch.linspace(beta_start, beta_end, diffusion_steps)
        alphas = 1 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        return alpha_bars
    elif schedule == 'scaled_linear':
        beta_start = 0.00085 * scale
        beta_end = 0.012 * scale
        betas = torch.linspace(beta_start**0.5, beta_end**0.5, diffusion_steps, dtype=torch.float32) ** 2
        alphas = 1 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        return alpha_bars
    elif schedule == 'squaredcos_cap_v2':
        t = torch.linspace(0, diffusion_steps, diffusion_steps + 1)
        alpha_bars = torch.cos(((t / diffusion_steps) + 0.008) / 1.008 * torch.pi / 2) ** 2
        return alpha_bars
    else:
        raise NotImplementedError

def get_SNR(diffusion_steps=1000, schedule='squaredcos_cap_v2'):
    alpha_bars = get_alpha_bar(diffusion_steps, schedule)
    snr = alpha_bars / (1 - alpha_bars)
    log_snr = torch.log(snr)
    plt.figure(figsize=(8, 6))
    plt.plot(range(diffusion_steps + 1), log_snr.numpy(), marker='o', linestyle='--', color='blue')
    plt.xlabel("Sampling Step", fontsize=18)
    plt.ylabel("lnSNR", fontsize=18)
    plt.grid(True)
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.savefig(f"analysis/SNR_{schedule}.png", dpi=300)
    plt.show()
    return log_snr

# ============================
# SDXL COCO Dataset
# ============================
class CocoSDXLDataset(Dataset):
    def __init__(self, image_dir, ann_file, pipeline,
                 image_size=(1024, 1024), diffusion_steps=1000, step_start=0, step_end=1000,
                 base_seed: Optional[int] = 42, device='cuda', cfg: bool = True):

        self.dataset = CocoCaptions(
            root=image_dir,
            annFile=ann_file,
            transform=transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor()
            ])
        )
        self.pipeline = pipeline
        self.image_size = image_size
        self.diffusion_steps = diffusion_steps
        self.step_start = step_start
        self.step_end = step_end
        self.device = device
        self.alpha_bar = get_alpha_bar(diffusion_steps, 'scaled_linear')
        self.cfg = bool(cfg)

        self._gen = None
        if base_seed is not None:
            self._gen = torch.Generator(device=self.device).manual_seed(int(base_seed))

        self._t_by_index: Dict[int, int] = {}

    def __len__(self):
        return len(self.dataset)

    def _randint(self, low: int, high: int) -> int:
        if self._gen is None:
            return int(torch.randint(low, high, (1,), device=self.device).item())
        return int(torch.randint(low, high, (1,), generator=self._gen, device=self.device).item())

    def _t_for_index(self, idx: int) -> int:
        if idx in self._t_by_index:
            return int(self._t_by_index[idx])
        return self._randint(self.step_start, self.step_end)

    @torch.no_grad()
    def _encode_and_pack_conditions(self, prompt: str, dtype: torch.dtype):
        """
        Uses self.pipeline helpers to build SDXL conditioning.
        Returns:
          encoder_hidden_states: [B_or_2B, T, C]
          added_cond_kwargs: {"text_embeds": [B_or_2B, P], "time_ids": [B_or_2B, D]}
        """
        # 1) Encode text; ask for both cond/uncond (we’ll pack per cfg)
        pe, neg_pe, pooled, neg_pooled = self.pipeline.encode_prompt(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=None,
            clip_skip=None
        )

        # 2) SDXL extra conditioning
        time_ids = self.pipeline._get_add_time_ids(
            self.image_size, (0, 0), self.image_size,
            dtype=pe.dtype,
            text_encoder_projection_dim=self.pipeline.text_encoder_2.config.projection_dim,
        ).to(self.device)
        if time_ids.ndim == 1:
            time_ids = time_ids.unsqueeze(0)  # [1, D]

        if self.cfg:
            # Pack as [uncond, cond]
            enc_states = torch.cat([neg_pe, pe], dim=0)                 # [2, T, C]
            text_embeds = torch.cat([neg_pooled, pooled], dim=0)        # [2, P]
            time_ids = time_ids.repeat(2, 1)                            # [2, D]
        else:
            enc_states = pe                                             # [1, T, C]
            text_embeds = pooled                                        # [1, P]
            # time_ids already [1, D]

        # Cast to desired dtype if needed
        enc_states = enc_states.to(dtype=dtype, device=self.device)
        text_embeds = text_embeds.to(dtype=dtype, device=self.device)
        time_ids = time_ids.to(dtype=dtype, device=self.device)

        added = {"text_embeds": text_embeds, "time_ids": time_ids}
        return enc_states, added

    def __getitem__(self, idx):
        img, captions = self.dataset[idx]
        prompt = captions[0] if captions else "a photo"

        # ---- Encode conditions via the pipeline (inside the dataset) ----
        # we’ll use fp16 inputs to UNet usually; match that here
        unet_in_dtype = torch.float16

        with torch.no_grad():
            encoder_hidden_states, added_cond_kwargs = self._encode_and_pack_conditions(
                prompt=prompt, dtype=unet_in_dtype
            )

        # ---- VAE encode (use pipeline for correct scaling) ----
        img = img.unsqueeze(0).to(self.device)
        needs_upcast = (
            self.pipeline.vae.dtype == torch.float16
            and getattr(self.pipeline.vae.config, "force_upcast", False)
        )
        if needs_upcast:
            self.pipeline.upcast_vae()
            img = img.to(torch.float32)

        with torch.no_grad():
            z0 = self.pipeline.vae.encode(img).latent_dist.sample()
            z0 = z0 * self.pipeline.vae.config.scaling_factor          # [1, C, H', W']

        if needs_upcast:
            self.pipeline.vae.to(dtype=torch.float16)

        # ---- Sample timestep and construct x_t (no CFG mixing here) ----
        t_val = self._t_for_index(idx)
        alpha_bar_t = torch.tensor([self.alpha_bar[t_val]], device=self.device, dtype=z0.dtype)

        if self.cfg:
            latents_in = z0.repeat(2, 1, 1, 1)                         # [2, C, H', W']
        else:
            latents_in = z0                                            # [1, C, H', W']

        eps = torch.randn(
            latents_in.shape,
            device=self.device,
            dtype=latents_in.dtype,
            generator=self._gen,
        )

        x_t = alpha_bar_t.sqrt() * latents_in + (1.0 - alpha_bar_t).sqrt() * eps
        x_t = self.pipeline.scheduler.scale_model_input(x_t, t_val)
        
        t_tensor = torch.tensor([t_val], device=self.device, dtype=torch.long)

        # when cfg=True, duplicate it once (for [uncond, cond])
        if self.cfg:
            t_tensor = torch.cat([t_tensor, t_tensor], dim=0)  # [2]

        return {
            "x": x_t.to(unet_in_dtype),
            "t": t_tensor,
            "encoder_hidden_states": encoder_hidden_states,              # UNet arg
            "added_cond_kwargs": added_cond_kwargs,                      # UNet kwarg
            "cfg": self.cfg,
            "eps": eps.to(unet_in_dtype),
        }

    def set_fixed_t_plan_for_subset(self, subset_indices: List[int], stages: List[Tuple[int, int]],
                                     per_stage_rep: str = "uniform", plan_seed: Optional[int] = 42):
        max_t = self.diffusion_steps - 1
        clamped = [(max(0, min(max_t, lo)), max(0, min(max_t, hi))) for lo, hi in stages]
        stages = [(min(lo, hi), max(lo, hi)) for lo, hi in clamped]

        K, S = len(subset_indices), len(stages)
        base = K // S
        rem = K - base * S
        rng = random.Random(plan_seed)

        order = list(range(S))
        rng.shuffle(order)
        extra = set(order[:rem])

        t_list = []
        for s, (lo, hi) in enumerate(stages):
            quota = base + (1 if s in extra else 0)
            if per_stage_rep == "midpoint":
                t_mid = (lo + hi) // 2
                t_list.extend([t_mid] * quota)
            else:
                t_list.extend([rng.randint(lo, hi) for _ in range(quota)])

        rng.shuffle(t_list)
        for idx, t in zip(subset_indices, t_list):
            self._t_by_index[int(idx)] = int(t)
            
# ============================
# DataLoader Builder
# ============================
def dataloader_builder(dataset: Dataset, batchsize: int = 32, nsamples: int = 1024,
                       same_subset: bool = True, base_seed: int = 42,
                       stages: Optional[List[Tuple[int, int]]] = None,
                       per_stage_rep: str = "uniform", plan_seed: Optional[int] = 42, shuffle = False) -> DataLoader:

    if same_subset:
        rng = random.Random(base_seed)
    else:
        rng = random.Random(int(time.time() * 1e6) & 0xFFFFFFFF)

    total = len(dataset)
    indices = list(range(total))
    rng.shuffle(indices)
    indices = indices[:min(nsamples, total)]

    if stages is not None and hasattr(dataset, 'set_fixed_t_plan_for_subset'):
        dataset.set_fixed_t_plan_for_subset(
            subset_indices=indices,
            stages=stages,
            per_stage_rep=per_stage_rep,
            plan_seed=plan_seed
        )

    return DataLoader(Subset(dataset, indices), batch_size=batchsize, shuffle=shuffle)

# ============================
# CLI Entry Point
# ============================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--schedule', type=str, default='squaredcos_cap_v2')
    args = parser.parse_args()
    get_SNR(schedule=args.schedule)