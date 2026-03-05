#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Callable, Dict, List, Optional, Tuple, Literal, Any
import copy, hashlib, logging, random, json, math
import inspect

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

# ---- your utilities (unchanged imports) ----
from evo_pruning_utils_sdxl import (
    build_layerdrop_schedule_from_orders,
    build_secondorder_schedule_from_orders,
    apply_secondorder_schedule,
    select_obs_bank_for_ratios,
)

logger = logging.getLogger(__name__)

from typing import Callable, Dict, List, Optional, Tuple, Literal, Any
import logging, math
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

MetricType = Literal[
    "latent_mse","latent_abs","latent_cos",
    "img_mse","img_ssim","img_fft","latent_abs_fft","fft_cos_bound",
    # NR-IQA (pyiqa-backed)
    "img_niqe","img_clipiqa","img_topiq","img_hyperiqa","img_liqe","img_qalign","img_qualiclip"
]

class FitnessFinalSDXL:
    """
    Final-only teacher–student fitness using StableDiffusionXLPipeline, with
    deterministic latents controlled purely by torch.Generator seeds (no manual latents).

    Determinism:
      - For every call, we construct a list of K generators, one per image,
        with seeds: base_seed + latent_seed_offset + i.
      - Teacher and student calls both build generators from the same plan,
        so they see identical noise.

    To refresh probes (i.e., change latents/prompts), call refresh_probes(gen=...).
    """

    _NR_METRICS = ("img_niqe","img_clipiqa","img_topiq","img_hyperiqa","img_liqe","img_qalign","img_qualiclip")
    _PYIQA_ALIASES: Dict[str, List[str]] = {
        "img_niqe":      ["niqe"],
        "img_clipiqa":   ["clipiqa", "clipiqa+","clipiqa+_vitb16","clipiqa+_vitl14"],
        "img_topiq":     ["topiq", "topiq_nr", "topiq-fr", "topiq-nr", "topiq_fr"],
        "img_hyperiqa":  ["hyperiqa"],
        "img_liqe":      ["liqe"],
        "img_qalign":    ["qalign","q-align","q-align-nr"],
        "img_qualiclip": ["qualiclip","quali-clip","clip-iqa-itm"],
    }
    _LOWER_IS_BETTER = {"img_niqe"}

    def __init__(
        self,
        *,
        pipe,
        num_steps: int,
        stages,
        cfg_scale: Optional[float],
        probe_batch: int,
        height: Optional[int] = None,
        width: Optional[int] = None,
        base_seed: int = 1234,
        context_provider=None,
        metric: "MetricType" = "latent_mse",
        iqa_device: Optional[str] = None,
        iqa_options: Optional[Dict[str, Any]] = None,
        fft_highpass_radius_frac: float = 0.25,
        fft_zero_mean: bool = True,
        abs_fft_weight: float = 0.5,
        cos_lower_bound: float = 0.88,
    ):
        self.pipe = pipe
        self.num_steps = int(num_steps)
        self.cfg_scale = (None if cfg_scale is None else float(cfg_scale))
        self.K = int(probe_batch)
        self.base_seed = int(base_seed)
        self.latent_seed_offset = 0  # increases when you call refresh_probes(gen=...)
        self.context_provider = context_provider
        self.stages = stages

        # size defaults to pipeline default (SDXL: default_sample_size * vae_scale_factor)
        default_hw = self.pipe.default_sample_size * self.pipe.vae_scale_factor
        self.height = int(height if height is not None else default_hw)
        self.width  = int(width  if width  is not None else default_hw)

        valid_names = {
            "latent_mse","latent_abs","latent_cos",
            "img_mse","img_ssim","img_fft","latent_abs_fft","fft_cos_bound",
            *self._NR_METRICS
        }
        if metric not in valid_names:
            raise ValueError(f"Unknown metric: {metric}")
        self.metric = metric

        # FFT / combo knobs
        assert 0.0 < fft_highpass_radius_frac < 0.5
        self.fft_highpass_radius_frac = float(fft_highpass_radius_frac)
        self.fft_zero_mean = bool(fft_zero_mean)
        assert 0.0 <= abs_fft_weight <= 1.0
        self.abs_fft_weight = float(abs_fft_weight)
        self.cos_lower_bound = float(cos_lower_bound)

        # caches (teacher outputs)
        self._prompts: Optional[Dict[str, Any]] = None
        self._teacher_latent: Optional[torch.Tensor] = None
        self._teacher_image: Optional[torch.Tensor] = None  # BCHW [0,1]

        # NR-IQA registry
        self._iqa_device = iqa_device
        self._iqa_options = dict(iqa_options or {})
        self._iqa_models: Dict[str, Any] = {}
        self._iqa_ready: Dict[str, bool] = {}

        # Build initial probes immediately
        self.refresh_probes()

        if self.metric in self._NR_METRICS:
            self._ensure_iqa_metric(self.metric)

        self._debug_levels_str: Optional[str] = None

    # ---------- Public API ----------
    def refresh_probes(self, gen: Optional[int] = None):
        """
        Refresh prompt strings and, optionally, the latent seed plan.

        If gen is not None, we bump latent_seed_offset so subsequent runs use new noise.
        """
        if gen is not None:
            self.latent_seed_offset = int(gen) * 10000
        self._build_prompts()   # strings only
        self._teacher_latent = None
        self._teacher_image = None

    def __call__(self, _unet_ignored: nn.Module, schedule: Dict) -> float:
        needs_t_lat   = self.metric in {"latent_mse","latent_abs","latent_cos","latent_abs_fft","fft_cos_bound"}
        needs_t_image = self.metric in {"img_mse","img_ssim"}
        needs_s_lat   = self.metric in {"latent_mse","latent_abs","latent_cos","latent_abs_fft","fft_cos_bound"}
        needs_s_image = (
            self.metric in {"img_mse","img_ssim","img_fft"} or
            self.metric in self._NR_METRICS or
            self.metric in {"latent_abs_fft","fft_cos_bound"}
        )

        if needs_t_lat:
            t_lat = self._get_teacher_latent()
        if needs_t_image:
            t_img = self._get_teacher_image()

        obs_bank = getattr(self.pipe.unet, "_pending_proj_bank", None) if hasattr(self.pipe.unet, "_pending_proj_bank") else None

        s_lat = None
        s_img = None
        if needs_s_lat:
            s_lat = self._run_pipeline(output_type="latent", schedule=schedule, obs_bank=obs_bank)
        if needs_s_image:
            s_img = self._run_pipeline(output_type="np", schedule=schedule, obs_bank=obs_bank)
            # if getattr(self, "_debug_levels_str", None):
            #     try:
            #         from pathlib import Path
            #         out_dir = Path("./debug") / str(self._debug_levels_str)
            #         self._save_tensor_images(s_img, out_dir)
            #     except Exception as _e:
            #         logger.exception("Failed to save debug images to ./debug/%s: %s",
            #                          str(self._debug_levels_str), str(_e))
            #     finally:
            #         self._debug_levels_str = None

        if self.metric == "latent_mse":
            return -float(((t_lat - s_lat) ** 2).mean().item())
        if self.metric == "latent_abs":
            return -float((t_lat - s_lat).abs().mean().item())
        if self.metric == "latent_cos":
            return float(self._cosine_batch(t_lat, s_lat).mean().item())
        if self.metric == "img_mse":
            return -float(((t_img - s_img) ** 2).mean().item())
        if self.metric == "img_ssim":
            return float(self._ssim_batch(t_img, s_img).mean().item())
        if self.metric == "img_fft":
            return float(self._fft_highfreq_fraction(s_img, self.fft_highpass_radius_frac, self.fft_zero_mean).mean().item())
        if self.metric == "latent_abs_fft":
            mae_neg = -float((t_lat - s_lat).abs().mean().item())
            fft_v = self._fft_highfreq_fraction(s_img, self.fft_highpass_radius_frac, self.fft_zero_mean).mean().item()
            return float(self.abs_fft_weight * mae_neg + (1.0 - self.abs_fft_weight) * fft_v)
        if self.metric == "fft_cos_bound":
            cosv = float(self._cosine_batch(t_lat, s_lat).mean().item())
            if cosv < self.cos_lower_bound:
                return -1e6 + cosv
            fft_v = self._fft_highfreq_fraction(s_img, self.fft_highpass_radius_frac, self.fft_zero_mean).mean().item()
            return float((cosv - self.cos_lower_bound) + 0.1 * fft_v)

        if self.metric in self._NR_METRICS:
            metric_fn = self._ensure_iqa_metric(self.metric)
            dev = getattr(self._iqa_models[self.metric], "device", self._device())
            with torch.no_grad():
                score = metric_fn(s_img.to(device=dev, dtype=torch.float32))
            score = score.mean().item() if torch.is_tensor(score) else float(score)
            return -score if self.metric in self._LOWER_IS_BETTER else score

        raise ValueError(f"Unknown metric: {self.metric}")

    # ---------- Internals ----------
    def _device(self) -> torch.device:
        return self.pipe._execution_device

    def _unet_dtype(self) -> torch.dtype:
        return next(self.pipe.unet.parameters()).dtype

    def _seed_for_index(self, i: int) -> int:
        # Per-image seed to ensure identical latents across runs
        return int(self.base_seed + self.latent_seed_offset + i)

    def _make_generators(self) -> List[torch.Generator]:
        dev = self._device()
        return [torch.Generator(device=dev).manual_seed(self._seed_for_index(i)) for i in range(self.K)]

    def _build_prompts(self):
        """Store prompt strings only; let the pipeline handle encoding/CFG packing."""
        P = {"prompt": [""] * self.K,
             "prompt_2": None,
             "negative_prompt": None,
             "negative_prompt_2": None}
        if self.context_provider is not None:
            try:
                ctx = self.context_provider(K=self.K, seed=self.base_seed + self.latent_seed_offset) or {}
                for k in P.keys():
                    v = ctx.get(k, None)
                    if v is None:
                        continue
                    if isinstance(v, str):
                        v = [v] * self.K
                    P[k] = v
            except Exception as e:
                logger.exception("context_provider failed; falling back to empty prompts. Err: %s", str(e))
        self._prompts = P

    def _run_pipeline(self, *, output_type: Literal["latent","np"], schedule: Optional[Dict], obs_bank: Optional[Dict]):
        P = self._prompts or {"prompt": ["" for _ in range(self.K)],
                              "prompt_2": None,
                              "negative_prompt": None,
                              "negative_prompt_2": None}

        # IMPORTANT: do NOT pass latents; pass deterministic generators instead.
        # A list of generators preserves per-image determinism under CFG packing.
        gens = self._make_generators()

        kwargs: Dict[str, Any] = dict(
            prompt=P["prompt"],
            prompt_2=P["prompt_2"],
            negative_prompt=P["negative_prompt"],
            negative_prompt_2=P["negative_prompt_2"],
            num_inference_steps=self.num_steps,
            guidance_scale=(self.cfg_scale if self.cfg_scale is not None else 1.0),
            generator=gens,            # <= here
            latents=None,              # <= do NOT supply latents
            height=self.height,
            width=self.width,
            output_type=("latent" if output_type == "latent" else "np"),
            return_dict=True,
        )
        try:
            self._apply_schedule_and_bank(self.pipe.unet, schedule, obs_bank, self.stages)
            out = self.pipe(**kwargs)
        finally:
            self._clear_schedule_and_bank(self.pipe.unet)

        if output_type == "latent":
            lat = out.images
            if not torch.is_tensor(lat):
                raise RuntimeError("Expected latents tensor when output_type='latent'.")
            return lat

        imgs = out.images
        if isinstance(imgs, list):
            import numpy as _np
            imgs = _np.stack(imgs, axis=0)
        if not isinstance(imgs, np.ndarray):
            raise RuntimeError("Expected numpy array for output_type='np'.")

        t = torch.from_numpy(imgs)
        if t.dtype != torch.float32:
            t = t.float()
        if t.max() > 1.5:
            t = t / 255.0
        t = t.permute(0, 3, 1, 2).contiguous()
        return t.to(dtype=torch.float32, device=self._device())

    def _get_teacher_latent(self) -> torch.Tensor:
        if self._teacher_latent is None:
            self._teacher_latent = self._run_pipeline(output_type="latent", schedule=None, obs_bank=None)
        return self._teacher_latent

    def _get_teacher_image(self) -> torch.Tensor:
        if self._teacher_image is None:
            self._teacher_image = self._run_pipeline(output_type="np", schedule=None, obs_bank=None)
        return self._teacher_image

    # ---------- metrics helpers ----------
    @staticmethod
    def _cosine_batch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        af = a.reshape(a.size(0), -1); bf = b.reshape(b.size(0), -1)
        an = torch.nn.functional.normalize(af, dim=1)
        bn = torch.nn.functional.normalize(bf, dim=1)
        return (an * bn).sum(dim=1)

    @staticmethod
    def _gaussian_window(window_size: int = 11, sigma: float = 1.5, channel: int = 3, device=None, dtype=None):
        coords = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1)/2.0
        g = torch.exp(-(coords**2) / (2 * sigma * sigma))
        w = (g / g.sum()).unsqueeze(0)
        w = (w.t() @ w); w = w / w.sum()
        return w.expand(channel,1,window_size,window_size).contiguous()

    @staticmethod
    def _conv2d_same(x, w):
        pad = w.size(-1)//2
        return torch.nn.functional.conv2d(x, w, padding=pad, groups=x.size(1))

    @classmethod
    def _ssim_batch(cls, x: torch.Tensor, y: torch.Tensor, window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
        C1, C2 = 0.01**2, 0.03**2
        B, C, H, W = x.shape
        xd = x.to(dtype=torch.float32); yd = y.to(dtype=torch.float32)
        w = cls._gaussian_window(window_size, sigma, C, device=xd.device, dtype=xd.dtype)
        mu_x = cls._conv2d_same(xd, w); mu_y = cls._conv2d_same(yd, w)
        mu_x2, mu_y2, mu_xy = mu_x*mu_x, mu_y*mu_y, mu_x*mu_y
        sig_x2 = cls._conv2d_same(xd*xd, w) - mu_x2
        sig_y2 = cls._conv2d_same(yd*yd, w) - mu_y2
        sig_xy = cls._conv2d_same(xd*yd, w) - mu_xy
        ssim_map = ((2*mu_xy + C1) * (2*sig_xy + C2)) / ((mu_x2 + mu_y2 + C1) * (sig_x2 + sig_y2 + C2))
        return ssim_map.mean(dim=(1,2,3))

    @staticmethod
    def _fft_highfreq_fraction(img: torch.Tensor, radius_frac: float = 0.25, zero_mean: bool = True, eps: float = 1e-12) -> torch.Tensor:
        assert 0.0 < radius_frac <= 0.5
        B, C, H, W = img.shape
        x = img - img.mean(dim=(2,3), keepdim=True) if zero_mean else img
        F = torch.fft.fft2(x, dim=(-2,-1))
        F = torch.fft.fftshift(F, dim=(-2,-1))
        P = (F.real**2 + F.imag**2)
        cy, cx = H//2, W//2
        yy = torch.arange(H, device=img.device).view(-1,1).expand(H,W)
        xx = torch.arange(W, device=img.device).view(1,-1).expand(H,W)
        r = torch.sqrt((yy - cy).float()**2 + (xx - cx).float()**2)
        r_max = torch.sqrt(torch.tensor((H/2.0)**2 + (W/2.0)**2, device=img.device, dtype=torch.float32))
        r_cut = float(radius_frac) * r_max
        hp = (r >= r_cut).to(P.dtype)
        total = P.sum(dim=(1,2,3)).clamp_min(eps)
        high = (P * hp).sum(dim=(1,2,3))
        return (high / total).clamp(0.0, 1.0)

    # ---------- pyiqa glue ----------
    def _ensure_iqa_metric(self, name: str) -> Callable[[torch.Tensor], torch.Tensor]:
        if name in self._iqa_ready and self._iqa_ready[name]:
            return self._iqa_models[name]
        try:
            import pyiqa
        except Exception as e:
            logger.warning("pyiqa not available; NR metric '%s' cannot be used.", name)
            raise
        aliases = self._PYIQA_ALIASES.get(name, [])
        last_err = None
        dev = self._iqa_device or str(self._device())
        for cand in aliases:
            try:
                model = pyiqa.create_metric(cand, device=dev, as_loss=False, **self._iqa_options)
                def _fn(x, _model=model):
                    return _model(x)
                self._iqa_models[name] = _fn
                self._iqa_ready[name] = True
                logger.info("Registered NR-IQA metric '%s' via pyiqa model '%s' on '%s'.", name, cand, dev)
                return _fn
            except Exception as err:
                last_err = err
                continue
        raise RuntimeError(f"Could not initialize pyiqa metric for '{name}'. Tried {aliases}. Last error: {last_err}")

    # ---------- schedule/bank helpers (unchanged) ----------
    @staticmethod
    def _apply_schedule_and_bank(
        unet: nn.Module,
        schedule: Optional[Dict],
        obs_bank: Optional[Dict],
        stages: Optional[List[Tuple[int,int]]] = None
    ):
        """
        Always reset first via unet.clear_all_accel(), then apply exactly one schedule
        (layerdrop OR second-order struct) plus an optional OBS projection bank.
        """
        # Hard reset
        if hasattr(unet, "clear_all_accel"):
            unet.clear_all_accel()

        # Apply schedule if provided
        if schedule is not None:
            # Heuristic to detect second-order struct schedules:
            # A dict-of-dicts where inner dict has keys like attn/attn1/attn2/mlp.
            is_second_order = False
            if isinstance(schedule, dict) and schedule:
                any_t_val = next(iter(schedule.values()))
                if isinstance(any_t_val, dict):
                    if {"attn", "attn1", "attn2", "mlp"} & set(any_t_val.keys()):
                        is_second_order = True

            if is_second_order:
                # Stage-based second-order
                if hasattr(unet, "set_secondorder_stage_schedule"):
                    unet.set_secondorder_stage_schedule(schedule, stages=stages)
                else:
                    raise RuntimeError("UNet missing set_secondorder_stage_schedule(schedule).")
            else:
                if hasattr(unet, "set_layerdrop_schedule"):
                    unet.set_layerdrop_schedule(schedule, stages)
                else:
                    raise RuntimeError("UNet missing set_layerdrop_schedule(schedule).")

        # Apply OBS bank if provided
        if obs_bank is not None and hasattr(unet, "set_projection_bank"):
            if stages is not None:
                unet.set_projection_bank(obs_bank, stages=stages)
            else:
                unet.set_projection_bank(obs_bank)

    @staticmethod
    def _clear_schedule_and_bank(unet: nn.Module):
        # One call to rule them all.
        if hasattr(unet, "clear_all_accel"):
            unet.clear_all_accel()

    # ---------- misc ----------
    @staticmethod
    def _save_tensor_images(img: torch.Tensor, out_dir: "Path"):
        out_dir.mkdir(parents=True, exist_ok=True)
        x = img.detach().to("cpu", dtype=torch.float32)
        if x.max() > 1.5:
            x = x / 255.0
        x = x.clamp(0, 1)
        B = x.shape[0]
        from PIL import Image
        for i in range(B):
            arr = (x[i] * 255.0).round().to(torch.uint8).permute(1, 2, 0).numpy()
            Image.fromarray(arr).save(out_dir / f"{i:03d}.png")


# ======================================================================================
# Evolutionary search (LEVELS ONLY) — unchanged surface
# ======================================================================================

class EvoLayerDropSearchSDXL:
    """
    Levels-only EA over stages for SDXL UNet.

    Note: `eval_fn` is expected to have signature `eval_fn(unet, schedule) -> float`,
    and we pass `self.unet` in. The provided FitnessFinalSDXL keeps this signature by
    ignoring the unet arg.
    """

    def __init__(
        self,
        unet: nn.Module,
        stages: List[Tuple[int, int]],
        *,
        rng_seed: int = 42,
        verbose: bool = True,
        mode: str = "layerdrop",            # "layerdrop" | "secondorder"
        mode_kwargs: Optional[Dict] = None, # so_head_dim, so_num_heads, so_protect_ends
        H_override: Optional[int] = None,   # if you want to force H (cap)
    ):
        self.unet = unet
        self.stages = stages
        self.S = len(stages)
        self.verbose = verbose

        random.seed(rng_seed); np.random.seed(rng_seed); torch.manual_seed(rng_seed)

        # payloads
        self.orders_per_stage: Dict[int, List[int]] = {}  # for layerdrop
        self.scores_per_stage: Dict[int, List[float]] = {}
        self.so_orders: Optional[Dict] = None             # for secondorder

        assert mode in ("layerdrop", "secondorder")
        self.mode = mode
        self.mode_kwargs = mode_kwargs or {}
        self.so_head_dim = int(self.mode_kwargs.get("so_head_dim", 64))
        self.so_num_heads = int(self.mode_kwargs.get("so_num_heads", 16))
        self.so_protect_ends = int(self.mode_kwargs.get("so_protect_ends", 0))
        # Default to 1000, so it's easy to find bug
        self.H = int(self.mode_kwargs.get("H", 1000))

        # Level cap H
        if H_override is not None:
            self.H = int(max(1, H_override))
        # else:
        #     self.H = (self.so_num_heads if self.mode == "secondorder" else len(self.orders_per_stage[0]))

        # OBS
        self.obs_repo: Optional[dict] = None
        self.obs_round_mode = "nearest"
        self._bank_cache = {}
        self._bank_cache_max = 0

        # enable struct forward path if available
        if self.mode == "secondorder" and hasattr(self.unet, "enable_struct_prune_forward"):
            try:
                self.unet.enable_struct_prune_forward()
            except Exception:
                pass

        # fitness memoization
        self._fitness_cache: Dict[str, float] = {}
        self._fitness_cache_val: Dict[str, float] = {}

        # middle results
        self.middle_dir: Optional[Path] = None

    # ---------------- OBS helpers ----------------
    def set_obs_repo(self, repo):
        self.obs_repo = repo

    def set_obs_round_mode(self, mode: str = "nearest"):
        assert mode in ("nearest", "floor", "ceil")
        self.obs_round_mode = mode

    def _bank_key(self, ratios_dict):
        return tuple(sorted((int(k), float(v)) for k, v in ratios_dict.items()))

    def _get_or_make_cpu_bank(self, ratios_dict):
        key = self._bank_key(ratios_dict)

        # If cache disabled, just build every time
        if self._bank_cache_max <= 0:
            return select_obs_bank_for_ratios(
                repo=self.obs_repo,
                ratios=ratios_dict,
                stages=self.stages,
                round_mode=self.obs_round_mode,
            )

        bank = self._bank_cache.get(key)
        if bank is None:
            bank = select_obs_bank_for_ratios(
                repo=self.obs_repo,
                ratios=ratios_dict,
                stages=self.stages,
                round_mode=self.obs_round_mode,
            )
            if len(self._bank_cache) >= self._bank_cache_max:
                self._bank_cache.pop(next(iter(self._bank_cache)))
            self._bank_cache[key] = bank
        return bank

    def _evaluate_candidate(self, levels_per_stage, eval_fn, schedule):
        H = max(1, int(self.H))
        ratios = {int(s): float(max(0, min(H, int(levels_per_stage.get(s, 0))))) / float(H)
                  for s in range(len(self.stages))}
        try:
            # ---- DEBUG: set levels string only; fitness will write to ./debug/{levels}/
            levels_str = "_".join(str(int(levels_per_stage.get(s, 0))) for s in range(self.S))
            try:
                setattr(eval_fn, "_debug_levels_str", levels_str)
            except Exception:
                pass

            # OBS bank (second-order)
            if self.mode == "secondorder" and self.obs_repo and hasattr(self.unet, "set_projection_bank"):
                bank_cpu = self._get_or_make_cpu_bank(ratios)
                setattr(self.unet, "_pending_proj_bank", bank_cpu)

            return eval_fn(self.unet, schedule)
        finally:
            if hasattr(self.unet, "_pending_proj_bank"):
                try:
                    delattr(self.unet, "_pending_proj_bank")
                except Exception:
                    pass

    # ---------------- calibration setters ----------------
    def set_layerdrop_orders(self, orders_per_stage: Dict[int, List[int]], scores_per_stage: Dict[int, List[float]]) -> None:
        self.orders_per_stage = orders_per_stage
        self.scores_per_stage = scores_per_stage

    def set_secondorder_orders(self, so_orders: Dict) -> None:
        self.so_orders = so_orders

    # ---------------- main search ----------------
    def run(
        self,
        *,
        generations: int,
        offspring: int,
        target_level: int,
        survivors_per_selection: List[int],
        eval_fn: Callable[[nn.Module, Dict], float],
        start_level: Optional[int] = None,
        mutation_max_levels: int = 2,
        mutation_max_times: int = 5,
        init_population: Optional[List[Dict[int, int]]] = None,
        patience: Optional[int] = None,
        eval_fn_val: Optional[Callable[[nn.Module, Dict], float]] = None,
        log_dir: Optional[Path] = None,
        log_float_precision: int = 4,
        log_every_gen: bool = True,
        single_log_name: str = "evolution.json",
        refresh_every: int = 0,     # ← NEW: 0 = never, N = every N generations
    ) -> Tuple[Dict[int, int], Dict, float]:

        if self.mode == "layerdrop":
            assert self.orders_per_stage, "Call set_layerdrop_orders(...) before run()."
        else:
            assert self.so_orders is not None, "Call set_secondorder_orders(...) before run()."

        TL = int(max(0, min(self.H, int(target_level))))
        SL = None if start_level is None else int(max(0, min(self.H, int(start_level))))
        max_lv = int(max(1, int(mutation_max_levels)))

        stage_sizes = [int(v) for v in survivors_per_selection]
        first_stage_size = stage_sizes[0] + offspring

        evo_log_dir = self._ensure_log_dir(log_dir)
        log_out_path = evo_log_dir / single_log_name
        evo_all = {
            "meta": {
                "mode": self.mode,
                "stages": self.stages,
                "generations": int(generations),
                "offspring": int(offspring),
                "survivors_per_selection": [int(x) for x in stage_sizes],
                "mutation_max_levels": int(max_lv),
                "target_level_int": int(TL),
                "start_level_int": (None if SL is None else int(SL)),
                "H": int(self.H),
                "k_semantics": "levels_per_stage",
            },
            "generations": []
        }

        if init_population is None:
            population = [self._init_levels_total(TL) for _ in range(first_stage_size)]
            if SL is not None and SL != TL:
                population[0] = self._init_levels_total(SL)
        else:
            pop_raw: List[Dict[int, int]] = []
            for k in init_population:
                r = {s: int(max(0, min(self.H, int(k.get(s, 0))))) for s in range(self.S)}
                pop_raw.append(self._retarget_sum(r, target_sum=TL * self.S))
            population = pop_raw[:first_stage_size]
            while len(population) < first_stage_size:
                population.append(self._init_levels_total(TL))

        best_score = -float("inf")
        best_L: Optional[Dict[int, int]] = None
        best_sched: Optional[Dict] = None
        stale = 0

        for g in range(generations):
            # ----- NEW: refresh probes at the cadence you want -----
            if refresh_every and (g % int(refresh_every) == 0) and hasattr(eval_fn, "refresh_probes"):
                # Use (g+1) so gen 0 (initial) stays the base plan unless you also refresh there.
                eval_fn.refresh_probes(gen=g + 1)
                
            if self.verbose:
                logger.info("[Generation %d/%d] avg_target_level=%d (H=%d)", g + 1, generations, TL, self.H)

            gen_log = {"generation": g + 1, "target_level": int(TL), "selections": []}

            pool = population
            for sel_idx, survivors_n in enumerate(stage_sizes):
                if self.verbose:
                    logger.info(
                        "  [Selection %d/%d] pool=%d → survivors=%d (+offspring=%d)",
                        sel_idx + 1, len(stage_sizes), len(pool), survivors_n,
                        (offspring if sel_idx < len(stage_sizes) - 1 else 0)
                    )

                evaluated: List[Tuple[float, Dict[int, int]]] = []
                for indiv in pool:
                    score = self._fitness_L_with_cache(indiv, eval_fn, self._fitness_cache)
                    print(f"    Evaluated indiv levels={self._levels_list(indiv)} score={score:.6f}")
                    evaluated.append((score, indiv))

                evaluated.sort(key=lambda x: x[0], reverse=True)
                survivors = [copy.deepcopy(ind) for (_, ind) in evaluated[:survivors_n]]

                sel_log = {
                    "selection_index": sel_idx,
                    "pool_size": len(evaluated),
                    "survivors_n": survivors_n,
                    "pool": [
                        {
                            "score": round(float(score), 6),
                            "levels": self._levels_list(ind),
                            "ratios": self._ratios_list(ind, prec=log_float_precision),
                        } for (score, ind) in evaluated
                    ],
                    "survivors": [
                        {
                            "score": round(float(evaluated[i][0]), 6) if i < len(evaluated) else None,
                            "levels": self._levels_list(survivors[i]),
                            "ratios": self._ratios_list(survivors[i], prec=log_float_precision),
                        } for i in range(len(survivors))
                    ],
                }
                gen_log["selections"].append(sel_log)

                if evaluated and evaluated[0][0] > best_score:
                    best_score = evaluated[0][0]
                    best_L = copy.deepcopy(evaluated[0][1])
                    best_sched = self._L_to_schedule(best_L)
                    stale = 0
                    if getattr(self, "middle_dir", None):
                        try:
                            self.middle_dir.mkdir(parents=True, exist_ok=True)
                            gi = g + 1
                            with open(self.middle_dir / f"schedule_gen{gi:03d}.json", "w") as f:
                                json.dump(self._to_py(best_sched), f, indent=2)
                            with open(self.middle_dir / f"levels_per_stage_gen{gi:03d}.json", "w") as f:
                                json.dump(self._to_py(best_L), f, indent=2)
                            logger.info("  [MID] Saved middle_results for gen=%d (score=%.6f)", gi, best_score)
                        except Exception as _e:
                            logger.exception("  [MID] Save failed: %s", str(_e))
                else:
                    stale += 1

                if self.verbose and evaluated:
                    top_train = evaluated[0][0]
                    if eval_fn_val is not None:
                        val_evals: List[Tuple[float, Dict[int, int]]] = []
                        for _, indiv in evaluated:
                            s_val = self._fitness_L_with_cache(indiv, eval_fn_val, self._fitness_cache_val)
                            val_evals.append((s_val, indiv))
                        val_evals.sort(key=lambda x: x[0], reverse=True)
                        top_val = val_evals[0][0]
                        same_top = (self._L_key(evaluated[0][1]) == self._L_key(val_evals[0][1]))
                        logger.info("    top train=%.6f | top val=%.6f | best train=%.6f | same_top=%s",
                                    top_train, top_val, best_score, "YES" if same_top else "NO")
                    else:
                        logger.info("    top train=%.6f | best train=%.6f", top_train, best_score)

                if sel_idx < len(stage_sizes) - 1:
                    children = [self._mutate_levels(random.choice(survivors), max_lv, mutation_max_times)
                                for _ in range(offspring)]
                    pool = survivors + children
                else:
                    next_seed = survivors
                    extra = first_stage_size - len(next_seed)
                    children = [self._mutate_levels(random.choice(next_seed), max_lv, mutation_max_times)
                                for _ in range(max(0, extra))]
                    population = next_seed + children

            if best_L is not None:
                gen_log["best_so_far"] = {
                    "score": round(float(best_score), 6),
                    "levels": self._levels_list(best_L),
                    "ratios": self._ratios_list(best_L, prec=log_float_precision),
                }

            evo_all["generations"].append(gen_log)
            if log_every_gen:
                try:
                    with open(log_out_path, "w") as f:
                        json.dump(evo_all, f, indent=2)
                    if self.verbose:
                        logger.info("  [LOG] updated %s (gen %03d)", str(log_out_path), g + 1)
                except Exception as _e:
                    logger.exception("  [LOG] failed to update %s: %s", str(log_out_path), str(_e))

            if patience is not None and stale >= patience:
                if self.verbose:
                    logger.info("[Early stop] No improvement for %d selection steps.", patience)
                break

        assert best_L is not None and best_sched is not None, "Search produced no candidate."
        return best_L, best_sched, float(best_score)

    # ---------------- internals ----------------
    def _fitness_L_with_cache(self, L: Dict[int, int], eval_fn, cache: Dict[str, float]) -> float:
        key = self._L_key(L)
        if key in cache:
            return cache[key]

        schedule = self._L_to_schedule(L)
        score = self._evaluate_candidate(L, eval_fn, schedule)
        cache[key] = score
        return score

    def _L_to_schedule(self, L: Dict[int, int]) -> Dict:
        ratios = {s: float(max(0, min(self.H, L.get(s, 0)))) / float(max(1, self.H)) for s in range(self.S)}
        if self.mode == "layerdrop":
            return build_layerdrop_schedule_from_orders(
                orders_per_stage=self.orders_per_stage,
                stages=self.stages,
                ratios=ratios,
                protect_ends=0,
            )
        else:
            return build_secondorder_schedule_from_orders(
                orders_per_stage=self.so_orders,
                stages=self.stages,
                ratios=ratios,
                # head_dim=self.so_head_dim,
                # num_heads=self.so_num_heads,
                protect_ends=self.so_protect_ends,
            )

    def _levels_list(self, L: Dict[int, int]) -> List[int]:
        return [int(max(0, min(self.H, L.get(s, 0)))) for s in range(self.S)]

    def _ratios_list(self, L: Dict[int, int], prec: int = 4) -> List[float]:
        H = float(max(1, self.H))
        return [round(float(max(0, min(self.H, L.get(s, 0)))) / H, prec) for s in range(self.S)]

    def _ensure_log_dir(self, log_dir: Optional[Path]) -> Path:
        if log_dir is not None:
            d = Path(log_dir)
        elif getattr(self, "middle_dir", None):
            d = Path(self.middle_dir) / "evolution"
        else:
            d = Path("./search/evolution")
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _L_key(self, L: Dict[int, int]) -> str:
        s = ",".join(str(int(max(0, min(self.H, L.get(i, 0))))) for i in range(self.S))
        return hashlib.md5(s.encode("utf-8")).hexdigest()

    # ---------- integer-level helpers ----------
    def _init_levels_total(self, TL: int) -> Dict[int, int]:
        TL = int(max(0, min(self.H, TL)))
        S = self.S; B = TL * S
        w = np.random.gamma(shape=1.0, scale=1.0, size=S).astype(np.float64)
        return self._shape_to_levels(w, B)

    def _retarget_sum(self, L: Dict[int, int], target_sum: int) -> Dict[int, int]:
        S, H = self.S, self.H
        out = [int(max(0, min(H, int(L.get(s, 0))))) for s in range(S)]
        cur = sum(out)
        if cur == target_sum:
            return {s: out[s] for s in range(S)}
        if cur < target_sum:
            order = sorted(range(S), key=lambda s: (H - out[s]), reverse=True)
            i = 0
            while cur < target_sum and i < len(order):
                s = order[i]
                if out[s] < H:
                    out[s] += 1; cur += 1
                else:
                    i += 1
            i = 0
            while cur < target_sum:
                s = i % S
                if out[s] < H:
                    out[s] += 1; cur += 1
                i += 1
        else:
            order = sorted(range(S), key=lambda s: out[s], reverse=True)
            i = 0
            while cur > target_sum and i < len(order):
                s = order[i]
                if out[s] > 0:
                    out[s] -= 1; cur -= 1
                else:
                    i += 1
            i = 0
            while cur > target_sum:
                s = i % S
                if out[s] > 0:
                    out[s] -= 1; cur -= 1
                i += 1
        return {s: out[s] for s in range(S)}

    def _shape_to_levels(self, weights: np.ndarray, B: int) -> Dict[int, int]:
        S, H = self.S, self.H
        w = np.maximum(0.0, np.asarray(weights, dtype=np.float64))
        if not np.any(w):
            base = min(H, B // S)
            L = [base] * S
            rem = B - base * S
            idx = 0
            while rem > 0:
                if L[idx % S] < H:
                    L[idx % S] += 1; rem -= 1
                idx += 1
            return {s: L[s] for s in range(S)}
        q = w / (w.sum() + 1e-12)
        x = np.floor(q * B + 1e-6).astype(int).tolist()
        L = [min(H, max(0, x[s])) for s in range(S)]
        cur = sum(L)
        if cur > B:
            order = sorted(range(S), key=lambda s: L[s], reverse=True)
            i = 0
            while cur > B and i < len(order):
                s = order[i]
                dec = min(L[s], cur - B)
                if dec > 0:
                    L[s] -= dec; cur -= dec
                i += 1
        elif cur < B:
            order = sorted(range(S), key=lambda s: (H - L[s]), reverse=True)
            i = 0
            while cur < B and i < len(order):
                s = order[i]
                inc = min(H - L[s], B - cur)
                if inc > 0:
                    L[s] += inc; cur += 1 * inc
                i += 1
            i = 0
            while cur < B:
                s = i % S
                if L[s] < H:
                    L[s] += 1; cur += 1
                i += 1
        return {s: int(L[s]) for s in range(S)}

    def _mutate_levels(self, parent: Dict[int, int], max_levels: int, max_times: int) -> Dict[int, int]:
        child = {s: int(max(0, min(self.H, int(parent.get(s, 0))))) for s in range(self.S)}
        S, H = self.S, self.H
        if S == 1:
            child[0] = max(0, min(H, child[0])); return child
        num_mutations = min(random.randint(1, max_times), random.randint(1, max_times))
        trials = 0; successful = 0; max_trials = 32 * num_mutations
        while successful < num_mutations and trials < max_trials:
            i, j = random.sample(range(S), 2)
            if child[i] <= 0 or child[j] >= H:
                trials += 1; continue
            m = random.randint(1, max_levels)
            move = min(m, child[i], H - child[j])
            if move <= 0:
                trials += 1; continue
            child[i] -= move; child[j] += move
            successful += 1; trials += 1
        return child

    @staticmethod
    def _to_py(o):
        import numpy as _np
        if isinstance(o, (int, float, str, type(None), bool)): return o
        if isinstance(o, _np.integer): return int(o)
        if isinstance(o, _np.floating): return float(o)
        if isinstance(o, _np.ndarray): return o.tolist()
        if isinstance(o, dict): return {EvoLayerDropSearchSDXL._to_py(k): EvoLayerDropSearchSDXL._to_py(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)): return [EvoLayerDropSearchSDXL._to_py(x) for x in o]
        return o


# ===== Initial population helper (same API as before) ================================

InitStrategy = Literal["random", "uniform", "heuristic_only", "hybrid", "warm_hybrid"]

def build_init_population_levels_sdxl(
    search: "EvoLayerDropSearchSDXL",
    *,
    survivors_per_selection: List[int],
    offspring: int,
    target_level: int,
    start_level: Optional[int] = None,
    strategy: InitStrategy = "hybrid",
    include_patterns: Optional[List[str]] = None,
    random_fraction: float = 0.6,
    warm_starts: Optional[List[Dict[int, int]]] = None,
    rng: Optional[random.Random] = None,
) -> List[Dict[int, int]]:
    """
    Identical behavior to the DiT version; unchanged from your previous code.
    """
    R = rng or random
    S = len(search.stages)
    H = search.H
    first_stage_size = max(1, int(survivors_per_selection[0] + offspring))
    TL = int(max(0, min(H, int(target_level))))
    SL = None if start_level is None else int(max(0, min(H, int(start_level))))
    B = TL * S
    MAX_UNIQUE_TRIES = 50 * first_stage_size

    def project_to_budget(L: Dict[int, int]) -> Dict[int, int]:
        return search._retarget_sum(L, target_sum=B)

    def _uniform() -> Dict[int, int]:
        return {s: TL for s in range(S)}

    def _rand() -> Dict[int, int]:
        w = np.random.gamma(shape=1.0, scale=1.0, size=S).astype(np.float64)
        return search._shape_to_levels(w, B)

    def pat_front():
        w = np.array([(S - s) for s in range(S)], dtype=np.float64)
        return search._shape_to_levels(w, B)

    def pat_back():
        w = np.array([(s + 1) for s in range(S)], dtype=np.float64)
        return search._shape_to_levels(w, B)

    def pat_middle():
        mu, sigma = (S - 1) / 2.0, max(1.0, S / 4.0)
        w = np.array([math.exp(-0.5 * ((s - mu) / sigma) ** 2) for s in range(S)], dtype=np.float64)
        return search._shape_to_levels(w, B)

    def pat_ramp_up():
        w = np.array([s + 1 for s in range(S)], dtype=np.float64)
        return search._shape_to_levels(w, B)

    def pat_ramp_down():
        w = np.array([S - s for s in range(S)], dtype=np.float64)
        return search._shape_to_levels(w, B)

    def pat_zigzag():
        w = np.array([1.2 if (s % 2 == 0) else 0.8 for s in range(S)], dtype=np.float64)
        return search._shape_to_levels(w, B)

    def pat_ends():
        mu, sigma = (S - 1) / 2.0, max(1.0, S / 3.5)
        mid = np.array([math.exp(-0.5 * ((s - mu) / sigma) ** 2) for s in range(S)], dtype=np.float64)
        mx = float(mid.max()) if mid.size else 1.0
        inv = (mx - mid + 1e-6)
        return search._shape_to_levels(inv, B)

    def _random_split_with_caps(m: int, total: int, cap: int) -> np.ndarray:
        if m <= 0 or total <= 0:
            return np.zeros((max(0, m),), dtype=np.int64)
        total = int(min(total, cap * m))
        if total <= 0:
            return np.zeros((m,), dtype=np.int64)
        w = np.random.gamma(shape=1.0, scale=1.0, size=m)
        if not np.isfinite(w).all() or w.sum() <= 0:
            w = np.ones((m,), dtype=np.float64)
        x = np.floor((w / w.sum()) * total).astype(np.int64)
        np.clip(x, 0, cap, out=x)
        diff = int(total - int(x.sum()))
        if diff > 0:
            idxs = list(range(m)); R.shuffle(idxs)
            for i in idxs:
                if diff <= 0: break
                if x[i] < cap:
                    x[i] += 1; diff -= 1
            while diff > 0:
                cand = [i for i in range(m) if x[i] < cap]
                if not cand: break
                i = R.choice(cand); x[i] += 1; diff -= 1
        elif diff < 0:
            diff = -diff
            for i in np.argsort(-x):
                if diff <= 0: break
                take = min(int(x[i]), diff)
                x[i] -= take; diff -= take
        np.clip(x, 0, cap, out=x)
        assert int(x.sum()) == total
        return x

    def pat_mosaicdiff():
        nA = max(0, int(round(0.10 * S)))
        nB = max(0, int(round(0.45 * S)))
        nC = S - nA - nB
        if nC <= 0 and S > 0:
            need = 1 - nC
            takeB = min(need, nB); nB -= takeB; need -= takeB
            if need > 0:
                takeA = min(need, nA); nA -= takeA; need -= takeA
            nC = S - nA - nB
        A_idx = list(range(0, nA))
        B_idx = list(range(nA, nA + nB))
        C_idx = list(range(nA + nB, S))

        L = np.zeros((S,), dtype=np.int64); rem = B
        if len(C_idx) and rem > 0:
            allocC = min(rem, H * len(C_idx))
            xC = _random_split_with_caps(len(C_idx), allocC, H)
            for j, s in enumerate(C_idx): L[s] += int(xC[j])
            rem -= allocC
        if len(A_idx) and rem > 0:
            allocA = min(rem, H * len(A_idx))
            xA = _random_split_with_caps(len(A_idx), allocA, H)
            for j, s in enumerate(A_idx): L[s] += int(xA[j])
            rem -= allocA
        if len(B_idx) and rem > 0:
            allocB = min(rem, H * len(B_idx))
            xB = _random_split_with_caps(len(B_idx), allocB, H)
            for j, s in enumerate(B_idx): L[s] += int(xB[j])
            rem -= allocB

        cur = int(L.sum())
        if cur != B:
            delta = B - cur
            if delta > 0:
                for part in (C_idx, A_idx, B_idx):
                    if delta <= 0: break
                    R.shuffle(part)
                    for s in part:
                        if delta <= 0: break
                        if L[s] < H: L[s] += 1; delta -= 1
            else:
                delta = -delta
                for part in (B_idx, A_idx, C_idx):
                    if delta <= 0: break
                    for s in sorted(part, key=lambda i: L[i], reverse=True):
                        if delta <= 0: break
                        take = min(L[s], delta); L[s] -= take; delta -= take

        np.clip(L, 0, H, out=L)
        assert int(L.sum()) == B
        return {s: int(L[s]) for s in range(S)}

    name2pat = {
        "front": pat_front,
        "back": pat_back,
        "middle": pat_middle,
        "ramp_up": pat_ramp_up,
        "ramp_down": pat_ramp_down,
        "zigzag": pat_zigzag,
        "ends": pat_ends,
    }
    default_patterns = ["front", "back", "middle", "ramp_up", "ramp_down", "zigzag", "ends"]
    pats = include_patterns or default_patterns

    pop: List[Dict[int, int]] = []
    seen = set()

    def push_unique(L: Dict[int, int]) -> bool:
        key = search._L_key(L)
        if key in seen: return False
        seen.add(key); pop.append(L); return True

    def add_random_until(target_size: int):
        attempts = 0
        while len(pop) < target_size and attempts < MAX_UNIQUE_TRIES:
            attempts += 1
            push_unique(_rand())

    if strategy == "random":
        add_random_until(first_stage_size)
    elif strategy == "uniform":
        push_unique(_uniform()); add_random_until(first_stage_size)
    elif strategy == "heuristic_only":
        name = pats[0]; fn = name2pat.get(name)
        if fn:
            attempts = 0
            while len(pop) < first_stage_size and attempts < MAX_UNIQUE_TRIES:
                attempts += 1
                push_unique(fn())
        while len(pop) < first_stage_size:
            pop.append(fn())
    elif strategy == "hybrid":
        if SL is not None and SL != TL: push_unique({s: SL for s in range(S)})
        push_unique(_uniform())
        for name in pats:
            fn = name2pat.get(name)
            if fn: push_unique(fn())
        remaining = max(0, first_stage_size - len(pop))
        n_rand = max(0, int(round(remaining * (random_fraction if 0 <= random_fraction <= 1 else 0.6))))
        add_random_until(len(pop) + n_rand); add_random_until(first_stage_size)
    elif strategy == "warm_hybrid":
        if warm_starts:
            for w in warm_starts:
                L0 = {int(s): int(max(0, min(H, int(v)))) for s, v in w.items()}
                push_unique(project_to_budget(L0))
        if len(pop) == 0:
            push_unique(_uniform())
        for name in pats:
            fn = name2pat.get(name)
            if fn: push_unique(fn())
        remaining = max(0, first_stage_size - len(pop))
        n_rand = max(0, int(round(remaining * (random_fraction if 0 <= random_fraction <= 1 else 0.6))))
        add_random_until(len(pop) + n_rand); add_random_until(first_stage_size)
    else:
        raise ValueError(f"Unknown init strategy: {strategy}")

    if len(pop) < first_stage_size:
        need = first_stage_size - len(pop)
        if pop:
            for i in range(need):
                parent = pop[i % len(pop)]
                child = dict(parent)
                if S >= 2:
                    i_s, j_s = R.sample(range(S), 2)
                    if child[i_s] > 0 and child[j_s] < H:
                        child[i_s] -= 1; child[j_s] += 1
                    else:
                        child = project_to_budget(child)
                pop.append(child)
        else:
            for _ in range(need): pop.append(_uniform())

    return pop[:first_stage_size]