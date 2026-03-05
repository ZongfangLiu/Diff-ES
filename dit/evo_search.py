# evo_search.py
# -*- coding: utf-8 -*-
"""
Evolutionary search engine used by both LayerDrop and Second-Order/Wanda pruning.

Levels are the first-class genome:
  - Each individual is per-stage integer levels L_s in [0, H].
  - Global budget is conserved by level-switch mutations; no projection needed.
  - Ratios are derived on the fly as r_s = L_s / H for schedule building & OBS.

Modes:
  - LayerDrop: H = depth (# Transformer blocks)
  - Second-Order / Wanda: H = # attention heads
"""

from __future__ import annotations
from typing import Callable, Dict, List, Optional, Tuple, Literal
import copy, hashlib, logging, random, json
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
import math

from evo_pruning_utils import (
    build_layerdrop_schedule_from_orders,
    build_secondorder_schedule_from_orders,
    apply_secondorder_schedule,
    select_obs_bank_for_ratios,              # still used (we pass derived ratios)
)

logger = logging.getLogger(__name__)


# ======================================================================================
# Fitness on sampling trajectory (latent space or image space)  [UNCHANGED API]
# ======================================================================================
class FitnessOnTrajectory:
    """
    Trajectory-based teacher–student fitness measured either in latent space
    or in image space (after VAE decode).

    fitness_metric:
      - "latent_mse" (default):     NEG MSE in latent space (higher is better)
      - "latent_abs":               NEG MAE in latent space (higher is better)
      - "latent_cos":               cosine similarity in latent space (higher is better)
      - "latent_snr":               signal-to-noise ratio vs teacher latent (higher is better)
      - "latent_snr_cosine":        blend of SNR and cosine (higher is better)
      - "img_mse":                  NEG MSE in image space (after VAE decode)
      - "img_ssim":                 SSIM in image space (after VAE decode)
      - "img_fft":                  high-frequency energy fraction in image space (student only)
      - "img_niqe":                 NEG NIQE via pyiqa (student only; higher is better)
      - "img_clipiqa":              CLIP-IQA via pyiqa (student only; dir auto-handled)
      - "img_topiq":                TOPIQ via pyiqa (student only; dir auto-handled)
      - "img_hyperiqa":             HyperIQA via pyiqa (student only; dir auto-handled)
      - "img_liqe":                 LIQE via pyiqa (student only; dir auto-handled)
      - "img_qalign":               Q-Align via pyiqa (student only; dir auto-handled)
      - "img_qualiclip":            QualiCLIP via pyiqa (student only; dir auto-handled)
      - "fft_cos_bound":            img_fft score with cosine-to-teacher lower bound
      - "latent_abs_fft":           blend of NEG latent_abs and img_fft (requires VAE)
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        diffusion,                 # your GaussianDiffusion (wrapped by respace)
        num_steps: int,
        cfg_scale: float,
        probe_batch: int,
        image_size: int,
        num_classes: int,
        mode: str = "final",       # "final" | "suffix" | "full"
        suffix_steps: Optional[int] = None,
        suffix_frac: float = 0.5,
        late_weighting: str = "cosine",  # "cosine" | "linear" | "uniform"
        include_eps_term: bool = False,  # only meaningful for latent_mse/latent_cos
        eps_weight: float = 0.3,
        device: torch.device = torch.device("cuda"),
        base_seed: int = 1234,
        progress: bool = False,
        refresh_every: int = 0,
        stages: Optional[List[Tuple[int, int]]] = None,  # needed for second-order
        fitness_metric: str = "latent_mse",              # see list above
        vae: Optional["AutoencoderKL"] = None,           # required for image-space metrics

        # ---- SNR knobs ----
        snr_mode: Literal["raw", "bounded"] = "bounded",
        snr_weight: float = 0.5,                         # used only for latent_snr_cosine

        # ---- FFT knobs ----
        fft_highpass_radius_frac: float = 0.25,          # fraction of Nyquist radius to define HP mask
        fft_zero_mean: bool = True,                      # subtract per-image mean before FFT

        # ---- Cosine bound (fft_cos_bound) ----
        cos_lower_bound: float = 0.88,

        # ---- latent_abs_fft blend weight ----
        abs_fft_weight: float = 0.5,
    ):
        assert mode in ("final", "suffix", "full")
        assert late_weighting in ("cosine", "linear", "uniform")
        assert fitness_metric in (
            "latent_mse", "latent_abs", "latent_cos", "latent_snr", "latent_snr_cosine",
            "img_mse", "img_ssim", "img_fft",
            "img_niqe", "img_clipiqa", "img_topiq", "img_hyperiqa", "img_liqe", "img_qalign",
            "img_qualiclip",
            "fft_cos_bound", "latent_abs_fft"
        )
        assert snr_mode in ("raw", "bounded")
        assert 0.0 < fft_highpass_radius_frac < 0.5, "fft_highpass_radius_frac must be in (0, 0.5)"
        assert 0.0 <= abs_fft_weight <= 1.0

        self.model = model
        self.diffusion = diffusion
        self.num_steps = int(num_steps)
        self.cfg_scale = float(cfg_scale)
        self.K = int(probe_batch)
        self.image_size = int(image_size)
        self.latent = self.image_size // 8
        self.num_classes = int(num_classes)
        self.mode = mode
        self.suffix_steps = None if suffix_steps is None else int(suffix_steps)
        self.suffix_frac = float(suffix_frac)
        self.late_weighting = late_weighting
        self.include_eps_term = bool(include_eps_term)
        self.eps_weight = float(eps_weight)
        self.device = device or next(model.parameters()).device
        self.base_seed = int(base_seed)
        self.progress = bool(progress)
        self.refresh_every = int(refresh_every)
        self.stages = stages

        self.fitness_metric = fitness_metric
        self.vae = vae

        # SNR settings
        self.snr_mode = snr_mode
        self.snr_weight = float(snr_weight)

        # FFT settings
        self.fft_highpass_radius_frac = float(fft_highpass_radius_frac)
        self.fft_zero_mean = bool(fft_zero_mean)

        # cosine bound (only for fft_cos_bound)
        self.cos_lower_bound = float(cos_lower_bound)

        # blend weight for latent_abs_fft
        self.abs_fft_weight = float(abs_fft_weight)

        # internal probe cache
        self._z0 = None  # [K, C, H, W]
        self._y0 = None  # [K]

        # precompute weighting profile if needed
        self._build_weights()

        # --- teacher caches (invalidate on probe refresh) ---
        self._teacher_final_latent: Optional[torch.Tensor] = None            # [K, C, H, W]
        self._teacher_traj: Optional[List[Dict[str, torch.Tensor]]] = None   # [{"sample", "pred_xstart"}, ...]
        self._teacher_final_img: Optional[torch.Tensor] = None               # [K, 3, H, W], in [0,1]
        self._teacher_traj_imgs: Optional[List[torch.Tensor]] = None         # list of [K,3,H,W]

        # --- external IQA metrics (lazy) ---
        self._niqe_metric = None
        self._clipiqa_metric = None
        self._topiq_metric = None
        self._hyperiqa_metric = None
        self._liqe_metric = None
        self._qalign_metric = None
        self._qualiclip_metric = None

    # ---------- schedule helpers ----------
    def _is_secondorder_schedule(self, schedule: Dict) -> bool:
        if not schedule:
            return False
        any_t = next(iter(schedule))
        entry = schedule[any_t]
        return isinstance(entry, dict) and ("attn" in entry or "mlp" in entry)

    def _clear_all_schedules(self, model: nn.Module):
        if hasattr(model, "clear_layerdrop_schedule"):
            model.clear_layerdrop_schedule()
        else:
            setattr(model, "layerdrop_schedule", None)
        if hasattr(model, "clear_secondorder_schedule"):
            model.clear_secondorder_schedule()
        # OBS extras
        if hasattr(model, "clear_projection_bank"):
            try:
                model.clear_projection_bank()
            except Exception:
                pass
        if hasattr(model, "_pending_proj_bank"):
            try:
                delattr(model, "_pending_proj_bank")
            except Exception:
                pass

    def _apply_schedule(self, model: nn.Module, schedule: Dict):
        if schedule is None:
            self._clear_all_schedules(model)
            return

        if self._is_secondorder_schedule(schedule):
            if hasattr(model, "enable_struct_prune_forward"):
                try:
                    model.enable_struct_prune_forward()
                except Exception:
                    pass
            apply_secondorder_schedule(model, schedule, stages=self.stages)
        else:
            if hasattr(model, "set_layerdrop_schedule"):
                model.set_layerdrop_schedule(schedule)
            else:
                setattr(model, "layerdrop_schedule", schedule)

    # ---------- public API ----------
    def refresh_probes(self, gen: Optional[int] = None):
        seed = self.base_seed if gen is None else (self.base_seed + int(gen))
        self._build_probe_set(seed)
        # invalidate teacher caches
        self._teacher_final_latent = None
        self._teacher_traj = None
        # teacher images are derived from teacher latents
        self._teacher_final_img = None
        self._teacher_traj_imgs = None

    @torch.no_grad()
    def __call__(self, model: nn.Module, schedule: Dict) -> float:
        """
        Return a scalar score (higher is better).
        latent_mse/img_mse return NEG MSE; latent_abs returns NEG MAE; latent_cos/img_ssim return similarity.
        snr variants return SNR (raw or bounded/mixture). img_fft returns HF-energy fraction.
        img_niqe returns NEG NIQE. img_clipiqa/img_topiq/img_hyperiqa/img_liqe/img_qalign return metric values (direction auto-handled).
        fft_cos_bound: maximize img_fft subject to a cosine lower bound.
        """
        if self._z0 is None:
            self._build_probe_set(self.base_seed)

        # ----- Student -----
        self._apply_schedule(model, schedule)
        try:
            if self.mode == "final" and not self.include_eps_term:
                s_final = self._rollout_final(self._z0, self._y0)
            else:
                s_traj = self._rollout_progressive(self._z0, self._y0)
        finally:
            self._clear_all_schedules(model)

        # ----- Teacher (cached) -----
        teacher_needs_traj = (self.mode != "final") or self.include_eps_term
        if self.mode == "final" and not self.include_eps_term:
            if self.fitness_metric in (
                "latent_mse", "latent_abs", "latent_cos", "latent_snr", "latent_snr_cosine",
                "fft_cos_bound", "latent_abs_fft"
            ):
                t_final_lat = self._get_teacher_final_latent()
        else:
            if self.fitness_metric in (
                "latent_mse", "latent_abs", "latent_cos", "latent_snr", "latent_snr_cosine",
                "fft_cos_bound", "latent_abs_fft"
            ):
                t_traj = self._get_teacher_traj()

        # For teacher image decodes (only for img_mse/img_ssim)
        if self.fitness_metric in ("img_mse", "img_ssim"):
            if self.mode == "final" and not self.include_eps_term:
                t_final_img = self._get_teacher_final_img()
            else:
                t_traj_imgs = self._get_teacher_traj_imgs()

        # ===== Scoring: FINAL (no eps-term) =====
        if self.mode == "final" and not self.include_eps_term:
            if self.fitness_metric == "latent_mse":
                loss = torch.mean((t_final_lat - s_final) ** 2)
                return -float(loss.item())
            elif self.fitness_metric == "latent_abs":
                loss = torch.mean(torch.abs(t_final_lat - s_final))
                return -float(loss.item())
            elif self.fitness_metric == "latent_cos":
                return float(self._cosine_batch(t_final_lat, s_final).mean().item())
            elif self.fitness_metric == "latent_snr":
                snr = self._snr_batch(t_final_lat, s_final, mode=self.snr_mode).mean()
                return float(snr.item())
            elif self.fitness_metric == "latent_snr_cosine":
                snr = self._snr_batch(t_final_lat, s_final, mode=self.snr_mode).mean()
                cos = self._cosine_batch(t_final_lat, s_final).mean()
                score = self.snr_weight * snr + (1.0 - self.snr_weight) * cos
                return float(score.item())
            elif self.fitness_metric == "img_fft":
                self._require_vae()
                s_img = self._decode_latents(s_final)          # [K,3,H,W] in [0,1]
                hf = self._fft_highfreq_fraction(s_img, self.fft_highpass_radius_frac, self.fft_zero_mean)
                return float(hf.mean().item())
            elif self.fitness_metric == "img_niqe":
                self._require_vae()
                niqe = self._get_niqe_metric()
                s_img = self._decode_latents(s_final)
                val = niqe(s_img)  # lower is better → negate
                return -float(val.mean().item())
            elif self.fitness_metric == "img_clipiqa":
                self._require_vae()
                metric = self._get_clipiqa_metric()
                s_img = self._decode_latents(s_final)
                val = metric(s_img)
                if getattr(metric, "lower_better", False):
                    val = -val
                return float(val.mean().item())
            elif self.fitness_metric == "img_topiq":
                self._require_vae()
                metric = self._get_topiq_metric()
                s_img = self._decode_latents(s_final)
                val = metric(s_img)
                if getattr(metric, "lower_better", False):
                    val = -val
                return float(val.mean().item())
            elif self.fitness_metric == "img_hyperiqa":
                self._require_vae()
                metric = self._get_hyperiqa_metric()
                s_img = self._decode_latents(s_final)
                val = metric(s_img)
                if getattr(metric, "lower_better", False):
                    val = -val
                return float(val.mean().item())
            elif self.fitness_metric == "img_liqe":
                self._require_vae()
                metric = self._get_liqe_metric()
                s_img = self._decode_latents(s_final)
                val = metric(s_img)
                if getattr(metric, "lower_better", False):
                    val = -val
                return float(val.mean().item())
            elif self.fitness_metric == "img_qalign":
                self._require_vae()
                metric = self._get_qalign_metric()
                s_img = self._decode_latents(s_final)
                val = metric(s_img)
                if getattr(metric, "lower_better", False):
                    val = -val
                return float(val.mean().item())
            elif self.fitness_metric == "img_qualiclip":
                self._require_vae()
                metric = self._get_qualiclip_metric()
                s_img = self._decode_latents(s_final)
                val = metric(s_img)
                # QualiCLIP is “higher is better”, but keep a safety flip if pyiqa exposes lower_better
                if getattr(metric, "lower_better", False):
                    val = -val
                return float(val.mean().item())
            elif self.fitness_metric == "fft_cos_bound":
                self._require_vae()
                cos = self._cosine_batch(t_final_lat, s_final).mean()
                if float(cos.item()) < self.cos_lower_bound:
                    return -1e9
                s_img = self._decode_latents(s_final)
                hf = self._fft_highfreq_fraction(s_img, self.fft_highpass_radius_frac, self.fft_zero_mean)
                return float(hf.mean().item())
            elif self.fitness_metric == "latent_abs_fft":
                self._require_vae()
                mae = torch.mean(torch.abs(t_final_lat - s_final))  # scalar
                s_img = self._decode_latents(s_final)
                hf = self._fft_highfreq_fraction(s_img, self.fft_highpass_radius_frac, self.fft_zero_mean).mean()
                score = self.abs_fft_weight * (-mae) + (1.0 - self.abs_fft_weight) * hf
                return float(score.item())
            else:
                # image-space teacher metrics
                self._require_vae()
                s_img = self._decode_latents(s_final)
                if self.fitness_metric == "img_mse":
                    loss = torch.mean((self._get_teacher_final_img() - s_img) ** 2)
                    return -float(loss.item())
                else:  # img_ssim
                    return float(self._ssim_batch(self._get_teacher_final_img(), s_img).mean().item())

        # ===== Scoring: SUFFIX/FULL over progressive lists =====
        L = self._suffix_len()
        t_sel = list(reversed(t_traj[-L:])) if (teacher_needs_traj and self.fitness_metric in (
            "latent_mse","latent_abs","latent_cos","latent_snr","latent_snr_cosine","fft_cos_bound","latent_abs_fft"
        )) else None
        s_sel = list(reversed(s_traj[-L:]))

        scores = []
        if self.fitness_metric in ("latent_mse", "latent_abs", "latent_cos", "latent_snr", "latent_snr_cosine"):
            for k in range(L):
                t_lat = t_sel[k]["sample"]
                s_lat = s_sel[k]["sample"]
                if self.fitness_metric == "latent_mse":
                    val = torch.mean((t_lat - s_lat) ** 2) * (-1.0)
                elif self.fitness_metric == "latent_abs":
                    val = torch.mean(torch.abs(t_lat - s_lat)) * (-1.0)
                elif self.fitness_metric == "latent_cos":
                    val = self._cosine_batch(t_lat, s_lat).mean()
                elif self.fitness_metric == "latent_snr":
                    val = self._snr_batch(t_lat, s_lat, mode=self.snr_mode).mean()
                else:  # latent_snr_cosine
                    snr = self._snr_batch(t_lat, s_lat, mode=self.snr_mode).mean()
                    cos = self._cosine_batch(t_lat, s_lat).mean()
                    val = self.snr_weight * snr + (1.0 - self.snr_weight) * cos
                scores.append(self.w[k] * val)
        elif self.fitness_metric == "img_fft":
            self._require_vae()
            for k in range(L):
                s_img = self._decode_latents(s_sel[k]["pred_xstart"])
                val = self._fft_highfreq_fraction(s_img, self.fft_highpass_radius_frac, self.fft_zero_mean).mean()
                scores.append(self.w[k] * val)
        elif self.fitness_metric == "img_niqe":
            self._require_vae()
            niqe = self._get_niqe_metric()
            for k in range(L):
                s_img = self._decode_latents(s_sel[k]["pred_xstart"])
                val = -niqe(s_img).mean()  # lower is better → negate
                scores.append(self.w[k] * val)
        elif self.fitness_metric == "img_clipiqa":
            self._require_vae()
            metric = self._get_clipiqa_metric()
            for k in range(L):
                s_img = self._decode_latents(s_sel[k]["pred_xstart"])
                v = metric(s_img).mean()
                if getattr(metric, "lower_better", False):
                    v = -v
                scores.append(self.w[k] * v)
        elif self.fitness_metric == "img_topiq":
            self._require_vae()
            metric = self._get_topiq_metric()
            for k in range(L):
                s_img = self._decode_latents(s_sel[k]["pred_xstart"])
                v = metric(s_img).mean()
                if getattr(metric, "lower_better", False):
                    v = -v
                scores.append(self.w[k] * v)
        elif self.fitness_metric == "img_hyperiqa":
            self._require_vae()
            metric = self._get_hyperiqa_metric()
            for k in range(L):
                s_img = self._decode_latents(s_sel[k]["pred_xstart"])
                v = metric(s_img).mean()
                if getattr(metric, "lower_better", False):
                    v = -v
                scores.append(self.w[k] * v)
        elif self.fitness_metric == "img_liqe":
            self._require_vae()
            metric = self._get_liqe_metric()
            for k in range(L):
                s_img = self._decode_latents(s_sel[k]["pred_xstart"])
                v = metric(s_img).mean()
                if getattr(metric, "lower_better", False):
                    v = -v
                scores.append(self.w[k] * v)
        elif self.fitness_metric == "img_qalign":
            self._require_vae()
            metric = self._get_qalign_metric()
            for k in range(L):
                s_img = self._decode_latents(s_sel[k]["pred_xstart"])
                v = metric(s_img).mean()
                if getattr(metric, "lower_better", False):
                    v = -v
                scores.append(self.w[k] * v)
        elif self.fitness_metric == "img_qualiclip":
            self._require_vae()
            metric = self._get_qualiclip_metric()
            for k in range(L):
                s_img = self._decode_latents(s_sel[k]["pred_xstart"])
                v = metric(s_img).mean()
                if getattr(metric, "lower_better", False):
                    v = -v
                scores.append(self.w[k] * v)
        elif self.fitness_metric == "fft_cos_bound":
            self._require_vae()
            # Weighted cosine bound across progressive steps
            cos_scores = []
            for k in range(L):
                t_lat = t_sel[k]["sample"]
                s_lat = s_sel[k]["sample"]
                cos_scores.append(self.w[k] * self._cosine_batch(t_lat, s_lat).mean())
            cos_wmean = torch.stack(cos_scores).sum()
            if float(cos_wmean.item()) < self.cos_lower_bound:
                return -1e9
            # If bound passes, use weighted img FFT score
            fft_scores = []
            for k in range(L):
                s_img = self._decode_latents(s_sel[k]["pred_xstart"])
                fft_scores.append(self.w[k] * self._fft_highfreq_fraction(s_img, self.fft_highpass_radius_frac, self.fft_zero_mean).mean())
            score = torch.stack(fft_scores).sum()
            return float(score.item())
        elif self.fitness_metric == "latent_abs_fft":
            self._require_vae()
            for k in range(L):
                t_lat = t_sel[k]["sample"]
                s_lat = s_sel[k]["sample"]
                mae_neg = torch.mean(torch.abs(t_lat - s_lat)) * (-1.0)
                s_img = self._decode_latents(s_sel[k]["pred_xstart"])
                fft_val = self._fft_highfreq_fraction(s_img, self.fft_highpass_radius_frac, self.fft_zero_mean).mean()
                val = self.abs_fft_weight * mae_neg + (1.0 - self.abs_fft_weight) * fft_val
                scores.append(self.w[k] * val)
        else:
            # img_mse / img_ssim
            self._require_vae()
            t_imgs_all = self._get_teacher_traj_imgs()
            for k in range(L):
                s_img = self._decode_latents(s_sel[k]["pred_xstart"])
                t_img = t_imgs_all[len(t_imgs_all) - 1 - k]
                if self.fitness_metric == "img_mse":
                    val = torch.mean((t_img - s_img) ** 2) * (-1.0)
                else:  # img_ssim
                    val = self._ssim_batch(t_img, s_img).mean()
                scores.append(self.w[k] * val)

        score = torch.stack(scores).sum()

        # Eps-term remains only for latent_mse/latent_cos (unchanged)
        if self.include_eps_term and self.fitness_metric in ("latent_mse", "latent_cos") \
           and hasattr(self.diffusion, "_predict_eps_from_xstart"):
            eps_terms = []
            for k in range(L):
                t_scalar = self.num_steps - L + k
                eps_t = self.diffusion._predict_eps_from_xstart(
                    t_sel[k]["sample"],
                    torch.tensor([t_scalar] * self.K, device=self.device),
                    t_sel[k]["pred_xstart"]
                )
                eps_s = self.diffusion._predict_eps_from_xstart(
                    s_sel[k]["sample"],
                    torch.tensor([t_scalar] * self.K, device=self.device),
                    s_sel[k]["pred_xstart"]
                )
                if self.fitness_metric == "latent_mse":
                    val = torch.mean((eps_t - eps_s) ** 2) * (-1.0)
                else:
                    val = self._cosine_batch(eps_t, eps_s).mean()
                eps_terms.append(self.w[k] * val)
            if eps_terms:
                score = score + self.eps_weight * torch.stack(eps_terms).sum()

        return float(score.item())

    # ---------- teacher getters ----------
    def _get_teacher_final_latent(self) -> torch.Tensor:
        if self._teacher_final_latent is None:
            # Ensure clean teacher run
            self._clear_all_schedules(self.model)
            try:
                self._teacher_final_latent = self._rollout_final(self._z0, self._y0)
            finally:
                self._clear_all_schedules(self.model)
        return self._teacher_final_latent

    def _get_teacher_traj(self) -> List[Dict[str, torch.Tensor]]:
        if self._teacher_traj is None:
            self._clear_all_schedules(self.model)
            try:
                self._teacher_traj = self._rollout_progressive(self._z0, self._y0)
            finally:
                self._clear_all_schedules(self.model)
        return self._teacher_traj

    def _get_teacher_final_img(self) -> torch.Tensor:
        if self._teacher_final_img is None:
            self._require_vae()
            t_lat = self._get_teacher_final_latent()
            self._teacher_final_img = self._decode_latents(t_lat)
        return self._teacher_final_img

    def _get_teacher_traj_imgs(self) -> List[torch.Tensor]:
        if self._teacher_traj_imgs is None:
            self._require_vae()
            t_traj = self._get_teacher_traj()
            # Decode exactly what we’ll compare against (pred_xstart in image metrics)
            self._teacher_traj_imgs = [self._decode_latents(step["pred_xstart"]) for step in t_traj]
        return self._teacher_traj_imgs

    # ---------- internals + helpers ----------
    def _build_probe_set(self, seed: int):
        g = torch.Generator(device=self.device).manual_seed(int(seed))
        z = torch.randn(self.K, getattr(self.model, "in_channels", 4), self.latent, self.latent, generator=g, device=self.device)
        y = torch.randint(0, self.num_classes, (self.K,), generator=g, device=self.device)
        self._z0 = z
        self._y0 = y
        # probes changed -> invalidate teacher caches
        self._teacher_final_latent = None
        self._teacher_traj = None
        self._teacher_final_img = None
        self._teacher_traj_imgs = None

    def _suffix_len(self) -> int:
        if self.mode == "final":
            return 1
        if self.mode == "full":
            return self.num_steps
        if self.suffix_steps is not None:
            return max(1, min(self.num_steps, int(self.suffix_steps)))
        return max(1, min(self.num_steps, int(round(self.suffix_frac * self.num_steps))))

    def _build_weights(self):
        if self.mode == "final":
            self.w = None
            return
        L = self._suffix_len()
        if self.late_weighting == "uniform":
            w = torch.ones(L, dtype=torch.float32)
        elif self.late_weighting == "linear":
            w = torch.linspace(0.5, 1.0, steps=L, dtype=torch.float32)
        else:  # cosine
            t = torch.linspace(0, 1, steps=L, dtype=torch.float32)
            w = 0.5 * (1.0 - torch.cos(torch.pi * t))
        self.w = (w / w.sum()).to(self.device)

    @torch.no_grad()
    def _rollout_final(self, z_init: torch.Tensor, y_init: torch.Tensor) -> torch.Tensor:
        using_cfg = (self.cfg_scale is not None) and (self.cfg_scale > 1.0)
        if using_cfg:
            z = torch.cat([z_init, z_init], dim=0)
            y_null = torch.full((y_init.shape[0],), self.num_classes, device=z_init.device, dtype=y_init.dtype)
            y = torch.cat([y_init, y_null], dim=0)
            def model_fn(x, t, y):
                return self.model.forward_with_cfg(x, t, y, self.cfg_scale)
            out = self.diffusion.ddim_sample_loop(
                model_fn, z.shape, z, clip_denoised=False,
                model_kwargs=dict(y=y), progress=False, device=self.device,
            )
            img = out["sample"] if isinstance(out, dict) else out
            img, _ = img.chunk(2, dim=0)
            return img
        else:
            def model_fn(x, t, y):
                return self.model(x, t, y)
            out = self.diffusion.ddim_sample_loop(
                model_fn, z_init.shape, z_init, clip_denoised=False,
                model_kwargs=dict(y=y_init), progress=False, device=self.device,
            )
            img = out["sample"] if isinstance(out, dict) else out
            return img

    @torch.no_grad()
    def _rollout_progressive(self, z_init: torch.Tensor, y_init: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        using_cfg = (self.cfg_scale is not None) and (self.cfg_scale > 1.0)
        outs: List[Dict[str, torch.Tensor]] = []
        if using_cfg:
            z_pair = torch.cat([z_init, z_init], dim=0)
            y_null = torch.full((y_init.shape[0],), self.num_classes, device=z_init.device, dtype=y_init.dtype)
            y_pair = torch.cat([y_init, y_null], dim=0)
            def model_fn(x, t, y):
                return self.model.forward_with_cfg(x, t, y, self.cfg_scale)
            gen = self.diffusion.ddim_sample_loop_progressive(
                model_fn, z_pair.shape, noise=z_pair, clip_denoised=False,
                model_kwargs=dict(y=y_pair), progress=False, device=self.device,
            )
            for out in gen:
                s, _ = out["sample"].chunk(2, dim=0)
                px, _ = out["pred_xstart"].chunk(2, dim=0)
                outs.append({"sample": s, "pred_xstart": px})
        else:
            def model_fn(x, t, y):
                return self.model(x, t, y)
            gen = self.diffusion.ddim_sample_loop_progressive(
                model_fn, z_init.shape, noise=z_init, clip_denoised=False,
                model_kwargs=dict(y=y_init), progress=False, device=self.device,
            )
            for out in gen:
                outs.append({"sample": out["sample"], "pred_xstart": out["pred_xstart"]})
        return outs

    def _require_vae(self):
        if self.vae is None:
            raise RuntimeError("Fitness metric requires VAE but none was provided. Pass vae=... when constructing FitnessOnTrajectory.")

    @torch.no_grad()
    def _decode_latents(self, lat: torch.Tensor) -> torch.Tensor:
        imgs = self.vae.decode(lat / 0.18215).sample  # [-1,1]
        imgs = torch.clamp((imgs + 1.0) * 0.5, 0.0, 1.0)
        return imgs

    @staticmethod
    def _cosine_batch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a_f = a.reshape(a.size(0), -1)
        b_f = b.reshape(b.size(0), -1)
        a_n = torch.nn.functional.normalize(a_f, dim=1)
        b_n = torch.nn.functional.normalize(b_f, dim=1)
        return (a_n * b_n).sum(dim=1)

    # ---------------- SNR helpers ----------------
    @staticmethod
    def _snr_batch(t: torch.Tensor, s: torch.Tensor, mode: str = "bounded", eps: float = 1e-12) -> torch.Tensor:
        sig = (t * t).mean(dim=(1, 2, 3))
        diff = (t - s)
        noise = (diff * diff).mean(dim=(1, 2, 3)).clamp_min(eps)
        if mode == "raw":
            return sig / noise
        return sig / (sig + noise)

    # ---------------- IQA helpers ----------------
    def _get_niqe_metric(self):
        if self._niqe_metric is None:
            try:
                import pyiqa
            except ImportError as e:
                raise RuntimeError("img_niqe requires the 'pyiqa' package. Install with `pip install pyiqa`.") from e
            self._niqe_metric = pyiqa.create_metric('niqe', device=self.device)
        return self._niqe_metric

    def _get_clipiqa_metric(self):
        if self._clipiqa_metric is None:
            try:
                import pyiqa
            except ImportError as e:
                raise RuntimeError("img_clipiqa requires the 'pyiqa' package. Install with `pip install pyiqa`.") from e
            self._clipiqa_metric = pyiqa.create_metric('clipiqa', device=self.device)
        return self._clipiqa_metric

    def _get_topiq_metric(self):
        if self._topiq_metric is None:
            try:
                import pyiqa
            except ImportError as e:
                raise RuntimeError("img_topiq requires the 'pyiqa' package. Install with `pip install pyiqa`.") from e
            tried = []
            for name in ["topiq", "topiq_nr", "topiq-fr", "topiq-nr", "topiq_fr"]:
                try:
                    self._topiq_metric = pyiqa.create_metric(name, device=self.device)
                    break
                except Exception:
                    tried.append(name)
                    self._topiq_metric = None
            if self._topiq_metric is None:
                raise RuntimeError(
                    f"TOPIQ metric not found in pyiqa. Tried: {tried}. "
                    "Run `print(pyiqa.list_models())` to inspect available names."
                )
        return self._topiq_metric

    def _get_hyperiqa_metric(self):
        if self._hyperiqa_metric is None:
            try:
                import pyiqa
            except ImportError as e:
                raise RuntimeError("img_hyperiqa requires the 'pyiqa' package. Install with `pip install pyiqa`.") from e
            self._hyperiqa_metric = pyiqa.create_metric('hyperiqa', device=self.device)
        return self._hyperiqa_metric

    def _get_liqe_metric(self):
        if self._liqe_metric is None:
            try:
                import pyiqa
            except ImportError as e:
                raise RuntimeError("img_liqe requires the 'pyiqa' package. Install with `pip install pyiqa`.") from e
            self._liqe_metric = pyiqa.create_metric('liqe', device=self.device)
        return self._liqe_metric

    def _get_qalign_metric(self):
        """
        Lazily create and cache the Q-Align metric from pyiqa.
        Accepts tensors in [0,1], shape (N,3,H,W). We try a few common IDs
        for robustness across pyiqa versions.
        """
        if self._qalign_metric is None:
            try:
                import pyiqa
            except ImportError as e:
                raise RuntimeError("img_qalign requires the 'pyiqa' package. Install with `pip install pyiqa`.") from e
            tried = []
            for name in ["qalign", "q_align", "q-align", "qalign_nr", "qalign-v2", "qalign_v2", "qalign_gm"]:
                try:
                    self._qalign_metric = pyiqa.create_metric(name, device=self.device)
                    break
                except Exception:
                    tried.append(name)
                    self._qalign_metric = None
            if self._qalign_metric is None:
                raise RuntimeError(
                    f"Q-Align metric not found in pyiqa. Tried: {tried}. "
                    "Run `print(pyiqa.list_models())` to inspect available names."
                )
        return self._qalign_metric
    
    def _get_qualiclip_metric(self):
        """
        Lazily create and cache the QualiCLIP metric from pyiqa.
        Accepts tensors in [0,1], shape (N,3,H,W).
        Tries common IDs across pyiqa versions.
        """
        if self._qualiclip_metric is None:
            try:
                import pyiqa
            except ImportError as e:
                raise RuntimeError("img_qualiclip requires the 'pyiqa' package. Install with `pip install pyiqa`.") from e

            tried = []
            # Known names in pyiqa’s model cards: qualiclip, qualiclip+, and dataset variants
            for name in ["qualiclip", "qualiclip+", "qualiclip+-koniq", "qualiclip+-clive", "qualiclip+-flive", "qualiclip+-spaq"]:
                try:
                    self._qualiclip_metric = pyiqa.create_metric(name, device=self.device)
                    break
                except Exception:
                    tried.append(name)
                    self._qualiclip_metric = None
            if self._qualiclip_metric is None:
                raise RuntimeError(
                    f"QualiCLIP metric not found in pyiqa. Tried: {tried}. "
                    "Run `import pyiqa; print(pyiqa.list_models())` to see available names."
                )
        return self._qualiclip_metric

    # ---------------- FFT helper ----------------
    @staticmethod
    def _fft_highfreq_fraction(img: torch.Tensor, radius_frac: float = 0.25, zero_mean: bool = True, eps: float = 1e-12) -> torch.Tensor:
        """
        img: [B, C, H, W] in [0,1]. Returns per-sample high-frequency energy fraction in [0,1].
        radius_frac: cutoff radius as fraction of the FFT grid's max radius (related to Nyquist).
        zero_mean: subtract per-image mean before FFT to suppress DC.
        """
        assert 0.0 < radius_frac <= 0.5, "cutoff must be in (0, 0.5]"
        B, C, H, W = img.shape
        x = img
        if zero_mean:
            x = x - x.mean(dim=(2, 3), keepdim=True)

        F = torch.fft.fft2(x, dim=(-2, -1))
        F = torch.fft.fftshift(F, dim=(-2, -1))
        P = (F.real**2 + F.imag**2)  # [B,C,H,W]

        cy, cx = H // 2, W // 2
        yy = torch.arange(H, device=img.device).reshape(-1, 1).expand(H, W)
        xx = torch.arange(W, device=img.device).reshape(1, -1).expand(H, W)
        dy = (yy - cy).float()
        dx = (xx - cx).float()
        r = torch.sqrt(dy * dy + dx * dx)  # [H,W]

        r_max = torch.sqrt(torch.tensor((H / 2.0) ** 2 + (W / 2.0) ** 2, device=img.device, dtype=torch.float32))
        r_cut = float(radius_frac) * r_max
        hp = (r >= r_cut).to(P.dtype)  # [H,W]

        total = P.sum(dim=(1, 2, 3)).clamp_min(eps)          # [B]
        high = (P * hp).sum(dim=(1, 2, 3))                   # [B]
        frac = (high / total).clamp(0.0, 1.0)                # [B] in [0,1]
        return frac

    # ---------- SSIM helpers ----------
    @staticmethod
    def _gaussian_window(window_size: int = 11, sigma: float = 1.5, channel: int = 3, device=None, dtype=None):
        coords = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2.0
        g = torch.exp(-(coords**2) / (2 * sigma * sigma))
        g = (g / g.sum()).unsqueeze(0)
        w = (g.t() @ g)
        w = w / w.sum()
        w = w.expand(channel, 1, window_size, window_size).contiguous()
        return w

    @staticmethod
    def _conv2d_same(x, weight):
        padding = weight.size(-1) // 2
        return torch.nn.functional.conv2d(x, weight, padding=padding, groups=x.size(1))

    @classmethod
    def _ssim_batch(cls, x: torch.Tensor, y: torch.Tensor, window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
        C1, C2 = 0.01**2, 0.03**2
        B, C, H, W = x.shape
        dtype, device = x.dtype, x.device
        w = cls._gaussian_window(window_size, sigma, C, device=device, dtype=dtype)

        mu_x = cls._conv2d_same(x, w)
        mu_y = cls._conv2d_same(y, w)
        mu_x2 = mu_x * mu_x
        mu_y2 = mu_y * mu_y
        mu_xy = mu_x * mu_y

        sigma_x2 = cls._conv2d_same(x * x, w) - mu_x2
        sigma_y2 = cls._conv2d_same(y * y, w) - mu_y2
        sigma_xy = cls._conv2d_same(x * y, w) - mu_xy

        ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2))
        return ssim_map.mean(dim=(1, 2, 3))

# ======================================================================================
# Evolutionary search (LEVELS ONLY, UNIFORM STAGES)
# ======================================================================================
class EvoLayerDropSearch:
    def __init__(
        self,
        model: nn.Module,
        stages: List[Tuple[int, int]],
        *,
        rng_seed: int = 42,
        verbose: bool = True,
        mode: str = "layerdrop",             # "layerdrop" | "secondorder"
        mode_kwargs: Optional[Dict] = None,  # so_head_dim, so_num_heads, so_protect_ends
    ):
        self.model = model
        self.stages = stages
        self.S = len(stages)

        assert hasattr(model, "blocks") and isinstance(model.blocks, nn.ModuleList)
        self.depth = len(model.blocks)
        self.verbose = verbose

        random.seed(rng_seed); np.random.seed(rng_seed); torch.manual_seed(rng_seed)

        # Calibration payloads
        self.orders_per_stage: Dict[int, List[int]] = {}       # layerdrop
        self.scores_per_stage: Dict[int, List[float]] = {}     # layerdrop
        self.so_orders: Optional[Dict] = None                  # secondorder

        # Middle results dir (optional)
        self.middle_dir: Optional[Path] = None

        # Mode switch / caps
        assert mode in ("layerdrop", "secondorder")
        self.mode = mode
        self.mode_kwargs = mode_kwargs or {}
        self.so_head_dim = int(self.mode_kwargs.get("so_head_dim", 72))
        self.so_num_heads = int(self.mode_kwargs.get("so_num_heads", 16))
        self.so_protect_ends = int(self.mode_kwargs.get("so_protect_ends", 0))

        # H (per-stage level cap)
        self.H = self.depth if self.mode == "layerdrop" else max(1, int(self.so_num_heads))

        # OBS repo (optional; set via set_obs_repo)
        self.obs_repo: Optional[dict] = None

        # If we're in second-order mode and the model supports struct thin forward, enable it once.
        if self.mode == "secondorder" and hasattr(self.model, "enable_struct_prune_forward"):
            try:
                self.model.enable_struct_prune_forward()
            except Exception:
                print("Warning: could not enable_struct_prune_forward on model.")
                pass

        # fitness caches
        self._fitness_cache: Dict[str, float] = {}
        self._fitness_cache_val: Dict[str, float] = {}
        
        # OBS supports
        self.use_obs = False                 # <- default OFF
        self.obs_repo = None                 # already suggested earlier
        self.obs_round_mode = "nearest"
        self._bank_cache = {}
        self._bank_cache_max = 8
        
    # ----------------------------- OBS support -----------------------------
    def set_obs_repo(self, repo):
        self.obs_repo = repo

    def set_obs_round_mode(self, mode: str = "nearest"):
        assert mode in ("nearest", "floor", "ceil")
        self.obs_round_mode = mode
        
    def _bank_key(self, ratios_dict):
        # ratios_dict: {stage:int -> ratio:float}; normalize to a sorted tuple
        return tuple(sorted((int(k), float(v)) for k, v in ratios_dict.items()))

    def _get_or_make_cpu_bank(self, ratios_dict):
        key = self._bank_key(ratios_dict)
        bank = self._bank_cache.get(key)
        if bank is None:
            bank = select_obs_bank_for_ratios(
                repo=self.obs_repo,
                ratios=ratios_dict,
                stages=self.stages,
                round_mode=self.obs_round_mode,
            )
            # print("Bank created for ratios:", ratios_dict)
            # print("Bank keys:", list(bank.keys()))
            # print("Bank stage 0", len(bank.get(0)['attn'][1]['kept_idx']))
            if len(self._bank_cache) >= self._bank_cache_max:
                self._bank_cache.pop(next(iter(self._bank_cache)))  # simple eviction
            self._bank_cache[key] = bank
        return bank
    
    def _evaluate_candidate(self, levels_per_stage, eval_fn, schedule):
        """
        Minimal evaluator:
        - Derive ratios from integer levels.
        - If in secondorder mode *and* an OBS repo is present, install the OBS bank
            for these ratios before calling eval_fn.
        - Do NOT touch any schedules (assume they're fixed / handled elsewhere).
        - Only clear the projection bank afterward.
        """
        # 1) Ratios from integer levels
        H = max(1, int(self.H))
        ratios = {
            int(s): float(max(0, min(H, int(levels_per_stage.get(s, 0))))) / float(H)
            for s in range(len(self.stages))
        }

        try:
            # 2) OBS-only: install bank if available
            if self.mode == "secondorder" and self.obs_repo and hasattr(self.model, "set_projection_bank"):
                bank_cpu = self._get_or_make_cpu_bank(ratios)
                self.model.set_projection_bank(bank_cpu, stages=self.stages)

            # 3) Delegate to eval_fn (which handles/assumes schedules)
            return eval_fn(self.model, schedule)

        finally:
            # 4) Only clear the projection bank; leave schedules untouched
            if hasattr(self.model, "clear_projection_bank"):
                self.model.clear_projection_bank()


    # ----------------------------- Calibration setters -----------------------------

    def set_layerdrop_orders(self, orders_per_stage: Dict[int, List[int]], scores_per_stage: Dict[int, List[float]]) -> None:
        self.orders_per_stage = orders_per_stage
        self.scores_per_stage = scores_per_stage

    def set_secondorder_orders(self, so_orders: Dict) -> None:
        """
        Expected shape (dict of dicts), produced by pruning.calibrate_secondorder_orders.
        """
        self.so_orders = so_orders

    # ----------------------------- Public API -----------------------------

    def run(
        self,
        *,
        generations: int,
        offspring: int,
        target_level: int,                        # average level per stage (integer in [0, H])
        survivors_per_selection: List[int],
        eval_fn: Callable[[nn.Module, Dict], float],
        start_level: Optional[int] = None,        # optional warm start level (avg)
        mutation_max_levels: int = 2,
        mutation_max_times: int = 5,
        # No usage, for legacy
        mutation_n_valid: int = 10,
        init_population: Optional[List[Dict[int, int]]] = None,
        patience: Optional[int] = None,
        eval_fn_val: Optional[Callable[[nn.Module, Dict], float]] = None,
        log_dir: Optional[Path] = None,
        log_float_precision: int = 4,
        log_every_gen: bool = True,
        single_log_name: str = "evolution.json",
    ) -> Tuple[Dict[int, int], Dict, float]:
        """
        Returns:
            best_levels_per_stage, best_schedule, best_score

        Individuals are Dict[stage_id -> level in [0, H]] with sum_s L_s = target_level * S.
        """
        # Sanity per mode
        if self.mode == "layerdrop":
            assert self.orders_per_stage, "Call set_layerdrop_orders(...) before run()."
        else:
            assert self.so_orders is not None, "Call set_secondorder_orders(...) before run()."

        # Clamp inputs
        TL = int(max(0, min(self.H, int(target_level))))
        SL = None if start_level is None else int(max(0, min(self.H, int(start_level))))
        max_lv = int(max(1, int(mutation_max_levels)))

        stage_sizes = [int(v) for v in survivors_per_selection]
        first_stage_size = stage_sizes[0] + offspring

        # --- setup logging directory ---
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

        # ====== Initialize population (levels only; sum conserved) ======
        if init_population is None:
            population = [self._init_levels_total(TL) for _ in range(first_stage_size)]
            if SL is not None and SL != TL:
                # small warm-start trick: bias the very first one toward SL
                population[0] = self._init_levels_total(SL)
        else:
            # Coerce provided seeds to levels with correct sum; clamp [0,H]
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
            if self.verbose:
                logger.info("[Generation %d/%d] avg_target_level=%d (H=%d)", g + 1, generations, TL, self.H)

            gen_log = {
                "generation": g + 1,
                "target_level": int(TL),
                "selections": []
            }

            # (Optional) probe refresh
            if hasattr(eval_fn, "refresh_every") and getattr(eval_fn, "refresh_every", 0) > 0:
                if g > 0 and (g % int(eval_fn.refresh_every) == 0):
                    try:
                        eval_fn.refresh_probes(gen=g)
                        if eval_fn_val is not None and hasattr(eval_fn_val, "refresh_probes"):
                            eval_fn_val.refresh_probes(gen=g + 100000)
                        self._fitness_cache.clear()
                        self._fitness_cache_val.clear()
                        logger.info("  [Fitness] Refreshed probes at gen=%d", g)
                    except Exception as _e:
                        logger.exception("  [Fitness] Probe refresh failed: %s", str(_e))

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

                # --- Log full pool & survivors for this selection ---
                sel_log = {
                    "selection_index": sel_idx,
                    "pool_size": len(evaluated),
                    "survivors_n": survivors_n,
                    "pool": [
                        {
                            "score": round(float(score), 6),
                            "levels": self._levels_list(ind),
                            "ratios": self._ratios_list(ind, prec=log_float_precision),
                        }
                        for (score, ind) in evaluated
                    ],
                    "survivors": [
                        {
                            "score": round(float(evaluated[i][0]), 6) if i < len(evaluated) else None,
                            "levels": self._levels_list(survivors[i]),
                            "ratios": self._ratios_list(survivors[i], prec=log_float_precision),
                        }
                        for i in range(len(survivors))
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
                            gen_idx = g + 1
                            with open(self.middle_dir / f"schedule_gen{gen_idx:03d}.json", "w") as f:
                                json.dump(self._to_py(best_sched), f, indent=2)
                            with open(self.middle_dir / f"levels_per_stage_gen{gen_idx:03d}.json", "w") as f:
                                json.dump(self._to_py(best_L), f, indent=2)
                            logger.info("  [MID] Saved middle_results for gen=%d (score=%.6f)", gen_idx, best_score)
                        except Exception as _e:
                            logger.exception("  [MID] Save failed: %s", str(_e))
                else:
                    stale += 1

                # (optional) validation log
                if self.verbose and evaluated:
                    top_train_score, top_train_indiv = evaluated[0]
                    if eval_fn_val is not None:
                        val_evals: List[Tuple[float, Dict[int, int]]] = []
                        for _, indiv in evaluated:
                            s_val = self._fitness_L_with_cache(indiv, eval_fn_val, self._fitness_cache_val)
                            val_evals.append((s_val, indiv))
                        val_evals.sort(key=lambda x: x[0], reverse=True)

                        top_val_score, top_val_indiv = val_evals[0]
                        same_top = (self._L_key(top_train_indiv) == self._L_key(top_val_indiv))

                        logger.info(
                            "    top train=%.6f | top val=%.6f | best train=%.6f | same_top=%s",
                            top_train_score, top_val_score, best_score, "YES" if same_top else "NO",
                        )
                        if not same_top:
                            logger.info("    top-train L=%s", self._levels_list(top_train_indiv))
                            logger.info("    top-val   L=%s", self._levels_list(top_val_indiv))
                    else:
                        logger.info("    top train=%.6f | best train=%.6f", top_train_score, best_score)

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

            # --- attach best-so-far + write aggregate JSON ---
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

    # ----------------------------- Internals (LEVELS) -----------------------------

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

    def _fitness_L_with_cache(self, L: Dict[int, int], eval_fn, cache: Dict[str, float]) -> float:
        key = self._L_key(L)
        if key in cache:
            return cache[key]

        schedule = self._L_to_schedule(L)

        # If OBS is available, select thin weights using derived ratios
        pending_set = False
        if (self.mode == "secondorder") and (self.obs_repo is not None):
            try:
                ratios = {s: float(max(0, min(self.H, L.get(s, 0)))) / float(max(1, self.H)) for s in range(self.S)}
                bank = select_obs_bank_for_ratios(self.obs_repo, ratios, self.stages)
                setattr(self.model, "_pending_proj_bank", bank)
                pending_set = True
            except Exception:
                pending_set = False

        # score = float(eval_fn(self.model, schedule))
        score = self._evaluate_candidate(L, eval_fn, schedule)

        if pending_set and hasattr(self.model, "_pending_proj_bank"):
            try:
                delattr(self.model, "_pending_proj_bank")
            except Exception:
                pass

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
                head_dim=self.so_head_dim,
                num_heads=self.so_num_heads,
                protect_ends=self.so_protect_ends,
            )

    # ---------- Level-space helpers ----------

    def _init_levels_total(self, TL: int) -> Dict[int, int]:
        """
        Build an integer vector L_s ∈ [0,H] with sum_s L_s = TL * S.
        Stages are uniform; we keep exact budget without projection.
        """
        TL = int(max(0, min(self.H, TL)))
        S = self.S
        B = TL * S
        # Start from a shaped (patterned) distribution to get diversity:
        weights = np.random.gamma(shape=1.0, scale=1.0, size=S).astype(np.float64)
        return self._shape_to_levels(weights, B)

    def _retarget_sum(self, L: Dict[int, int], target_sum: int) -> Dict[int, int]:
        """Clamp to [0,H] then adjust ±1 to match target_sum exactly."""
        S = self.S
        H = self.H
        out = [int(max(0, min(H, int(L.get(s, 0))))) for s in range(S)]
        cur = sum(out)
        if cur == target_sum:
            return {s: out[s] for s in range(S)}
        if cur < target_sum:
            # increment stages with largest slack first
            order = sorted(range(S), key=lambda s: (H - out[s]), reverse=True)
            i = 0
            while cur < target_sum and i < len(order):
                s = order[i]
                if out[s] < H:
                    out[s] += 1; cur += 1
                else:
                    i += 1
            # if still short (shouldn't happen), do a round-robin with checks
            i = 0
            while cur < target_sum:
                s = i % S
                if out[s] < H:
                    out[s] += 1; cur += 1
                i += 1
        else:
            # decrement stages with highest value first
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
        """
        Convert nonnegative weights -> integer levels with sum B and per-stage cap H.
        """
        S, H = self.S, self.H
        w = np.maximum(0.0, np.asarray(weights, dtype=np.float64))
        if not np.any(w):
            # fallback to uniform then adjust
            base = min(H, B // S)
            L = [base] * S
            rem = B - base * S
            idx = 0
            while rem > 0:
                if L[idx % S] < H:
                    L[idx % S] += 1; rem -= 1
                idx += 1
            return {s: L[s] for s in range(S)}
        q = w / (w.sum() + 1e-12)  # normalized pattern
        x = np.floor(q * B + 1e-6).astype(int).tolist()
        # fix rounding to match B under cap
        L = [min(H, max(0, x[s])) for s in range(S)]
        cur = sum(L)
        # if over, decrement from largest first
        if cur > B:
            order = sorted(range(S), key=lambda s: L[s], reverse=True)
            i = 0
            while cur > B and i < len(order):
                s = order[i]
                dec = min(L[s], cur - B)
                if dec > 0:
                    L[s] -= dec; cur -= dec
                i += 1
        # if under, increment where there is slack
        elif cur < B:
            order = sorted(range(S), key=lambda s: (H - L[s]), reverse=True)
            i = 0
            while cur < B and i < len(order):
                s = order[i]
                inc = min(H - L[s], B - cur)
                if inc > 0:
                    L[s] += inc; cur += inc
                i += 1
            # final safety (rare): round-robin add with cap
            i = 0
            while cur < B:
                s = i % S
                if L[s] < H:
                    L[s] += 1; cur += 1
                i += 1
        return {s: int(L[s]) for s in range(S)}

    def _mutate_levels(self, parent: Dict[int, int], max_levels: int, max_times: int) -> Dict[int, int]:
        """
        Level-switch mutation: apply up to 'n' mutations (where n ~ min(randint(1,max_times), randint(1,max_times))),
        each moving up to 'max_levels' units from stage i to j,
        conserving the sum and staying within [0,H].
        """
        child = {s: int(max(0, min(self.H, int(parent.get(s, 0))))) for s in range(self.S)}
        S, H = self.S, self.H
        if S == 1:
            child[0] = max(0, min(H, child[0]))  # nothing to do
            return child

        num_mutations = min(random.randint(1, max_times), random.randint(1, max_times))
        trials = 0
        successful = 0
        max_trials = 32 * num_mutations  # total budget

        while successful < num_mutations and trials < max_trials:
            i, j = random.sample(range(S), 2)
            if child[i] <= 0 or child[j] >= H:
                trials += 1
                continue
            m = random.randint(1, max_levels)
            move = min(m, child[i], H - child[j])
            if move <= 0:
                trials += 1
                continue
            child[i] -= move
            child[j] += move
            successful += 1
            trials += 1

        return child

    def _L_key(self, L: Dict[int, int]) -> str:
        s = ",".join(str(int(max(0, min(self.H, L.get(i, 0))))) for i in range(self.S))
        return hashlib.md5(s.encode("utf-8")).hexdigest()

    @staticmethod
    def _to_py(o):
        import numpy as _np
        if isinstance(o, (int, float, str, type(None), bool)):
            return o
        if isinstance(o, _np.integer):
            return int(o)
        if isinstance(o, _np.floating):
            return float(o)
        if isinstance(o, _np.ndarray):
            return o.tolist()
        if isinstance(o, dict):
            return {EvoLayerDropSearch._to_py(k): EvoLayerDropSearch._to_py(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [EvoLayerDropSearch._to_py(x) for x in o]
        return o


# ===== Level-space initializers (patterns + random) ===============================
InitStrategy = Literal[
    "random",           # random composition projected to exact budget
    "uniform",          # identical level per stage (exact TL)
    "heuristic_only",   # deterministic patterns only
    "hybrid",           # uniform + patterns + random (recommended)
    "warm_hybrid",      # hybrid + warm-start seeds you provide
]

def build_init_population_levels(
    search: "EvoLayerDropSearch",
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
    Returns an initial population list[Dict[int,int]] sized to
    (survivors_per_selection[0] + offspring). Individuals are levels L_s in [0,H]
    with exact budget sum_s L_s = TL * S (stages are uniform).

    This version avoids infinite loops when the unique solution space is small:
    - capped attempts for uniqueness
    - fallback to duplicates or tiny mutations to fill to requested size
    """
    R = rng or random
    S = len(search.stages)
    H = search.H
    first_stage_size = max(1, int(survivors_per_selection[0] + offspring))

    TL = int(max(0, min(H, int(target_level))))
    SL = None if start_level is None else int(max(0, min(H, int(start_level))))
    B = TL * S

    # Try-limit for uniqueness: generous but finite
    # (if we fail to add unique items after this many attempts, we relax de-dup)
    MAX_UNIQUE_TRIES = 50 * first_stage_size

    def project_to_budget(L: Dict[int, int]) -> Dict[int, int]:
        return search._retarget_sum(L, target_sum=B)

    # ---- base constructors -----------------------------------------------------
    def _uniform() -> Dict[int, int]:
        return {s: TL for s in range(S)}  # exact already

    def _rand() -> Dict[int, int]:
        w = np.random.gamma(shape=1.0, scale=1.0, size=S).astype(np.float64)
        return search._shape_to_levels(w, B)

    # patterns via weights
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
        """Random integer split of 'total' into m bins (each <= cap), exact sum enforced."""
        if m <= 0 or total <= 0:
            return np.zeros((max(0, m),), dtype=np.int64)
        total = int(min(total, cap * m))
        if total <= 0:
            return np.zeros((m,), dtype=np.int64)

        # Dirichlet-like allocation then floor + exactness fix
        w = np.random.gamma(shape=1.0, scale=1.0, size=m)
        if not np.isfinite(w).all() or w.sum() <= 0:
            w = np.ones((m,), dtype=np.float64)
        x = np.floor((w / w.sum()) * total).astype(np.int64)
        np.clip(x, 0, cap, out=x)

        diff = int(total - int(x.sum()))
        if diff > 0:
            idxs = list(range(m))
            R.shuffle(idxs)
            for i in idxs:
                if diff <= 0:
                    break
                if x[i] < cap:
                    x[i] += 1
                    diff -= 1
            while diff > 0:
                cand = [i for i in range(m) if x[i] < cap]
                if not cand:
                    break
                i = R.choice(cand)
                x[i] += 1
                diff -= 1
        elif diff < 0:
            diff = -diff
            for i in np.argsort(-x):
                if diff <= 0:
                    break
                take = min(int(x[i]), diff)
                x[i] -= take
                diff -= take

        np.clip(x, 0, cap, out=x)
        # Asserts are safe; remove if you prefer no exceptions
        assert int(x.sum()) == total, f"split sum {int(x.sum())} != {total}"
        return x

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
        if key in seen:
            return False
        seen.add(key)
        pop.append(L)
        return True

    def add_random_until(target_size: int):
        """Try to add unique randoms up to target_size, with capped attempts."""
        attempts = 0
        while len(pop) < target_size and attempts < MAX_UNIQUE_TRIES:
            attempts += 1
            push_unique(_rand())

    # ----------------------------- assemble population --------------------------
    if strategy == "random":
        add_random_until(first_stage_size)

    elif strategy == "uniform":
        push_unique(_uniform())
        add_random_until(first_stage_size)

    # elif strategy == "heuristic_only":
    #     for name in pats:
    #         fn = name2pat.get(name)
    #         if fn:
    #             push_unique(fn())
    #     add_random_until(first_stage_size)
    elif strategy == "heuristic_only":
        name = pats[0]
        fn = name2pat.get(name)
        if fn:
            attempts = 0
            while len(pop) < first_stage_size and attempts < MAX_UNIQUE_TRIES:
                attempts += 1
                push_unique(fn())
        # If still short (rare due to de-dup), top up with more pattern draws
        while len(pop) < first_stage_size:
            pop.append(fn())

    elif strategy == "hybrid":
        # optional warm/ramp seed
        if SL is not None and SL != TL:
            push_unique({s: SL for s in range(S)})
        # 1) uniform anchor
        push_unique(_uniform())
        # 2) patterns
        for name in pats:
            fn = name2pat.get(name)
            if fn:
                push_unique(fn())
        # 3) random fraction + fill
        remaining = max(0, first_stage_size - len(pop))
        n_rand = max(0, int(round(remaining * (random_fraction if 0 <= random_fraction <= 1 else 0.6))))
        add_random_until(len(pop) + n_rand)
        add_random_until(first_stage_size)

    elif strategy == "warm_hybrid":
        if warm_starts:
            for w in warm_starts:
                L0 = {int(s): int(max(0, min(H, int(v)))) for s, v in w.items()}
                push_unique(project_to_budget(L0))
        if len(pop) == 0:
            push_unique(_uniform())
        for name in pats:
            fn = name2pat.get(name)
            if fn:
                push_unique(fn())
        remaining = max(0, first_stage_size - len(pop))
        n_rand = max(0, int(round(remaining * (random_fraction if 0 <= random_fraction <= 1 else 0.6))))
        add_random_until(len(pop) + n_rand)
        add_random_until(first_stage_size)

    else:
        raise ValueError(f"Unknown init strategy: {strategy}")

    # ----------------------------- fallback fill (no infinite loops) ------------
    if len(pop) < first_stage_size:
        need = first_stage_size - len(pop)
        # If we have at least 1 seed, synthesize via tiny integer mutations (may duplicate; OK).
        if pop:
            for i in range(need):
                parent = pop[i % len(pop)]
                # Small, balanced move of 1 level between two random stages
                child = dict(parent)
                if S >= 2:
                    i_s, j_s = R.sample(range(S), 2)
                    # move 1 level from a stage with >0 to another with <H
                    if child[i_s] > 0 and child[j_s] < H:
                        child[i_s] -= 1
                        child[j_s] += 1
                    else:
                        # if boundary prevents, just project (no-op) to keep exact budget
                        child = project_to_budget(child)
                pop.append(child)
        else:
            # Extremely degenerate case: generate uniform copies
            for _ in range(need):
                pop.append(_uniform())

    return pop[:first_stage_size]