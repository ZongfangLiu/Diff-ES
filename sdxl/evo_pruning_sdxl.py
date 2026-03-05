#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import re
import json
import math
import copy
import argparse
import logging
from pathlib import Path
from typing import Tuple, Dict, Optional, List

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from diffusers import StableDiffusionXLPipeline

# ---------- project imports (SDXL versions) ----------
from calibration_sdxl import CocoSDXLDataset, dataloader_builder
from evo_search_sdxl import (
    EvoLayerDropSearchSDXL,
    FitnessFinalSDXL,
    build_init_population_levels_sdxl,
)
from models_sdxl import UNet2DConditionPruned, PrunedBasicTransformerBlock, PrunedTransformer2DModel

from evo_pruning_utils_sdxl import (
    # LayerDrop
    apply_layerdrop_schedule,
    calibrate_layerdrop_orders_sdxl,
    build_layerdrop_schedule_from_orders,

    # Unified first-order (wanda / magnitude / activation)
    calibrate_firstorder_orders_sdxl,
    build_secondorder_schedule_from_orders,   # schedule shape shared by FO and SO
    apply_secondorder_schedule,

    # True second-order (OBS)
    calibrate_secondorder_orders_sdxl,       # returns (orders, repo)
    save_obs_bank,
    load_obs_bank,
    select_obs_bank_for_ratios,
)
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"   # <- important

# -------------------- Eval (lazy) --------------------
def _lazy_imports_for_eval():
    global T, FrechetInceptionDistance, SentenceTransformer, util, cv2, ssim, CocoCaptions
    import torchvision.transforms as T
    from torchmetrics.image.fid import FrechetInceptionDistance
    from sentence_transformers import SentenceTransformer, util
    import cv2
    from skimage.metrics import structural_similarity as ssim
    from torchvision.datasets import CocoCaptions


# -------------------- Logging & RNG --------------------
def setup_logger(log_path: str, level: str = "INFO"):
    Path(os.path.dirname(log_path)).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
    )
    logging.getLogger("tqdm").setLevel(logging.WARNING)


def _secs_to_breakdown(secs: Optional[float]):
    if secs is None:
        return {"sec": None, "min": None, "hr": None}
    return {
        "sec": round(float(secs), 3),
        "min": round(float(secs) / 60.0, 2),
        "hr":  round(float(secs) / 3600.0, 3),
    }


def reset_rng(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


# -------------------- Small utils --------------------
def sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._\-+,/]", "-", name)


def uniform_stage_ranges(num_stages: int, diffusion_steps: int = 1000):
    assert num_stages >= 1
    edges = [int(round(i * diffusion_steps / num_stages)) for i in range(num_stages + 1)]
    out = []
    for i in range(num_stages):
        lo = edges[i]
        hi = max(lo, edges[i + 1] - 1)
        out.append((lo, hi))
    return out


def build_stages_from_dividers(dividers, num_stages: int, diffusion_steps: int = 1000):
    if not dividers:
        return None
    if len(dividers) != num_stages - 1:
        raise ValueError(f"--stage-dividers must have num_stages-1 values; got {len(dividers)}")
    ds = int(diffusion_steps)
    dv = [int(d) if d > 1 else int(round(d * ds)) for d in dividers]  # accept absolute or fractional
    if any(d <= 0 or d >= ds for d in dv):
        raise ValueError(f"Stage dividers must be in (0,{ds})")
    if sorted(dv) != dv or len(set(dv)) != len(dv):
        raise ValueError("Stage dividers must be strictly increasing and unique")
    edges = [0] + dv + [ds]
    out = []
    for i in range(len(edges) - 1):
        lo = edges[i]
        hi = edges[i + 1] - 1
        lo = max(0, min(lo, ds - 1))
        hi = max(lo, min(hi, ds - 1))
        out.append((lo, hi))
    return out


def to_py(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, dict):
        return {to_py(k): to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_py(x) for x in o]
    return o


# -------------------- SDXL structure/introspection --------------------
def _count_basic_transformer_blocks(unet) -> int:
    """
    Count BasicTransformerBlock modules in SDXL UNet across down/mid/up attentions.
    Works both when the class is importable and when only `.transformer_blocks` lists exist.
    """
    total_a = 0
    try:
        from diffusers.models.attention import BasicTransformerBlock
        for m in unet.modules():
            if isinstance(m, BasicTransformerBlock):
                total_a += 1
    except Exception:
        pass

    # Fallback / cross-check via .transformer_blocks
    def _len_tb(x):
        tb = getattr(x, "transformer_blocks", None)
        return len(tb) if isinstance(tb, (list, tuple, torch.nn.ModuleList)) else 0

    total_b = 0
    for attr in ("down_blocks", "up_blocks"):
        group = getattr(unet, attr, None)
        if group is None:
            continue
        for blk in group:
            atts = getattr(blk, "attentions", None)
            if atts is None:
                continue
            for att in atts:
                total_b += _len_tb(att)

    mid = getattr(unet, "mid_block", None)
    if mid is not None:
        atts = getattr(mid, "attentions", None)
        if atts is not None:
            for att in atts:
                total_b += _len_tb(att)

    return max(total_a, total_b)


def _detect_sdxl_min_heads_and_dim(unet):
    """
    Robustly scan SDXL UNet to find the minimum (num_heads, head_dim)
    across all attention blocks. Works across diffusers versions without
    tripping FutureWarnings or int(None) crashes.
    """
    min_heads, min_hdim = None, None

    def _upd(nh, hd):
        nonlocal min_heads, min_hdim
        if nh is None or hd is None:
            return
        try:
            nh = int(nh)
            hd = int(hd)
        except Exception:
            return
        if nh <= 0 or hd <= 0:
            return
        min_heads = nh if min_heads is None else min(min_heads, nh)
        min_hdim  = hd if min_hdim  is None else min(min_hdim,  hd)

    # 1) Walk modules; prefer config when present
    for _, m in unet.named_modules():
        cfg = getattr(m, "config", None)

        # (a) Try config fields first (no deprecation warnings)
        if cfg is not None:
            nh = getattr(cfg, "num_attention_heads", None)
            hd = getattr(cfg, "attention_head_dim", None)
            if isinstance(hd, (list, tuple)):
                hd = min([int(x) for x in hd if x is not None and int(x) > 0], default=None)
            _upd(nh, hd)

        # (b) Common runtime fields on attention submodules
        for att_name in ("attn", "attn1", "attn2"):
            att = getattr(m, att_name, None)
            if att is None:
                continue
            nh = getattr(att, "num_heads", None)
            if nh is None:
                nh = getattr(att, "heads", None)
            hd = getattr(att, "head_dim", None)
            _upd(nh, hd)

        # (c) direct fields
        nh = getattr(m, "num_heads", None)
        hd = getattr(m, "head_dim", None)
        _upd(nh, hd)

    # 2) fallback from UNet config
    if (min_heads is None or min_hdim is None) and hasattr(unet, "config"):
        ucfg = unet.config
        hdim_cfg = getattr(ucfg, "attention_head_dim", None)
        if isinstance(hdim_cfg, (list, tuple)):
            cand_hdim = [int(x) for x in hdim_cfg if x is not None and int(x) > 0]
            min_hdim = min(cand_hdim) if cand_hdim else min_hdim
        elif isinstance(hdim_cfg, (int, float)) and hdim_cfg > 0:
            min_hdim = int(hdim_cfg)

        if min_hdim is not None:
            chs = getattr(ucfg, "block_out_channels", None)
            if isinstance(chs, (list, tuple)) and len(chs) > 0:
                approx_heads = [max(1, int(int(c) // int(min_hdim))) for c in chs if c]
                if approx_heads:
                    min_heads = min(approx_heads) if min_heads is None else min(min_heads, min(approx_heads))

    if min_hdim is None:
        min_hdim = 64
    if min_heads is None:
        try:
            chs = list(getattr(unet.config, "block_out_channels", []))
            if chs:
                approx_heads = [max(1, int(int(c) // int(min_hdim))) for c in chs]
                min_heads = min(approx_heads) if approx_heads else 10
            else:
                min_heads = 10
        except Exception:
            min_heads = 10

    return int(min_heads), int(min_hdim)


def _compute_level_cap(unet, prune_method: str) -> int:
    """
    For layerdrop: H = total BasicTransformerBlocks in UNet.
    For SO/FO:     H = min #heads across all attention blocks.
    """
    if prune_method == "layerdrop":
        H = _count_basic_transformer_blocks(unet)
        return max(1, int(H))
    else:
        min_heads, _ = _detect_sdxl_min_heads_and_dim(unet)
        return max(1, int(min_heads))


# -------------------- Helpers: broadcast single-order to all stages --------------------
def _broadcast_orders_to_all_stages(orders_one_stage, stages):
    """
    Take an 'orders' object calibrated over the whole 0..999 trajectory
    and return a per-stage mapping: {stage_idx: orders_one_stage, ...}.
    Use deepcopy to avoid later in-place mutation coupling stages.
    """
    S = len(stages)
    return {str(s): copy.deepcopy(orders_one_stage) for s in range(S)}

def _ensure_repo_per_stage(repo_or_single, stages):
    """
    OBS repos in your utils may be keyed per-stage. If we calibrated a single
    repo over 0..999, present a stage-indexed view so downstream works unchanged.
    """
    S = len(stages)
    # If it's already a dict keyed by stage indices, return as-is
    if isinstance(repo_or_single, dict) and "entries" in repo_or_single:
        ent = repo_or_single.get("entries", {})
        if all(isinstance(k, int) for k in ent.keys()) and set(ent.keys()).issuperset({0}):
            return repo_or_single
    # Otherwise, alias by rebuilding a minimal repo dict per stage from the single
    single = repo_or_single
    if isinstance(single, dict) and "entries" in single:
        entries_single = single["entries"]
        # If entries keyed only at sid 0 (or missing), replicate to all sids
        if 0 in entries_single and len(entries_single.keys()) == 1:
            new_entries = {int(s): copy.deepcopy(entries_single[0]) for s in range(S)}
        else:
            # assume 'entries' already global; just mirror per sid
            new_entries = {int(s): copy.deepcopy(entries_single) for s in range(S)}
        repo = dict(single)
        repo["entries"] = new_entries
        return repo
    # Fallback: wrap into a minimal structure
    return {"entries": {int(s): copy.deepcopy(single) for s in range(S)}, "cache_root": ""}


# -------------------- Exp name (kept similar to DiT) --------------------
def build_exp_name(
    *,
    prune_method: str,
    image_size: int,
    cfg_scale: float,
    num_sampling_steps: int,
    num_stages: int,
    target_level: float,
    seed: int,
    generations: int,
    offspring: int,
    survivors_per_selection,
    mutation_max_levels: int,
    mutation_n_valid: int,
    traj_fitness_mode: str,
    traj_suffix_steps,
    traj_suffix_frac: float,
    traj_late_weighting: str,
    traj_include_eps: bool,
    traj_eps_weight: float,
    traj_probe_batch: int,
    traj_refresh_every: int,
    traj_fitness_metric: str,
    abs_fft_weight: float,
    calib_importance: str,
    init_strategy: str,
    so_struct_speedup: bool,
    stage_dividers=None,
    start_level=None,
    calib_whole_trajectory: bool = False,   # <-- NEW: suffix when enabled
    loader_nsamples: int = None,            # <-- NEW: will show up as "-sample{N}"
) -> str:
    core = (
        f"Formal-{prune_method}"
        f"-size-{image_size}"
        f"-cfg-{cfg_scale}"
        f"-step-{num_sampling_steps}"
        f"-S-{num_stages}"
        f"-lvl-{float(target_level):.2f}"
        f"-init-{init_strategy}"
        f"-gen{generations}"
        f"-offs{offspring}"
        f"-surv{'-'.join(map(str, survivors_per_selection))}"
        f"-mutL{int(mutation_max_levels)}"
        f"-mutN{int(mutation_n_valid)}"
        + (f"-samples{int(loader_nsamples)}" if loader_nsamples is not None else "")  # <-- NEW
        + f"-fitmetrics-{traj_fitness_metric.replace('latent_','l_').replace('img_','i_')}"
        + (f"-AbsFftWeight{abs_fft_weight}" if "abs_fft" in traj_fitness_metric else "")
        + ("" if not stage_dividers else f"-div-{'_'.join(map(str, stage_dividers))}")
        + ("-speedup" if so_struct_speedup else "")
        + f"-seed-{seed}"
    )
    core += "-imp-" + ("cos" if calib_importance.lower().startswith("cos") else "mse")
    if start_level is not None:
        core += f"-start{float(start_level):.2f}"
    if calib_whole_trajectory:
        core += "-calibFULL"   # <-- visible tag when full-trajectory calibration is used

    fit = f"-mode-{traj_fitness_mode}"
    if traj_fitness_mode == "suffix":
        if traj_suffix_steps is not None:
            fit += f"-K{int(traj_suffix_steps)}"
        else:
            fit += f"-F{float(traj_suffix_frac):.2f}"
    fit += f"-w{traj_late_weighting[0]}"
    if traj_include_eps:
        fit += f"-eps{float(traj_eps_weight):.2f}"
    fit += f"-probe{int(traj_probe_batch)}"
    if traj_refresh_every and int(traj_refresh_every) > 0:
        fit += f"-ref{int(traj_refresh_every)}"
    return sanitize_name(core + fit)


# -------------------- COCO prompt set --------------------
def load_coco_prompts(coco_img_dir: str, coco_ann_file: str, k: int = 5000, seed: int = 42):
    _lazy_imports_for_eval()
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    from torchvision.datasets import CocoCaptions
    ds = CocoCaptions(root=os.path.expanduser(coco_img_dir),
                      annFile=os.path.expanduser(coco_ann_file))
    N = min(k, len(ds))
    try:
        ids = sorted(getattr(ds, 'ids'))
        id_to_idx = {img_id: i for i, img_id in enumerate(ds.ids)}
        mapped_indices = [id_to_idx[ids[i]] for i in range(N)]
    except Exception:
        mapped_indices = list(range(N))

    prompts = []
    for idx in mapped_indices:
        _, caps = ds[idx]
        prompts.append(caps[0] if (caps and len(caps) > 0) else "")
    return prompts


# -------------------- Evaluation helpers --------------------
def compute_fid_score(real_dir: str, gen_dir: str, device: torch.device) -> float:
    _lazy_imports_for_eval()
    from PIL import Image

    fid = FrechetInceptionDistance().to(device)

    def pil_to_uint8_chw(img: Image.Image) -> torch.Tensor:
        arr = np.array(img)
        if arr.ndim == 2:
            arr = np.stack([arr]*3, axis=-1)
        if arr.shape[2] == 4:
            arr = arr[:, :, :3]
        t = torch.from_numpy(arr)           # H,W,C uint8
        t = t.permute(2, 0, 1).contiguous() # C,H,W
        return t

    def iter_uint8_images(folder):
        files = sorted([f for f in os.listdir(folder) if f.lower().endswith((".png",".jpg",".jpeg",".bmp",".tiff"))])
        for fname in files:
            path = os.path.join(folder, fname)
            try:
                img = Image.open(path).convert("RGB")
            except Exception:
                continue
            x = pil_to_uint8_chw(img).unsqueeze(0)
            yield x.to(device=device, dtype=torch.uint8)

    for x in tqdm(iter_uint8_images(real_dir), desc="FID: real"):
        fid.update(x, real=True)
    for x in tqdm(iter_uint8_images(gen_dir), desc="FID: gen"):
        fid.update(x, real=False)

    return float(fid.compute().item())


def compute_clip_score_with_prompts(gen_dir: str, prompts_path: str, device: torch.device, batch_size: int = 64) -> float:
    _lazy_imports_for_eval()
    from PIL import Image as _Image
    model = SentenceTransformer("clip-ViT-B-16", device=str(device), cache_folder=str(Path("~/.cache/huggingface/hub").expanduser()))

    with open(prompts_path, "r") as f:
        prompts = json.load(f)

    gen_files = sorted([p for p in Path(gen_dir).glob("*.png")])
    N = min(len(prompts), len(gen_files))
    if N == 0:
        return float("nan")

    images = []
    for i in range(N):
        try:
            images.append(_Image.open(gen_files[i]).convert("RGB"))
        except Exception:
            images.append(None)

    img_embeds = []
    for off in tqdm(range(0, N, batch_size), desc="CLIP encode images"):
        batch_imgs = [im for im in images[off:off+batch_size] if im is not None]
        idxs = [i for i in range(off, min(off+batch_size, N)) if images[i] is not None]
        if not idxs:
            continue
        emb = model.encode(batch_imgs, convert_to_tensor=True, device=str(device), batch_size=len(batch_imgs))
        img_embeds.append((idxs, emb))

    img_tensor = None
    valid_indices = []
    for idxs, emb in img_embeds:
        valid_indices.extend(idxs)
        img_tensor = emb if img_tensor is None else torch.cat([img_tensor, emb], dim=0)

    valid_prompts = [prompts[i] for i in valid_indices]
    txt_tensor = model.encode(valid_prompts, convert_to_tensor=True, device=str(device), batch_size=batch_size)
    sims = util.cos_sim(img_tensor, txt_tensor).diagonal().cpu().numpy().tolist()
    return float(sum(sims) / len(sims)) if sims else float("nan")


def compute_ssim_against_baseline(baseline_dir: str, gen_dir: str) -> float:
    _lazy_imports_for_eval()
    files1 = sorted([f for f in os.listdir(baseline_dir) if f.lower().endswith(".png")])
    files2 = sorted([f for f in os.listdir(gen_dir) if f.lower().endswith(".png")])
    n = min(len(files1), len(files2))
    if n == 0:
        logging.warning("SSIM: no overlapping pairs found.")
        return float("nan")

    scores = []
    for i in tqdm(range(n), desc="SSIM pairs"):
        p1 = os.path.join(baseline_dir, files1[i])
        p2 = os.path.join(gen_dir, files2[i])
        img1 = cv2.imread(p1, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(p2, cv2.IMREAD_GRAYSCALE)
        if img1 is None or img2 is None:
            continue
        h = min(img1.shape[0], img2.shape[0])
        w = min(img1.shape[1], img2.shape[1])
        img1r = cv2.resize(img1, (w, h), interpolation=cv2.INTER_AREA)
        img2r = cv2.resize(img2, (w, h), interpolation=cv2.INTER_AREA)
        score, _ = ssim(img1r, img2r, full=True)
        scores.append(score)
    return float(sum(scores) / len(scores)) if scores else float("nan")

@torch.inference_mode()
def generate_samples_with_prompts(
    pipe,
    prompts,
    num_steps,
    out_dir,
    device,
    per_bs,
    base_seed,
    guidance_scale,
):
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    # Discover which indices already exist
    existing = set()
    for p in outp.glob("*.png"):
        stem = p.stem
        if len(stem) == 6 and stem.isdigit():
            try:
                existing.add(int(stem))
            except Exception:
                pass

    total = len(prompts)
    all_indices = list(range(total))
    missing_indices = [i for i in all_indices if i not in existing]

    if len(missing_indices) == 0:
        logging.info(f"[Sample] All {total} samples already exist in '{out_dir}'. Nothing to do.")
        return

    # Warmup (doesn't affect per-image seeds)
    _ = pipe(
        prompt=["warmup"],
        num_inference_steps=1,
        guidance_scale=guidance_scale,
        generator=torch.Generator(device=device).manual_seed(base_seed),
    )

    # Generate only the missing indices, in batches
    desc = f"Sampling(resume)->{os.path.basename(out_dir)}"
    for off in tqdm(range(0, len(missing_indices), per_bs), desc=desc):
        batch_idx = missing_indices[off : off + per_bs]
        batch_prompts = [prompts[i] for i in batch_idx]
        batch_gens = [torch.Generator(device=device).manual_seed(base_seed + i) for i in batch_idx]

        out = pipe(
            prompt=batch_prompts,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=batch_gens,
        )
        images = out.images

        # Save each returned image to its intended slot filename
        for i, img in zip(batch_idx, images):
            img.save(outp / f"{i:06d}.png")


# ---- in evo_pruning_sdxl.py ----
# from models_sdxl import UNet2DConditionPruned, PrunedBasicTransformerBlock, PrunedTransformer2DModel

def count_pruned_blocks(unet: torch.nn.Module):
    n_basic = 0
    n_wrap = 0
    for m in unet.modules():
        if isinstance(m, PrunedBasicTransformerBlock):
            n_basic += 1
        if isinstance(m, PrunedTransformer2DModel):
            n_wrap += 1
    return n_basic, n_wrap

def attach_pruned_unet(pipe, device, repo_or_path: str = None, **load_kwargs):
    """
    Replace pipe.unet with UNet2DConditionPruned, copying weights 1:1 and
    swapping *inner* BasicTransformerBlocks to pruned versions.
    """
    # Choose source: if a repo/path is given, load from it; otherwise reuse the pipe's own unet weights.
    if repo_or_path is not None:
        # Keep dtype consistent with pipeline
        if "torch_dtype" not in load_kwargs:
            load_kwargs["torch_dtype"] = getattr(pipe.unet, "dtype", torch.float16)
        if "subfolder" not in load_kwargs:
            load_kwargs["subfolder"] = "unet"
        pruned = UNet2DConditionPruned.from_pretrained_pruned(repo_or_path, **load_kwargs)
    else:
        # Build pruned unet from the live pipe.unet (no hub fetch)
        base = pipe.unet
        try:
            base_cfg = base.config.to_dict() if hasattr(base.config, "to_dict") else dict(base.config)
        except Exception:
            base_cfg = dict(base.config)
        pruned = UNet2DConditionPruned(**base_cfg, swap_on_init=False)
        pruned.load_state_dict(base.state_dict(), strict=True)
        pruned._install_pruned_transformers()
        # keep dtype/device
        try:
            p = next(base.parameters())
            pruned.to(device=p.device, dtype=p.dtype)
        except StopIteration:
            pruned.to(device=device)

    # Move to device
    pruned.to(device=device)

    # Sanity check: count pruned blocks
    n_basic, n_wrap = count_pruned_blocks(pruned)
    assert (n_basic > 0) or (n_wrap > 0), "Pruned blocks not installed (swap failed)."

    pipe.unet = pruned
    return pruned


# -------------------- Provider for FitnessFinalSDXL --------------------
def simple_coco_prompt_provider_factory(dataset_like, *, pick_from: str = "captions"):
    """
    Returns a provider(K:int, seed:int) -> dict matching FitnessFinalSDXL contract:
      {
        "prompt": List[str] | str,
        "prompt_2": List[str] | str | None,
        "negative_prompt": List[str] | str | None,
        "negative_prompt_2": List[str] | str | None,
      }
    It deterministically samples K items by `seed` and uses the first caption.
    """
    N = len(dataset_like)

    def get_caption_at(i: int) -> str:
        item = dataset_like[i]
        if isinstance(item, tuple) and len(item) >= 2:
            caps = item[1]
        else:
            caps = getattr(item, "captions", [])
        return caps[0] if caps else ""

    def provider(*, K: int, seed: int):
        rng = np.random.default_rng(seed)
        if N <= 0 or K <= 0:
            return {"prompt": [""] * max(1, K), "prompt_2": None, "negative_prompt": None, "negative_prompt_2": None}
        idxs = rng.choice(N, size=K, replace=(N < K))
        prompts = [get_caption_at(int(i)) for i in idxs]
        print(prompts)
        return {"prompt": prompts, "prompt_2": None, "negative_prompt": None, "negative_prompt_2": None}

    return provider


# --- NEW: simple list-based prompt provider (TRAIN prompts for search) ---
def simple_list_prompt_provider_factory(prompt_list: List[str]):
    """
    Returns a provider(K:int, seed:int) -> dict with non-empty prompts.
    Deterministic by seed; samples from a plain list of strings.
    """
    N = len(prompt_list)

    def provider(*, K: int, seed: int):
        if N <= 0 or K <= 0:
            return {"prompt": ["a high quality photo"] * max(1, K),
                    "prompt_2": None, "negative_prompt": None, "negative_prompt_2": None}
        rng = np.random.default_rng(seed)
        idxs = rng.choice(N, size=K, replace=(N < K))
        prompts = []
        for i in idxs:
            p = prompt_list[int(i)]
            p = (p.strip() if isinstance(p, str) else "")
            prompts.append(p if p else "a high quality photo")
        return {"prompt": prompts, "prompt_2": None, "negative_prompt": None, "negative_prompt_2": None}

    return provider

# -------------------- Dense fallback --------------------
def _force_dense_sampling(pipe, stages, note: str = ""):
    msg = "[Sample] Falling back to DENSE (full) model"
    if note:
        msg += f" because: {note}"
    logging.warning(msg)
    print(msg)

    # Best-effort clear of any pruning/banks/schedules
    try:
        if hasattr(pipe.unet, "clear_all_accel"):
            pipe.unet.clear_all_accel()
    except Exception:
        pass
    try:
        apply_layerdrop_schedule(pipe.unet, {}, stages=stages)
    except Exception:
        pass
    try:
        apply_secondorder_schedule(pipe.unet, {}, stages=stages)
    except Exception:
        pass
    try:
        if hasattr(pipe.unet, "set_projection_bank"):
            pipe.unet.set_projection_bank(None, stages=stages)
    except Exception:
        pass

    pipe.unet.train(False)

# -------------------- Main --------------------
def main(args):
    # device / seeds / logger
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reset_rng(args.seed)

    # experiment dirs
    exp_name = build_exp_name(
        prune_method=args.prune_method,
        image_size=args.image_size,
        cfg_scale=args.cfg_scale,
        num_sampling_steps=args.num_sampling_steps,
        num_stages=args.num_stages,
        target_level=float(args.target_level),
        seed=args.seed,
        generations=args.generations,
        offspring=args.offspring,
        survivors_per_selection=args.survivors_per_selection,
        mutation_max_levels=args.mutation_max_levels,
        mutation_n_valid=args.mutation_n_valid,
        traj_fitness_mode=args.traj_fitness_mode,
        traj_suffix_steps=args.traj_suffix_steps,
        traj_suffix_frac=args.traj_suffix_frac,
        traj_late_weighting=args.traj_late_weighting,
        traj_include_eps=args.traj_include_eps,
        traj_eps_weight=args.traj_eps_weight,
        traj_probe_batch=args.traj_probe_batch,
        traj_refresh_every=args.traj_refresh_every,
        traj_fitness_metric=args.traj_fitness_metric,
        abs_fft_weight=args.abs_fft_weight,
        calib_importance=args.calib_importance,
        init_strategy=args.init_strategy,
        so_struct_speedup=args.so_struct_speedup,
        stage_dividers=(args.stage_dividers if args.stage_dividers else None),
        start_level=args.start_level,
        calib_whole_trajectory=args.calib_whole_trajectory,   # <-- NEW
        loader_nsamples=args.loader_nsamples,
    )
    exp_dir    = os.path.join(args.experiments_dir, exp_name)
    logs_dir   = os.path.join(exp_dir, "logs")
    search_dir = os.path.join(exp_dir, "search")
    sample_dir = os.path.join(exp_dir, "samples")
    eval_dir   = os.path.join(exp_dir, "eval")
    for d in [logs_dir, search_dir, sample_dir, eval_dir]:
        Path(d).mkdir(parents=True, exist_ok=True)
    setup_logger(os.path.join(logs_dir, "run.log"), level=args.log_level)
    logging.info(f"[Exp] {exp_name}")
    
    timings = {
    "calibration_wall_time_sec": None,
    "search_wall_time_sec": None,
    "calibration_cache_hit": False,
    # NEW nested breakdowns
    "calibration": {"sec": None, "min": None, "hr": None},
    "search": {"sec": None, "min": None, "hr": None},}


    # pipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16" if torch.cuda.is_available() else None,
        use_safetensors=True,
        local_files_only=True,
    )
    attach_pruned_unet(pipe, device)
    # Offload helps keep VRAM in check; FitnessFinalSDXL uses pipe._execution_device correctly.
    # pipe.enable_model_cpu_offload()
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    diffusion_steps = 1000

    # stages
    if args.stage_dividers:
        stages = build_stages_from_dividers(args.stage_dividers, args.num_stages, diffusion_steps)
    else:
        stages = uniform_stage_ranges(args.num_stages, diffusion_steps)
    logging.info(f"[Stages] {stages}")

    # Determine level cap H
    H = _compute_level_cap(pipe.unet, args.prune_method)
    logging.info(f"[Levels] H (cap) = {H} (method='{args.prune_method}') | basic_blocks={_count_basic_transformer_blocks(pipe.unet)}")

    # Auto-detect (min) heads & head_dim for mixed 10/20-head SDXL; allow CLI override
    det_min_heads, det_min_hdim = _detect_sdxl_min_heads_and_dim(pipe.unet)
    so_num_heads = args.so_num_heads if args.so_num_heads is not None else det_min_heads
    so_head_dim  = args.so_head_dim  if args.so_head_dim  is not None else det_min_hdim
    logging.info(f"[Heads] min_num_heads={so_num_heads}, min_head_dim={so_head_dim}")

    # OBS repo root for this S (append _FULL when whole-trajectory calibration is used)
    bank_dir = os.path.join(
        os.path.dirname(os.path.abspath(args.obs_bank_root)),
        f"obs_bank_stage{args.num_stages}_sample{args.loader_nsamples}{'_FULL' if args.calib_whole_trajectory else ''}",
    )
    Path(bank_dir).mkdir(parents=True, exist_ok=True)

    if args.do_prune:
        # ----- Calibration dataset (shared) -----
        calib_dataset = CocoSDXLDataset(
            image_dir=os.path.expanduser(args.image_dir),
            ann_file=os.path.expanduser(args.ann_file),
            pipeline=pipe,
            image_size=(pipe.default_sample_size * pipe.vae_scale_factor,
                        pipe.default_sample_size * pipe.vae_scale_factor),
            diffusion_steps=diffusion_steps,
            step_start=0,
            step_end=diffusion_steps,
            device=device,
        )
        calib_dl = dataloader_builder(
            calib_dataset,
            batchsize=args.fitness_batches,
            nsamples=args.loader_nsamples,
            same_subset=True,
            stages=stages,
        )

        logging.info("Default image size: %dx%d", pipe.default_sample_size * pipe.vae_scale_factor,
                                            pipe.default_sample_size * pipe.vae_scale_factor)
        # ----- Build searcher -----
        search_mode = "layerdrop" if args.prune_method == "layerdrop" else "secondorder"
        search = EvoLayerDropSearchSDXL(
            unet=pipe.unet,
            stages=stages,
            rng_seed=args.seed,
            verbose=True,
            mode=search_mode,
            mode_kwargs=dict(
                so_head_dim=so_head_dim,
                so_num_heads=so_num_heads,
                so_protect_ends=args.so_protect_ends,
                H=H,
            ),
        )

        # ----- Calibrate orders -----
        calib_t0 = time.time()
        calib_cache_hit = False
        if args.prune_method == "layerdrop":
            orders_path = os.path.join(search_dir, "orders_layerdrop.json")
            scores_path = os.path.join(search_dir, "orders_layerdrop_scores.json")

            if os.path.exists(orders_path):
                calib_cache_hit = True
                orders = json.load(open(orders_path, "r"))
                scores = None
                if os.path.exists(scores_path):
                    try:
                        scores = json.load(open(scores_path, "r"))
                    except Exception:
                        scores = None
                if isinstance(next(iter(orders.keys())), str):
                    orders = {int(k): v for k, v in orders.items()}
                search.set_layerdrop_orders(orders, scores)
                logging.info("[LayerDrop] Loaded cached orders%s.", " + scores" if scores is not None else "")

            else:
                if args.calib_whole_trajectory:
                    # one full-range loader
                    dl_full = dataloader_builder(
                        CocoSDXLDataset(
                            image_dir=os.path.expanduser(args.image_dir),
                            ann_file=os.path.expanduser(args.ann_file),
                            pipeline=pipe,
                            image_size=(pipe.default_sample_size * pipe.vae_scale_factor,
                                        pipe.default_sample_size * pipe.vae_scale_factor),
                            diffusion_steps=diffusion_steps,
                            step_start=0,
                            step_end=diffusion_steps,
                            device=device,
                        ),
                        batchsize=args.fitness_batches,
                        nsamples=args.loader_nsamples,
                        same_subset=True,
                    )

                    full_orders, full_scores = calibrate_layerdrop_orders_sdxl(
                        unet=pipe.unet,
                        dataloader=dl_full,
                        stages=[(0, diffusion_steps - 1)],   # single "stage" covering whole trajectory
                        importance_metric=args.calib_importance,
                        cosine_eps=args.calib_cosine_eps,
                    )
                    # Broadcast the one-stage orders to all real stages
                    one = full_orders.get(0, full_orders.get("0", []))
                    orders = _broadcast_orders_to_all_stages(one, stages)
                    scores = None  # optional — scores are only used for logging
                    search.set_layerdrop_orders(orders, scores)
                else:
                    # original per-stage calibration
                    orders, scores = calibrate_layerdrop_orders_sdxl(
                        unet=pipe.unet,
                        dataloader=calib_dl,
                        stages=stages,
                        importance_metric=args.calib_importance,
                        cosine_eps=args.calib_cosine_eps,
                    )
                    search.set_layerdrop_orders(orders, scores)

                with open(orders_path, "w") as f:
                    json.dump(to_py(orders), f, indent=2)
                try:
                    if scores is not None:
                        with open(scores_path, "w") as f:
                            json.dump(to_py(scores), f, indent=2)
                except Exception:
                    pass

        elif args.prune_method == "secondorder":
            orders_path = os.path.join(bank_dir, "orders_obs.json")
            repo_path   = os.path.join(bank_dir, "repo.pt")

            if os.path.exists(orders_path) and os.path.exists(repo_path):
                calib_cache_hit = True
                so_orders = json.load(open(orders_path, "r"))
                repo = load_obs_bank(bank_dir)
                search.set_secondorder_orders(so_orders)
                search.set_obs_repo(repo)
                logging.info("[OBS] Loaded cached orders + repo.")

            else:
                if args.calib_whole_trajectory:
                    # Single full-range loader to derive one global OBS order + repo
                    dl_full = dataloader_builder(
                        CocoSDXLDataset(
                            image_dir=os.path.expanduser(args.image_dir),
                            ann_file=os.path.expanduser(args.ann_file),
                            pipeline=pipe,
                            image_size=(pipe.default_sample_size * pipe.vae_scale_factor,
                                        pipe.default_sample_size * pipe.vae_scale_factor),
                            diffusion_steps=diffusion_steps,
                            step_start=0,
                            step_end=diffusion_steps,
                            device=device,
                        ),
                        batchsize=args.fitness_batches,
                        nsamples=args.loader_nsamples,
                        shuffle=True,
                        same_subset=True,
                    )

                    # Calibrate with a single 'stage' that spans all timesteps
                    so_orders_one, repo_one = calibrate_secondorder_orders_sdxl(
                        unet=pipe.unet,
                        dataloaders_per_stage=[dl_full],
                        stages=[(0, diffusion_steps - 1)],
                        obs_cache_dir=bank_dir,
                        obs_level_max=H,
                        head_dim=so_head_dim,
                    )
                    # Broadcast orders to every stage index
                    one = so_orders_one.get("0", so_orders_one.get(0, {}))
                    so_orders = _broadcast_orders_to_all_stages(one, stages)
                    # Present the repo as per-stage (alias the same repo to every stage)
                    repo = _ensure_repo_per_stage(repo_one, stages)

                else:
                    # Original per-stage OBS calibration
                    obs_stage_loaders = []
                    for (lo, hi) in stages:
                        ds_stage = CocoSDXLDataset(
                            image_dir=os.path.expanduser(args.image_dir),
                            ann_file=os.path.expanduser(args.ann_file),
                            pipeline=pipe,
                            image_size=(pipe.default_sample_size * pipe.vae_scale_factor,
                                        pipe.default_sample_size * pipe.vae_scale_factor),
                            diffusion_steps=diffusion_steps,
                            step_start=int(lo),
                            step_end=int(hi) + 1,
                            device=device,
                        )
                        dl_stage = dataloader_builder(
                            ds_stage,
                            batchsize=args.fitness_batches,
                            nsamples=args.loader_nsamples,
                            shuffle=True,
                            same_subset=True,
                        )
                        obs_stage_loaders.append(dl_stage)

                    so_orders, repo = calibrate_secondorder_orders_sdxl(
                        unet=pipe.unet,
                        dataloaders_per_stage=obs_stage_loaders,
                        stages=stages,
                        obs_cache_dir=bank_dir,
                        obs_level_max=H,
                        head_dim=so_head_dim,
                    )

                search.set_secondorder_orders(so_orders)
                search.set_obs_repo(repo)
                with open(orders_path, "w") as f:
                    json.dump(to_py(so_orders), f, indent=2)
                save_obs_bank(repo, bank_dir)
                logging.info("[OBS] Saved orders + repo to %s", bank_dir)

        else:
            # First-order: wanda | magnitude | activation
            orders_file = f"orders_{args.prune_method}.json"
            orders_path = os.path.join(search_dir, orders_file)

            if os.path.exists(orders_path):
                calib_cache_hit = True

                fo_orders = json.load(open(orders_path, "r"))
                search.set_secondorder_orders(fo_orders)
                logging.info("[FO] Loaded %s", orders_file)
            else:
                if args.calib_whole_trajectory:
                    dl_full = dataloader_builder(
                        CocoSDXLDataset(
                            image_dir=os.path.expanduser(args.image_dir),
                            ann_file=os.path.expanduser(args.ann_file),
                            pipeline=pipe,
                            image_size=(pipe.default_sample_size * pipe.vae_scale_factor,
                                        pipe.default_sample_size * pipe.vae_scale_factor),
                            diffusion_steps=diffusion_steps,
                            step_start=0,
                            step_end=diffusion_steps,
                            device=device,
                        ),
                        batchsize=args.fitness_batches,
                        nsamples=args.loader_nsamples,
                        same_subset=True,
                    )

                    fo_orders_one = calibrate_firstorder_orders_sdxl(
                        unet=pipe.unet,
                        dataloader=dl_full,
                        stages=[(0, diffusion_steps - 1)],
                        method=args.prune_method,
                    )
                    # print(fo_orders_one)
                    one = fo_orders_one.get("0", fo_orders_one.get(0, {}))
                    fo_orders = _broadcast_orders_to_all_stages(one, stages)
                    # print(fo_orders["0"])
                else:
                    fo_orders = calibrate_firstorder_orders_sdxl(
                        unet=pipe.unet,
                        dataloader=calib_dl,
                        stages=stages,
                        method=args.prune_method,
                    )

                search.set_secondorder_orders(fo_orders)
                with open(orders_path, "w") as f:
                    json.dump(to_py(fo_orders), f, indent=2)
        timings["calibration_wall_time_sec"] = round(time.time() - calib_t0, 3)
        timings["calibration_cache_hit"] = bool(calib_cache_hit)
        timings["calibration"] = _secs_to_breakdown(timings["calibration_wall_time_sec"])

        # ----- Fitness (final-only, pipeline-centric) -----
        # Build TRAIN-set prompt pool for fitness/search and guarantee non-empty text
        probe_prompts = load_coco_prompts(
            coco_img_dir=args.image_dir,       # train2017
            coco_ann_file=args.ann_file,       # captions_train2017.json
            k=max(args.traj_probe_batch, 5000),
            seed=args.seed
        )
        probe_prompts = [(p.strip() if isinstance(p, str) else "") or "a high quality photo"
                        for p in probe_prompts]
        # Persist for exact reproducibility of fitness probes
        with open(os.path.join(exp_dir, "search_prompts_train.json"), "w") as f:
            json.dump(probe_prompts, f, ensure_ascii=False, indent=2)
        prompt_provider = simple_list_prompt_provider_factory(probe_prompts)

        eval_fn = FitnessFinalSDXL(
            pipe=pipe,
            num_steps=args.num_sampling_steps,
            stages=stages,
            cfg_scale=args.cfg_scale,
            probe_batch=args.traj_probe_batch,
            height=pipe.default_sample_size * pipe.vae_scale_factor,
            width=pipe.default_sample_size * pipe.vae_scale_factor,
            base_seed=args.traj_probe_seed,
            context_provider=prompt_provider,          # TRAIN captions, non-empty
            metric=args.traj_fitness_metric,           # must be one of supported set
            fft_highpass_radius_frac=0.25,
            fft_zero_mean=True,
            abs_fft_weight=args.abs_fft_weight,
            cos_lower_bound=0.88,
        )

        # ----- Evo search over integer LEVELS -----
        target_level_int = max(0, min(H, int(round(float(args.target_level)))))
        start_level_int  = None if args.start_level is None else max(0, min(H, int(round(float(args.start_level)))))

        init_pop = build_init_population_levels_sdxl(
            search,
            survivors_per_selection=args.survivors_per_selection,
            offspring=args.offspring,
            target_level=target_level_int,
            start_level=start_level_int,
            strategy=args.init_strategy,
            include_patterns=args.init_patterns,
            random_fraction=args.init_random_fraction,
            warm_starts=None,
        )
        search_t0 = time.time()
        best_L, _best_sched, best_score = search.run(
            generations=args.generations,
            offspring=args.offspring,
            target_level=target_level_int,
            survivors_per_selection=args.survivors_per_selection,
            eval_fn=eval_fn,
            start_level=start_level_int,
            mutation_max_levels=int(args.mutation_max_levels),
            mutation_max_times=int(args.mutation_n_valid),
            eval_fn_val=None,
            init_population=init_pop,
            log_dir=search_dir,
            refresh_every=args.traj_refresh_every,
        )
        timings["search_wall_time_sec"] = round(time.time() - search_t0, 3)
        timings["search"] = _secs_to_breakdown(timings["search_wall_time_sec"])

        logging.info("[Search] Best train (final-only) score = %.6f", best_score)

        # Persist per-stage LEVELS (PRIMARY) and ratios (DERIVED)
        S = len(stages)

        def _lvl(d, s):
            if isinstance(d, dict):
                return int(d.get(s, d.get(str(s), 0)))
            if isinstance(d, (list, tuple)):
                return int(d[s]) if 0 <= s < len(d) else 0
            return int(d) if isinstance(d, (int, float)) and s == 0 else 0

        best_levels = {int(s): max(0, min(H, _lvl(best_L, s))) for s in range(S)}
        best_ratios = {int(s): float(best_levels[s]) / float(max(1, H)) for s in range(S)}

        with open(os.path.join(search_dir, "levels_per_stage.json"), "w") as f:
            json.dump(to_py(best_levels), f, indent=2)
        with open(os.path.join(search_dir, "ratios_per_stage.json"), "w") as f:
            json.dump(to_py(best_ratios), f, indent=2)
        with open(os.path.join(search_dir, "timings.json"), "w") as f:
            json.dump(to_py(timings), f, indent=2)

        with open(os.path.join(search_dir, "meta.json"), "w") as f:
            meta = dict(
                prune_method=args.prune_method,
                num_sampling_steps=args.num_sampling_steps,
                number_of_stages=args.num_stages,
                stages=stages,
                generations=args.generations,
                offspring=args.offspring,
                survivors_per_selection=args.survivors_per_selection,
                mode=("layerdrop" if args.prune_method == "layerdrop" else "secondorder"),
                H=int(H),
                target_level_int=int(target_level_int),
                start_level_int=(None if args.start_level is None else int(start_level_int)),
                mutation_max_levels=int(args.mutation_max_levels),
                mutation_max_times=int(args.mutation_n_valid),
                cfg_scale=args.cfg_scale,
                calibration_loader_nsamples=args.loader_nsamples,
                fitness_type="final",
                traj_fitness_mode="final",
                traj_suffix_steps=None,
                traj_suffix_frac=None,
                traj_late_weighting="cosine",
                traj_include_eps=False,
                traj_eps_weight=0.0,
                traj_probe_batch=args.traj_probe_batch,
                traj_probe_seed=args.traj_probe_seed,
                seed=args.seed,
                k_semantics="levels_per_stage",
                traj_fitness_metric=args.traj_fitness_metric,
                calib_whole_trajectory=bool(args.calib_whole_trajectory),  # <-- record
            )
            json.dump(to_py(meta), f, indent=2)

    
    prompts_path = os.path.join(exp_dir, "prompts_5k.json")
    if (not os.path.exists(prompts_path)) or args.refresh_prompts:
        prompts = load_coco_prompts(args.coco_val_dir, args.coco_val_ann,
                                    k=args.num_eval_images, seed=args.seed)
        with open(prompts_path, "w") as f:
            json.dump(prompts, f, ensure_ascii=False, indent=2)
    else:
        prompts = json.load(open(prompts_path, "r"))
    # -------------------- Sampling --------------------
    # if args.do_sample or args.do_eval:
    if args.do_sample:
        pipe.to(device)
        # Build deterministic COCO prompts once (saved next to samples for eval re-use)
        # -------------------- Build and apply schedule for sampling --------------------
        try:
            ratios_path = os.path.join(search_dir, "ratios_per_stage.json")
            if not os.path.exists(ratios_path):
                raise FileNotFoundError(f"missing {os.path.basename(ratios_path)}")
            ratios = {int(k): float(v) for k, v in json.load(open(ratios_path, "r")).items()}

            # Heads/dim again for schedule builders
            so_num_heads = args.so_num_heads if args.so_num_heads is not None else det_min_heads
            so_head_dim  = args.so_head_dim  if args.so_head_dim  is not None else det_min_hdim

            if args.prune_method == "layerdrop":
                layerdrop_file = os.path.join(search_dir, "orders_layerdrop.json")
                if not os.path.exists(layerdrop_file):
                    raise FileNotFoundError(f"missing {os.path.basename(layerdrop_file)}")
                orders = json.load(open(layerdrop_file, "r"))
                if isinstance(next(iter(orders.keys())), str):
                    orders = {int(k): v for k, v in orders.items()}
                sched = build_layerdrop_schedule_from_orders(
                    orders_per_stage=orders,
                    stages=stages,
                    ratios=ratios,
                    protect_ends=0,
                )
                apply_layerdrop_schedule(pipe.unet, sched, stages=stages)
                logging.info("[Sample] Applied LayerDrop schedule.")

            elif args.prune_method == "secondorder":
                orders_obs = os.path.join(bank_dir, "orders_obs.json")
                repo_path  = os.path.join(bank_dir, "repo.pt")
                if not (os.path.exists(orders_obs) and os.path.exists(repo_path)):
                    raise FileNotFoundError(f"missing {os.path.basename(orders_obs)} or {os.path.basename(repo_path)}")
                so_orders = json.load(open(orders_obs, "r"))
                sched = build_secondorder_schedule_from_orders(
                    orders_per_stage=so_orders,
                    stages=stages,
                    ratios=ratios,
                    protect_ends=args.so_protect_ends,
                )
                repo = load_obs_bank(bank_dir)
                bank = select_obs_bank_for_ratios(
                    repo=repo,
                    ratios=ratios,
                    stages=stages,
                    round_mode=args.obs_round_mode,
                )
                if hasattr(pipe.unet, "set_projection_bank"):
                    pipe.unet.set_projection_bank(bank, stages=stages)
                apply_secondorder_schedule(pipe.unet, sched, stages=stages)
                logging.info("[Sample][OBS] Applied schedule + projection bank.")

            else:
                file_map = dict(
                    wanda=os.path.join(search_dir, "orders_wanda.json"),
                    magnitude=os.path.join(search_dir, "orders_magnitude.json"),
                    activation=os.path.join(search_dir, "orders_activation.json"),
                )
                orders_path = file_map[args.prune_method]
                if not os.path.exists(orders_path):
                    raise FileNotFoundError(f"missing {os.path.basename(orders_path)}")
                fo_orders = json.load(open(orders_path, "r"))
                sched = build_secondorder_schedule_from_orders(
                    orders_per_stage=fo_orders,
                    stages=stages,
                    ratios=ratios,
                    protect_ends=args.so_protect_ends,
                )
                apply_secondorder_schedule(pipe.unet, sched, stages=stages)
                logging.info(f"[Sample] Applied {args.prune_method} schedule.")

        except Exception as e:
            # ANY error → clear and go dense
            _force_dense_sampling(pipe, stages, note=str(e))

        # pipe.enable_model_cpu_offload()

        # Generate samples (either pruned or dense if we fell back)
        generate_samples_with_prompts(
            pipe=pipe,
            prompts=prompts[:args.num_eval_images],
            num_steps=args.num_sampling_steps,
            out_dir=sample_dir,
            device=device,
            per_bs=args.per_proc_batch_size,
            base_seed=args.seed,
            guidance_scale=args.cfg_scale,
        )

    # -------------------- Evaluation --------------------
    if args.do_eval:
        # pipe.enable_model_cpu_offload()
        pipe.to(device)
        metrics = {}
        # Ensure baseline (dense) samples exist for SSIM
        baseline_dir = os.path.join(args.experiments_dir, f"SDXL-base_baseline_samples_search_cfg{args.cfg_scale}_formal")
        need_baseline = (not os.path.exists(baseline_dir)) or \
                        (len([f for f in os.listdir(baseline_dir) if f.lower().endswith(".png")]) < args.num_eval_images)
        if need_baseline:
            pipe.unet.clear_all_accel()
            pipe.unet.train(False)
            generate_samples_with_prompts(
                pipe=pipe,
                prompts=prompts[:args.num_eval_images],
                num_steps=args.num_sampling_steps,
                out_dir=baseline_dir,
                device=device,
                per_bs=args.per_proc_batch_size,
                base_seed=args.seed,
                guidance_scale=args.cfg_scale,
            )

        # FID (real COCO val vs generated)
        try:
            metrics["FID_5k"] = compute_fid_score(
                real_dir=os.path.expanduser(args.fid_real_dir),
                gen_dir=sample_dir,
                device=device,
            )
            logging.info(f"FID_5k: {metrics['FID_5k']:.4f}")
        except Exception as e:
            logging.exception("FID evaluation failed")
            metrics["FID_error"] = str(e)

        # CLIP (aligned with used prompts)
        try:
            metrics["CLIP_img_text_5k"] = compute_clip_score_with_prompts(
                gen_dir=sample_dir,
                prompts_path=os.path.join(exp_dir, "prompts_5k.json"),
                device=device,
            )
            logging.info(f"CLIP image-text (5k): {metrics['CLIP_img_text_5k']:.4f}")
        except Exception as e:
            logging.exception("CLIP evaluation failed")
            metrics["CLIP_error"] = str(e)

        # SSIM vs dense baseline
        try:
            metrics["SSIM_vs_baseline"] = compute_ssim_against_baseline(
                baseline_dir=baseline_dir, gen_dir=sample_dir
            )
            logging.info(f"SSIM vs baseline: {metrics['SSIM_vs_baseline']:.4f}")
        except Exception as e:
            logging.exception("SSIM evaluation failed")
            metrics["SSIM_error"] = str(e)

        # Save metrics
        with open(os.path.join(eval_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        logging.info(f"Saved metrics -> {os.path.join(eval_dir, 'metrics.json')}")

    logging.info("Done.")


# -------------------- CLI --------------------
def build_arg_parser():
    p = argparse.ArgumentParser(description="Single-GPU SDXL evolutionary pruning + sampling + (optional) eval")

    # Core paths
    p.add_argument("--experiments-dir", type=str, default="./experiments")
    p.add_argument("--image-dir", type=str, default="~/datasets/coco/train2017")
    p.add_argument("--ann-file", type=str, default="~/datasets/coco/annotations/captions_train2017.json")
    p.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=False)

    # COCO eval set (prompts source) + control
    p.add_argument("--coco-val-dir", type=str, default="~/datasets/coco/val2017")
    p.add_argument("--coco-val-ann", type=str, default="~/datasets/coco/annotations/captions_val2017.json")
    p.add_argument("--num-eval-images", type=int, default=5000)
    p.add_argument("--refresh-prompts", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--fid-real-dir", type=str, default="~/datasets/coco/val2017")

    # Method selection
    p.add_argument("--prune-method", type=str, default="layerdrop",
                   choices=["layerdrop", "secondorder", "wanda", "magnitude", "activation"])

    # Stages
    p.add_argument("--num-stages", type=int, default=20)
    p.add_argument("--stage-dividers", type=float, nargs="+", default=None,
                   help="Either fractional [0..1] or absolute timesteps (exclude 0 and 1000). Exactly num_stages-1 values.")

    # Levels & search
    p.add_argument("--target-level", type=float, required=True)
    p.add_argument("--start-level", type=float, default=None)
    p.add_argument("--generations", type=int, default=60)
    p.add_argument("--offspring", type=int, default=16)
    p.add_argument("--survivors-per-selection", type=int, nargs="+", default=[4])
    p.add_argument("--mutation-max-levels", type=int, default=2)
    p.add_argument("--mutation-n-valid", type=int, default=1)
    p.add_argument("--init-strategy", type=str, default="hybrid",
                   choices=["random","uniform","heuristic_only","hybrid","dirichlet","warm_hybrid"])
    p.add_argument("--init-patterns", type=str, nargs="+",
                   default=["front","back","middle","ramp_up","ramp_down","zigzag","ends"])
    p.add_argument("--init-random-fraction", type=float, default=0.4)

    # Calibration / loader
    p.add_argument("--fitness-batches", type=int, default=256)
    p.add_argument("--loader-nsamples", type=int, default=1024)
    p.add_argument("--calib-importance", type=str, default="cosine", choices=["mse","cosine"])
    p.add_argument("--calib-cosine-eps", type=float, default=1e-8)

    # NEW: calibration scope switch
    p.add_argument("--calib-whole-trajectory", action=argparse.BooleanOptionalAction, default=False,
                   help="If true, calibrate orders once on full [0..1000) and reuse the same order for every sub-stage. "
                        "Experiment name gets '-calibFULL' and OBS bank folder appends '_FULL'.")

    # SDXL / sampling
    p.add_argument("--image-size", type=int, default=1024)
    p.add_argument("--num-sampling-steps", type=int, default=20)
    p.add_argument("--cfg-scale", type=float, default=7.5)
    p.add_argument("--per-proc-batch-size", type=int, default=4)

    # OBS / runtime specifics
    p.add_argument("--so-head-dim", type=int, default=None)
    p.add_argument("--so-num-heads", type=int, default=None)
    p.add_argument("--so-protect-ends", type=int, default=0)
    p.add_argument("--obs-round-mode", type=str, default="nearest", choices=["floor","nearest","ceil"])
    p.add_argument("--obs-bank-root", type=str, default="./pretrained_models/obs_bank")
    p.add_argument("--so-struct-speedup", action=argparse.BooleanOptionalAction, default=True)

    # Fitness (final-only) knobs
    p.add_argument("--traj-fitness-mode", type=str, default="final", choices=["final"])  # locked
    p.add_argument("--traj-suffix-steps", type=int, default=None)   # unused, kept for exp name compat
    p.add_argument("--traj-suffix-frac", type=float, default=0.5)   # unused, kept for exp name compat
    p.add_argument("--traj-late-weighting", type=str, default="cosine", choices=["cosine"])  # locked
    p.add_argument("--traj-include-eps", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--traj-eps-weight", type=float, default=0.0)
    p.add_argument("--traj-probe-batch", type=int, default=64)
    p.add_argument("--traj-probe-seed", type=int, default=1234)
    p.add_argument("--traj-refresh-every", type=int, default=0)
    p.add_argument("--traj-fitness-metric", type=str, default="latent_mse",
                   choices=["latent_mse","latent_abs","latent_cos",
                            "img_mse","img_ssim","img_fft","latent_abs_fft", "fft_cos_bound",
                            "img_niqe","img_clipiqa","img_topiq","img_hyperiqa","img_liqe","img_qalign","img_qualiclip"])
    p.add_argument("--abs-fft-weight", type=float, default=0.5)

    # Modes
    p.add_argument("--do-prune", action=argparse.BooleanOptionalAction, default=True,
                   help="If false, skip both calibration and evolutionary search and run dense (unpruned) baseline.")
    p.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--do-eval", action=argparse.BooleanOptionalAction, default=True)

    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG","INFO","WARNING","ERROR","CRITICAL"])
    return p


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    main(args)