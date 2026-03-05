# evo_pruning_utils_sdxl.py
# -*- coding: utf-8 -*-

import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm

from collections import defaultdict
from typing import Iterable, Callable, Dict, List, Literal, Tuple, Union, Optional
from pathlib import Path
import os, json, re
import time
import math
import torch.nn.functional as F


# ===================== Types =====================
Schedule          = Union[Dict[int, Iterable[int]], Callable[[int], Iterable[int]]]
ScheduleSOEntry   = Dict[str, Dict[int, List[int]]]               # {"attn1":{blk:[heads]}, "attn2":{...}, "mlp":{blk:[channels]}}
ScheduleSO        = Dict[int, ScheduleSOEntry]                    # {sid: ScheduleSOEntry}
FirstOrderMethod  = Literal["wanda", "activation", "magnitude"]
ActStat           = Literal["mean_abs", "rms"]


# ===================== SDXL helpers =====================

def _t_to_stage_ids(t_tensor: torch.Tensor, stages: List[Tuple[int, int]]) -> torch.Tensor:
    assert t_tensor.ndim == 1, "Expecting t as a 1D tensor of shape (B,)."
    stage_ids = torch.full_like(t_tensor, fill_value=-1, dtype=torch.long)
    for sid, (lo, hi) in enumerate(stages):
        mask = (t_tensor >= lo) & (t_tensor <= hi)
        stage_ids[mask] = sid
    return stage_ids


def _sdxl_forward_unet(
    unet: nn.Module,
    batch: Dict[str, torch.Tensor],
):
    """
    Robust SDXL UNet forward that tolerates:
      - per-item CFG packing (x/t/conds with leading 2) collated into [B,2,...]
      - or already-flat [B_or_2B,...]
    It normalizes everything to the shapes diffusers expects and forwards.
    """
    sample  = batch.get("x", batch.get("latents"))
    t       = batch.get("t")
    encs    = batch.get("encoder_hidden_states", batch.get("cond"))
    added   = batch.get("added_cond_kwargs", None)
    amask   = batch.get("attention_mask", None)

    # ---- Normalize shapes ----
    # sample: ([B,2,C,H,W] | [B,C,H,W]) -> [B_or_2B, C, H, W]
    if sample is None:
        raise ValueError("Batch missing 'x' or 'latents' for UNet input.")
    if sample.dim() == 5 and sample.shape[1] == 2:
        B, two, C, H, W = sample.shape
        sample = sample.reshape(B * two, C, H, W)

    # t: ([B,2] | [B]) -> [B_or_2B]
    if t is None:
        raise ValueError("Batch missing 't' (timesteps) for UNet input.")
    if torch.is_tensor(t):
        if t.dim() == 2 and t.shape[1] == 2:
            t = t.reshape(-1)
        elif t.dim() == 0:
            t = t.view(1)
    else:
        t = torch.tensor([int(t)], device=sample.device, dtype=torch.long)

    # encoder_hidden_states: ([B,2,T,C] | [B,T,C]) -> [B_or_2B, T, C]
    if encs is not None and torch.is_tensor(encs):
        if encs.dim() == 4 and encs.shape[1] == 2:
            B, two, T, C = encs.shape
            encs = encs.reshape(B * two, T, C)

    # added_cond_kwargs: pack any [B,2,*] to [B_or_2B,*]
    if isinstance(added, dict):
        added = dict(added)
        for k in ("text_embeds", "time_ids"):
            x = added.get(k, None)
            if torch.is_tensor(x):
                if x.dim() == 3 and x.shape[1] == 2:   # [B,2,P]
                    B, two, P = x.shape
                    added[k] = x.reshape(B * two, P)
                elif x.dim() == 2 and x.shape[0] == 2 and sample.shape[0] != 2:
                    added[k] = x.repeat(sample.shape[0] // 2, 1)

    kwargs = {}
    if encs is not None:   kwargs["encoder_hidden_states"] = encs
    if added is not None:  kwargs["added_cond_kwargs"]     = added
    if amask is not None:  kwargs["attention_mask"]        = amask

    # If t is a scalar or length-1, expand to batch
    if t.dim() == 1 and t.numel() == 1 and sample.shape[0] > 1:
        t = t.expand(sample.shape[0]).contiguous()

    # Ensure dtype/device consistency for t
    t = t.to(device=sample.device, dtype=torch.long)

    return unet(sample, t, **kwargs)


def _iter_pruned_blocks(unet: nn.Module):
    """
    Yield (blk_id:int, block_module:Transformer2DModel-like) in the same flat order that
    models_sdxl.PrunedTransformer2DModel assigned (_blk_id).
    """
    seen = {}
    for name, m in unet.named_modules():
        if hasattr(m, "_blk_id"):
            bid = int(getattr(m, "_blk_id"))
            if bid not in seen:
                seen[bid] = m
    for bid in sorted(seen.keys()):
        yield bid, seen[bid]


def _infer_heads_from_attn(attn_module: nn.Module) -> Tuple[int, int]:
    """
    Return (num_heads H, head_dim d) for an SDXL Transformer2DModel attention submodule.
    """
    H = getattr(attn_module, "num_attention_heads", None)
    d = getattr(attn_module, "attention_head_dim", None)
    if H is None or d is None:
        if hasattr(attn_module, "to_q") and isinstance(attn_module.to_q, nn.Linear):
            Cq = int(attn_module.to_q.weight.shape[0])
            d = 64 if d is None else int(d)
            H = max(1, Cq // d)
        else:
            H, d = 8, 64
    return int(H), int(d)


# ===================== LayerDrop (stage-aware calibration) =====================

@torch.no_grad()
def calibrate_layerdrop_orders_sdxl(
    unet: nn.Module,
    dataloader,
    stages: List[Tuple[int, int]],
    importance_metric: Literal["mse", "cosine"] = "cosine",
    cosine_eps: float = 1e-8,
) -> Tuple[Dict[int, List[int]], Dict[int, List[float]]]:
    """
    Compute per-stage ranking of Transformer2D blocks (least->most important)
    for SDXL UNet by measuring per-block drift ||out - in|| at the block level.
    """
    device = next(unet.parameters()).device

    blk_ids = [bid for bid, _ in _iter_pruned_blocks(unet)]
    depth   = len(blk_ids)
    id2idx  = {bid: i for i, bid in enumerate(blk_ids)}
    print("depth =", depth, "blocks found for LayerDrop calibration.")

    stage_scores = {sid: torch.zeros(depth, dtype=torch.float64) for sid in range(len(stages))}
    current_stage_ids = {"vec": None}

    def make_block_hook(bid: int):
        def hook(_module, inputs, output):
            x_in  = inputs[0]
            x_out = output
            B = x_out.shape[0]
            a = x_out.detach().reshape(B, -1).float()
            b = x_in.detach().reshape(B, -1).float()

            if importance_metric == "mse":
                vals = (a - b).pow(2).mean(dim=1)
            else:
                cos = F.cosine_similarity(a, b, dim=1, eps=1e-6)
                cos = cos.clamp(-1.0, 1.0)
                cos[~torch.isfinite(cos)] = 0.0
                vals = 1.0 - cos

            vals = torch.where(torch.isfinite(vals), vals, torch.zeros_like(vals))

            stage_vec = current_stage_ids["vec"]
            if stage_vec is None:
                return
            idx = id2idx[bid]
            for sid in range(len(stages)):
                m = (stage_vec == sid)
                if m.any():
                    stage_scores[sid][idx] += vals[m].sum().double().cpu()
        return hook

    hooks: List[torch.utils.hooks.RemovableHandle] = []
    try:
        for bid, blk in _iter_pruned_blocks(unet):
            hooks.append(blk.register_forward_hook(make_block_hook(bid)))

        for batch in tqdm(dataloader, desc="[LayerDrop-Calibrate SDXL]"):
            t = batch["t"].to(device, non_blocking=True).view(-1)
            current_stage_ids["vec"] = _t_to_stage_ids(t, stages)
            _ = _sdxl_forward_unet(unet, {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()})
    finally:
        for h in hooks:
            h.remove()

    orders_per_stage: Dict[int, List[int]] = {}
    scores_per_stage: Dict[int, List[float]] = {}
    for sid in range(len(stages)):
        scores = stage_scores[sid].numpy().tolist()
        scores_per_stage[sid] = scores
        order_local_idx = list(np.argsort(scores))  # indices into [0..depth-1]
        order_block_ids = [blk_ids[i] for i in order_local_idx]
        orders_per_stage[sid] = order_block_ids

    return orders_per_stage, scores_per_stage


def build_layerdrop_schedule_from_orders(
    orders_per_stage: Dict[int, List[int]],
    stages: List[Tuple[int, int]],
    ratios: Union[float, Dict[int, float]],
    protect_ends: int = 1,
) -> Dict[int, List[int]]:
    """
    Build a *stage-based* schedule:
        schedule[sid] -> List[int]  (block ids to drop throughout that stage)
    """
    schedule: Dict[int, List[int]] = {}
    any_stage = next(iter(orders_per_stage))
    depth = len(orders_per_stage[any_stage])

    protected = set()
    if protect_ends > 0 and depth > 2 * protect_ends:
        protected = set(range(protect_ends)) | set(range(depth - protect_ends, depth))

    for sid, (_lo, _hi) in enumerate(stages):
        ratio = ratios[sid] if isinstance(ratios, dict) else ratios
        ratio = float(max(0.0, min(1.0, ratio)))
        k = int(depth * ratio)

        order_by_pos = [bid for pos, bid in enumerate(orders_per_stage[sid]) if pos not in protected]
        drop_ids = sorted(order_by_pos[:k])

        schedule[sid] = drop_ids

    return schedule


def apply_layerdrop(unet: nn.Module, drop_indices: Iterable[int]) -> None:
    """
    Static layer drop: always skip these block ids every step.
    """
    if hasattr(unet, "set_layerdrop"):
        unet.set_layerdrop(drop_indices)
    else:
        unet.drop_block_ids = set(int(i) for i in drop_indices)


def apply_layerdrop_schedule(unet: nn.Module, schedule: Schedule, stages=None) -> None:
    # schedule is stage-based: {sid: [blk_ids]}
    schedule = {int(k): [int(i) for i in v] for k, v in (schedule or {}).items()}
    if not schedule:
        if hasattr(unet, "clear_layerdrop_schedule"):
            unet.clear_layerdrop_schedule()
        elif hasattr(unet, "set_layerdrop_schedule"):
            unet.set_layerdrop_schedule({}, stages=None)
        return

    if hasattr(unet, "set_layerdrop_stage_schedule"):
        # Preferred explicit stage API if your UNet implements it
        unet.set_layerdrop_stage_schedule(schedule)
        print("Using UNet.set_layerdrop_stage_schedule — stage schedule installed.")
    elif hasattr(unet, "set_layerdrop_schedule"):
        # Back-compat: pass stage schedule plus stages description
        unet.set_layerdrop_schedule(schedule, stages=stages)
        print("Using UNet.set_layerdrop_schedule — stage schedule installed.")
    else:
        # Fallback stash
        unet.layerdrop_schedule = schedule
        if hasattr(unet, "layerdrop_stages"):
            unet.layerdrop_stages = stages


# ===================== FIRST-ORDER (SDXL) =====================

@torch.no_grad()
def calibrate_firstorder_orders_sdxl(
    unet: nn.Module,
    dataloader,
    stages: List[Tuple[int, int]],
    *,
    method: FirstOrderMethod = "wanda",
) -> Dict[str, Dict[str, Dict[str, List[int]]]]:
    """
    Stage-aware FIRST-ORDER structural ranking for SDXL UNet blocks.

    Returns (least->most important):
      {
        "<sid>": {
          "<blk_id>": {
            "attn1_heads_order": [head indices],
            "attn2_heads_order": [head indices],
            "mlp_channels_order": [channel indices],
          }, ...
        }, ...
      }
    """
    device = next(unet.parameters()).device
    S = len(stages)

    attn1_in_m2, attn1_cnt = defaultdict(dict), defaultdict(dict)
    attn2_in_m2, attn2_cnt = defaultdict(dict), defaultdict(dict)
    mlp_in_m2,  mlp_cnt  = defaultdict(dict), defaultdict(dict)

    toout_abs_rowsum = {}
    fc2_abs_rowsum   = {}

    heads_info = {}
    mlp_in_dims = {}

    stage_vec_ref = {"vec": None}
    hooks: List[torch.utils.hooks.RemovableHandle] = []

    def _maybe_rowsum_abs(W: torch.Tensor) -> torch.Tensor:
        return W.detach().abs().sum(dim=0)

    def _register_for_block(bid: int, blk: nn.Module):
        # Attn1
        if hasattr(blk, "attn1") and isinstance(getattr(blk.attn1, "to_out", None), (nn.Sequential, nn.Module)):
            to_out = getattr(blk.attn1, "to_out", None)
            to_out1 = to_out[0] if isinstance(to_out, (nn.Sequential, nn.ModuleList)) else to_out
            if isinstance(to_out1, nn.Linear):
                toout_abs_rowsum[(bid, "attn1")] = _maybe_rowsum_abs(to_out1.weight).cpu()
                H1, d1 = _infer_heads_from_attn(blk.attn1)
                heads_info.setdefault(bid, [H1, d1, None, None])
                def pre_hook(_m, inputs):
                    x = inputs[0]
                    if x is None: return
                    if x.dim() == 3:
                        x2 = (x ** 2).mean(dim=1)
                    elif x.dim() == 2:
                        x2 = x ** 2
                    else:
                        return
                    sv = stage_vec_ref["vec"]
                    if sv is None: return
                    C = x2.shape[-1]
                    for sid in range(S):
                        m = (sv == sid)
                        if m.any():
                            v = x2[m].mean(dim=0).double().cpu()
                            if (bid not in attn1_in_m2[sid]) or (attn1_in_m2[sid][bid].numel() != C):
                                attn1_in_m2[sid][bid] = torch.zeros(C, dtype=torch.float64)
                                attn1_cnt[sid][bid] = 0
                            attn1_in_m2[sid][bid] += v
                            attn1_cnt[sid][bid]   += 1
                hooks.append(to_out1.register_forward_pre_hook(pre_hook))

        # Attn2
        if hasattr(blk, "attn2") and blk.attn2 is not None and isinstance(getattr(blk.attn2, "to_out", None), (nn.Sequential, nn.Module)):
            to_out = getattr(blk.attn2, "to_out", None)
            to_out2 = to_out[0] if isinstance(to_out, (nn.Sequential, nn.ModuleList)) else to_out
            if isinstance(to_out2, nn.Linear):
                toout_abs_rowsum[(bid, "attn2")] = _maybe_rowsum_abs(to_out2.weight).cpu()
                H2, d2 = _infer_heads_from_attn(blk.attn2)
                if bid in heads_info:
                    heads_info[bid][2], heads_info[bid][3] = H2, d2
                else:
                    heads_info[bid] = [None, None, H2, d2]
                def pre_hook(_m, inputs):
                    x = inputs[0]
                    if x is None: return
                    if x.dim() == 3:
                        x2 = (x ** 2).mean(dim=1)
                    elif x.dim() == 2:
                        x2 = x ** 2
                    else:
                        return
                    sv = stage_vec_ref["vec"]
                    if sv is None: return
                    C = x2.shape[-1]
                    for sid in range(S):
                        m = (sv == sid)
                        if m.any():
                            v = x2[m].mean(dim=0).double().cpu()
                            if (bid not in attn2_in_m2[sid]) or (attn2_in_m2[sid][bid].numel() != C):
                                attn2_in_m2[sid][bid] = torch.zeros(C, dtype=torch.float64)
                                attn2_cnt[sid][bid] = 0
                            attn2_in_m2[sid][bid] += v
                            attn2_cnt[sid][bid]   += 1
                hooks.append(to_out2.register_forward_pre_hook(pre_hook))

        # MLP.fc2
        if hasattr(blk, "ff") and isinstance(getattr(blk.ff, "net", None), nn.ModuleList) and len(blk.ff.net) >= 3:
            fc2 = blk.ff.net[2]
            if isinstance(fc2, nn.Linear):
                fc2_abs_rowsum[bid] = _maybe_rowsum_abs(fc2.weight).cpu()
                def pre_hook(_m, inputs):
                    x = inputs[0]
                    if x is None: return
                    if x.dim() == 3:
                        x2 = (x ** 2).mean(dim=1)
                    elif x.dim() == 2:
                        x2 = x ** 2
                    else:
                        return
                    sv = stage_vec_ref["vec"]
                    if sv is None: return
                    C = x2.shape[-1]
                    for sid in range(S):
                        m = (sv == sid)
                        if m.any():
                            v = x2[m].mean(dim=0).double().cpu()
                            if (bid not in mlp_in_m2[sid]) or (mlp_in_m2[sid][bid].numel() != C):
                                mlp_in_m2[sid][bid] = torch.zeros(C, dtype=torch.float64)
                                mlp_cnt[sid][bid]   = 0
                            mlp_in_m2[sid][bid] += v
                            mlp_cnt[sid][bid]   += 1
                hooks.append(fc2.register_forward_pre_hook(pre_hook))
                mlp_in_dims[bid] = int(fc2.weight.shape[1])

    for bid, blk in _iter_pruned_blocks(unet):
        _register_for_block(bid, blk)

    try:
        for batch in tqdm(dataloader, desc="[FirstOrder-Calibrate SDXL]"):
            device = next(unet.parameters()).device
            t = batch["t"].to(device, non_blocking=True).view(-1)
            stage_vec_ref["vec"] = _t_to_stage_ids(t, stages)
            _ = _sdxl_forward_unet(unet, {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()})
    finally:
        for h in hooks:
            try: h.remove()
            except Exception: pass

    out: Dict[str, Dict[str, Dict[str, List[int]]]] = {}
    for sid in range(S):
        out[str(sid)] = {}
        for bid, blk in _iter_pruned_blocks(unet):
            entry = {"attn1_heads_order": [], "attn2_heads_order": [], "mlp_channels_order": []}

            # attn1
            if (bid, "attn1") in toout_abs_rowsum:
                rowsum = toout_abs_rowsum[(bid, "attn1")]
                H, d = heads_info.get(bid, [None, None, None, None])[:2]
                if H is None or d is None:
                    C_in = rowsum.numel()
                    d = 64
                    H = max(1, C_in // d)
                if method == "wanda":
                    A = attn1_in_m2[sid].get(bid)
                    cnt = attn1_cnt[sid].get(bid, 0)
                    if A is not None and cnt > 0:
                        x2 = (A / max(1, cnt))
                        S_in = rowsum.to(x2.dtype) * x2
                    else:
                        S_in = rowsum
                elif method == "magnitude":
                    S_in = rowsum
                else:
                    A = attn1_in_m2[sid].get(bid)
                    cnt = attn1_cnt[sid].get(bid, 0)
                    S_in = (A / max(1, cnt)) if (A is not None and cnt > 0) else rowsum

                head_scores = []
                C = int(S_in.numel())
                H = max(1, min(H, C // max(d,1)))
                for h in range(H):
                    s, e = h * d, min((h + 1) * d, C)
                    head_scores.append(S_in[s:e].sum())
                heads_order = torch.argsort(torch.stack(head_scores), dim=0).tolist() if head_scores else list(range(H))
                entry["attn1_heads_order"] = [int(h) for h in heads_order]

            # attn2
            if (bid, "attn2") in toout_abs_rowsum:
                rowsum = toout_abs_rowsum[(bid, "attn2")]
                H, d = heads_info.get(bid, [None, None, None, None])[2:]
                if H is None or d is None:
                    C_in = rowsum.numel()
                    d = 64
                    H = max(1, C_in // d)
                if method == "wanda":
                    A = attn2_in_m2[sid].get(bid)
                    cnt = attn2_cnt[sid].get(bid, 0)
                    if A is not None and cnt > 0:
                        x2 = (A / max(1, cnt))
                        S_in = rowsum.to(x2.dtype) * x2
                    else:
                        S_in = rowsum
                elif method == "magnitude":
                    S_in = rowsum
                else:
                    A = attn2_in_m2[sid].get(bid)
                    cnt = attn2_cnt[sid].get(bid, 0)
                    S_in = (A / max(1, cnt)) if (A is not None and cnt > 0) else rowsum
                head_scores = []
                C = int(S_in.numel())
                H = max(1, min(H, C // max(d,1)))
                for h in range(H):
                    s, e = h * d, min((h + 1) * d, C)
                    head_scores.append(S_in[s:e].sum())
                heads_order = torch.argsort(torch.stack(head_scores), dim=0).tolist() if head_scores else list(range(H))
                entry["attn2_heads_order"] = [int(h) for h in heads_order]

            # mlp.fc2
            if bid in fc2_abs_rowsum:
                rowsum = fc2_abs_rowsum[bid]
                if method == "wanda":
                    A = mlp_in_m2[sid].get(bid)
                    cnt = mlp_cnt[sid].get(bid, 0)
                    if A is not None and cnt > 0:
                        S_in = rowsum.to(A.dtype) * (A / max(1, cnt))
                    else:
                        S_in = rowsum
                elif method == "magnitude":
                    S_in = rowsum
                else:
                    A = mlp_in_m2[sid].get(bid)
                    cnt = mlp_cnt[sid].get(bid, 0)
                    S_in = (A / max(1, cnt)) if (A is not None and cnt > 0) else rowsum
                ch_order = torch.argsort(S_in, dim=0).tolist() if S_in.numel() else []
                entry["mlp_channels_order"] = [int(c) for c in ch_order]

            out[str(sid)][str(bid)] = entry

    return out


def build_secondorder_schedule_from_orders_sdxl(
    orders_per_stage: Dict[str, Dict[str, Dict[str, List[int]]]],
    stages: List[Tuple[int, int]],
    ratios: Union[float, Dict[int, float]],
    *,
    protect_ends: int = 0,
) -> ScheduleSO:
    """
    Build a **stage-based** SDXL schedule:
      schedule[sid] = {
        "attn1": { block_id: [head_ids_to_zero] },
        "attn2": { block_id: [head_ids_to_zero] },
        "mlp":   { block_id: [channel_ids_to_zero] },
      }
    """
    schedule: ScheduleSO = {}

    for sid, (_lo, _hi) in enumerate(stages):
        r = ratios[sid] if isinstance(ratios, dict) else ratios
        r = float(max(0.0, min(1.0, r)))
        stage_dict = orders_per_stage.get(str(sid), {})

        entry_attn1: Dict[int, List[int]] = {}
        entry_attn2: Dict[int, List[int]] = {}
        entry_mlp:   Dict[int, List[int]] = {}

        for blk_str, blk_dict in stage_dict.items():
            blk = int(blk_str)
            # heads attn1
            heads1 = list(blk_dict.get("attn1_heads_order", []))
            if protect_ends > 0 and heads1:
                H = len(heads1)
                keep = set(range(protect_ends)) | set(range(H - protect_ends, H))
                heads1 = [h for h in heads1 if h not in keep and 0 <= h < H]
            n_drop_h1 = int(round(r * len(heads1)))
            drop_h1 = sorted(heads1[:n_drop_h1])

            # heads attn2
            heads2 = list(blk_dict.get("attn2_heads_order", []))
            if protect_ends > 0 and heads2:
                H = len(heads2)
                keep = set(range(protect_ends)) | set(range(H - protect_ends, H))
                heads2 = [h for h in heads2 if h not in keep and 0 <= h < H]
            n_drop_h2 = int(round(r * len(heads2)))
            drop_h2 = sorted(heads2[:n_drop_h2])

            # mlp
            ch = list(blk_dict.get("mlp_channels_order", []))
            if protect_ends > 0 and ch:
                F = len(ch)
                keep = set(range(protect_ends)) | set(range(F - protect_ends, F))
                ch = [c for c in ch if c not in keep]
            n_drop_c = int(round(r * len(ch)))
            drop_c = sorted(ch[:n_drop_c])

            if drop_h1: entry_attn1[blk] = drop_h1
            if drop_h2: entry_attn2[blk] = drop_h2
            if drop_c:  entry_mlp[blk]   = drop_c

        schedule[sid] = {"attn1": entry_attn1, "attn2": entry_attn2, "mlp": entry_mlp}

    return schedule


# ===================== Second-Order (OBS) — SDXL =====================

@torch.no_grad()
def calibrate_secondorder_orders_sdxl(
    unet: nn.Module,
    dataloaders_per_stage,               # list/tuple, len == len(stages)
    stages: List[Tuple[int, int]],
    *,
    obs_cache_dir: Optional[str] = "./pretrained_models/obs_bank_sdxl",
    obs_level_max: Optional[int] = 10,
    head_dim: int = 64,
) -> Tuple[Dict[str, Dict[str, Dict[str, List[int]]]], dict]:
    """
    SDXL OBS calibration (stage-wise). Produces orders and an OBS repo manifest.
    """
    assert isinstance(dataloaders_per_stage, (list, tuple)) and len(dataloaders_per_stage) == len(stages), \
        "dataloaders_per_stage must be a list/tuple with one loader per stage."

    cache_root = Path(obs_cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    (cache_root / "meta.json").write_text(json.dumps({
        "stages": [(int(lo), int(hi)) for (lo, hi) in stages],
        "targets": ["attn1.to_out.0", "attn2.to_out.0", "ff.net.2"],
        "model": "sdxl-unet",
    }, indent=2))

    _blk_pat = re.compile(r"attentions_(\d+)_")

    def _blk_id_from_layer_dir(dirname: str) -> int:
        m = _blk_pat.search(dirname)
        if m:
            return int(m.group(1))
        m2 = re.search(r"block_(\d+)", dirname)
        return int(m2.group(1)) if m2 else -1

    traj_manifests: List[Dict[str, List[List[int]]]] = []
    for sid, loader in enumerate(dataloaders_per_stage):
        stage_dir = cache_root / f"stage_{sid}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        print(f"[OBS-Calibrate SDXL] Stage {sid} → {stage_dir}")

        manifest = simple_prune_traj_sdxl(
            unet,
            loader,
            save_dir=str(stage_dir),
            obs_level_max=(obs_level_max if obs_level_max is not None else 1),
            default_head_dim=head_dim,
        )
        traj_manifests.append(manifest)

    orders: Dict[str, Dict[str, Dict[str, List[int]]]] = {}

    for sid, stage_dir in enumerate(sorted([p for p in cache_root.glob("stage_*") if p.is_dir()],
                                           key=lambda p: int(re.findall(r"stage_(\d+)", p.name)[0]))):
        orders[str(sid)] = {}

        for ldir in [p for p in stage_dir.iterdir() if p.is_dir()]:
            bid = _blk_id_from_layer_dir(ldir.name)
            if bid < 0:
                continue

            k_files = sorted(ldir.glob("k_*.pt"), key=lambda p: int(re.findall(r"k_(\d+)", p.name)[0]))
            if not k_files:
                continue

            sample_pack = torch.load(k_files[0], map_location="cpu")
            typ = str(sample_pack.get("type", ""))
            gsize = int(sample_pack.get("group_size", (head_dim if "attn" in typ else 1)))

            kept_seq = []
            for f in k_files:
                pack = torch.load(f, map_location="cpu")
                kept_seq.append(set(int(i) for i in pack["kept_idx"]))

            entry = orders[str(sid)].setdefault(str(bid), {
                "attn1_heads_order": [],
                "attn2_heads_order": [],
                "mlp_channels_order": [],
            })

            if typ.startswith("attn1") or typ.startswith("attn2"):
                dropped_heads: List[int] = []
                prev = kept_seq[0]
                for ks in kept_seq[1:]:
                    dropped_cols = list(prev - ks)
                    for c in dropped_cols:
                        h = int(c) // max(1, gsize)
                        if h not in dropped_heads:
                            dropped_heads.append(h)
                    prev = ks
                if len(kept_seq) > 0:
                    kept_len = 0
                    try:
                        pack0 = torch.load(k_files[0], map_location="cpu")
                        kept_len = int(pack0["weight"].shape[1])
                    except Exception:
                        pass
                    H_guess = (kept_len // max(1, gsize)) if kept_len > 0 else 0
                    if H_guess > 0:
                        for h in range(H_guess):
                            if h not in dropped_heads:
                                dropped_heads.append(h)

                if typ.startswith("attn1"):
                    entry["attn1_heads_order"] = [int(h) for h in dropped_heads]
                else:
                    entry["attn2_heads_order"] = [int(h) for h in dropped_heads]

            else:
                dropped_channels: List[int] = []
                prev = kept_seq[0]
                for ks in kept_seq[1:]:
                    dropped_cols = [int(c) for c in (prev - ks)]
                    for c in dropped_cols:
                        if c not in dropped_channels:
                            dropped_channels.append(c)
                    prev = ks
                try:
                    W0 = sample_pack["weight"]
                    F_in = int(W0.shape[1])
                    for c in range(F_in):
                        if c not in dropped_channels:
                            dropped_channels.append(c)
                except Exception:
                    pass
                entry["mlp_channels_order"] = [int(c) for c in dropped_channels]

    repo = build_obs_repo_from_cache_sdxl(cache_root)
    return orders, repo


def apply_secondorder_schedule(
    unet: nn.Module,
    schedule: ScheduleSO,
    stages: Optional[List[Tuple[int, int]]] = None
) -> None:
    """
    Install a **stage-based** second-order structural schedule on UNet.
    """
    def _norm_sched(sch: ScheduleSO) -> ScheduleSO:
        out: ScheduleSO = {}
        for sk, ent in (sch or {}).items():
            sid = int(sk)
            a1 = {int(b): [int(h) for h in hs] for b, hs in ent.get("attn1", {}).items()}
            a2 = {int(b): [int(h) for h in hs] for b, hs in ent.get("attn2", {}).items()}
            ml = {int(b): [int(c) for c in cs] for b, cs in ent.get("mlp",   {}).items()}
            out[sid] = {"attn1": a1, "attn2": a2, "mlp": ml}
        return out

    if not schedule:
        if hasattr(unet, "set_secondorder_struct_schedule"):
            unet.set_secondorder_struct_schedule({}, stages=None)
        if hasattr(unet, "set_projection_bank"):
            try: unet._install_projection_bank({}, stages=None)
            except Exception: pass
        return

    sched_norm = _norm_sched(schedule)
    if hasattr(unet, "set_secondorder_struct_schedule"):
        # Pass stage-based schedule plus stages layout for reference
        unet.set_secondorder_struct_schedule(sched_norm, stages)
    else:
        unet._so_struct_schedule = sched_norm
        unet._so_struct_stages   = stages


# ===================== OBS (SDXL) =====================

class TapDiffOBS:
    """
    OBS core for Linear (Conv2d kept for parity).
    """
    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev, dtype=torch.double)
        self.scaler_row = torch.zeros((self.columns), device=self.dev, dtype=torch.double)
        self.nsamples = 0

    def add_batch(self, inp, out):
        inp = inp.detach()
        out = out.detach()
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        if isinstance(self.layer, nn.Conv2d):
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)
        inp = inp.double()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        self.H += 2 / self.nsamples * (inp.matmul(inp.t())).double()

    def invert(self, H, percentdamp=.01):
        try:
            Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H))
            if torch.isnan(Hinv).any():
                raise RuntimeError
        except RuntimeError:
            diagmean = torch.mean(torch.diag(H))
            print('Hessian not full rank.')
            tmp = (percentdamp * diagmean) * torch.eye(self.columns, device=self.dev)
            Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H + tmp))
        return Hinv

    def prepare(self):
        print("Preparing pruning...")
        W = self.layer.weight.data.clone().double()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        H = self.H.double()
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0
        Hinv = self.invert(H)
        Losses = torch.zeros([self.rows, self.columns + 1], device=self.dev)
        return W, H, Hinv, Losses

    def prune_struct(self, pruned, size=1, return_mode: str = "final"):
        assert return_mode in ("final", "traj")
        milestones = pruned[:] if isinstance(pruned, (list, tuple)) else [int(pruned)]

        W, H, Hinv, _ = self.prepare()
        count = self.columns // size
        rangecount = torch.arange(count, device=self.dev)
        rangecolumns = torch.arange(self.columns, device=self.dev)

        mask = torch.zeros(count, device=self.dev).bool()
        mask1 = None
        if size > 1:
            mask1 = torch.zeros(self.columns, device=self.dev).bool()

        res = []
        kept_indices_list = []
        Losses = torch.zeros(count + 1, device=self.dev)

        def _kept_indices():
            if size == 1:
                return torch.arange(self.columns, device=self.dev)[~mask]
            else:
                return torch.arange(self.columns, device=self.dev)[~mask1]

        def _record_snapshot(dropped_k: int, print_loss: bool = True):
            kept = _kept_indices()
            Wk_thin = W.index_select(1, kept)
            res.append(Wk_thin.to(dtype=torch.float32, device="cpu"))
            kept_indices_list.append(kept.cpu().tolist())
            if print_loss:
                if dropped_k == 0:
                    print(f"{0:4d} error", 0.0)
                else:
                    print(f"{dropped_k:4d} error", torch.sum(Losses[:dropped_k + 1]).item() / 2)

        if size == 1:
            for dropped in range(count):
                diag = torch.diagonal(Hinv)
                scores = torch.sum(W ** 2, 0) / diag
                scores[mask] = float('inf')
                j = torch.argmin(scores)
                Losses[dropped + 1] = scores[j]
                row = Hinv[j, :]
                d = diag[j]
                W -= ((W[:, j] / d).unsqueeze(1)).matmul(row.unsqueeze(0))
                mask[j] = True
                W[:, mask] = 0
                row /= torch.sqrt(d)
                Hinv -= row.unsqueeze(1).matmul(row.unsqueeze(0))
                k_now = dropped + 1
                while milestones and k_now == milestones[0]:
                    _record_snapshot(k_now)
                    milestones.pop(0)
                    if not milestones:
                        return res, kept_indices_list
        else:
            for dropped in range(count):
                blocks = Hinv.reshape(count, size, count, size)
                blocks = blocks[rangecount, :, rangecount, :]
                try:
                    invblocks = torch.cholesky_inverse(torch.linalg.cholesky(blocks))
                except Exception:
                    invblocks = torch.linalg.pinv(blocks, hermitian=True)

                W1 = W.reshape((self.rows, count, size)).transpose(0, 1)
                lambd = torch.bmm(W1, invblocks)
                scores = torch.sum(lambd * W1, (1, 2))
                scores[mask] = float('inf')
                j = torch.argmin(scores)

                Losses[dropped + 1] = scores[j]
                rows = Hinv[(size * j):(size * (j + 1)), :]
                d = invblocks[j]
                W -= lambd[j].matmul(rows)

                mask[j] = True
                mask1[(size * j):(size * (j + 1))] = True
                W[:, mask1] = 0

                Hinv -= rows.t().matmul(d.matmul(rows))
                Hinv[rangecolumns[mask1], rangecolumns[mask1]] = 1

                k_now = dropped + 1
                while milestones and k_now == milestones[0]:
                    _record_snapshot(k_now)
                    milestones.pop(0)
                    if not milestones:
                        return res, kept_indices_list

        return res, kept_indices_list

    def free(self):
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()


def _find_linear_targets_sdxl(unet: nn.Module) -> Dict[str, nn.Linear]:
    """
    Collect target Linears:
      - "*attn1.to_out.0"
      - "*attn2.to_out.0"
      - "*ff.net.2"
    """
    targets = {}
    for name, m in unet.named_modules():
        if name.endswith("attn1.to_out.0") and isinstance(m, nn.Linear):
            targets[name] = m
        elif name.endswith("attn2.to_out.0") and isinstance(m, nn.Linear):
            targets[name] = m
        elif name.endswith("ff.net.2") and isinstance(m, nn.Linear):
            targets[name] = m
    return targets


@torch.no_grad()
def simple_prune_traj_sdxl(
    unet: nn.Module,
    dataloader,
    save_dir: str,
    *,
    obs_level_max: int,
    default_head_dim: int = 64,
):
    """
    Accumulate OBS stats for SDXL target Linear layers and save *full trajectories*
    k=1..obs_level_max (attn heads) or spaced channel counts (mlp.fc2).
    """
    layers = _find_linear_targets_sdxl(unet)
    tapdiff_objects = {name: TapDiffOBS(layer) for name, layer in layers.items()}

    hooks = []
    def add_batch_factory(name):
        def add_batch_hook(layer, inp, out):
            tapdiff_objects[name].add_batch(inp[0], out)
        return add_batch_hook

    for name, layer in layers.items():
        hooks.append(layer.register_forward_hook(add_batch_factory(name)))

    device = next(unet.parameters()).device
    for batch in tqdm(dataloader, desc="[OBS-Accumulate SDXL]"):
        batch_dev = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        _ = _sdxl_forward_unet(unet, batch_dev)

    for h in hooks:
        try: h.remove()
        except Exception: pass

    save_root = Path(save_dir)
    save_root.mkdir(parents=True, exist_ok=True)

    traj_manifest = {}

    for name, tapdiff in tapdiff_objects.items():
        layer = layers[name]
        if name.endswith("to_out.0"):
            head_dim = default_head_dim
            num_heads = None
            C_in = int(layer.weight.shape[1])
            if C_in % default_head_dim == 0:
                num_heads = C_in // default_head_dim
            size = head_dim
            pruned = [i + 1 for i in range(obs_level_max)]
            thin_type = "attn1_to_out" if ".attn1." in name else "attn2_to_out"
        else:
            size = 1
            F_in = int(layer.weight.shape[1])
            pruned = [max(1, round(((i + 1) / obs_level_max) * F_in)) for i in range(obs_level_max)]
            pruned = sorted(list(set(pruned)))
            thin_type = "mlp_fc2"

        t0 = time.time()
        weights_seq, kept_seq = tapdiff.prune_struct(pruned=pruned, size=size, return_mode="traj")
        print(f"Pruned layer: {name}")
        print("  weights_seq len:", len(weights_seq), " kept_seq len:", len(kept_seq))
        # print("weights_seq :", weights_seq)
        # print("kept_seq :", kept_seq)

        safe_name = name.replace('.', '_')
        layer_dir = save_root / safe_name
        layer_dir.mkdir(parents=True, exist_ok=True)

        traj_manifest[name] = kept_seq

        for i, Wk in enumerate(weights_seq):
            kept = kept_seq[i]
            k = pruned[i]
            payload = {
                "weight":    Wk,
                "kept_idx":  kept,
                "group_size": int(size),
                "k":         int(k),
                "type":      thin_type,
                "layer_name": name,
            }
            if "attn" in thin_type and num_heads is not None:
                payload["num_heads"] = int(num_heads)
            torch.save(payload, layer_dir / f"k_{k}.pt")

        t1 = time.time()
        print(f"Saved OBS trajectory for {name} -> {layer_dir}, time: {t1 - t0:.2f}s")
        tapdiff.free()

    return traj_manifest


# ===================== OBS bank I/O (SDXL) =====================

def _read_thin_meta(thin_obj: dict) -> dict:
    meta = {
        "group_size": int(thin_obj.get("group_size", 1)),
        "kept_len": int(thin_obj["weight"].shape[1]),
        "type": str(thin_obj.get("type", "")),
        "k": int(thin_obj.get("k", 0)),
        "has_bias": thin_obj.get("bias", None) is not None,
    }
    if "num_heads" in thin_obj:
        meta["num_heads"] = int(thin_obj["num_heads"])
    return meta


def build_obs_repo_from_cache_sdxl(cache_root: Path) -> dict:
    """
    Scan `cache_root` for SDXL thin packs saved by simple_prune_traj_sdxl:

        <cache_root>/stage_<sid>/<safe_layer_name>/k_*.pt

    Design goals:
      - Always keep one entry per *actual layer* (no collapsing by a guessed block id).
      - Prefer `layer_name` stored inside each k_*.pt as the key.
      - Fall back to the folder name (per-folder unique), NEVER to a coarse int id.
      - Store useful meta: group_size, num_heads (attn), F_in (mlp), k_min/k_max.

    Output format:
      repo = {
        "cache_root": <abs path>,
        "entries": {
          sid: {
            "<layer_name>": {
              "attn1": { "ks":{k:int->path}, "group_size":..., "num_heads":..., "k_min":..., "k_max":... },
              "attn2": {...},
              "mlp":   { "ks":{k:int->path}, "group_size":..., "F_in":...,    "k_min":..., "k_max":... },
            },
            ...
          },
          ...
        },
        "meta": {...}   # reserved
      }
    """
    cache_root = Path(cache_root)
    meta = {}
    entries: Dict[int, Dict[str, Dict[str, dict]]] = {}

    # Find stage dirs; if none, treat cache_root as a single stage 0.
    stage_dirs = sorted(
        [p for p in cache_root.glob("stage_*") if p.is_dir()],
        key=lambda p: int(re.findall(r"stage_(\d+)", p.name)[0]) if re.findall(r"stage_(\d+)", p.name) else -1,
    )
    if not stage_dirs:
        stage_dirs = [cache_root]

    for s_dir in stage_dirs:
        sid_match = re.findall(r"stage_(\d+)", s_dir.name)
        sid = int(sid_match[0]) if sid_match else 0
        entries.setdefault(sid, {})

        # Each immediate subdir is a layer folder
        for ldir in [p for p in s_dir.iterdir() if p.is_dir()]:
            # Collect all k_*.pt files
            k_files = sorted(
                [p for p in ldir.glob("k_*.pt") if p.is_file()],
                key=lambda p: int(re.findall(r"k_(\d+)", p.name)[0]) if re.findall(r"k_(\d+)", p.name) else 0,
            )
            if not k_files:
                continue

            # Load one sample pack to read metadata
            try:
                sample_pack = torch.load(k_files[0], map_location="cpu")
            except Exception:
                # If we can't read the sample pack, skip this folder
                continue

            # ---------- Determine key (layer identifier) ----------
            layer_key = None
            if isinstance(sample_pack, dict) and isinstance(sample_pack.get("layer_name", None), str):
                # Preferred: exact module path
                layer_key = sample_pack["layer_name"]  # e.g. "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_out.0"
            if not layer_key:
                # Fallback: use the folder name; still unique per folder, but no collapsing by guessed block id.
                layer_key = ldir.name

            # ---------- Determine kind (attn1 / attn2 / mlp) ----------
            kind = None
            if isinstance(sample_pack, dict):
                t = str(sample_pack.get("type", ""))
                if t.startswith("attn1"):
                    kind = "attn1"
                elif t.startswith("attn2"):
                    kind = "attn2"
                elif "mlp" in t:
                    kind = "mlp"

            if kind is None:
                # Very defensive fallback based on folder naming
                name = ldir.name
                if "attn1" in name and "to_out" in name:
                    kind = "attn1"
                elif "attn2" in name and "to_out" in name:
                    kind = "attn2"
                elif "ff_net_2" in name or "mlp_fc2" in name or name.endswith("ff_net_2"):
                    kind = "mlp"

            if kind is None:
                # Unknown type → skip
                continue

            # ---------- Collect all ks ----------
            ks: Dict[int, str] = {}
            for f in k_files:
                m = re.findall(r"k_(\d+)", f.name)
                if not m:
                    continue
                k_val = int(m[0])
                ks[k_val] = str(f.resolve())
            if not ks:
                continue

            k_min = min(ks.keys())
            k_max = max(ks.keys())

            # ---------- Store meta ----------
            entries.setdefault(sid, {})
            per_layer = entries[sid].setdefault(layer_key, {})

            info: Dict[str, Union[int, Dict[int, str]]] = {
                "ks": ks,
                "k_min": k_min,
                "k_max": k_max,
            }

            if isinstance(sample_pack, dict):
                g = int(sample_pack.get("group_size", 1))
                info["group_size"] = g
                if kind in ("attn1", "attn2") and "num_heads" in sample_pack:
                    info["num_heads"] = int(sample_pack["num_heads"])
                if kind == "mlp":
                    try:
                        W0 = sample_pack["weight"]
                        info["F_in"] = int(W0.shape[1])
                    except Exception:
                        pass

            per_layer[kind] = info

    repo = {
        "cache_root": str(cache_root.resolve()),
        "entries": entries,
        "meta": meta,
    }
    return repo



def save_obs_bank(repo: dict, out_dir: str) -> str:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "repo.pt"
    repo = dict(repo)
    repo["cache_root"] = str(Path(repo.get("cache_root", out_dir)).resolve())
    torch.save(repo, path)
    return str(path)


def load_obs_bank(bank_path_or_dir: str) -> dict:
    p = Path(bank_path_or_dir)
    if p.is_dir():
        p = p / "repo.pt"
    repo = torch.load(str(p), map_location="cpu")
    if not ("entries" in repo and "cache_root" in repo):
        root = Path(repo.get("root", p.parent))
        repo = build_obs_repo_from_cache_sdxl(root)
    assert "entries" in repo and "cache_root" in repo, "Invalid OBS repo."
    return repo


def _round_to_int(x: float, mode: str) -> int:
    if mode == "floor": return int(math.floor(x))
    if mode == "ceil":  return int(math.ceil(x))
    return int(round(x))


def select_obs_bank_for_ratios_sdxl(repo, ratios, stages, round_mode: str = "nearest") -> dict:
    assert round_mode in ("nearest", "floor", "ceil")
    entries = repo.get("entries", {})
    bank: Dict[int, dict] = {}

    def _round_to_int_local(x: float) -> int:
        if round_mode == "floor":
            return int(math.floor(x))
        if round_mode == "ceil":
            return int(math.ceil(x))
        return int(round(x))

    for sid, _stage in enumerate(stages):
        r = float(ratios[sid] if isinstance(ratios, dict) else ratios)
        stage_map = entries.get(sid, {})
        out_a1, out_a2, out_m = {}, {}, {}

        if r <= 0.0:
            # Entire stage uses full (unpruned) structure → no projections
            bank[sid] = {"attn1": out_a1, "attn2": out_a2, "mlp": out_m}
            continue

        for key, kinds in stage_map.items():  # key is layer_name (str) in the new repo
            # ----- ATTENTION (attn1 / attn2) -----
            for kind_name in ("attn1", "attn2"):
                if kind_name not in kinds:
                    continue
                kinfo = kinds[kind_name]
                ks = sorted(kinfo["ks"].keys())
                if not ks:
                    continue

                k_max = int(kinfo.get("k_max", ks[-1]))
                k_min = int(kinfo.get("k_min", ks[0]))

                # ratio is fraction of OBS depth
                k_target = _round_to_int_local(r * k_max)
                # clamp into valid range
                if k_target <= 0:
                    # too small → effectively no drop, skip this layer
                    continue
                if k_target < k_min:
                    k_target = k_min
                if k_target > k_max:
                    k_target = k_max

                # pick available k closest to k_target
                k_sel = min(ks, key=lambda k: abs(k - k_target))
                pack = torch.load(kinfo["ks"][k_sel], map_location="cpu")

                payload = {
                    "proj_w":  pack["weight"].to(torch.float32),
                    "kept_idx": [int(i) for i in pack["kept_idx"]],
                    "head_dim": int(pack.get("group_size", kinfo.get("group_size", 64))),
                }
                if kind_name == "attn1":
                    out_a1[key] = payload
                else:
                    out_a2[key] = payload

            # ----- MLP (fc2) -----
            if "mlp" in kinds:
                kinfo = kinds["mlp"]
                ks = sorted(kinfo["ks"].keys())
                if not ks:
                    continue

                k_max = int(kinfo.get("k_max", ks[-1]))
                k_min = int(kinfo.get("k_min", ks[0]))

                k_target = _round_to_int_local(r * k_max)
                if k_target <= 0:
                    continue
                if k_target < k_min:
                    k_target = k_min
                if k_target > k_max:
                    k_target = k_max

                k_sel = min(ks, key=lambda k: abs(k - k_target))
                pack = torch.load(kinfo["ks"][k_sel], map_location="cpu")

                out_m[key] = {
                    "fc2_w":   pack["weight"].to(torch.float32),
                    "kept_idx": [int(i) for i in pack["kept_idx"]],
                }

        bank[sid] = {"attn1": out_a1, "attn2": out_a2, "mlp": out_m}

    return bank



# ===================== Glue for your UNet2DConditionPruned =====================

def install_projection_bank(
    unet: nn.Module,
    repo_or_bank: Union[str, dict],
    stages: Optional[List[Tuple[int, int]]],
    ratios_per_stage: Optional[Dict[int, float]] = None,
    round_mode: str = "nearest",
):
    """
    Install an OBS bank onto UNet. If `repo_or_bank` is a repo dict (entries/paths),
    and `ratios_per_stage` is provided, we will *load & select* the tensors now.
    If it's already a loaded bank (stage->attn1/2/mlp with tensors), we install directly.
    """
    if isinstance(repo_or_bank, str) or ("entries" in repo_or_bank and "cache_root" in repo_or_bank):
        repo = load_obs_bank(repo_or_bank) if isinstance(repo_or_bank, str) else repo_or_bank
        assert ratios_per_stage is not None, "ratios_per_stage is required when passing an OBS repo."
        bank = select_obs_bank_for_ratios_sdxl(repo, ratios_per_stage, stages, round_mode=round_mode)
    else:
        bank = repo_or_bank

    if hasattr(unet, "set_projection_bank"):
        unet.set_projection_bank(bank, stages)
        def print_bank_summary(unet, T_check=5):
            by_t = getattr(unet, "_pb_by_t", None)
            depth = getattr(unet, "_pb_depth", None)
            print(f"[bank] compiled={by_t is not None}  depth={depth}")
            if by_t is None:
                return
            T = len(by_t)
            for t in range(min(T_check, T)):
                row = by_t[t]
                if row is None:
                    print(f"t={t}: <no bank>")
                    continue
                a1, a2, mlp = row
                def _cnt(lst): return sum(1 for x in lst if x is not None)
                print(f"t={t}: attn1={_cnt(a1)} attn2={_cnt(a2)} mlp={_cnt(mlp)}")

        # after install_projection_bank(...)
        print_bank_summary(unet, T_check=10)
    else:
        unet._proj_bank = bank
        unet._proj_bank_stages = stages


# LayerDrop
calibrate_layerdrop_orders = calibrate_layerdrop_orders_sdxl

# First-order
calibrate_firstorder_orders = calibrate_firstorder_orders_sdxl

# Second-order schedule builder (explicit attn1/attn2)
build_secondorder_schedule_from_orders = build_secondorder_schedule_from_orders_sdxl

# ---- Back-compat alias so upstream calls work unchanged ----
calibrate_secondorder_orders = calibrate_secondorder_orders_sdxl

# OBS trajectory dumper (SDXL)
simple_prune_traj_dit = simple_prune_traj_sdxl  # keep symbol name so upstream calls remain valid

# OBS repo builder (SDXL)
build_obs_repo_from_cache = build_obs_repo_from_cache_sdxl

# OBS bank selector (SDXL)
select_obs_bank_for_ratios = select_obs_bank_for_ratios_sdxl