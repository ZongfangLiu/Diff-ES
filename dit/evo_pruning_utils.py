# evo_pruning_utils.py
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm

from collections import defaultdict
from typing import Iterable, Callable, Dict, List, Literal, Tuple, Union, Optional
from pathlib import Path
import os, json, re
import time
from collections import defaultdict
import math


# ===================== LayerDrop (stage-aware calibration) =====================

Schedule = Union[Dict[int, Iterable[int]], Callable[[int], Iterable[int]]]

def _t_to_stage_ids(t_tensor: torch.Tensor, stages: List[Tuple[int, int]]) -> torch.Tensor:
    """
    Map each timestep in t_tensor to a stage index (0..len(stages)-1).
    If a t falls in no stage, we mark it as -1 and ignore it.
    """
    assert t_tensor.ndim == 1, "Expecting t as a 1D tensor of shape (B,)."
    stage_ids = torch.full_like(t_tensor, fill_value=-1, dtype=torch.long)
    for sid, (lo, hi) in enumerate(stages):
        mask = (t_tensor >= lo) & (t_tensor <= hi)
        stage_ids[mask] = sid
    return stage_ids


@torch.no_grad()
def calibrate_layerdrop_orders(
    model: nn.Module,
    dataloader,
    stages: List[Tuple[int, int]],
    cfg_scale: float = 4.0,
    importance_metric: Literal["mse", "cosine"] = "cosine",
    cosine_eps: float = 1e-8,
) -> Tuple[Dict[int, List[int]], Dict[int, List[float]]]:
    """
    Compute a per-stage ranking (order) of Transformer blocks from least->most important.

    How importance is measured:
      For each batch, and for each DiT block i:
        - We compute the per-sample mean squared delta: ||out - in||^2 across non-batch dims.
        - We accumulate that value to the stage of each sample's timestep t.
      Intuition: blocks that "move" the representation less (under that stage's noise level)
                 are safer to drop.

    Args:
        model: your DiT instance (already supports layer drop; we won't use it here).
        dataloader: the SAME calibration dataloader you use elsewhere in this repo.
                    It must yield dicts with keys {'x','t','y'} as in your code.
        stages: list of (lo, hi) inclusive step ranges, e.g. [(0,199), (200,599), (600,999)].
        cfg_scale: if >1.0, we will invoke model.forward_with_cfg (matching your pipeline).

    Returns:
        orders_per_stage: dict[stage_index] -> list of block indices sorted
                          LEAST important first (i.e., drop order).
        scores_per_stage: dict[stage_index] -> list of raw scores per block (same index order)
                          so you can inspect/plot/debug later.

    Notes:
      - This uses forward hooks on each Block (not per-layer), so it’s cheap and robust.
      - We do NOT need to run the full diffusion sampler; we just forward the model
        on (x, t, y) batches the dataloader provides (as your pruning code already does).
      - We respect CFG the same way your pruning utils do (duplicating x/t/y).
    """
    device = next(model.parameters()).device
    using_cfg = cfg_scale > 1.0

    assert hasattr(model, "blocks"), "Expected model.blocks (ModuleList of DiTBlock)."
    num_blocks = len(model.blocks)

    stage_scores = {sid: torch.zeros(num_blocks, dtype=torch.float64) for sid in range(len(stages))}
    current_stage_ids = {"vec": None}  # mutated by the loop so hooks can read it

    def make_block_hook(block_idx: int):
        def hook(_module, inputs, output):
            x_in = inputs[0]  # DiTBlock forward signature: block(x, c)
            B = output.shape[0]

            if importance_metric == "mse":
                # mean over all non-batch dims
                delta = (output - x_in).detach()
                vals = delta.reshape(B, -1).pow(2).mean(dim=1)  # (B,)
            else:
                # cosine change = 1 - cos(out, in), flattened per sample
                a = output.detach().reshape(B, -1)
                b = x_in.detach().reshape(B, -1)
                # F.cosine_similarity is already numerically stable; eps guards zero-norm edge cases
                a_norm = a.norm(dim=1).clamp_min(cosine_eps)
                b_norm = b.norm(dim=1).clamp_min(cosine_eps)
                cos = (a * b).sum(dim=1) / (a_norm * b_norm)
                # clamp tiny drift beyond [-1, 1] from fp error
                cos = cos.clamp(-1.0, 1.0)
                vals = 1.0 - cos  # (B,)

            stage_vec = current_stage_ids["vec"]  # (B,), long
            if stage_vec is None:
                return
            for sid in range(len(stages)):
                mask = (stage_vec == sid)
                if mask.any():
                    stage_scores[sid][block_idx] += vals[mask].sum().double().cpu()
        return hook

    hooks = [blk.register_forward_hook(make_block_hook(i)) for i, blk in enumerate(model.blocks)]

    try:
        for batch in tqdm(dataloader, desc="[LayerDrop-Calibrate]"):
            x = batch["x"].to(device, non_blocking=True)
            t = batch["t"].to(device, non_blocking=True).view(-1)
            y = batch["y"].to(device, non_blocking=True).view(-1)

            if using_cfg:
                x = torch.cat([x, x], dim=0)
                t = torch.cat([t, t], dim=0)
                y_null = torch.full_like(y, fill_value=getattr(model, "num_classes", 1000))
                y = torch.cat([y, y_null], dim=0)

            current_stage_ids["vec"] = _t_to_stage_ids(t, stages)

            if using_cfg:
                _ = model.forward_with_cfg(x, t, y, cfg_scale)
            else:
                _ = model(x, t, y)
    finally:
        for h in hooks:
            h.remove()

    orders_per_stage: Dict[int, List[int]] = {}
    scores_per_stage: Dict[int, List[float]] = {}
    for sid in range(len(stages)):
        scores = stage_scores[sid].numpy().tolist()
        scores_per_stage[sid] = scores
        order = list(np.argsort(scores))  # ascending => drop-first
        orders_per_stage[sid] = order

    return orders_per_stage, scores_per_stage


def build_layerdrop_schedule_from_orders(
    orders_per_stage: Dict[int, List[int]],
    stages: List[Tuple[int, int]],
    ratios: Union[float, Dict[int, float]],
    protect_ends: int = 1,
) -> Dict[int, List[int]]:
    """
    Turn per-stage orders into a per-step schedule of indices to skip.

    Args:
        orders_per_stage: dict[stage] -> list of block ids sorted least->most important.
        stages: same list you passed to calibration ([(lo,hi), ...]).
        ratios: either a single float to apply to all stages, or a dict {stage: ratio}.
        protect_ends: keep this many blocks at each end off-limits for dropping.

    Returns:
        schedule: dict[step] -> list of block indices to skip at that step.
    """
    schedule: Dict[int, List[int]] = {}
    # derive depth from any order list
    any_stage = next(iter(orders_per_stage))
    depth = len(orders_per_stage[any_stage])

    # Build a protected mask
    protected = set(range(protect_ends)) | set(range(depth - protect_ends, depth))

    for sid, (lo, hi) in enumerate(stages):
        ratio = ratios[sid] if isinstance(ratios, dict) else ratios
        ratio = float(max(0.0, min(1.0, ratio)))
        k = int(depth * ratio)

        # take k from the drop-first list but skip protected indices
        order = [i for i in orders_per_stage[sid] if i not in protected]
        drop_ids = order[:k]
        drop_ids.sort()

        for step in range(int(lo), int(hi) + 1):
            schedule[step] = drop_ids

    return schedule


def apply_layerdrop(model: nn.Module, drop_indices: Iterable[int]) -> None:
    """
    Static layer drop: always skip these blocks every step.
    """
    if hasattr(model, "set_layerdrop"):
        model.set_layerdrop(drop_indices)
    else:
        model.drop_block_ids = set(int(i) for i in drop_indices)

def apply_layerdrop_schedule(model: nn.Module, schedule: Schedule, stages=None) -> None:
    # Transfer to int keys, JSON keys are str by default
    schedule = {int(k): v for k, v in schedule.items()}
    if hasattr(model, "set_layerdrop_schedule"):
        model.set_layerdrop_schedule(schedule, stages=stages)
        print("Using built-in set_layerdrop_schedule, schedule loaded.")
    else:
        model.layerdrop_schedule = schedule
        if hasattr(model, "layerdrop_stages"):
            model.layerdrop_stages = stages
            
# ==============================================================================


# ===================== Second-Order (no-update) — NEW =====================

ScheduleSO = Dict[int, Dict[str, Dict[int, List[int]]]]  # {t: {"attn": {blk:[heads]}, "mlp": {blk:[channels]}}}

@torch.no_grad()
def calibrate_secondorder_orders(
    model: nn.Module,
    dataloader,  # MUST be a list/tuple of per-stage loaders for OBS
    stages: List[Tuple[int, int]],
    *,
    cfg_scale: float = 4.0,
    pruner: str = "obs",                # OBS only
    method: str = "greedy",             # kept for API compatibility (unused)
    head_dim: int = 64,                 # fallback; inferred per block if possible
    num_heads: int = 24,                # fallback; inferred per block if possible
    # ----- OBS-specific -----
    obs_cache_dir: Optional[str] = "./pretrained_models/obs_bank",
    obs_percent_damp: float = 0.02,     # Tikhonov damping as % of diag mean (kept for compatibility)
    obs_save_precision: str = "float16",# "float32" | "float16" | "bfloat16"
    obs_level_max: int = None           # optional max level for OBS pruning (default: num_heads)
) -> Dict:
    """
    Second-Order (no-update) calibration — OBS ONLY.

    - Requires `dataloader` to be a List/Tuple with len == len(stages), one loader per stage.
    - Runs per-stage accumulation and saves full OBS trajectories for targeted layers.
    - Returns:
        (orders, repo)
      where:
        orders[str(stage)][str(block)] = {
          "attn_heads_order": List[int],
          "mlp_channels_order": List[int],
        }
        repo is a lightweight manifest built from saved thin packs (no tensor payloads).
    """
    if pruner.lower() != "obs":
        raise ValueError(
            "calibrate_secondorder_orders now supports only pruner='obs'. "
            "Use calibrate_firstorder_orders(..., method='activation' | 'wanda' | 'magnitude') "
            "for first-order methods."
        )

    obs_level_max = obs_level_max if obs_level_max is not None else num_heads

    assert hasattr(model, "blocks") and isinstance(model.blocks, nn.ModuleList)
    # -------------------- OBS path using simple_prune_traj_dit --------------------
    assert isinstance(dataloader, (list, tuple)) and len(dataloader) == len(stages), \
        "For OBS, dataloader must be a per-stage list/tuple with len == len(stages)."

    cache_root = Path(obs_cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    (cache_root / "meta.json").write_text(json.dumps({
        "stages": [(int(lo), int(hi)) for (lo, hi) in stages],
        "calibration_mode": "per-stage",
        "targets": ["attn.proj", "mlp.fc2"],
    }, indent=2))

    # helper to extract block index from a layer name like "blocks.7.attn.proj"
    _blk_pat = re.compile(r"blocks\.(\d+)\.")

    def _blk_idx_from_name(layer_name: str) -> int:
        m = _blk_pat.search(layer_name)
        return int(m.group(1)) if m else -1

    orders: Dict = {}

    for sid, loader in enumerate(dataloader):
        print(f"[OBS-Calibrate] Stage {sid}:")
        stage_dir = cache_root / f"stage_{sid}"
        stage_dir.mkdir(parents=True, exist_ok=True)

        # Save full trajectories under stage dir and get kept sequences
        traj_manifest = simple_prune_traj_dit(
            model, loader, cfg_scale,
            head_size=head_dim, head_num=num_heads,
            save_dir=str(stage_dir),
            obs_level_max=obs_level_max,
            attname="attn.proj", fcname="mlp.fc2"
        )

        # Build per-block orders from kept sequences (difference between k-1 and k)
        orders[str(sid)] = {}

        for layer_name, kept_seq in traj_manifest.items():
            blk_idx = _blk_idx_from_name(layer_name)
            if blk_idx < 0:
                continue  # skip if not a standard block

            entry = orders[str(sid)].setdefault(str(blk_idx), {
                "attn_heads_order": [],
                "mlp_channels_order": []
            })

            # head/channels order: read group size from name
            is_attn = ("attn.proj" in layer_name)
            if is_attn:
                # head drop order
                d = head_dim
                Hh = num_heads
                dropped_order = []
                prev_kept = set(kept_seq[0])
                # iterate along trajectory
                for k in range(1, min(Hh, len(kept_seq)-1) + 1):
                    kept_k = set(kept_seq[k])
                    dropped_cols = list(prev_kept - kept_k)
                    for c in dropped_cols:
                        h = int(c) // int(d)
                        if h not in dropped_order:
                            dropped_order.append(h)
                    prev_kept = kept_k
                # append remaining
                for h in range(Hh):
                    if h not in dropped_order:
                        dropped_order.append(h)
                entry["attn_heads_order"] = dropped_order
            else:
                # fc2 channels order
                F_in = len(kept_seq[0])
                dropped_order = []
                prev_kept = set(kept_seq[0])
                for k in range(1, len(kept_seq)):
                    kept_k = set(kept_seq[k])
                    dropped_cols = [int(c) for c in (prev_kept - kept_k)]
                    dropped_order.extend(dropped_cols)
                    prev_kept = kept_k
                # append remaining
                for c in range(F_in):
                    if c not in dropped_order:
                        dropped_order.append(int(c))
                entry["mlp_channels_order"] = dropped_order

    repo = build_obs_repo_from_cache(cache_root)
    return orders, repo


# ===================== OBS bank I/O + selection =====================

def _read_thin_meta(thin_obj: dict) -> dict:
    """
    Extract minimal metadata from a saved thin pack (k_*.pt).
    Keys present in packs saved by calibrate_secondorder_orders (OBS):
      - weight: Tensor
      - bias: Optional[Tensor]
      - kept_idx: List[int]
      - group_size: int
      - (optional) num_heads: int     # only for attn_proj
      - type: "attn_proj" | "mlp_fc2"
      - k: int
    """
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


def build_obs_repo_from_cache(cache_root: Path) -> dict:
    """
    Scan a cache folder structure created by calibrate_secondorder_orders(pruner='obs') and
    build a portable manifest (no Tensor payloads), e.g.:

    repo = {
      "cache_root": "<abs path>",
      "meta": {...},
      "entries": {
        stage: {
          block: {
            "attn_proj": {
              "group_size": d,
              "num_heads": H,            # if present
              "ks": { k: "<file path>" } # path to k_*.pt packs
            },
            "mlp_fc2": {
              "group_size": 1,
              "F_in": inferred_from_k0,
              "ks": { k: "<file path>" }
            }
          }
        }
      }
    }
    """
    cache_root = Path(cache_root)
    meta_path = cache_root / "meta.json"
    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            meta = {}

    entries = {}

    stage_dirs = sorted([p for p in cache_root.glob("stage_*") if p.is_dir()],
                        key=lambda p: int(re.findall(r"stage_(\d+)", p.name)[0]))
    for s_dir in stage_dirs:
        sid = int(re.findall(r"stage_(\d+)", s_dir.name)[0])
        entries.setdefault(sid, {})

        # ---- Preferred nested layout: stage/block_{bid}/{attn_proj|mlp_fc2}/k_*.pt
        block_dirs = sorted([p for p in s_dir.glob("block_*") if p.is_dir()],
                            key=lambda p: int(re.findall(r"block_(\d+)", p.name)[0]))
        if block_dirs:
            for b_dir in block_dirs:
                bid = int(re.findall(r"block_(\d+)", b_dir.name)[0])
                entries[sid].setdefault(bid, {})
                for kind in ("attn_proj", "mlp_fc2"):
                    kdir = b_dir / kind
                    if not kdir.exists():
                        continue
                    ks = {}
                    sample_pack = None
                    for f in sorted(kdir.glob("k_*.pt"),
                                    key=lambda p: int(re.findall(r"k_(\d+)", p.name)[0])):
                        ks[int(re.findall(r"k_(\d+)", f.name)[0])] = str(f.resolve())
                        if sample_pack is None:
                            try:
                                sample_pack = torch.load(f, map_location="cpu")
                            except Exception:
                                pass
                    if not ks:
                        continue
                    info = {"ks": ks}
                    if sample_pack is not None:
                        meta_k = _read_thin_meta(sample_pack)
                        info["group_size"] = meta_k["group_size"]
                        if kind == "attn_proj" and "num_heads" in meta_k:
                            info["num_heads"] = meta_k["num_heads"]
                        if kind == "mlp_fc2":
                            try:
                                W0 = sample_pack["weight"]
                                info["F_in"] = int(W0.shape[1])
                            except Exception:
                                pass
                    entries[sid][bid][kind] = info

        # ---- Fallback flat layout: stage/blocks_{bid}_{attn_proj|mlp_fc2}/k_*.pt
        # Only used if nested path didn’t populate anything.
        if not entries[sid]:
            layer_dirs = [p for p in s_dir.iterdir() if p.is_dir()]
            # match e.g. "blocks_7_attn_proj" or "blocks_7_mlp_fc2"
            pat = re.compile(r"blocks_(\d+)_(attn_proj|mlp_fc2)$")
            for ldir in layer_dirs:
                m = pat.match(ldir.name)
                if not m:
                    continue
                bid = int(m.group(1))
                kind = m.group(2)  # 'attn_proj' or 'mlp_fc2'
                ks = {}
                sample_pack = None
                for f in sorted(ldir.glob("k_*.pt"),
                                key=lambda p: int(re.findall(r"k_(\d+)", p.name)[0])):
                    ks[int(re.findall(r"k_(\d+)", f.name)[0])] = str(f.resolve())
                    if sample_pack is None:
                        try:
                            sample_pack = torch.load(f, map_location="cpu")
                        except Exception:
                            pass
                if not ks:
                    continue
                entries[sid].setdefault(bid, {})
                info = {"ks": ks}
                if sample_pack is not None:
                    meta_k = _read_thin_meta(sample_pack)
                    info["group_size"] = meta_k["group_size"]
                    if kind == "attn_proj" and "num_heads" in meta_k:
                        info["num_heads"] = meta_k["num_heads"]
                    if kind == "mlp_fc2":
                        try:
                            W0 = sample_pack["weight"]
                            info["F_in"] = int(W0.shape[1])
                        except Exception:
                            pass
                entries[sid][bid][kind] = info

    repo = {
        "cache_root": str(cache_root.resolve()),
        "meta": meta,
        "entries": entries,
    }
    return repo


def save_obs_bank(repo: dict, out_dir: str) -> str:
    """
    Save a portable manifest for a bank (no tensors) to <out_dir>/repo.pt and return its path.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "repo.pt"
    # ensure absolute paths so we can load from anywhere
    repo = dict(repo)
    repo["cache_root"] = str(Path(repo.get("cache_root", out_dir)).resolve())
    torch.save(repo, path)
    return str(path)


def load_obs_bank(bank_path_or_dir: str) -> dict:
    p = Path(bank_path_or_dir)
    if p.is_dir():
        p = p / "repo.pt"
    repo = torch.load(str(p), map_location="cpu")  # keep as-is; warning is harmless

    # Accept older lightweight manifest and upgrade it on the fly
    if not ("entries" in repo and "cache_root" in repo):
        root = Path(repo.get("root", p.parent))
        repo = build_obs_repo_from_cache(root)

    assert "entries" in repo and "cache_root" in repo, "Invalid OBS repo: missing keys."
    print()
    return repo


def _round_to_int(x: float, mode: str) -> int:
    if mode == "floor":
        return int(math.floor(x))
    if mode == "ceil":
        return int(math.ceil(x))
    return int(round(x))


@torch.no_grad()
def select_obs_bank_for_ratios(
    repo: dict,
    ratios: Dict[int, float],
    stages: List[Tuple[int, int]],
    round_mode: str = "nearest",
) -> dict:
    """
    Pick (and load) the per-stage thin tensors given stage-level ratios.

    Returns a RAM-resident bank ready for `DiT.set_projection_bank(bank, stages)`:

      bank[stage] = {
        "attn": { blk: { "proj_w": Tensor, "kept_idx": List[int], "head_dim": int } },
        "mlp":  { blk: { "fc2_w":  Tensor, "kept_idx": List[int] } }
      }

    Mapping ratio -> k (#groups/cols dropped):
      - attn: k = round_mode(r * H)       where H is num_heads for that block
      - mlp:  k = round_mode(r * F_in)    where F_in is fc2 input width
    We clamp k to available ks in the repo; if k not present, use the nearest available.
    """
    assert round_mode in ("nearest", "floor", "ceil")
    entries = repo.get("entries", {})
    bank: Dict[int, dict] = {}

    for sid, _stage in enumerate(stages):
        r = float(ratios[sid] if isinstance(ratios, dict) else ratios)
        stage_map = entries.get(sid, {})
        out_attn, out_mlp = {}, {}

        for bid, kinds in stage_map.items():
            # ----- Attention
            if "attn_proj" in kinds:
                kinfo = kinds["attn_proj"]
                ks = sorted(kinfo["ks"].keys())
                if not ks:
                    pass
                else:
                    H = int(kinfo.get("num_heads", 0))
                    d = int(kinfo.get("group_size", 1))
                    # If num_heads was not stored, infer from kept size at k=0
                    if H <= 0:
                        # load k=0 once to infer
                        k0_path = kinfo["ks"][min(ks)]
                        pack0 = torch.load(k0_path, map_location="cpu")
                        kept_len = int(pack0["weight"].shape[1])
                        d = int(pack0.get("group_size", d))
                        H = max(1, kept_len // d)
                    k_target = _round_to_int(r * H, round_mode)
                    k_target = max(min(k_target, ks[-1]), ks[0])
                    # choose nearest k if not exact
                    k_sel = min(ks, key=lambda k: abs(k - k_target))
                    pack = torch.load(kinfo["ks"][k_sel], map_location="cpu")
                    out_attn[int(bid)] = {
                        "proj_w":  pack["weight"].to(torch.float32),  # keep compute dtype flexible; upcast ok
                        "kept_idx": [int(i) for i in pack["kept_idx"]],
                        "head_dim": int(pack.get("group_size", d)),
                    }

            # ----- MLP
            if "mlp_fc2" in kinds:
                kinfo = kinds["mlp_fc2"]
                ks = sorted(kinfo["ks"].keys())
                if not ks:
                    pass
                else:
                    # infer F_in from k=0
                    F_in = int(kinfo.get("F_in", 0))
                    if F_in <= 0:
                        k0_path = kinfo["ks"][min(ks)]
                        pack0 = torch.load(k0_path, map_location="cpu")
                        F_in = int(pack0["weight"].shape[1])
                    k_target = _round_to_int(r * F_in, round_mode)
                    k_target = max(min(k_target, ks[-1]), ks[0])
                    k_sel = min(ks, key=lambda k: abs(k - k_target))
                    pack = torch.load(kinfo["ks"][k_sel], map_location="cpu")
                    out_mlp[int(bid)] = {
                        "fc2_w":   pack["weight"].to(torch.float32),
                        "kept_idx": [int(i) for i in pack["kept_idx"]],
                    }

        bank[sid] = {"attn": out_attn, "mlp": out_mlp}

    return bank


def build_secondorder_schedule_from_orders(
    orders_per_stage: Dict,                   # produced by calibrate_secondorder_orders
    stages: List[Tuple[int, int]],
    ratios: Union[float, Dict[int, float]],
    *,
    head_dim: int,
    num_heads: int,
    protect_ends: int = 0,
) -> ScheduleSO:
    """
    schedule[t] = {
      "attn": { block_id: [head_ids_to_zero] },
      "mlp":  { block_id: [channel_ids_to_zero] },
    }
    """
    schedule: ScheduleSO = {}
    # rough depth guess from dict structure
    stage0 = orders_per_stage.get("0", {})
    depth_guess = max([int(k) for k in stage0.keys()]) + 1 if stage0 else 0

    for sid, (lo, hi) in enumerate(stages):
        r = ratios[sid] if isinstance(ratios, dict) else ratios
        r = float(max(0.0, min(1.0, r)))
        stage_dict = orders_per_stage.get(str(sid), {})

        for t in range(int(lo), int(hi) + 1):
            entry_attn: Dict[int, List[int]] = {}
            entry_mlp: Dict[int, List[int]] = {}

            for blk in range(depth_guess):
                blk_dict = stage_dict.get(str(blk))
                if not blk_dict:
                    continue

                # heads
                heads_order = list(blk_dict.get("attn_heads_order", []))
                if protect_ends > 0 and heads_order:
                    keep = set(range(protect_ends)) | set(range(num_heads - protect_ends, num_heads))
                    heads_order = [h for h in heads_order if h not in keep and 0 <= h < num_heads]
                n_drop_h = int(round(r * len(heads_order)))
                drop_heads = sorted(heads_order[:n_drop_h])

                # channels
                ch_order = list(blk_dict.get("mlp_channels_order", []))
                if protect_ends > 0 and ch_order:
                    F = len(ch_order)
                    keep = set(range(protect_ends)) | set(range(F - protect_ends, F))
                    ch_order = [c for c in ch_order if c not in keep]
                n_drop_c = int(round(r * len(ch_order)))
                drop_ch = sorted(ch_order[:n_drop_c])

                if drop_heads:
                    entry_attn[blk] = drop_heads
                if drop_ch:
                    entry_mlp[blk] = drop_ch

            schedule[t] = {"attn": entry_attn, "mlp": entry_mlp}

    return schedule


# ===================== First-Order =====================

FirstOrderMethod = Literal["wanda", "activation", "magnitude"]
ActStat = Literal["mean_abs", "rms"]
# ---------- Core: first-order calibration ----------

@torch.no_grad()
def calibrate_firstorder_orders(
    model: nn.Module,
    dataloader,
    stages: List[Tuple[int, int]],
    *,
    method: Literal["wanda", "magnitude", "activation"] = "wanda",
    cfg_scale: float = 4.0,
    head_dim: int = 64,      # fallback; inferred per block if possible
    num_heads: int = 24,     # fallback; inferred per block if possible
) -> Dict:
    assert hasattr(model, "blocks") and isinstance(model.blocks, nn.ModuleList)
    device = next(model.parameters()).device
    using_cfg = (cfg_scale > 1.0) and hasattr(model, "forward_with_cfg")
    depth = len(model.blocks)
    S = len(stages)

    # ---------- helpers ----------
    def _t_to_stage_ids_simple(t_tensor: torch.Tensor, stages: List[Tuple[int, int]]) -> torch.Tensor:
        assert t_tensor.ndim == 1
        sid = torch.full_like(t_tensor, fill_value=-1, dtype=torch.long)
        for s, (lo, hi) in enumerate(stages):
            sid[(t_tensor >= lo) & (t_tensor <= hi)] = s
        return sid

    def _infer_heads(C_out: int, fallback_H: int, fallback_d: int, blk_attn: Optional[nn.Module]) -> Tuple[int, int]:
        H, d = fallback_H, fallback_d
        if blk_attn is not None:
            nh = getattr(blk_attn, "num_heads", None)
            if nh is not None and C_out % int(nh) == 0:
                H = int(nh)
                d = C_out // H
                return H, d
        if fallback_d > 0 and C_out % fallback_d == 0:
            H = C_out // fallback_d
            d = fallback_d
        elif fallback_H > 0 and C_out % fallback_H == 0:
            H = fallback_H
            d = C_out // H
        else:
            H, d = 1, C_out
        return H, d

    # ---------- per-stage accumulators (only what we need for chosen method) ----------
    need_attn_in  = method in ("wanda", "activation")
    need_attn_out = method == "activation"
    need_mlp_in   = method in ("wanda", "activation")

    attn_in_m2  = [[None for _ in range(depth)] for _ in range(S)]  # [F_in]
    attn_in_cnt = [[0    for _ in range(depth)] for _ in range(S)]

    attn_out_m2  = [[None for _ in range(depth)] for _ in range(S)] # [C_out]
    attn_out_cnt = [[0    for _ in range(depth)] for _ in range(S)]

    mlp_in_m2  = [[None for _ in range(depth)] for _ in range(S)]   # [F_in]
    mlp_in_cnt = [[0    for _ in range(depth)] for _ in range(S)]

    stage_vec_ref = {"vec": None}
    hooks: List[torch.utils.hooks.RemovableHandle] = []

    # ---------- hooks ----------
    def make_attn_proj_pre_hook(blk_idx: int):
        def pre_hook(_module: nn.Module, inputs):
            if not need_attn_in:
                return None
            x = inputs[0]
            if x is None: return None
            if x.dim() == 3:
                x2 = x.pow(2).mean(dim=1)  # [B, F]
            elif x.dim() == 2:
                x2 = x.pow(2)              # [B, F]
            else:
                return None
            sv = stage_vec_ref["vec"]
            if sv is None: return None
            B, F = x2.shape
            for sid in range(S):
                m = (sv == sid)
                if m.any():
                    if attn_in_m2[sid][blk_idx] is None or attn_in_m2[sid][blk_idx].numel() != F:
                        attn_in_m2[sid][blk_idx] = torch.zeros(F, dtype=torch.float64)
                        attn_in_cnt[sid][blk_idx] = 0
                    attn_in_m2[sid][blk_idx] += x2[m].mean(dim=0).double().cpu()
                    attn_in_cnt[sid][blk_idx] += 1
            return None
        return pre_hook

    def make_attn_proj_post_hook(blk_idx: int):
        def post_hook(_module: nn.Module, inputs, output):
            if not need_attn_out:
                return None
            y = output
            if y is None: return None
            if y.dim() == 3:
                y2 = y.pow(2).mean(dim=1)  # [B, C_out]
            elif y.dim() == 2:
                y2 = y.pow(2)              # [B, C_out]
            else:
                return None
            sv = stage_vec_ref["vec"]
            if sv is None: return None
            B, Fout = y2.shape
            for sid in range(S):
                m = (sv == sid)
                if m.any():
                    if attn_out_m2[sid][blk_idx] is None or attn_out_m2[sid][blk_idx].numel() != Fout:
                        attn_out_m2[sid][blk_idx] = torch.zeros(Fout, dtype=torch.float64)
                        attn_out_cnt[sid][blk_idx] = 0
                    attn_out_m2[sid][blk_idx] += y2[m].mean(dim=0).double().cpu()
                    attn_out_cnt[sid][blk_idx] += 1
            return None
        return post_hook

    def make_mlp_fc2_pre_hook(blk_idx: int):
        def pre_hook(_module: nn.Module, inputs):
            if not need_mlp_in:
                return None
            x = inputs[0]
            if x is None: return None
            if x.dim() == 3:
                x2 = x.pow(2).mean(dim=1)  # [B, F]
            elif x.dim() == 2:
                x2 = x.pow(2)              # [B, F]
            else:
                return None
            sv = stage_vec_ref["vec"]
            if sv is None: return None
            B, F = x2.shape
            for sid in range(S):
                m = (sv == sid)
                if m.any():
                    if mlp_in_m2[sid][blk_idx] is None or mlp_in_m2[sid][blk_idx].numel() != F:
                        mlp_in_m2[sid][blk_idx] = torch.zeros(F, dtype=torch.float64)
                        mlp_in_cnt[sid][blk_idx] = 0
                    mlp_in_m2[sid][blk_idx] += x2[m].mean(dim=0).double().cpu()
                    mlp_in_cnt[sid][blk_idx] += 1
            return None
        return pre_hook

    # Register only necessary hooks to save overhead
    for i, blk in enumerate(model.blocks):
        if hasattr(blk, "attn") and isinstance(getattr(blk.attn, "proj", None), nn.Linear):
            if need_attn_in:
                hooks.append(blk.attn.proj.register_forward_pre_hook(make_attn_proj_pre_hook(i)))
            if need_attn_out:
                hooks.append(blk.attn.proj.register_forward_hook(make_attn_proj_post_hook(i)))
        if hasattr(blk, "mlp") and isinstance(getattr(blk.mlp, "fc2", None), nn.Linear):
            if need_mlp_in:
                hooks.append(blk.mlp.fc2.register_forward_pre_hook(make_mlp_fc2_pre_hook(i)))

    # ---------- calibration pass ----------
    try:
        for batch in tqdm(dataloader, desc="[FirstOrder-Calibrate]"):
            x = batch["x"].to(device, non_blocking=True)
            t = batch["t"].to(device, non_blocking=True).view(-1)
            y = batch["y"].to(device, non_blocking=True).view(-1)

            if using_cfg:
                x = torch.cat([x, x], dim=0)
                t = torch.cat([t, t], dim=0)
                y_null = torch.full_like(y, fill_value=getattr(model, "num_classes", 1000))
                y = torch.cat([y, y_null], dim=0)

            stage_vec_ref["vec"] = _t_to_stage_ids_simple(t, stages)

            if using_cfg:
                _ = model.forward_with_cfg(x, t, y, cfg_scale)
            else:
                _ = model(x, t, y)
    finally:
        for h in hooks:
            try:
                h.remove()
            except Exception:
                pass

    # ---------- build STRUCTURAL orders (least -> most important) ----------
    out: Dict = {}
    for sid in range(S):
        out[str(sid)] = {}
        for blk_idx, blk in enumerate(model.blocks):
            entry = {"attn_heads_order": [], "mlp_channels_order": []}

            # ---- Attention heads (group contiguous output spans of size head_dim) ----
            if hasattr(blk, "attn") and isinstance(getattr(blk.attn, "proj", None), nn.Linear):
                proj: nn.Linear = blk.attn.proj
                Wabs = proj.weight.detach().abs()  # [C_out, C_in]

                # Per-output scores S_out
                if method == "wanda":
                    A_in = attn_in_m2[sid][blk_idx]; cnt = attn_in_cnt[sid][blk_idx]
                    if A_in is not None and cnt > 0:
                        A = (A_in / max(1, cnt)).to(Wabs.device, dtype=Wabs.dtype)  # [C_in], second moment
                        S_out = Wabs @ A                                          # [C_out]
                    else:
                        S_out = Wabs.sum(dim=1)  # fallback: magnitude
                elif method == "magnitude":
                    S_out = Wabs.sum(dim=1)
                else:  # activation
                    Y = attn_out_m2[sid][blk_idx]; cnt = attn_out_cnt[sid][blk_idx]
                    if Y is not None and cnt > 0:
                        S_out = (Y / max(1, cnt)).to(Wabs.device, dtype=Wabs.dtype)  # [C_out], E[(Wx)^2]
                    else:
                        S_out = Wabs.sum(dim=1)  # fallback

                C_out = int(S_out.numel())
                H, d = _infer_heads(C_out, num_heads, head_dim, getattr(blk, "attn", None))
                H = max(1, H); d = max(1, min(d, C_out))
                if H * d != C_out and d > 0:
                    # keep contiguous groups; truncate last partial group if any
                    H = C_out // d

                head_scores = []
                for h in range(H):
                    s, e = h * d, min((h + 1) * d, C_out)
                    head_scores.append(S_out[s:e].sum())
                head_scores = torch.stack(head_scores) if head_scores else torch.zeros(0, device=Wabs.device)
                heads_order = torch.argsort(head_scores, dim=0).tolist() if head_scores.numel() else list(range(H))
                entry["attn_heads_order"] = [int(h) for h in heads_order]

            # ---- MLP.fc2 input channels (each input dim is one structural channel) ----
            if hasattr(blk, "mlp") and isinstance(getattr(blk.mlp, "fc2", None), nn.Linear):
                fc2: nn.Linear = blk.mlp.fc2
                Wabs = fc2.weight.detach().abs()  # [F_out, F_in]

                if method == "wanda":
                    A_in = mlp_in_m2[sid][blk_idx]; cnt = mlp_in_cnt[sid][blk_idx]
                    if A_in is not None and cnt > 0:
                        A = (A_in / max(1, cnt)).to(Wabs.device, dtype=Wabs.dtype)  # [F_in], second moment
                        S_in = Wabs.sum(dim=0) * A                                   # [F_in]
                    else:
                        S_in = Wabs.sum(dim=0)  # fallback
                elif method == "magnitude":
                    S_in = Wabs.sum(dim=0)
                else:  # activation
                    A_in = mlp_in_m2[sid][blk_idx]; cnt = mlp_in_cnt[sid][blk_idx]
                    if A_in is not None and cnt > 0:
                        S_in = (A_in / max(1, cnt)).to(Wabs.device, dtype=Wabs.dtype)  # [F_in], E[x^2]
                    else:
                        S_in = Wabs.sum(dim=0)

                ch_order = torch.argsort(S_in, dim=0).tolist() if S_in.numel() else []
                entry["mlp_channels_order"] = [int(c) for c in ch_order]

            out[str(sid)][str(blk_idx)] = entry

    return out


# ---------- Back-compat shim ----------

@torch.no_grad()
def calibrate_wanda_orders(
    model: nn.Module,
    dataloader,
    stages: List[Tuple[int, int]],
    *,
    cfg_scale: float = 4.0,
    head_dim: int = 64,
    num_heads: Optional[int] = None,
    act_stat: ActStat = "mean_abs",
    device: Optional[torch.device] = None,
    max_batches: Optional[int] = None,
) -> Dict[str, Dict[int, Dict[str, List[int]]]]:
    """
    Deprecated: use calibrate_firstorder_orders(..., method='wanda').
    """
    return calibrate_firstorder_orders(
        model=model,
        dataloader=dataloader,
        stages=stages,
        method="wanda",
        cfg_scale=cfg_scale,
        head_dim=head_dim,
        num_heads=num_heads,
        act_stat=act_stat,
        device=device,
        max_batches=max_batches,
    )


def apply_secondorder_schedule(
    model: nn.Module,
    schedule: Dict,
    stages: Optional[List[Tuple[int, int]]] = None
) -> None:
    """
    Apply a per-timestep second-order pruning schedule.

    Modes:
      1) Legacy (default): if model.so_struct_forward is False or absent
         -> installs forward hooks that mask inputs (simulation; no speedup).
      2) Struct runtime (recommended): if model.so_struct_forward is True
         -> stores a normalized per-step schedule on the model; DiT.forward will run thin
            GEMMs with preloaded OBS-updated thin matrices (real speedup).
            If obs_cache_dir is provided, we pre-load the needed thin matrices for
            each (stage, block, k) into a per-schedule bank ONCE.

    NOTE: To keep LayerDrop separate, we clear any existing layerdrop schedule here.
    """
    # ---- small helpers --------------------------------------------------------
    def _normalize_so_schedule_keys(sched: Dict) -> Dict[int, Dict[str, Dict[int, List[int]]]]:
        """Coerce JSON string keys to ints for timesteps and block indices."""
        norm: Dict[int, Dict[str, Dict[int, List[int]]]] = {}
        for t_k, entry in (sched or {}).items():
            t_i = int(t_k)
            attn_src = entry.get("attn", {})
            mlp_src  = entry.get("mlp",  {})
            attn = {int(bk): [int(h) for h in heads] for bk, heads in attn_src.items()}
            mlp  = {int(bk): [int(c) for c in chans] for bk, chans in mlp_src.items()}
            norm[t_i] = {"attn": attn, "mlp": mlp}
        return norm

    # ---- clear previous second-order state (both modes) -----------------------
    if hasattr(model, "_so_hooks") and model._so_hooks:
        for h in model._so_hooks:
            try:
                h.remove()
            except Exception:
                pass
    model._so_hooks = []

    if hasattr(model, "_so_struct_schedule"):
        model._so_struct_schedule = None
    if hasattr(model, "_so_struct_stages"):
        model._so_struct_stages = None
    model._so_current_t = None
    model._so_current_stage = -1

    if hasattr(model, "clear_layerdrop_schedule"):
        try:
            model.clear_layerdrop_schedule()
        except Exception:
            model.layerdrop_schedule = None

    # If schedule is empty -> clear + return
    if not schedule:
        print("No second-order schedule provided; cleared previous state.")
        def _clear_secondorder_schedule():
            if hasattr(model, "_so_hooks") and model._so_hooks:
                for h in model._so_hooks:
                    try:
                        h.remove()
                    except Exception:
                        pass
            model._so_hooks = []
            if hasattr(model, "_so_struct_schedule"):
                model._so_struct_schedule = None
            if hasattr(model, "_so_struct_stages"):
                model._so_struct_stages = None
            model._so_current_t = None
            model._so_current_stage = -1
            if hasattr(model, "clear_obs_bank"):
                try:
                    model.clear_obs_bank()
                except Exception:
                    pass
        model.clear_secondorder_schedule = _clear_secondorder_schedule
        return

    # ---- struct runtime branch (preferred) -----------------------------------
    if bool(getattr(model, "so_struct_forward", False)):
        sched_norm = _normalize_so_schedule_keys(schedule)
        # model._so_struct_schedule = sched_norm
        # model._so_struct_stages = [(int(lo), int(hi)) for (lo, hi) in (stages or [])]
        model.set_secondorder_schedule(sched_norm, stages)

        # unified clearer
        def _clear_secondorder_schedule():
            if hasattr(model, "_so_hooks") and model._so_hooks:
                for h in model._so_hooks:
                    try:
                        h.remove()
                    except Exception:
                        pass
            model._so_hooks = []
            if hasattr(model, "_so_struct_schedule"):
                model._so_struct_schedule = None
            if hasattr(model, "_so_struct_stages"):
                model._so_struct_stages = None
            if hasattr(model, "_so_struct_compiled"):
                model._so_struct_compiled = None
            model._so_current_t = None
            model._so_current_stage = -1
            if hasattr(model, "clear_obs_bank"):
                try:
                    model.clear_obs_bank()
                except Exception:
                    pass

        model.clear_secondorder_schedule = _clear_secondorder_schedule
        return
 
 # ===================== OBS (Optimal Brain Surgeon) utils =====================
class TapDiffOBS:
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
        # print("nsamples:", self.nsamples, " tmp:", tmp)
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        self.H += 2 / self.nsamples * (inp.matmul(inp.t())).double()

    def invert(self, H, percentdamp=.01):
        try:
            Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H))
            # print('Hessian inv successful.')
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
        # print(f"Layer :{self.layer} shape: {self.layer.weight.shape}")
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
        """
        Structured OBS pruning.

        Args:
            pruned (List[int]): milestone k values (number of groups/columns dropped)
                                at which to record snapshots. Used only when
                                return_mode == "final".
            size (int): group size. 1 => column pruning; >1 => block/group pruning.
            return_mode (str): 
                - "final": (default) record snapshots only at milestones in `pruned`
                        (k=0 included ONLY if 0 is in `pruned`, matching old behavior).
                - "traj" : record the full trajectory: k=0 state + every step.

        Returns:
            res (List[Tensor]): list of weight tensors (same shape as layer.weight)
            kept_indices_list (List[Tensor]): list of 1D LongTensors of kept indices
        """
        assert return_mode in ("final", "traj"), "return_mode must be 'final' or 'traj'"

        milestones = pruned[:] if isinstance(pruned, (list, tuple)) else [int(pruned)]

        W, H, Hinv, _ = self.prepare()
        count = self.columns // size
        rangecount = torch.arange(count, device=self.dev)
        rangecolumns = torch.arange(self.columns, device=self.dev)

        # Masks
        mask = torch.zeros(count, device=self.dev).bool()
        mask1 = None
        if size > 1:
            mask1 = torch.zeros(self.columns, device=self.dev).bool()

        res = []
        kept_indices_list = []
        Losses = torch.zeros(count + 1, device=self.dev)  # Losses[k] stores score at drop k (1..count)

        def _kept_indices():
            if size == 1:
                return torch.arange(self.columns, device=self.dev)[~mask]
            else:
                return torch.arange(self.columns, device=self.dev)[~mask1]

        def _record_snapshot(dropped_k: int, print_loss: bool = False):
            kept = _kept_indices()
            # kept_indices_list.append(kept.to(dtype=torch.int64, device="cpu"))
            # THIN ON GPU, then downcast to fp32 for storage
            Wk_thin = W.index_select(1, kept)                 # GPU op
            res.append(Wk_thin.to(dtype=torch.float32, device="cpu"))
            kept_indices_list.append(kept.cpu().tolist())
            
            if print_loss:
                # Print cumulative loss up to k (classic OBS uses sum(Losses)/2)
                if dropped_k == 0:
                    print(f"{0:4d} error", 0.0)
                else:
                    print(f"{dropped_k:4d} error", torch.sum(Losses[:dropped_k + 1]).item() / 2)

        # ==================== Case 1: size = 1 (column-wise) ====================
        if size == 1:
            for dropped in range(count):
                diag = torch.diagonal(Hinv)
                scores = torch.sum(W ** 2, 0) / diag
                scores[mask] = float('inf')
                j = torch.argmin(scores)

                # store loss for this drop at index dropped+1
                Losses[dropped + 1] = scores[j]

                # OBS update on W
                row = Hinv[j, :]
                d = diag[j]
                W -= ((W[:, j] / d).unsqueeze(1)).matmul(row.unsqueeze(0))

                # Mark dropped and zero columns
                mask[j] = True
                W[:, mask] = 0

                # Update Hinv (rank-1 downdate)
                row /= torch.sqrt(d)
                Hinv -= row.unsqueeze(1).matmul(row.unsqueeze(0))

                k_now = dropped + 1

                # if return_mode == "traj":
                #     _record_snapshot(k_now)
                # else:
                    # "final": record only at requested milestones
                while milestones and k_now == milestones[0]:
                    _record_snapshot(k_now)
                    milestones.pop(0)
                    if not milestones:
                        return res, kept_indices_list

        # ==================== Case 2: size > 1 (block/group pruning) ====================
        else:
            for dropped in range(count):
                blocks = Hinv.reshape(count, size, count, size)
                blocks = blocks[rangecount, :, rangecount, :]  # (count, size, size)
                try:
                    invblocks = torch.cholesky_inverse(torch.linalg.cholesky(blocks))
                except Exception:
                    invblocks = torch.linalg.pinv(blocks, hermitian=True)

                W1 = W.reshape((self.rows, count, size)).transpose(0, 1)  # (count, rows, size)
                lambd = torch.bmm(W1, invblocks)                          # (count, rows, size)
                scores = torch.sum(lambd * W1, (1, 2))                    # (count,)
                scores[mask] = float('inf')
                j = torch.argmin(scores)

                # store loss for this drop
                Losses[dropped + 1] = scores[j]

                # Update W with chosen group's block
                rows = Hinv[(size * j):(size * (j + 1)), :]
                d = invblocks[j]
                W -= lambd[j].matmul(rows)

                # Mark dropped group and expanded columns
                mask[j] = True
                mask1[(size * j):(size * (j + 1))] = True
                W[:, mask1] = 0

                # Update Hinv with block downdate
                Hinv -= rows.t().matmul(d.matmul(rows))
                Hinv[rangecolumns[mask1], rangecolumns[mask1]] = 1

                k_now = dropped + 1

                # if return_mode == "traj":
                #     _record_snapshot(k_now)
                # else:
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


def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        new_name = name + '.' + name1 if name != '' else name1
        res.update(find_layers(child, layers=layers, name=new_name))
    return res

@torch.no_grad()
def simple_prune_traj_dit(model: nn.Module, dataloader, cfg_scale: float, 
                          head_size: int, head_num: int,
                          save_dir: str,
                          obs_level_max: int,
                          attname: str = "attn.proj", fcname: str = "mlp.fc2"):
    """
    Same accumulation flow as simple_prune_dit, but instead of applying the final
    weights into the model, we SAVE the entire OBS trajectory (k=0..max_k) for
    each targeted layer to 'save_dir'. No biases are saved.
    """
    layers = find_layers(model, layers=[nn.Linear])  # keep identical preamble
    tapdiff_objects = {name: TapDiffOBS(layer) for name, layer in layers.items()}  # use OBS core

    assert cfg_scale >= 1.0, "In almost all cases, cfg_scale be >= 1.0"
    using_cfg = cfg_scale > 1.0

    def add_batch_factory(name):
        def add_batch_hook(layer, inp, out):
            tapdiff_objects[name].add_batch(inp[0], out)
        return add_batch_hook

    hooks = []
    for name, layer in layers.items():
        hook = layer.register_forward_hook(add_batch_factory(name))
        hooks.append(hook)

    # Accumulate statistics (identical)
    for batch in tqdm(dataloader):
        device = next(model.parameters()).device
        x = batch['x'].to(device)
        t = batch['t'].to(device).view(-1)
        y = batch['y'].to(device).view(-1)

        x = torch.cat([x, x], 0)
        t = torch.cat([t, t], 0)
        y_null = torch.tensor([getattr(model, "num_classes", 1000)] * y.shape[0], device=device)
        y = torch.cat([y, y_null], 0)
        if using_cfg:
            _ = model.forward_with_cfg(x, t, y, cfg_scale)
        else:
            _ = model.forward(x, t, y)

    for h in hooks:
        h.remove()

    def prundim(name):
        if attname in name:
            return head_size
        if fcname in name:
            return 1
        return 0

    # ------------------------ Prune (trajectory saving) ------------------------
    save_root = Path(save_dir)
    save_root.mkdir(parents=True, exist_ok=True)

    # We also return a small in-memory manifest mapping layer_name -> kept_idx sequence
    traj_manifest = {}

    for name, tapdiff in tapdiff_objects.items():
        size = prundim(name)
        if size > 0 and isinstance(layers[name], nn.Linear):  # ensure Linear target
            layer = layers[name]
            if attname in name:
                # number of groups = head_num
                # max_k = head_num
                pruned = [i + 1 for i in range(obs_level_max)]
            else:
                # number of groups = number of input channels
                # max_k = layer.in_features
                pruned = [round(((i + 1) / obs_level_max) * layer.in_features )for i in range(obs_level_max)]

            t0 = time.time()
            weights_seq, kept_seq = tapdiff.prune_struct(pruned=pruned, size=size, return_mode="traj")
            print("Pruned layer:", name)
            # print("max_k:", max_k, " group size:", size)
            print("weights_seq len:", len(weights_seq), " kept_seq len:", len(kept_seq))

            # derive directory for this layer
            # (normalize layer name into a safe path; also split by standard DiT naming if available)
            safe_name = name.replace('.', '_')
            # Save under e.g. <save_dir>/<layer_name>/k_<k>.pt
            layer_dir = save_root / safe_name
            layer_dir.mkdir(parents=True, exist_ok=True)

            traj_manifest[name] = kept_seq  # keep for order derivation upstream

            for i, Wk in enumerate(weights_seq):
                kept = kept_seq[i]  # 1D Long indices (element-level)
                k = pruned[i]
                print(f"  Saving k={k} with {len(kept)} kept indices.")
                payload = {
                    # Prefer using the thinned weight for OBS bank consumption:
                    "weight": Wk,
                    "kept_idx": kept,
                    "group_size": int(size),
                    "k": int(k),
                    "type": ("attn_proj" if attname in name else "mlp_fc2"),
                    "layer_name": name,
                }
                torch.save(payload, layer_dir / f"k_{k}.pt")
            t1 = time.time()

            print(f"Saved OBS trajectory for layer {name} to {layer_dir}, time taken: {t1 - t0:.2f}s")

            tapdiff_objects[name].free()

    return traj_manifest
