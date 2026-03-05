#!/usr/bin/env python
# -*- coding: utf-8 -*-
from ast import arg
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import argparse
import glob
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from diffusers.models import AutoencoderKL
from torch.utils.data import DataLoader
from tqdm import tqdm

# ==== project imports ====
from calibration import ImageDiTDataset, dataloader_builder
from diffusion import create_diffusion
from download import find_model
from evo_search import EvoLayerDropSearch, FitnessOnTrajectory, build_init_population_levels
from models import DiT_models
from evo_pruning_utils import (
    # LayerDrop
    apply_layerdrop_schedule,
    calibrate_layerdrop_orders,
    build_layerdrop_schedule_from_orders,

    # Unified first-order (wanda / magnitude / activation)
    calibrate_firstorder_orders,
    build_secondorder_schedule_from_orders,   # schedule shape shared by FO and SO
    apply_secondorder_schedule,

    # True second-order (OBS)
    calibrate_secondorder_orders,             # OBS path: returns (orders, repo)
    save_obs_bank,
    load_obs_bank,
    select_obs_bank_for_ratios,
)
import socket, random

def _pick_free_port(default_base=29500, spread=1000):
    """Return a random free TCP port on localhost (used for MASTER_PORT)."""
    env_port = os.environ.get("MASTER_PORT")
    if env_port:
        return int(env_port)
    for _ in range(20):
        port = default_base + random.randint(0, spread)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]

# ======================================================================================
# DDP helpers
# ======================================================================================

def ddp_is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()

def ddp_init_if_needed() -> None:
    if ddp_is_initialized():
        return
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")

def ddp_rank() -> int:
    return dist.get_rank() if ddp_is_initialized() else 0

def ddp_world_size() -> int:
    return dist.get_world_size() if ddp_is_initialized() else 1

def ddp_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))

def ddp_barrier() -> None:
    if ddp_is_initialized():
        try:
            if torch.cuda.is_available():
                dist.barrier(device_ids=[ddp_local_rank()])
            else:
                dist.barrier()
        except TypeError:
            dist.barrier()

# ======================================================================================
# Logging
# ======================================================================================

def setup_logger(log_file: str, *, console: bool, level: str = "INFO") -> None:
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)
    if console:
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        root.addHandler(sh)
    logging.getLogger("tqdm").setLevel(logging.WARNING)
    root.info("Logger initialized at %s", level.upper())

# ======================================================================================
# Small utils
# ======================================================================================

def sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._\-+,/]", "-", name)

def create_npz_from_sample_folder(sample_dir: str, out_npz_path: str, num: int) -> str:
    Path(out_npz_path).parent.mkdir(parents=True, exist_ok=True)
    arrs: List[np.ndarray] = []
    iterator = tqdm(range(num), desc="Packaging NPZ", disable=(ddp_rank() != 0))
    for i in iterator:
        p = os.path.join(sample_dir, f"{i:06d}.png")
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing sample {p}")
        with Image.open(p) as im:
            arrs.append(np.asarray(im).astype(np.uint8))
    arr = np.stack(arrs)
    np.savez(out_npz_path, arr_0=arr)
    logging.info("Saved NPZ -> %s (shape=%s)", out_npz_path, arr.shape)
    return out_npz_path

def uniform_stage_ranges(num_stages: int, diffusion_steps: int = 1000) -> List[Tuple[int, int]]:
    assert num_stages >= 1
    edges = [int(round(i * diffusion_steps / num_stages)) for i in range(num_stages + 1)]
    stages: List[Tuple[int, int]] = []
    for i in range(num_stages):
        lo = edges[i]
        hi = max(lo, edges[i + 1] - 1)
        stages.append((lo, hi))
    return stages

def to_py(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, dict):
        return {to_py(k): to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_py(x) for x in o]
    return o

# ======================================================================================
# Experiment naming (kept stable)
# ======================================================================================

def build_exp_name(
    model: str,
    ckpt: Optional[str],
    image_size: int,
    vae: str,
    cfg_scale: float,
    num_sampling_steps: int,
    num_stages: int,
    target_level: float,
    seed: int,
    generations: int,
    offspring: int,
    survivors_per_selection: List[int],
    mutation_max_levels: int,
    fitness_batches: int,
    fitness_reduce: str,
    start_level: Optional[float],
    *,
    prune_method: str,
    stage_dividers: Optional[List[int]] = None,
    traj_fitness_mode: str = "final",
    traj_suffix_steps: Optional[int] = None,
    traj_suffix_frac: Optional[float] = None,
    traj_late_weighting: str = "cosine",
    traj_include_eps: bool = False,
    traj_eps_weight: float = 0.30,
    traj_probe_batch: int = 64,
    traj_refresh_every: int = 0,
    traj_fitness_metric: str = "latent_mse",
    abs_fft_weight: float = 0.5,
    use_validation: bool = False,
    calib_importance: str = "mse",
    init_strategy: str = "hybrid",
    so_struct_speedup: bool = False,
    mutation_n_valid: int = 10
) -> str:
    core = (
        f"{prune_method}"
        f"-size-{image_size}"
        f"-vae-{vae}"
        f"-cfg-{cfg_scale}"
        f"-step-{num_sampling_steps}"
        f"-S-{num_stages}"
        f"-lvl-{target_level:.2f}"
        f"-init-{init_strategy}"
        f"-gen{generations}"
        f"-offs{offspring}"
        f"-surv{'-'.join(map(str, survivors_per_selection))}"
        f"-mutL{int(mutation_max_levels)}"
        f"-mutN{int(mutation_n_valid)}"
        + f"-fitmetrics-{traj_fitness_metric.replace('latent_','l_').replace('img_','i_')}"
        + (f"-AbsFftWeight{abs_fft_weight}" if traj_fitness_metric == "l_abs_fft" or traj_fitness_metric == "latent_abs_fft" else "")
        + ("" if not stage_dividers else f"-div-{'_'.join(map(str, stage_dividers))}")
        + ("" if not so_struct_speedup else "-speedup")
        + f"-seed-{seed}"
    )
    imp_tag = "cos" if calib_importance.lower().startswith("cos") else "mse"
    core += f"-imp-{imp_tag}"
    if start_level is not None:
        core += f"-start{start_level:.2f}"

    fit = f"-mode-{traj_fitness_mode}"
    if traj_fitness_mode == "suffix":
        if traj_suffix_steps is not None:
            fit += f"-K{int(traj_suffix_steps)}"
        elif traj_suffix_frac is not None:
            fit += f"-F{float(traj_suffix_frac):.2f}"
        else:
            fit += "-F0.50"
    fit += f"-w{traj_late_weighting[0]}"
    if traj_include_eps:
        fit += f"-eps{float(traj_eps_weight):.2f}"
    fit += f"-probe{int(traj_probe_batch)}"
    if traj_refresh_every and int(traj_refresh_every) > 0:
        fit += f"-ref{int(traj_refresh_every)}"
    fit += ("-val" if use_validation else "-noval")
    return sanitize_name(core + fit)

def build_stages_from_dividers(
    dividers: Optional[List[int]],
    num_stages: int,
    *,
    diffusion_steps: int = 1000,
) -> Optional[List[Tuple[int, int]]]:
    if not dividers:
        return None
    if len(dividers) != max(0, int(num_stages) - 1):
        raise ValueError(
            f"--stage-dividers expects exactly num_stages-1 values; "
            f"got {len(dividers)} but num_stages={num_stages} (need {max(0, num_stages-1)})."
        )
    ds = int(diffusion_steps)
    dv = [int(x) for x in dividers]
    if any(d <= 0 or d >= ds for d in dv):
        raise ValueError(f"Stage dividers must be strictly inside (0, {ds}), received: {dv}")
    if sorted(dv) != dv or len(set(dv)) != len(dv):
        raise ValueError(f"Stage dividers must be strictly increasing with no duplicates, received: {dv}")
    edges = [0] + dv + [ds]
    stages: List[Tuple[int, int]] = []
    for i in range(len(edges) - 1):
        lo = edges[i]
        hi = edges[i + 1] - 1
        lo = max(0, min(lo, ds - 1))
        hi = max(lo, min(hi, ds - 1))
        stages.append((lo, hi))
    return stages

# ======================================================================================
# WORKER MODE
# ======================================================================================

def worker_main(args: argparse.Namespace) -> None:
    ddp_init_if_needed()
    rank = ddp_rank()
    world_size = ddp_world_size()
    local_rank = ddp_local_rank()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    Path(args.experiments_dir).mkdir(parents=True, exist_ok=True)
    # Set up logger early
    tmp_log_dir = os.path.join(args.experiments_dir, "logs")
    Path(tmp_log_dir).mkdir(parents=True, exist_ok=True)
    log_file = os.path.join(tmp_log_dir, f"run_rank{rank}.log")
    setup_logger(log_file, console=(rank == 0), level=args.log_level)
    ddp_barrier()

    # seeds & backend
    torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
    torch.backends.cudnn.allow_tf32 = bool(args.tf32)
    seed = args.seed * (world_size if world_size else 1) + rank
    print("[SEED]", seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # model / diffusion / VAE
    latent_size = args.image_size // 8
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae_id = f"./pretrained_models/sd-vae-ft-{args.vae}"
    vae = AutoencoderKL.from_pretrained(vae_id).to(device)
    model = DiT_models[args.model](input_size=latent_size, num_classes=args.num_classes).to(device)
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()

    # auto-detect head count & dim
    try:
        attn0 = model.blocks[0].attn
        det_heads = getattr(attn0, "num_heads", None)
        det_hdim  = getattr(attn0, "head_dim",  None)
        if det_hdim is None and det_heads:
            det_hdim = model.blocks[0].hidden_size // det_heads
        if det_heads:
            args.so_num_heads = det_heads
        if det_hdim:
            args.so_head_dim = det_hdim
        logging.info(f"[Heads] auto-detected: num_heads={args.so_num_heads}, head_dim={args.so_head_dim}")
    except Exception as _e:
        logging.warning(f"[Heads] could not auto-detect heads/dim, keeping CLI values: {_e}")

    # struct speed-up toggle for methods that mask heads/channels at runtime
    if args.prune_method in ("secondorder", "wanda", "magnitude", "activation"):
        try:
            if bool(args.so_struct_speedup):
                if hasattr(model, "enable_struct_prune_forward"):
                    model.enable_struct_prune_forward()
                else:
                    setattr(model, "so_struct_forward", True)
                logging.info("[Runtime] Structural speed-up ENABLED.")
            else:
                if hasattr(model, "disable_struct_prune_forward"):
                    model.disable_struct_prune_forward()
                else:
                    setattr(model, "so_struct_forward", False)
                logging.info("[Runtime] Legacy masking hooks (no speed-up).")
        except Exception as e:
            logging.warning("[Runtime] Failed toggling struct speed-up flag, falling back to legacy: %s", e)

    # stages (uniform or custom)
    if args.stage_dividers is not None:
        stages = build_stages_from_dividers(args.stage_dividers, args.num_stages, diffusion_steps=1000)
    else:
        stages = uniform_stage_ranges(args.num_stages, diffusion_steps=1000)
    logging.info("Stages: %s", stages)

    # Determine H (level cap)
    depth = len(model.blocks)
    H = depth if args.prune_method == "layerdrop" else max(1, int(args.so_num_heads))
    logging.info("[Levels] H (cap) = %d", H)
    # mutation_n_valid = args.mutation_n_valid if args.mutation_n_valid is not None else 2 * args.num_stages
    mutation_n_valid = args.mutation_n_valid


    # EXP NAME
    exp_name = build_exp_name(
        model=args.model,
        ckpt=args.ckpt,
        image_size=args.image_size,
        vae=args.vae,
        cfg_scale=args.cfg_scale,
        num_sampling_steps=args.num_sampling_steps,
        num_stages=args.num_stages,
        stage_dividers=args.stage_dividers,
        target_level=float(args.target_level),
        seed=args.seed,
        generations=args.generations,
        offspring=args.offspring,
        survivors_per_selection=args.survivors_per_selection,
        mutation_max_levels=args.mutation_max_levels,
        mutation_n_valid=mutation_n_valid,
        fitness_batches=args.fitness_batches,
        fitness_reduce=args.fitness_reduce,
        start_level=(None if args.start_level is None else float(args.start_level)),
        prune_method=args.prune_method,
        traj_fitness_mode=args.traj_fitness_mode,
        traj_suffix_steps=args.traj_suffix_steps,
        traj_suffix_frac=args.traj_suffix_frac,
        traj_late_weighting=args.traj_late_weighting,
        traj_include_eps=bool(args.traj_include_eps),
        traj_eps_weight=args.traj_eps_weight,
        traj_probe_batch=args.traj_probe_batch,
        traj_refresh_every=args.traj_refresh_every,
        traj_fitness_metric=args.traj_fitness_metric,
        abs_fft_weight=args.abs_fft_weight,
        use_validation=bool(args.use_validation),
        calib_importance=args.calib_importance,
        init_strategy=args.init_strategy,
        so_struct_speedup=args.so_struct_speedup,
    )
    exp_dir = os.path.join(args.experiments_dir, exp_name)
    logs_dir = os.path.join(exp_dir, "logs")
    search_dir = os.path.join(exp_dir, "search")
    sample_dir = os.path.join(exp_dir, "samples")
    npz_dir = os.path.join(exp_dir, "npz")
    bank_dir = os.path.join(
        os.path.dirname(os.path.abspath(args.obs_bank_root)),
        f"obs_bank_stage{args.num_stages}_samples{args.loader_nsamples}"
    )
    logging.info("[OBS] Bank directory: %s", bank_dir)
    for d in [logs_dir, search_dir, sample_dir, npz_dir, bank_dir]:
        Path(d).mkdir(parents=True, exist_ok=True)
    logging.info("Experiment directory: %s", exp_dir)

    # Back-compat notice for deprecated ratio flag
    if any(arg.startswith("--mutation-max-ratio") or arg == "--mutation-max-ratio" for arg in sys.argv):
        logging.warning("[Back-compat] --mutation-max-ratio is ignored. Use --mutation-max-levels.")

    # ------------------------- timing containers -------------------------
    prune_timing = dict(
        calib_seconds=None,
        search_seconds=None,
        total_prune_seconds=None,
    )

    # ------------------------- evolutionary search -------------------------
    method = args.prune_method

    if args.do_prune:
        logging.info("[Prune] Calibration + evolutionary search (method=%s)", method)

        # start global pruning timer (calibration + evolutionary search)
        t_prune_start = time.time()

        logging.info(f"Speed-up: {getattr(model, 'so_struct_forward', False)}")

        # Calibration DATASET
        calib_dl = DataLoader(
            ImageDiTDataset(
                image_dir=os.path.expanduser("~/datasets/imagenet-1k/train"),
                vae=vae,
                image_size=args.image_size,
                num_classes=args.num_classes,
                diffusion_steps=1000,
                step_start=0,
                step_end=1000,
                device=device,
            ),
            batch_size=args.fitness_batches,
            shuffle=True,
        )

        calib_dl_for_calib = dataloader_builder(
            calib_dl,
            batchsize=args.fitness_batches,
            nsamples=args.loader_nsamples,
            shuffle=True,
            same_subset=True,
            stages=stages
        )

        # Optional validation fitness
        if args.use_validation and ddp_rank() == 0:
            eval_fn_val = FitnessOnTrajectory(
                model=model, diffusion=diffusion, num_steps=args.num_sampling_steps,
                cfg_scale=args.cfg_scale, probe_batch=args.traj_probe_batch,
                image_size=args.image_size, num_classes=args.num_classes,
                mode=args.traj_fitness_mode, suffix_steps=args.traj_suffix_steps,
                suffix_frac=args.traj_suffix_frac, late_weighting=args.traj_late_weighting,
                include_eps_term=args.traj_include_eps, eps_weight=args.traj_eps_weight,
                device=device, base_seed=args.traj_probe_seed + 999,
                progress=False, refresh_every=args.traj_refresh_every,
                stages=stages,
                fitness_metric=args.traj_fitness_metric,
                vae=vae,
                abs_fft_weight=args.abs_fft_weight
            )
            logging.info("[Val] Validation trajectory fitness ready.")
        else:
            eval_fn_val = None

        if ddp_rank() == 0:
            # Searcher mode: "layerdrop" uses block indices; others share SO/FO schema
            search_mode = "layerdrop" if method == "layerdrop" else "secondorder"
            search = EvoLayerDropSearch(
                model=model,
                stages=stages,
                rng_seed=seed,
                verbose=True,
                mode=search_mode,
                mode_kwargs=dict(
                    so_head_dim=args.so_head_dim,
                    so_num_heads=args.so_num_heads,
                    so_protect_ends=args.so_protect_ends,
                ),
            )

            # --------- Calibrate orders (timed) ----------
            t_calib_start = time.time()

            if method == "layerdrop":
                orders, scores = calibrate_layerdrop_orders(
                    model=model, dataloader=calib_dl_for_calib, stages=stages,
                    cfg_scale=args.cfg_scale, importance_metric=args.calib_importance,
                    cosine_eps=args.calib_cosine_eps
                )
                search.set_layerdrop_orders(orders, scores)
                with open(os.path.join(search_dir, "orders_layerdrop.json"), "w") as f:
                    json.dump(to_py(orders), f, indent=2)

            elif method == "secondorder":
                # OBS ONLY
                logging.info("[Prune][OBS] Calibrating OBS orders + thin packs...")
                orders_path = os.path.join(bank_dir, "orders_obs.json")
                obs_repo_path = os.path.join(bank_dir, "repo.pt")

                if os.path.exists(orders_path) and os.path.exists(obs_repo_path):
                    so_orders = json.load(open(orders_path, "r"))
                    repo      = load_obs_bank(bank_dir)
                    search.set_secondorder_orders(so_orders)
                    search.set_obs_repo(repo)
                    logging.info("[Prune][OBS] Loaded orders (%s) + repo (%s).", orders_path, obs_repo_path)
                else:
                    # Build one DataLoader per stage (OBS requires this)
                    obs_stage_loaders = []
                    for sid, (lo, hi) in enumerate(stages):
                        ds_stage = ImageDiTDataset(
                            image_dir=os.path.expanduser("~/datasets/imagenet-1k/train"),
                            vae=vae,
                            image_size=args.image_size,
                            num_classes=args.num_classes,
                            diffusion_steps=1000,
                            step_start=int(lo),
                            step_end=int(hi) + 1,   # exclusive upper bound
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

                    # returns (orders, repo) in OBS mode
                    so_orders, repo = calibrate_secondorder_orders(
                        model=model,
                        dataloader=obs_stage_loaders,
                        stages=stages,
                        cfg_scale=args.cfg_scale,
                        pruner="obs",
                        head_dim=args.so_head_dim,
                        num_heads=args.so_num_heads,
                        obs_cache_dir=bank_dir,
                    )
                    search.set_secondorder_orders(so_orders)
                    search.set_obs_repo(repo)
                    with open(orders_path, "w") as f:
                        json.dump(to_py(so_orders), f, indent=2)
                    saved_manifest = save_obs_bank(repo, bank_dir)
                    logging.info("[Prune][OBS] Saved orders -> %s and repo -> %s", orders_path, saved_manifest)

            else:
                # First-order family: wanda | magnitude | activation
                orders_file = f"orders_{method}.json"  # e.g., orders_wanda.json
                orders_path = os.path.join(search_dir, orders_file)
                if os.path.exists(orders_path):
                    fo_orders = json.load(open(orders_path, "r"))
                    search.set_secondorder_orders(fo_orders)  # same schema as SO
                    logging.info("[Prune][FO] Loaded %s", orders_file)
                else:
                    logging.info("[Prune][FO] Calibrating first-order orders (method=%s)...", method)
                    fo_orders = calibrate_firstorder_orders(
                        model=model,
                        dataloader=calib_dl_for_calib,
                        stages=stages,
                        method=method,                      # wanda | magnitude | activation
                        cfg_scale=args.cfg_scale,
                        head_dim=args.so_head_dim,
                        num_heads=args.so_num_heads,
                    )
                    search.set_secondorder_orders(fo_orders)
                    with open(orders_path, "w") as f:
                        json.dump(to_py(fo_orders), f, indent=2)

            # end calibration timer
            prune_timing["calib_seconds"] = time.time() - t_calib_start

            # --------- Trajectory-based fitness ----------
            eval_fn = FitnessOnTrajectory(
                model=model, diffusion=diffusion, num_steps=args.num_sampling_steps,
                cfg_scale=args.cfg_scale, probe_batch=args.traj_probe_batch,
                image_size=args.image_size, num_classes=args.num_classes,
                mode=args.traj_fitness_mode, suffix_steps=args.traj_suffix_steps,
                suffix_frac=args.traj_suffix_frac, late_weighting=args.traj_late_weighting,
                include_eps_term=args.traj_include_eps, eps_weight=args.traj_eps_weight,
                device=device, base_seed=args.traj_probe_seed,
                progress=False, refresh_every=args.traj_refresh_every,
                stages=stages,
                fitness_metric=args.traj_fitness_metric,
                vae=vae,
                abs_fft_weight=args.abs_fft_weight
            )

            # --------- Evo search (LEVELS) ----------
            target_level_int = int(round(float(args.target_level)))
            start_level_int  = None if args.start_level is None else int(round(float(args.start_level)))
            target_level_int = max(0, min(H, target_level_int))
            if start_level_int is not None:
                start_level_int = max(0, min(H, start_level_int))

            t0 = time.time()
            init_pop = build_init_population_levels(
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
            print("Initial population size:", len(init_pop))

            # time just the evolutionary search loop
            t_search_start = time.time()
            best_L, _best_schedule, best_score = search.run(
                generations=args.generations,
                offspring=args.offspring,
                target_level=target_level_int,
                survivors_per_selection=args.survivors_per_selection,
                eval_fn=eval_fn,
                start_level=start_level_int,
                mutation_max_levels=int(args.mutation_max_levels),
                mutation_max_times=int(mutation_n_valid),
                mutation_n_valid=int(mutation_n_valid),
                eval_fn_val=eval_fn_val,
                init_population=init_pop,
                log_dir=search_dir,
            )
            t_search_end = time.time()
            prune_timing["search_seconds"] = t_search_end - t_search_start

            elapsed_min = (time.time() - t0) / 60.0

            logging.info("[Prune] Search done in %.1f min. Best train (traj) score = %.6f", elapsed_min, best_score)

            # Persist per-stage LEVELS (PRIMARY)
            H = search.H
            S = len(stages)

            def _lvl(d, s):
                if isinstance(d, dict):
                    return int(d.get(s, d.get(str(s), 0)))
                if isinstance(d, (list, tuple)):
                    return int(d[s]) if 0 <= s < len(d) else 0
                return int(d) if isinstance(d, (int, float)) and s == 0 else 0

            best_levels = {int(s): max(0, min(H, _lvl(best_L, s))) for s in range(S)}
            best_ratios = {int(s): float(best_levels[s]) / float(max(1, H)) for s in range(S)}

            Path(search_dir).mkdir(parents=True, exist_ok=True)
            levels_path = os.path.join(search_dir, "levels_per_stage.json")
            ratios_path = os.path.join(search_dir, "ratios_per_stage.json")
            with open(levels_path, "w") as f:
                json.dump(to_py(best_levels), f, indent=2)
            with open(ratios_path, "w") as f:
                json.dump(to_py(best_ratios), f, indent=2)

            logging.info("[Prune] Saved levels (%s) and ratios (%s).", os.path.basename(levels_path), os.path.basename(ratios_path))

            # Save meta
            with open(os.path.join(search_dir, "meta.json"), "w") as f:
                meta = dict(
                    prune_method=method,
                    num_sampling_steps=args.num_sampling_steps,
                    number_of_stages=args.num_stages,
                    stages=stages,
                    generations=args.generations,
                    offspring=args.offspring,
                    survivors_per_selection=args.survivors_per_selection,
                    use_levels=True,
                    mode=("layerdrop" if method == "layerdrop" else "secondorder"),
                    depth=len(model.blocks),
                    num_heads=(args.so_num_heads if method in ("secondorder","wanda","magnitude","activation") else None),
                    H=int(H),
                    target_level_int=int(args.target_level),
                    start_level_int=(None if args.start_level is None else int(args.start_level)),
                    mutation_max_levels=getattr(args, "mutation_max_levels", 2),
                    mutation_n_valid=int(mutation_n_valid),
                    cfg_scale=args.cfg_scale,
                    calibration_loader_nsamples=2 * args.loader_nsamples,
                    fitness_type="trajectory",
                    traj_fitness_mode=args.traj_fitness_mode,
                    traj_suffix_steps=args.traj_suffix_steps,
                    traj_suffix_frac=args.traj_suffix_frac,
                    traj_late_weighting=args.traj_late_weighting,
                    traj_include_eps=bool(args.traj_include_eps),
                    traj_eps_weight=args.traj_eps_weight,
                    traj_probe_batch=args.traj_probe_batch,
                    traj_probe_seed=args.traj_probe_seed,
                    used_validation=bool(args.use_validation),
                    seed=seed,
                    k_semantics="levels_per_stage",
                    traj_fitness_metric=args.traj_fitness_metric,
                )
                json.dump(to_py(meta), f, indent=2)

            if prune_timing["total_prune_seconds"] is None:
                prune_timing["total_prune_seconds"] = time.time() - t_prune_start
            # --------- Save timing info ---------
            def _add_units(sec: Optional[float]):
                if sec is None:
                    return dict(seconds=None, minutes=None, hours=None)
                return dict(
                    seconds=float(sec),
                    minutes=float(sec) / 60.0,
                    hours=float(sec) / 3600.0,
                )

            timings_payload = {
                "calibration": _add_units(prune_timing.get("calib_seconds")),
                "search":      _add_units(prune_timing.get("search_seconds")),
                "total_prune": _add_units(prune_timing.get("total_prune_seconds")),
                "prune_method": method,
                "num_stages": args.num_stages,
                "target_level": float(args.target_level),
                "seed": seed,
            }

            timings_path = os.path.join(search_dir, "timings.json")
            with open(timings_path, "w") as f_t:
                json.dump(to_py(timings_payload), f_t, indent=2)
            logging.info("[Prune] Saved timing info -> %s", timings_path)

        ddp_barrier()

        # end global pruning timer (only rank 0 records)
        # if ddp_rank() == 0:
        #     prune_timing["total_prune_seconds"] = time.time() - t_prune_start

    ddp_barrier()

    # ------------------------- sampling -------------------------
    if args.do_sample:
        logging.info("Starting sampling...")
        logging.info(f"Speed-up: {getattr(model, 'so_struct_forward', False)}")
        meta_path = os.path.join(search_dir, "meta.json")
        meta = json.load(open(meta_path, "r")) if os.path.exists(meta_path) else {"prune_method": method}
        applied_method = meta.get("prune_method", method)

        try:
            # discover files
            levels_path = os.path.join(search_dir, "levels_per_stage.json")
            ratios_path = os.path.join(search_dir, "ratios_per_stage.json")
            orders_ld   = os.path.join(search_dir, "orders_layerdrop.json")
            orders_obs  = os.path.join(bank_dir,  "orders_obs.json")
            obs_repo    = os.path.join(bank_dir,  "repo.pt")
            orders_wa   = os.path.join(search_dir, "orders_wanda.json")
            orders_mag  = os.path.join(search_dir, "orders_magnitude.json")
            orders_act  = os.path.join(search_dir, "orders_activation.json")

            # stages
            stages_from_meta = meta.get("stages", None)
            stages_rt = [tuple(x) for x in (stages_from_meta if stages_from_meta else stages)]

            # Prefer LEVELS; derive ratios on the fly
            ratios: Dict[int, float] = {}
            if os.path.exists(levels_path):
                raw = json.load(open(levels_path, "r"))
                if isinstance(raw, dict):
                    levels = {int(k): int(v) for k, v in raw.items()}
                else:
                    levels = {int(i): int(v) for i, v in enumerate(raw)}
                ratios = {int(s): float(max(0, min(H, int(levels.get(s, 0))))) / float(max(1, H))
                          for s in range(len(stages_rt))}
                logging.info("[Sample] Using levels_per_stage.json (H=%d).", H)
            elif os.path.exists(ratios_path):
                ratios_raw = json.load(open(ratios_path, "r"))
                if isinstance(ratios_raw, dict):
                    ratios = {int(k): float(v) for k, v in ratios_raw.items()}
                else:
                    ratios = {int(i): float(v) for i, v in enumerate(ratios_raw)}
                logging.info("[Sample] Using ratios_per_stage.json (legacy).")
            else:
                logging.warning("[Sample] No per-stage levels/ratios found; sampling unpruned.")
                ratios = {}

            if applied_method == "layerdrop":
                if not os.path.exists(orders_ld):
                    logging.warning("[Sample] Missing %s; sampling with unpruned model.", orders_ld)
                else:
                    orders = json.load(open(orders_ld, "r"))
                    if isinstance(next(iter(orders.keys())), str):
                        orders = {int(k): v for k, v in orders.items()}
                    sched = build_layerdrop_schedule_from_orders(
                        orders_per_stage=orders,
                        stages=stages_rt,
                        ratios=ratios,
                        protect_ends=0,
                    )
                    apply_layerdrop_schedule(model, sched, stages=stages_rt)
                    total_drops = sum(len(v) for v in sched.values())
                    logging.info("[Sample] Applied LayerDrop schedule (drops=%d).", total_drops)

            elif applied_method == "secondorder":
                if os.path.exists(orders_obs) and os.path.exists(obs_repo):
                    so_orders = json.load(open(orders_obs, "r"))
                    sched = build_secondorder_schedule_from_orders(
                        orders_per_stage=so_orders,
                        stages=stages_rt,
                        ratios=ratios,
                        head_dim=args.so_head_dim,
                        num_heads=args.so_num_heads,
                        protect_ends=args.so_protect_ends,
                    )
                    repo = load_obs_bank(bank_dir)
                    bank = select_obs_bank_for_ratios(
                        repo=repo,
                        ratios=ratios,
                        stages=stages_rt,
                        round_mode=args.obs_round_mode,
                    )
                    if hasattr(model, "set_projection_bank"):
                        model.set_projection_bank(bank, stages=stages_rt)
                    apply_secondorder_schedule(
                        model, sched, stages=stages_rt)
                    logging.info("[Sample][OBS] Applied schedule + projection bank.")
                else:
                    logging.warning("[Sample] No OBS assets found; sampling unpruned.")

            else:
                # First-order family at sampling
                file_map = dict(wanda=orders_wa, magnitude=orders_mag, activation=orders_act)
                orders_path = file_map[applied_method]
                if not os.path.exists(orders_path):
                    logging.warning("[Sample] Missing %s; sampling with unpruned model.", os.path.basename(orders_path))
                else:
                    fo_orders = json.load(open(orders_path, "r"))
                    sched = build_secondorder_schedule_from_orders(
                        orders_per_stage=fo_orders,
                        stages=stages_rt,
                        ratios=ratios,
                        head_dim=args.so_head_dim,
                        num_heads=args.so_num_heads,
                        protect_ends=args.so_protect_ends,
                    )
                    apply_secondorder_schedule(model, sched, stages=stages_rt)
                    any_t = next(iter(sched)) if len(sched) else None
                    if any_t is not None:
                        logging.info(
                            "[Sample] Applied %s schedule (t=%s: attn=%d, mlp=%d).",
                            applied_method, any_t,
                            len(sched[any_t].get("attn", {})),
                            len(sched[any_t].get("mlp", {})),
                        )

        except Exception as e:
            logging.exception("[Sample] Failed to build/apply schedule: %s", e)

        if ddp_rank() == 0:
            Path(sample_dir).mkdir(parents=True, exist_ok=True)
            Path(npz_dir).mkdir(parents=True, exist_ok=True)

        per_gpu = args.per_proc_batch_size
        world = ddp_world_size()
        global_bs = per_gpu * world
        total_samples = int(math.ceil(args.num_fid_samples / global_bs) * global_bs)
        if ddp_rank() == 0:
            logging.info("[Sample] Sampling %d images (global_bs=%d).", total_samples, global_bs)

        samples_per_gpu = total_samples // world
        iterations = samples_per_gpu // per_gpu
        assert samples_per_gpu % per_gpu == 0

        base_offset = 0
        if args.resume and args.resume_append and ddp_rank() == 0:
            existing = glob.glob(os.path.join(sample_dir, "*.png"))
            if existing:
                max_idx = max(int(os.path.splitext(os.path.basename(p))[0]) for p in existing)
                base_offset = ((max_idx + 1 + global_bs - 1) // global_bs) * global_bs
                logging.info("[Sample] Resume-append: found %d PNGs, max=%d, base_offset=%d",
                             len(existing), max_idx, base_offset)

        base_offset_t = torch.tensor(int(base_offset), device=device)
        if ddp_is_initialized():
            dist.broadcast(base_offset_t, src=0)
        base_offset = int(base_offset_t.item())

        using_cfg = args.cfg_scale > 1.0
        latent = args.image_size // 8
        total_written = 0

        iterator = tqdm(range(iterations), disable=(ddp_rank() != 0))
        for _ in iterator:
            z = torch.randn(per_gpu, model.in_channels, latent, latent, device=device)
            y = torch.randint(0, args.num_classes, (per_gpu,), device=device)
            # print(y)

            if using_cfg:
                z = torch.cat([z, z], dim=0)
                y_null = torch.tensor([args.num_classes] * per_gpu, device=device)
                y = torch.cat([y, y_null], dim=0)
                sample_fn = model.forward_with_cfg
                model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
            else:
                sample_fn = model.forward
                model_kwargs = dict(y=y)

            samples = diffusion.ddim_sample_loop(
                sample_fn,
                z.shape,
                z,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                progress=False,
                device=device,
            )

            if using_cfg:
                samples, _ = samples.chunk(2, dim=0)

            samples = vae.decode(samples / 0.18215).sample
            samples = torch.clamp(127.5 * samples + 128.0, 0, 255)
            samples = samples.permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

            for i, sample in enumerate(samples):
                idx = base_offset + i * ddp_world_size() + ddp_rank() + total_written
                if idx >= args.num_fid_samples:
                    continue
                Image.fromarray(sample).save(os.path.join(sample_dir, f"{idx:06d}.png"))

            total_written += global_bs

        # OBS cleanup (if used)
        try:
            if hasattr(model, "clear_projection_bank"):
                model.clear_projection_bank()
        except Exception:
            pass
        ddp_barrier()

        if ddp_rank() == 0:
            pngs = glob.glob(os.path.join(sample_dir, "*.png"))
            have = sum(1 for p in pngs if int(os.path.splitext(os.path.basename(p))[0]) < args.num_fid_samples)
            if have == args.num_fid_samples:
                exp_npz_path = os.path.join(npz_dir, f"{exp_name}.npz")
                create_npz_from_sample_folder(sample_dir, exp_npz_path, args.num_fid_samples)
                Path(args.npz_root).mkdir(parents=True, exist_ok=True)
                central_npz = Path(args.npz_root) / f"{exp_name}.npz"
                try:
                    if central_npz.exists() or central_npz.is_symlink():
                        os.remove(central_npz)
                    os.symlink(os.path.abspath(exp_npz_path), central_npz)
                    logging.info("[Sample] NPZ symlinked -> %s", central_npz)
                except Exception:
                    shutil.copy2(exp_npz_path, central_npz)
                    logging.info("[Sample] NPZ copied -> %s", central_npz)
            else:
                logging.info("[Sample] Only %d/%d PNGs present; skipping NPZ packaging.", have, args.num_fid_samples)

    ddp_barrier()
    if ddp_is_initialized():
        dist.destroy_process_group()
    if ddp_rank() == 0:
        logging.info("Done worker phase.")

# ======================================================================================
# ORCHESTRATOR MODE
# ======================================================================================

def _args_to_forwardable_list(args: argparse.Namespace) -> List[str]:
    skip = {
        "mode",
        "search_nproc", "sample_nproc",
        "search_cuda", "sample_cuda",
        "do_eval", "eval_cmd",
        "do_prune", "do_sample",
    }
    def flag(name: str) -> str:
        return f"--{name.replace('_', '-')}"
    out: List[str] = []
    for k, v in vars(args).items():
        if k in skip:
            continue
        if v is None:
            continue
        if isinstance(v, bool):
            out.append(flag(k) if v else f"--no-{k.replace('_','-')}")
            continue
        if isinstance(v, list):
            out.append(flag(k))
            out.extend(map(str, v))
            continue
        out.append(f"{flag(k)}={v}")
    return out

def _run_subprocess(cmd: List[str], extra_env: Optional[dict] = None) -> None:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    env.setdefault("PYTHONUNBUFFERED", "1")
    logging.info("Running: %s", " ".join(cmd))
    subprocess.check_call(cmd, env=env)

def orchestrate_main(args: argparse.Namespace) -> None:
    Path(args.experiments_dir).mkdir(parents=True, exist_ok=True)
    log_file = os.path.join(args.experiments_dir, "logs", f"orchestrate.log")
    setup_logger(log_file, console=True, level=args.log_level)

    forwarded = _args_to_forwardable_list(args)
    cuda_list = _parse_cuda_list(args.cudas)
    visible_cuda = ",".join(cuda_list) if cuda_list else os.environ.get("CUDA_VISIBLE_DEVICES", "")
    num_sample_procs = len(cuda_list) if cuda_list else (args.gpus if args.gpus else 1)
    mutation_n_valid = args.mutation_n_valid if args.mutation_n_valid is not None else 2 * args.num_stages

    common_env = {}
    if visible_cuda:
        common_env["CUDA_VISIBLE_DEVICES"] = visible_cuda

    # Phase 1: SEARCH (force single GPU)
    if args.do_prune:
        search_env = dict(common_env)  # same visible GPUs (or none -> inherits)
        nproc_search = 1  # always 1 GPU for search
        search_cmd = [
            "conda", "run", "--no-capture-output", "-n", args.search_env,
            sys.executable, "-m", "torch.distributed.run",
            f"--nproc_per_node={nproc_search}",
            f"--master_port={_pick_free_port()}",
            os.path.abspath(__file__),
            "--mode=worker",
            "--do-prune",
            "--no-do-sample",
        ] + forwarded
        _run_subprocess(search_cmd, extra_env=search_env)

    # Phase 2: SAMPLE (use all GPUs)
    if args.do_sample:
        sample_env = dict(common_env)
        sample_cmd = [
            "conda", "run", "--no-capture-output", "-n", args.sample_env,
            sys.executable, "-m", "torch.distributed.run",
            f"--nproc_per_node={num_sample_procs}",
            f"--master_port={_pick_free_port()}",
            os.path.abspath(__file__),
            "--mode=worker",
            "--no-do-prune",
            "--do-sample",
        ] + forwarded
        _run_subprocess(sample_cmd, extra_env=sample_env)

    # Phase 3: EVAL
    if args.do_eval:
        logging.info("Starting evaluation...")
        exp_name = build_exp_name(
            model=args.model,
            ckpt=args.ckpt,
            image_size=args.image_size,
            vae=args.vae,
            cfg_scale=args.cfg_scale,
            num_sampling_steps=args.num_sampling_steps,
            num_stages=args.num_stages,
            stage_dividers=args.stage_dividers,
            target_level=float(args.target_level),
            seed=args.seed,
            generations=args.generations,
            offspring=args.offspring,
            survivors_per_selection=args.survivors_per_selection,
            mutation_max_levels=args.mutation_max_levels,
            mutation_n_valid=mutation_n_valid,
            fitness_batches=args.fitness_batches,
            fitness_reduce=args.fitness_reduce,
            start_level=(None if args.start_level is None else float(args.start_level)),
            prune_method=args.prune_method,
            traj_fitness_mode=args.traj_fitness_mode,
            traj_suffix_steps=args.traj_suffix_steps,
            traj_suffix_frac=args.traj_suffix_frac,
            traj_late_weighting=args.traj_late_weighting,
            traj_include_eps=bool(args.traj_include_eps),
            traj_eps_weight=args.traj_eps_weight,
            traj_probe_batch=args.traj_probe_batch,
            traj_refresh_every=args.traj_refresh_every,
            traj_fitness_metric=args.traj_fitness_metric,
            abs_fft_weight=args.abs_fft_weight,
            use_validation=bool(args.use_validation),
            calib_importance=args.calib_importance,
            init_strategy=args.init_strategy,
            so_struct_speedup=args.so_struct_speedup,
        )
        exp_dir  = os.path.join(args.experiments_dir, exp_name)
        npz_path = os.path.join(exp_dir, "npz", f"{exp_name}.npz")

        if not os.path.exists(npz_path):
            logging.warning("[Eval] NPZ not found at %s; skipping evaluation.", npz_path)
        elif not args.ref_npz or not os.path.exists(args.ref_npz):
            logging.warning("[Eval] Reference NPZ (--ref-npz) missing or invalid; skipping evaluation.")
        else:
            eval_dir = os.path.join(exp_dir, "eval")
            os.makedirs(eval_dir, exist_ok=True)
            out_json = os.path.join(eval_dir, "metrics.json")
            out_txt  = os.path.join(eval_dir, "metrics.txt")
            eval_script = os.path.join(os.path.dirname(__file__), "evaluator.py")
            cmd = [
                "conda", "run", "-n", args.eval_env,
                "python", eval_script,
                args.ref_npz,
                npz_path,
                "--out-json", out_json,
                "--out-text", out_txt,
            ]
            logging.info("[Eval] Running evaluator.py in conda env '%s'", args.eval_env)
            logging.info("CMD: %s", " ".join(cmd))
            _run_subprocess(cmd, extra_env=dict(common_env))  # <<< use same CUDA visibility
            if os.path.exists(out_json):
                try:
                    with open(out_json, "r") as f:
                        metrics = json.load(f)
                    logging.info("=== Evaluation Results ===")
                    for k in sorted(metrics.keys()):
                        logging.info("%-20s : %s", k, metrics[k])
                    logging.info("Saved metrics -> %s", out_json)
                    if os.path.exists(out_txt):
                        logging.info("Saved text    -> %s", out_txt)
                except Exception as e:
                    logging.warning("[Eval] Completed but failed to parse metrics.json: %s", e)

    logging.info("Orchestration complete.")

# ======================================================================================
# CLI
# ======================================================================================

def _parse_cuda_list(cudas: str) -> List[str]:
    return [x.strip() for x in cudas.split(",") if x.strip()] if cudas else []

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evolutionary search + sampling (DDP-friendly) + Orchestrator")

    # orchestrator vs worker
    parser.add_argument("--mode", type=str, default="orchestrate", choices=["worker", "orchestrate"])
    # NEW unified CUDA controls
    parser.add_argument(
        "--cudas", dest="cudas", type=str, default="",
        help="Comma-separated GPU ids to use, e.g. '0,1,3'. "
            "Search uses 1 GPU from this list; sampling uses them all; eval sees the same."
    )
    parser.add_argument(
        "--gpus", dest="gpus", type=int, default=None,
        help="Optional override for #GPUs for sampling when --cudas is empty. "
            "Search still uses 1."
    )
    parser.add_argument("--do-eval", "--do_eval", dest="do_eval",
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ref-npz", "--ref_npz", dest="ref_npz",
                        type=str, default="./npz_files/VIRTUAL_imagenet256_labeled.npz")
    parser.add_argument("--eval-env", "--eval_env", dest="eval_env",
                        type=str, default="eval")
    parser.add_argument("--search-env", "--search_env", dest="search_env",
                        type=str, default="DiT")
    parser.add_argument("--sample-env", "--sample_env", dest="sample_env",
                        type=str, default="DiT")

    # directories
    parser.add_argument("--experiments-dir", "--experiments_dir", dest="experiments_dir",
                        type=str, default="./experiments")
    parser.add_argument("--npz-root", "--npz_root", dest="npz_root",
                        type=str, default="./npz_files")

    # model / sampling
    parser.add_argument("--model", type=str, default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--image-size", "--image_size", dest="image_size",
                        type=int, default=256)
    parser.add_argument("--num-classes", "--num_classes", dest="num_classes",
                        type=int, default=1000)
    parser.add_argument("--per-proc-batch-size", "--per_proc_batch_size", dest="per_proc_batch_size",
                        type=int, default=16)
    parser.add_argument("--num-fid-samples", "--num_fid_samples", dest="num_fid_samples",
                        type=int, default=50000)
    parser.add_argument("--num-sampling-steps", "--num_sampling_steps", dest="num_sampling_steps",
                        type=int, default=20)
    parser.add_argument("--cfg-scale", "--cfg_scale", dest="cfg_scale",
                        type=float, default=2.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume-append", "--resume_append", dest="resume_append",
                        action=argparse.BooleanOptionalAction, default=False)

    # method selection (simple & explicit)
    parser.add_argument("--prune-method", "--prune_method", dest="prune_method",
                        type=str, default="layerdrop",
                        choices=["layerdrop", "secondorder", "wanda", "magnitude", "activation"])

    # evolutionary search toggles
    parser.add_argument("--do-prune", "--do_prune", dest="do_prune",
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--do-sample", "--do_sample", dest="do_sample",
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-stages", "--num_stages", dest="num_stages",
                        type=int, default=20)
    parser.add_argument("--stage-dividers", "--stage_dividers", dest="stage_dividers",
                        type=int, nargs="+", default=None,
        help="Internal stage dividers (exclude 0 and 1000). Exactly num_stages-1 ints, strictly increasing.")
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--offspring", type=int, default=16)
    parser.add_argument("--survivors-per-selection", "--survivors_per_selection", dest="survivors_per_selection",
                        type=int, nargs="+", default=[4])

    # LEVEL-FIRST knobs
    parser.add_argument("--target-level", "--target_level", dest="target_level",
                        type=float, required=True,
                        help="Global *average* target in integer levels (rounded if float).")
    parser.add_argument("--start-level", "--start_level", dest="start_level",
                        type=float, default=None,
                        help="Optional warm/gradual start in integer levels (rounded if float).")
    parser.add_argument("--mutation-max-levels", "--mutation_max_levels", dest="mutation_max_levels",
                        type=int, default=2,
                        help="Maximum integer levels to move per mutation step.")
    parser.add_argument("--mutation-n-valid", "--mutation_n_valid", dest="mutation_n_valid",
                    type=int, default=1,
                    help="Number of successful level-transfer moves to apply per mutation.")

    parser.add_argument("--fitness-batches", "--fitness_batches", dest="fitness_batches",
                        type=int, default=256)
    parser.add_argument("--fitness-reduce", "--fitness_reduce", dest="fitness_reduce",
                        type=str, default="mean", choices=["mean", "median"])
    parser.add_argument("--loader-nsamples", "--loader_nsamples", dest="loader_nsamples",
                        type=int, default=1024)
    
    parser.add_argument("--init-strategy", "--init_strategy", dest="init_strategy",
                        type=str, default="hybrid",
                        choices=["random","uniform","heuristic_only","hybrid","dirichlet","warm_hybrid"])
    parser.add_argument("--init-patterns", "--init_patterns", dest="init_patterns",
                        type=str, nargs="+", default=["front","back","middle","ramp_up","ramp_down","zigzag","ends"],
                        help="Patterns to include for initialization (e.g., front, back, ...). "
                             "If omitted, the code uses its internal defaults.")
    parser.add_argument("--init-random-fraction", "--init_random_fraction", dest="init_random_fraction",
                        type=float, default=0.4,
                        help="Fraction of random individuals to include when strategy='hybrid'.")

    # OBS / runtime specifics
    parser.add_argument("--so-head-dim", "--so_head_dim", dest="so_head_dim",
                        type=int, default=72, help="Per-head hidden dim.")
    parser.add_argument("--so-num-heads", "--so_num_heads", dest="so_num_heads",
                        type=int, default=16, help="Number of attention heads.")
    parser.add_argument("--so-protect-ends", "--so_protect_ends", dest="so_protect_ends",
                        type=int, default=0, help="Protect this many groups at both ends.")
    parser.add_argument("--obs-round-mode", "--obs_round_mode", dest="obs_round_mode",
                        type=str, default="nearest", choices=["floor", "nearest", "ceil"])
    parser.add_argument("--obs-bank-root", "--obs_bank_root", dest="obs_bank_root",
                        type=str, default="./pretrained_models/obs_bank")
    parser.add_argument("--so-struct-speedup", "--so_struct_speedup", dest="so_struct_speedup",
                        action=argparse.BooleanOptionalAction, default=True)

    # validation toggle for fitness
    parser.add_argument("--use-validation", "--use_validation", dest="use_validation",
                        action=argparse.BooleanOptionalAction, default=False)

    # trajectory-fitness flags
    parser.add_argument("--traj-fitness-mode", "--traj_fitness_mode", dest="traj_fitness_mode",
                        type=str, default="final", choices=["final", "suffix", "full"])
    parser.add_argument("--traj-suffix-steps", "--traj_suffix_steps", dest="traj_suffix_steps",
                        type=int, default=None)
    parser.add_argument("--traj-suffix-frac", "--traj_suffix_frac", dest="traj_suffix_frac",
                        type=float, default=0.5)
    parser.add_argument("--traj-late-weighting", "--traj_late_weighting", dest="traj_late_weighting",
                        type=str, default="cosine", choices=["cosine", "linear", "uniform"])
    parser.add_argument("--traj-include-eps", "--traj_include_eps", dest="traj_include_eps",
                        action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--traj-eps-weight", "--traj_eps_weight", dest="traj_eps_weight",
                        type=float, default=0.3)
    parser.add_argument("--traj-probe-batch", "--traj_probe_batch", dest="traj_probe_batch",
                        type=int, default=64)
    parser.add_argument("--traj-probe-seed", "--traj_probe_seed", dest="traj_probe_seed",
                        type=int, default=1234)
    parser.add_argument("--traj-refresh-every", "--traj_refresh_every", dest="traj_refresh_every",
                        type=int, default=0)
    parser.add_argument("--traj-fitness-metric", "--traj_fitness_metric", dest="traj_fitness_metric",
                        type=str, default="latent_mse",
                        choices=["latent_mse", "latent_cos", "latent_snr", "latent_snr_cosine", 
                                 "img_mse", "img_ssim", "img_fft", "img_niqe", "img_clipiqa", "img_topiq",
                                 "img_hyperiqa", "img_liqe", "img_qalign", "img_qualiclip",
                                 "fft_cos_bound", "latent_abs", "latent_abs_fft"])
    parser.add_argument("--abs-fft-weight", "--abs_fft_weight", dest="abs_fft_weight",
                        type=float, default=0.5)

    # layerdrop calibration knobs
    parser.add_argument("--calib-importance", "--calib_importance", dest="calib_importance",
                        type=str, default="cosine", choices=["mse", "cosine"])
    parser.add_argument("--calib-cosine-eps", "--calib_cosine_eps", dest="calib_cosine_eps",
                        type=float, default=1e-8)

    # logging
    parser.add_argument("--log-level", "--log_level", dest="log_level",
                        type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    return parser

def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.mode == "orchestrate":
        orchestrate_main(args)
    else:
        worker_main(args)

if __name__ == "__main__":
    main()
