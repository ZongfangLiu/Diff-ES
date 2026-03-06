# Diff-ES: Stage-Wise Structural Diffusion Pruning via Evolutionary Search.

- [ArXiv Preprint](https://arxiv.org/abs/2603.05105)

## Overview
This repository contains two pruning pipelines:

- `dit/`: DiT pruning + distributed sampling (`evo_pruning_ddp.py`).
- `sdxl/`: SDXL pruning + sampling + evaluation (`evo_pruning_sdxl.py`).

Both pipelines follow the same idea: calibrate stage-wise importance, run evolutionary search for per-stage pruning levels, then sample/evaluate with the discovered schedule.

## Environment Setup
### 1) Requirements
- Linux with NVIDIA GPU and CUDA.
- Conda (recommended, required by DiT orchestrator flow).
- Python 3.10+.

### 2) Create environment(s)
`dit/evo_pruning_ddp.py` launches phases via `conda run -n <env>`.
Use separate environments for search/sampling and evaluation.

#### Search/Sample env (`DiT`)
```bash
conda create -n DiT python=3.10 -y
conda activate DiT
pip install -r requirements.txt
```

#### Eval env (`eval`, recommended)
`dit/evaluator.py` uses TensorFlow (`tensorflow.compat.v1`), so isolating eval avoids package-version conflicts with the main PyTorch stack:

```bash
conda create -n eval python=3.10 -y
conda activate eval
pip install "tensorflow==2.15.*" "numpy<2" scipy requests tqdm
```

Then set:
- `--search-env DiT`
- `--sample-env DiT`
- `--eval-env eval`

### 3) Data and checkpoints
#### DiT (`dit/evo_pruning_ddp.py`)
- ImageNet train set is expected at `~/datasets/imagenet-1k/train` (currently hardcoded in script).
- VAE path defaults to `./pretrained_models/sd-vae-ft-ema` (or `...-mse` with `--vae mse`).
- DiT checkpoint defaults to `DiT-XL-2-256x256.pt` unless `--ckpt` is provided.
- DiT evaluation expects a reference NPZ (`--ref-npz`, default `./npz_files/VIRTUAL_imagenet256_labeled.npz`).
- If you enable DiT evaluation, ensure `dit/evaluator.py` is available.
- If you see `ModuleNotFoundError: download`, add `download.py` (with `find_model`) into `dit/`.

#### SDXL (`sdxl/evo_pruning_sdxl.py`)
- COCO paths (defaults):
  - `~/datasets/coco/train2017`
  - `~/datasets/coco/annotations/captions_train2017.json`
  - `~/datasets/coco/val2017`
  - `~/datasets/coco/annotations/captions_val2017.json`
- SDXL base model is loaded as `stabilityai/stable-diffusion-xl-base-1.0` with offline flags in code, so download/cache it before running.

## Running
Run commands from repository root.

### DiT example (50% sparsity style)
```bash
python dit/evo_pruning_ddp.py \
  --cudas 0 \
  --prune-method secondorder \
  --target-level 8 \
  --num-stages 10 \
  --init-strategy hybrid \
  --generations 50 \
  --per-proc-batch-size 16 \
  --traj-fitness-metric img_topiq \
  --mutation-max-levels 5 \
  --loader-nsamples 1024 \
  --fitness-batches 128 \
  --mutation-n-valid 1 \
  --traj-probe-batch 64 \
  --search-env DiT \
  --sample-env DiT \
  --no-do-eval
```

Remove `--no-do-eval` when your DiT evaluation dependencies and reference NPZ are ready, and set `--eval-env eval`.

### SDXL example (30% sparsity style)
```bash
python sdxl/evo_pruning_sdxl.py \
  --prune-method secondorder \
  --target-level 3 \
  --num-stages 10 \
  --init-strategy hybrid \
  --generations 100 \
  --per-proc-batch-size 4 \
  --experiments-dir ./experiments_1 \
  --traj-fitness-metric img_ssim \
  --mutation-max-levels 3 \
  --loader-nsamples 1024 \
  --fitness-batches 16 \
  --traj-probe-batch 4 \
  --mutation-n-valid 1 \
  --image-dir ~/datasets/coco/train2017 \
  --ann-file ~/datasets/coco/annotations/captions_train2017.json \
  --coco-val-dir ~/datasets/coco/val2017 \
  --coco-val-ann ~/datasets/coco/annotations/captions_val2017.json \
  --fid-real-dir ~/datasets/coco/val2017
```

## Explanation of Main Arguments
- `--prune-method`: pruning backend (`layerdrop`, `secondorder`, `wanda`, `magnitude`, `activation`).
- `--target-level`: global average pruning target in level space.
- `--num-stages`: number of diffusion timeline segments.
- `--generations`, `--offspring`: evolutionary search budget.
- `--loader-nsamples`, `--fitness-batches`: calibration/evaluation data budget per run.
- `--traj-fitness-metric`: objective used during search.

## Output Structure
Each run creates `experiments/<exp_name>/` (or your `--experiments-dir`) with artifacts such as:

- `logs/`: runtime logs.
- `search/`: discovered schedules (`levels_per_stage.json`, `ratios_per_stage.json`, `meta.json`, `timings.json`).
- `samples/`: generated images.
- `npz/`: packaged samples for DiT FID flow.
- `eval/`: metrics (when evaluation is enabled).
