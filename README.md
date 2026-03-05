# Diff-ES
Diff-ES: Stage-Wise Structural Diffusion Pruning via Evolutionary Search

```
# Example command for DiT, 50% Sparsity
python evo_pruning_ddp.py --cudas 0 \
--prune-method secondorder --target_level 8 --num-stages 10 --init-strategy hybrid --generations 50 --per-proc-batch-size 16 \
--traj-fitness-metric img_topiq --mutation-max-levels 5 --loader-nsamples 1024 --fitness-batches 128 --mutation-n-valid 1 --traj-probe-batch 64

# Example command for SDXL, 30% Sparsity
python evo_pruning_sdxl.py --prune-method secondorder --target-level 3 --num-stages 10 \
--init-strategy hybrid --generations 100 --per-proc-batch-size 4 --experiments-dir ./experiments_1 \
--traj-fitness-metric img_ssim --mutation-max-levels 3 --loader-nsamples 1024 --fitness-batches 16 --traj-probe-batch 4 --mutation-n-valid 1
```