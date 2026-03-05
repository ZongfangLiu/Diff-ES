import math
from os import device_encoding
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp

# -----------------------------------------------------------------------------
# Small helper
# -----------------------------------------------------------------------------

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

# -----------------------------------------------------------------------------
# Embedding layers
# -----------------------------------------------------------------------------

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings

# -----------------------------------------------------------------------------
# Core DiT
# -----------------------------------------------------------------------------

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.

    OBS integration (final design):
    - OBS bank provides thin projection weights (attn.proj, mlp.fc2) per stage/block.
    - You must still apply the thin schedule; OBS overrides only replace the sliced weights.
    """

    # ------------------------------- init -----------------------------------
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

        # ---- LayerDrop (static & per-timestep schedule) ----
        self.drop_block_ids: set[int] = set()
        self.layerdrop_schedule = None
        self.layerdrop_stages: Optional[List[Tuple[int,int]]] = None

        # ---- Second-Order "struct" runtime (thin GEMMs) ----
        self.so_struct_forward: bool = False
        self._so_struct_schedule: Optional[Dict[int, Dict[str, Dict[int, List[int]]]]] = None
        self._so_struct_stages: Optional[List[Tuple[int,int]]] = None
        self._so_struct_compiled: Optional[List[Optional[List[Optional[Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]]]]]] = None
        # compiled[t][blk] -> (qkv_rows, proj_cols, mlp_keep) or None

        # ---- OBS projection bank (stage-aware), with fast lookups ----
        self._proj_bank = None               # tensors on device
        self._proj_bank_cpu = None           # original (optional)
        self._proj_bank_stages: Optional[List[Tuple[int,int]]] = None
        self._proj_bank_t2s: Optional[List[int]] = None

        # fast OBS lookup tables
        self._pb_by_stage: Optional[List[Tuple[List[Optional[Tuple[Any, Optional[torch.Tensor], Optional[int]]]],
                                               List[Optional[Tuple[Any, Optional[torch.Tensor]]]]]]] = None
        self._pb_by_t: Optional[List[Optional[Tuple[List[Optional[Tuple[Any, Optional[torch.Tensor], Optional[int]]]],
                                                    List[Optional[Tuple[Any, Optional[torch.Tensor]]]]]]]] = None
        self._pb_depth: Optional[int] = None

        self._current_stage = -1
        self._using_proj_bank: bool = False

        # legacy-compat flag used in some code paths
        self.strict_equiv_mlp: bool = False

    # ------------------------------- public toggles --------------------------
    def enable_struct_prune_forward(self):
        """Enable thin-GEMM path driven by a per-step second-order schedule (required for OBS)."""
        self.so_struct_forward = True

    def clear_secondorder_struct_runtime(self):
        """Disable struct path and clear compiled schedule."""
        self.so_struct_forward = False
        self._so_struct_schedule = None
        self._so_struct_stages = None
        self._so_struct_compiled = None

    # LayerDrop APIs
    def set_layerdrop(self, drop_ids):
        self.drop_block_ids = set(int(i) for i in drop_ids)

    def clear_layerdrop(self):
        self.drop_block_ids.clear()

    def set_layerdrop_schedule(self, schedule, stages=None):
        """
        `schedule`: dict {native_timestep -> [block_ids]} or callable(step)->indices
        `stages`:   list of (lo, hi) native step ranges, used by stage-grouped forwarding
        """
        self.layerdrop_schedule = schedule
        self.layerdrop_stages = [(int(lo), int(hi)) for (lo, hi) in (stages or [])] if stages else None

    def clear_layerdrop_schedule(self):
        self.layerdrop_schedule = None
        self.layerdrop_stages = None

    # ------------------------------- schedule compilation --------------------
    def set_secondorder_struct_schedule(self, schedule: dict, stages: Optional[List[Tuple[int,int]]] = None):
        """
        Preferred entry point: install a per-step thin schedule and compile it to
        fast index tensors used during forward.
        """
        self._so_struct_schedule = schedule or {}
        self._so_struct_stages = [(int(lo), int(hi)) for (lo, hi) in (stages or [])] if stages else None
        self._compile_struct_schedule()

    # Back-compat alias some utilities might call
    def set_secondorder_schedule(self, schedule: dict, stages: Optional[List[Tuple[int,int]]] = None):
        self.set_secondorder_struct_schedule(schedule, stages)

    def _compile_struct_schedule(self):
        """Turn Python lists/dicts into per-(t,block) LongTensors for fast index_select."""
        sch = self._so_struct_schedule
        if not sch:
            self._so_struct_compiled = None
            return

        depth = len(self.blocks)
        # infer dims from first block
        attn0 = self.blocks[0].attn
        H = int(getattr(attn0, "num_heads", self.num_heads))
        C = int(getattr(self.blocks[0], "hidden_size", attn0.qkv.weight.shape[1]))
        d = C // H
        Fhid = int(self.blocks[0].mlp.fc2.in_features)
        dev = attn0.qkv.weight.device

        max_t = max(int(t) for t in sch.keys()) if isinstance(sch, dict) else -1
        compiled: List[Optional[List[Optional[Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]]]]] = [None] * (max_t + 1)

        full_heads = torch.arange(H, dtype=torch.long, device=dev)
        full_dspan = torch.arange(d, dtype=torch.long, device=dev)
        full_hidden = torch.arange(Fhid, dtype=torch.long, device=dev)

        for t in range(max_t + 1):
            ent = sch.get(t, {}) if isinstance(sch, dict) else {}
            attn_map = ent.get("attn", {}) or {}
            mlp_map  = ent.get("mlp",  {}) or {}
            comp_for_t: List[Optional[Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]]] = [None] * depth

            for b in range(depth):
                dh = attn_map.get(b, None)
                dm = mlp_map.get(b, None)

                qkv_rows = proj_cols = mlp_keep = None

                if dh:
                    drops = torch.tensor(sorted(set(int(x) for x in dh)), dtype=torch.long, device=dev)
                    mask = torch.ones(H, dtype=torch.bool, device=dev)
                    mask[drops] = False
                    kept_heads = full_heads[mask]              # [Hk]
                    head_spans = (kept_heads[:, None] * d + full_dspan[None, :]).reshape(-1)  # [Hk*d]
                    q_rows = head_spans
                    k_rows = head_spans + C
                    v_rows = head_spans + 2 * C
                    qkv_rows = torch.cat([q_rows, k_rows, v_rows], dim=0)                     # [3*Hk*d]
                    proj_cols = head_spans                                                     # [Hk*d]

                if dm:
                    drops = torch.tensor(sorted(set(int(x) for x in dm)), dtype=torch.long, device=dev)
                    mask = torch.ones(Fhid, dtype=torch.bool, device=dev)
                    mask[drops] = False
                    mlp_keep = full_hidden[mask]                                              # [K]

                comp_for_t[b] = (qkv_rows, proj_cols, mlp_keep)
            compiled[t] = comp_for_t

        self._so_struct_compiled = compiled

    # ------------------------------- OBS bank install ------------------------
    def set_projection_bank(self, bank: dict, stages: Optional[List[Tuple[int, int]]] = None):
        """
        Install a stage->(attn/mlp per block) projection bank produced by OBS calibration.
        Copies tensors to this model replica's device and dtype (DDP-safe),
        and builds fast per-step lookup tables.
        """
        # Save CPU copy & stages
        self._proj_bank_cpu = bank or {}
        self._proj_bank_stages = [(int(lo), int(hi)) for (lo, hi) in (stages or [])] if stages else None

        # Dense t->stage table
        self._proj_bank_t2s = None
        if self._proj_bank_stages:
            max_hi = max(int(hi) for (_, hi) in self._proj_bank_stages)
            T = max_hi + 1
            t2s = [-1] * T
            for sid, (lo, hi) in enumerate(self._proj_bank_stages):
                lo_i, hi_i = max(0, int(lo)), min(T - 1, int(hi))
                for t in range(lo_i, hi_i + 1):
                    t2s[t] = sid
            self._proj_bank_t2s = t2s

        # Move tensors to device/dtype
        dev = next(self.parameters()).device
        ref_dtype = self.blocks[0].attn.proj.weight.dtype if len(self.blocks) > 0 else torch.float32
        self._proj_bank = self._bank_to_device(self._proj_bank_cpu, device=dev, dtype=ref_dtype)

        # Build fast lookup tables
        self._build_fast_obs_tables()

        self._using_proj_bank = bool(self._proj_bank)
        self._current_stage = -1

    def clear_projection_bank(self):
        self._proj_bank = None
        self._proj_bank_cpu = None
        self._proj_bank_stages = None
        self._proj_bank_t2s = None
        self._pb_by_stage = None
        self._pb_by_t = None
        self._pb_depth = None
        self._using_proj_bank = False
        self._current_stage = -1

    @staticmethod
    def _bank_to_device(bank: dict, *, device, dtype):
        """Copy tensors inside `bank` to device/dtype; keep indices as Python ints."""
        if not bank:
            return {}
        out: Dict[int, Dict[str, Dict[int, Dict[str, Any]]]] = {}
        for sid, stage_map in bank.items():
            s_out = {"attn": {}, "mlp": {}}
            for bid, entry in stage_map.get("attn", {}).items():
                ne = {}
                if entry.get("proj_w") is not None:
                    ne["proj_w"]  = entry["proj_w"].to(device=device, dtype=dtype, non_blocking=True)
                if "kept_idx" in entry: ne["kept_idx"] = [int(i) for i in entry["kept_idx"]]
                if "head_dim" in entry: ne["head_dim"] = int(entry["head_dim"])
                s_out["attn"][int(bid)] = ne
            for bid, entry in stage_map.get("mlp", {}).items():
                ne = {}
                if entry.get("fc2_w") is not None:
                    ne["fc2_w"]   = entry["fc2_w"].to(device=device, dtype=dtype, non_blocking=True)
                if "kept_idx" in entry: ne["kept_idx"] = [int(i) for i in entry["kept_idx"]]
                s_out["mlp"][int(bid)] = ne
            out[int(sid)] = s_out
        return out

    def _build_fast_obs_tables(self):
        """Precompute O(1) t/block → override tuples; no tensor duplication."""
        if not self._proj_bank:
            self._pb_by_stage = None
            self._pb_by_t = None
            self._pb_depth = None
            return

        depth = len(self.blocks)
        self._pb_depth = depth
        dev = next(self.parameters()).device  # model's device

        # Per-stage, per-block lists
        pb_by_stage: List[Tuple[List[Optional[Tuple[Any, Optional[torch.Tensor], Optional[int]]]],
                                List[Optional[Tuple[Any, Optional[torch.Tensor]]]]]] = []

        for sid in sorted(self._proj_bank.keys()):
            stage_map = self._proj_bank[sid]
            attn_map = stage_map.get("attn", {}) or {}
            mlp_map  = stage_map.get("mlp",  {}) or {}

            attn_list: List[Optional[Tuple[Any, Optional[torch.Tensor], Optional[int]]]] = [None] * depth
            mlp_list : List[Optional[Tuple[Any, Optional[torch.Tensor]]]] = [None] * depth

            for b, entry in attn_map.items():
                kept_tensor = (torch.as_tensor(entry["kept_idx"], dtype=torch.long, device=dev)
                               if "kept_idx" in entry else None)
                attn_list[int(b)] = (
                    entry.get("proj_w", None),
                    kept_tensor,
                    entry.get("head_dim", None),
                )

            for b, entry in mlp_map.items():
                kept_tensor = (torch.as_tensor(entry["kept_idx"], dtype=torch.long, device=dev)
                               if "kept_idx" in entry else None)
                mlp_list[int(b)] = (
                    entry.get("fc2_w", None),
                    kept_tensor,
                )

            pb_by_stage.append((attn_list, mlp_list))

        self._pb_by_stage = pb_by_stage

        # Per-timestep table (references to per-stage lists)
        self._pb_by_t = None
        if self._proj_bank_t2s is not None:
            T = len(self._proj_bank_t2s)
            by_t: List[Optional[Tuple[List[Optional[Tuple[Any, Optional[torch.Tensor], Optional[int]]]],
                                      List[Optional[Tuple[Any, Optional[torch.Tensor]]]]]]] = [None] * T
            for t, sid in enumerate(self._proj_bank_t2s):
                by_t[t] = pb_by_stage[sid] if sid >= 0 else None
            self._pb_by_t = by_t

    # ------------------------------- init helpers ----------------------------
    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    # ------------------------------- utils -----------------------------------
    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    # ------------------------------- forward ---------------------------------
    def forward(self, x, t, y):
        # Lazy safety: if caller set raw schedule fields directly, compile once before use
        if self.so_struct_forward and self._so_struct_schedule is not None and self._so_struct_compiled is None:
            self._compile_struct_schedule()
        # Embed inputs
        x = self.x_embedder(x) + self.pos_embed   # (N, T, D)
        t_emb = self.t_embedder(t)                # (N, D)
        y_emb = self.y_embedder(y, self.training) # (N, D)
        c = t_emb + y_emb                         # conditioning

        # LayerDrop
        active_drop = set(self.drop_block_ids)
        step = int(t[0].item()) if torch.is_tensor(t) else int(t)

        if self.layerdrop_schedule is not None:
            if callable(self.layerdrop_schedule):
                sched_ids = self.layerdrop_schedule(step) or ()
            else:
                sched_ids = self.layerdrop_schedule.get(step, ()) or ()
            try:
                active_drop |= set(int(i) for i in sched_ids)
            except Exception:
                pass

        # Pull per-step struct slice once
        sched_for_step = None
        if self.so_struct_forward and self._so_struct_schedule is not None:
            if isinstance(self._so_struct_schedule, dict):
                sched_for_step = self._so_struct_schedule.get(int(step), None)

        # Blocks
        for i, block in enumerate(self.blocks):
            if i in active_drop:
                continue
            if self.so_struct_forward and sched_for_step is not None:
                x = self._forward_block_struct(block, x, c, i, step, sched_for_step)
            else:
                x = block(x, c)

        # Final layer
        x = self.final_layer(x, c)
        x = self.unpatchify(x)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.__call__(combined, t, y)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

    # ----------------- struct runtime internals (thin GEMMs) -----------------
    def _forward_block_struct(self, block, x, c, blk_idx: int, step: int, sched_for_step: dict):
        """
        Thin-by-masking path with optional OBS projection overrides.
        Requires: self.so_struct_forward == True and schedule compiled/installed.
        """
        # precomputed compiled indices
        qkv_rows = proj_cols = mlp_keep = None
        if self._so_struct_compiled is not None and 0 <= int(step) < len(self._so_struct_compiled):
            comp_for_t = self._so_struct_compiled[int(step)]
            if comp_for_t is not None and 0 <= int(blk_idx) < len(comp_for_t):
                tpl = comp_for_t[int(blk_idx)]
                if tpl is not None:
                    qkv_rows, proj_cols, mlp_keep = tpl

        # adaLN-Zero
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            block.adaLN_modulation(c).chunk(6, dim=1)

        # Fast OBS override lookup (no dict walking)
        attn_tuple = mlp_tuple = None
        headdim_override = None
        if self._pb_by_t is not None and 0 <= int(step) < len(self._pb_by_t):
            sid_pack = self._pb_by_t[int(step)]
            if sid_pack is not None:
                attn_list, mlp_list = sid_pack
                if 0 <= int(blk_idx) < (self._pb_depth or len(attn_list)):
                    attn_tuple = attn_list[int(blk_idx)]
                    mlp_tuple  = mlp_list[int(blk_idx)]

        # If bank provides explicit kept_idx, override compiled indices to match bank exactly
        # (This mirrors the slow version's "trust the bank" behavior.)
        C_in = block.attn.qkv.weight.shape[1]
        if attn_tuple is not None and attn_tuple[1] is not None:
            kept_idx = attn_tuple[1]  # LongTensor on correct device
            qkv_rows = torch.cat((kept_idx, kept_idx + C_in, kept_idx + 2 * C_in), dim=0)
            proj_cols = kept_idx
            headdim_override = attn_tuple[2] if (len(attn_tuple) > 2) else None

        if mlp_tuple is not None and mlp_tuple[1] is not None:
            mlp_keep = mlp_tuple[1]

        # Attention
        x1 = modulate(block.norm1(x), shift_msa, scale_msa)
        if qkv_rows is not None and proj_cols is not None:
            attn_out = self._attn_forward_thin_fast(block.attn, x1, qkv_rows, proj_cols, attn_tuple, headdim_override)
        else:
            attn_out = block.attn(x1)
        x = x + gate_msa.unsqueeze(1) * attn_out

        # MLP
        x2 = modulate(block.norm2(x), shift_mlp, scale_mlp)
        if mlp_keep is not None:
            mlp_out = self._mlp_forward_thin_fast(block.mlp, x2, mlp_keep, mlp_tuple)
        else:
            mlp_out = block.mlp(x2)
        x = x + gate_mlp.unsqueeze(1) * mlp_out
        return x

    # ----- fast thin helpers using compiled indices -----
    def _attn_forward_thin_fast(
        self, attn: Attention, x: torch.Tensor,
        qkv_rows: torch.Tensor, proj_cols: torch.Tensor,
        attn_tuple: Optional[Tuple[Any, Optional[torch.Tensor], Optional[int]]],
        headdim_override: Optional[int] = None
    ) -> torch.Tensor:
        """
        x: [B, N, C]
        qkv_rows: LongTensor selecting rows from attn.qkv.weight
        proj_cols: LongTensor selecting columns from attn.proj.weight
        attn_tuple: None or (proj_w, kept_idx_tensor_or_None, head_dim_or_None)
        """
        B, N, C = x.shape
        # EXACTLY like slow version: if bank provided head_dim, use it to compute d
        d = headdim_override if headdim_override is not None else (C // int(getattr(attn, "num_heads", 1)))
        Hk = proj_cols.numel() // max(d, 1)

        # Thin qkv
        qkv_w = attn.qkv.weight.index_select(0, qkv_rows)
        qkv_b = attn.qkv.bias.index_select(0, qkv_rows) if attn.qkv.bias is not None else None
        qkv = F.linear(x, qkv_w, qkv_b).reshape(B, N, 3, Hk, d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_scale = (d ** -0.5)
        attn_scores = (q * attn_scale) @ k.transpose(-2, -1)
        attn_weights = attn_scores.softmax(dim=-1)
        ctx = attn_weights @ v
        ctx = ctx.transpose(1, 2).reshape(B, N, Hk * d)

        # proj: OBS override if provided (bank already thin; use directly)
        if attn_tuple is not None and attn_tuple[0] is not None:
            proj_w = attn_tuple[0]
            proj_b = attn.proj.bias if attn.proj.bias is not None else None
            return F.linear(ctx, proj_w, proj_b)
        else:
            proj_w_n = attn.proj.weight.index_select(1, proj_cols)
            proj_b = attn.proj.bias if attn.proj.bias is not None else None
            return F.linear(ctx, proj_w_n, proj_b)

    def _mlp_forward_thin_fast(
        self, mlp: Mlp, x: torch.Tensor,
        mlp_keep: torch.Tensor,
        mlp_tuple: Optional[Tuple[Any, Optional[torch.Tensor]]]
    ) -> torch.Tensor:
        """
        mlp_keep: LongTensor for hidden indices kept at fc1/fc2
        mlp_tuple: None or (fc2_w, kept_idx_tensor_or_None)
        """
        # thin fc1
        fc1_w = mlp.fc1.weight.index_select(0, mlp_keep)
        fc1_b = mlp.fc1.bias.index_select(0, mlp_keep) if mlp.fc1.bias is not None else None
        h = F.linear(x, fc1_w, fc1_b)
        h = mlp.act(h)
        if hasattr(mlp, "drop1") and mlp.drop1 is not None:
            h = mlp.drop1(h)

        # thin fc2: bank already thin; use directly if present
        if mlp_tuple is not None and mlp_tuple[0] is not None:
            fc2_w = mlp_tuple[0]
            fc2_b = mlp.fc2.bias if mlp.fc2.bias is not None else None
        else:
            fc2_w = mlp.fc2.weight.index_select(1, mlp_keep)
            fc2_b = mlp.fc2.bias if mlp.fc2.bias is not None else None

        y = F.linear(h, fc2_w, fc2_b)
        if hasattr(mlp, "drop2") and mlp.drop2 is not None:
            y = mlp.drop2(y)
        return y

# -----------------------------------------------------------------------------
# Positional embeddings (from MAE)
# -----------------------------------------------------------------------------

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # w first
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

# -----------------------------------------------------------------------------
# Stitch Pruning
# -----------------------------------------------------------------------------

class ModelStitch(nn.Module):
    def __init__(self, denoisers, diffusion_steps=1000):
        super(ModelStitch, self).__init__()
        self.denoisers = nn.ModuleList(denoisers)
        self.dividers = None
        self.in_channels = denoisers[0].in_channels
        self.diffusion_steps = diffusion_steps
        self.cnt = torch.zeros(3, dtype=torch.int32)

    def forward_with_cfg(self, x, t, y, cfg_scale):
        time_step = t[0].item()
        progress = 1 - time_step / self.diffusion_steps
        allocation = torch.tensor(self.dividers)
        
        idx = torch.searchsorted(allocation, progress, right=True).item()
        if idx > len(allocation):
            idx = len(allocation)
        return self.denoisers[idx].forward_with_cfg(x, t, y, cfg_scale)
    
    def forward(self, x, t, y):
        time_step = t[0].item()
        progress = 1 - time_step / self.diffusion_steps
        allocation = torch.tensor(self.dividers)
        
        idx = torch.searchsorted(allocation, progress, right=True).item()
        if idx > len(allocation):
            idx = len(allocation)
        return self.denoisers[idx].forward(x, t, y)

# -----------------------------------------------------------------------------
# DiT configs
# -----------------------------------------------------------------------------

def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiT_XL_8(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiT_L_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiT_L_4(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def DiT_L_8(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


DiT_models = {
    'DiT-XL/2': DiT_XL_2,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/8': DiT_XL_8,
    'DiT-L/2':  DiT_L_2,   'DiT-L/4':  DiT_L_4,   'DiT-L/8':  DiT_L_8,
    'DiT-B/2':  DiT_B_2,   'DiT-B/4':  DiT_B_4,   'DiT-B/8':  DiT_B_8,
    'DiT-S/2':  DiT_S_2,   'DiT-S/4':  DiT_S_4,   'DiT-S/8':  DiT_S_8,
}