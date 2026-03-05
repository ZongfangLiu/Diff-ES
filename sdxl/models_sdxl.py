# models_sdxl.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple, Union, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from contextvars import ContextVar
from dataclasses import dataclass  # <-- added

# ---- Diffusers imports (kept minimal for version robustness) ----
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.models.transformers.transformer_2d import Transformer2DModel
import math
import re
# BasicTransformerBlock location differs by diffusers versions
try:
    from diffusers.models.attention import BasicTransformerBlock as _BasicBlock  # diffusers >= 0.25
except Exception:
    _BasicBlock = None  # type: ignore


# Context var to expose the active UNet to child blocks during forward()
_CTX_UNET: ContextVar["UNet2DConditionPruned | None"] = ContextVar("_CTX_UNET", default=None)


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _to_device_dtype(t: Optional[torch.Tensor], device, dtype):
    if t is None:
        return None
    return t.to(device=device, dtype=dtype, non_blocking=True)


def _as_long(t: Optional[Union[List[int], torch.Tensor]], device) -> Optional[torch.Tensor]:
    if t is None:
        return None
    if isinstance(t, torch.Tensor):
        return t.to(device=device, dtype=torch.long, non_blocking=True)
    return torch.as_tensor(t, device=device, dtype=torch.long)


def _maybe_int(x, default=None):
    try:
        return int(x)
    except Exception:
        return default


# -----------------------------------------------------------------------------
# Small index-only cache packs (no weight copies)
# -----------------------------------------------------------------------------

@dataclass
class _IdxAttnPack:
    proj_cols: torch.Tensor     # [Hk*d] columns (Q output channels kept)
    kept_rows: torch.Tensor     # [Hk*d] rows per-head for Q/K/V and to_out columns
    head_dim: int               # per-head dim d


@dataclass
class _IdxMLPPack:
    kept_idx: torch.Tensor      # [|S|] inner hidden units kept


# -----------------------------------------------------------------------------
# Shared pruning/thinning mixin
# -----------------------------------------------------------------------------
class _PruneMixin:
    """
    Mixin that implements:
      - thin attention & MLP forwards (stage-aware via parent UNet lookups)
      - LayerDrop-aware block forward (expects attrs: attn1, attn2, ff, norm1/2/3)
      - relies on parent UNet via context var (no back-reference stored on the block)
    """

    _blk_id: int  # set by the swapper

    # ------------ context helper ------------

    def _get_unet(self) -> "UNet2DConditionPruned":
        U = _CTX_UNET.get()
        if U is None:
            raise RuntimeError("No active UNet in context (did UNet.forward() set the context var?)")
        return U

    # ---------------------- Helpers: Shapes & GEGLU ----------------------

    @staticmethod
    def _to_seq(x: torch.Tensor) -> Tuple[torch.Tensor, Optional[Tuple[int, int, int, int]]]:
        # Accepts [B, C, H, W] or already [B, N, C]
        if x.dim() == 4:
            B, C, H, W = x.shape
            x_seq = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
            return x_seq, (B, C, H, W)
        return x, None

    @staticmethod
    def _from_seq(x_seq: torch.Tensor, hwc: Optional[Tuple[int, int, int, int]]) -> torch.Tensor:
        if hwc is None:
            return x_seq
        B, C, H, W = hwc
        return x_seq.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

    @staticmethod
    def _geglu(h: torch.Tensor) -> torch.Tensor:
        # GEGLU: split, h1 * GELU(h2)
        inner = h.shape[-1] // 2
        h1, h2 = h.split([inner, inner], dim=-1)
        return h1 * F.gelu(h2)

    # Ada/LayerNorm tolerant application
    @staticmethod
    def _apply_norm(norm_module: nn.Module, x: torch.Tensor, temb: Optional[torch.Tensor]) -> torch.Tensor:
        try:
            return norm_module(x, temb)  # AdaLayerNorm-like
        except TypeError:
            return norm_module(x)        # Plain LayerNorm/GroupNorm

    # -------- cross-attn kwargs sanitization & scale extraction --------

    @staticmethod
    def _sanitize_ca_kwargs(ca_kwargs: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Keep only keys that diffusers attention processors accept.
        Notably: 'scale' (or 'softmax_scale'). Drop 'encoder_attention_mask'.
        """
        if not ca_kwargs:
            return None
        ca_kwargs = dict(ca_kwargs)
        # Drop unsupported keys that trigger logs in AttnProcessor2_0
        ca_kwargs.pop("encoder_attention_mask", None)

        # Normalize 'softmax_scale' alias -> 'scale'
        if "scale" not in ca_kwargs and "softmax_scale" in ca_kwargs:
            ca_kwargs["scale"] = ca_kwargs.pop("softmax_scale")

        return ca_kwargs or None

    @staticmethod
    def _extract_softmax_scale(ca_kwargs: Optional[Dict[str, Any]]) -> Optional[float]:
        if not ca_kwargs:
            return None
        s = ca_kwargs.get("scale", None)
        try:
            return float(s) if s is not None else None
        except Exception:
            return None

    # ---------------------- Attention core wrapper & numeric guards ----------------------

    @staticmethod
    def _sanitize_qkv_(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, max_abs: float = 1e3):
        # In-place numeric sanitization to avoid NaN/Inf explosions in kernels
        q.nan_to_num_(nan=0.0, posinf=1e4, neginf=-1e4)
        k.nan_to_num_(nan=0.0, posinf=1e4, neginf=-1e4)
        v.nan_to_num_(nan=0.0, posinf=1e4, neginf=-1e4)

        def _clip_(x: torch.Tensor):
            m = x.abs().amax()
            if torch.isfinite(m) and (m > max_abs):
                x.mul_(max_abs / m)

        _clip_(q); _clip_(k); _clip_(v)

    @staticmethod
    def _attn_core_safe(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, is_causal: bool = False) -> torch.Tensor:
        """
        Try Flash/Mem-Efficient SDPA → xFormers → math SDPA.
        q,k,v: [B, H, N, d] contiguous, finite.
        Returns: [B, H, N, d]
        """
        import torch
        import torch.nn.functional as F

        # Fast path: Flash / Mem-efficient SDPA
        try:
            from torch.nn.attention import sdpa_kernel
            with sdpa_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=False):
                out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=is_causal)
            if torch.isfinite(out).all():
                return out
        except Exception:
            pass

        # xFormers fallback (if available)
        try:
            import xformers.ops as xops
            B, H, Nq, d = q.shape
            Nk = k.shape[2]
            qh = q.reshape(B * H, Nq, d).contiguous()
            kh = k.reshape(B * H, Nk, d).contiguous()
            vh = v.reshape(B * H, Nk, d).contiguous()
            out = xops.memory_efficient_attention(qh, kh, vh, p=0.0)
            out = out.reshape(B, H, Nq, d)
            if torch.isfinite(out).all():
                return out
        except Exception:
            pass

        # Last resort: math SDPA
        try:
            from torch.nn.attention import sdpa_kernel
            with sdpa_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
                out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=is_causal)
            return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception:
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=is_causal)
            return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    # ---------------------- Thin Attention (self/cross) ----------------------

    def _attn_thin_forward(
        self,
        attn_module: nn.Module,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        kind: Literal["attn1", "attn2"],
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        U = self._get_unet()
        t = U._current_t
        b = self._blk_id

        # --- Try indices-only cached pack first
        sid = U._sid_for_t(t)
        cache_pack: Optional[_IdxAttnPack] = None
        if getattr(U, "_idx_attn_cache", None) is not None and 0 <= sid < len(U._idx_attn_cache):
            row = U._idx_attn_cache[sid]
            if row is not None and 0 <= b < len(row) and row[b] is not None:
                cache_pack = row[b][0] if kind == "attn1" else row[b][1]

        # stage/bank lookup (still needed for override and fallback)
        _qkv_rows, proj_cols, head_dim, proj_w_override = U._lookup_attn_thin(t, b, kind)

        # ---- Fast path: no thinning requested → call stock attention (SDPA path unchanged)
        if proj_cols is None and cache_pack is None:
            q_in, shape_info = self._to_seq(hidden_states)
            k_in = None
            if encoder_hidden_states is not None:
                k_in, _ = self._to_seq(encoder_hidden_states)
            ca_kwargs = self._sanitize_ca_kwargs(cross_attention_kwargs)
            if kind == "attn1":
                out = attn_module(q_in, encoder_hidden_states=None, **(ca_kwargs or {}))
            else:
                out = attn_module(q_in, encoder_hidden_states=k_in, **(ca_kwargs or {}))
            return self._from_seq(out, shape_info)

        # ---- Thinned path (pre-slice BEFORE projections)
        q_in, shape_info = self._to_seq(hidden_states)
        kv_in = self._to_seq(encoder_hidden_states)[0] if encoder_hidden_states is not None else q_in

        to_q, to_k, to_v = attn_module.to_q, attn_module.to_k, attn_module.to_v
        dev = to_q.weight.device

        # Use cached indices when available; otherwise compute once on-the-fly
        if cache_pack is not None:
            kept_rows = cache_pack.kept_rows
            d = int(cache_pack.head_dim)
            # print("Using cached attn thinning indices for stage", sid, "block", b, "kind", kind)
            # print()
        else:
            # Derive dims robustly from block props
            # print("Computing attn thinning indices for stage", sid, "block", b, "kind", kind)
            props = U._blk_props[b]
            d = int(head_dim) if head_dim is not None else int(props["d"])

            # Sanitize requested columns against Q output dim (equal to Cq)
            Cq_full = int(to_q.weight.shape[0])  # out_features of to_q
            proj_cols = torch.as_tensor(proj_cols, device=dev, dtype=torch.long)
            proj_cols = proj_cols[(proj_cols >= 0) & (proj_cols < Cq_full)]
            if proj_cols.numel() == 0:
                return self._from_seq(torch.zeros_like(q_in), shape_info)

            # Map kept columns → kept heads (unique, sorted)
            H_full = max(1, Cq_full // max(1, d))
            kept_heads = torch.div(proj_cols, max(1, d), rounding_mode="floor").unique(sorted=True)
            kept_heads = kept_heads[(kept_heads >= 0) & (kept_heads < H_full)]
            if kept_heads.numel() == 0:
                return self._from_seq(torch.zeros_like(q_in), shape_info)

            # Build rows for the kept channels of Q/K/V (same layout per head)
            full_d = torch.arange(d, device=dev, dtype=torch.long)
            kept_rows = (kept_heads[:, None] * d + full_d[None, :]).reshape(-1)  # length = Hk*d

        # Early compute only needed rows (avoid full qkv FCs)
        q = F.linear(
            q_in,
            to_q.weight.index_select(0, kept_rows),
            to_q.bias.index_select(0, kept_rows) if to_q.bias is not None else None,
        )  # [B,Nq,Hk*d]
        kv = kv_in
        k = F.linear(
            kv,
            to_k.weight.index_select(0, kept_rows),
            to_k.bias.index_select(0, kept_rows) if to_k.bias is not None else None,
        )  # [B,Nk,Hk*d]
        v = F.linear(
            kv,
            to_v.weight.index_select(0, kept_rows),
            to_v.bias.index_select(0, kept_rows) if to_v.bias is not None else None,
        )  # [B,Nk,Hk*d]

        B, Nq, _ = q.shape
        Nk = k.shape[1]
        Hk = int(kept_rows.numel() // max(1, d))

        # Pack to [B,Hk,N, d]
        q = q.view(B, Nq, Hk, d).permute(0, 2, 1, 3).contiguous()
        k = k.view(B, Nk, Hk, d).permute(0, 2, 1, 3).contiguous()
        v = v.view(B, Nk, Hk, d).permute(0, 2, 1, 3).contiguous()
        if Hk == 0 or d == 0:
            return self._from_seq(torch.zeros_like(q_in), shape_info)

        # numeric guards
        self._sanitize_qkv_(q, k, v)

        # run attention (Flash/MemEff → xFormers → math)
        ca_kwargs = self._sanitize_ca_kwargs(cross_attention_kwargs)
        softmax_scale = self._extract_softmax_scale(ca_kwargs)
        if softmax_scale is not None:
            s = float(softmax_scale)
            if not torch.isfinite(torch.tensor(s)):
                s = 1.0
            s = max(min(s, 10.0), 1e-3)
            q = q * s  # SDPA already applies 1/sqrt(d)

        ctx = self._attn_core_safe(q, k, v, is_causal=False)  # [B,Hk,Nq,d]
        ctx = ctx.transpose(1, 2).reshape(B, Nq, Hk * d)      # [B,Nq,Hk*d]

        # output projection: either bank override or to_out thin columns
        to_out = attn_module.to_out
        if isinstance(to_out, nn.Sequential):
            to_out_lin = to_out[0]
        elif isinstance(to_out, nn.ModuleList):
            to_out_lin = next((m for m in to_out if hasattr(m, "weight")), None)
            if to_out_lin is None:
                raise AttributeError("to_out ModuleList has no weighted submodule")
        else:
            to_out_lin = to_out

        cols = kept_rows  # [Hk*d]
        # Preserve bank override behavior
        if (proj_w_override is not None) and (proj_w_override.shape[1] == cols.numel()):
            out_seq = F.linear(ctx, proj_w_override, to_out_lin.bias)
        else:
            proj_w = to_out_lin.weight.index_select(1, cols)  # [C_out, Hk*d]
            out_seq = F.linear(ctx, proj_w, to_out_lin.bias)

        return self._from_seq(out_seq, shape_info)

    # ---------------------- Thin MLP (GEGLU-aware with fallback) ----------------------

    def _mlp_thin_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # print("MLP thinning forward called")
        U = self._get_unet()
        t = U._current_t
        b = self._blk_id

        # Prefer cached indices; still fetch override from lookup
        sid = U._sid_for_t(t)
        # print("MLP thinning for stage", sid, "block", b)
        mlp_keep_cached: Optional[torch.Tensor] = None
        # print(U._idx_mlp_cache)
        if getattr(U, "_idx_mlp_cache", None) is not None and 0 <= sid < len(U._idx_mlp_cache):
            row = U._idx_mlp_cache[sid]
            # print("MLP thinning cache row:", row)
            if row is not None and 0 <= b < len(row) and row[b] is not None:
                mlp_keep_cached = row[b].kept_idx

        # Always get current override (proj bank may provide fc2_w)
        mlp_keep_lookup, fc2_w_override = U._lookup_mlp_thin(t, b)
        mlp_keep = mlp_keep_cached if mlp_keep_cached is not None else mlp_keep_lookup

        # If nothing to thin (no indices and no override), use stock FF
        if mlp_keep is None and fc2_w_override is None:
            return self.ff(hidden_states)

        # SDXL GEGLU: ff.net[0] = GEGLU (with .proj), ff.net[2] = fc2 Linear
        try:
            geglu = self.ff.net[0]
            fc1 = geglu.proj          # Linear: dim -> 2*inner
            fc2 = self.ff.net[2]      # Linear: inner -> dim
        except Exception:
            # Fallback if structure differs
            return self.ff(hidden_states)

        device = fc1.weight.device
        inner_full = int(fc2.weight.shape[1])  # full inner
        dim_out = int(fc2.weight.shape[0])

        # Normalize/clean indices
        if mlp_keep is not None:
            mlp_keep = mlp_keep.to(device=device, dtype=torch.long)
            mlp_keep = mlp_keep[(mlp_keep >= 0) & (mlp_keep < inner_full)].unique(sorted=True)
            if mlp_keep.numel() == 0:
                # Early-out: zero residual contribution
                B = hidden_states.shape[0]
                N = hidden_states.shape[2] if hidden_states.dim() == 4 else hidden_states.shape[1]
                return torch.zeros((B, N, dim_out), device=hidden_states.device, dtype=hidden_states.dtype)
        else:
            # No kept_idx → cannot safely align an override; fall back to full path
            if fc2_w_override is not None:
                fc2_w_override = None

        # Fast path when no thinning after all
        if mlp_keep is None and fc2_w_override is None:
            return self.ff(hidden_states)

        # Compute only necessary rows of fc1 for GEGLU:
        # For a subset S of hidden units, GEGLU needs rows S and S+inner_full
        # (fc1 outputs 2*inner_full; first half is "linear", second half gated then GELU)
        if mlp_keep is None:
            # Shouldn't happen due to guards above; just run full FF
            return self.ff(hidden_states)

        rows_fc1 = torch.cat([mlp_keep, mlp_keep + inner_full], dim=0)
        fc1_w = fc1.weight.index_select(0, rows_fc1)
        fc1_b = fc1.bias.index_select(0, rows_fc1) if fc1.bias is not None else None

        # h_sub = [B,N, 2*|S|]
        h_sub = F.linear(hidden_states, fc1_w, fc1_b)
        # Recompose GEGLU on the subset → [B,N, |S|]
        s = h_sub.shape[-1] // 2
        h_lin = h_sub[..., :s]
        h_gate = h_sub[..., s:]
        h = h_lin * F.gelu(h_gate)

        # fc2 on the thinned activations
        if fc2_w_override is not None:
            # Use override only if width matches |S|
            if int(fc2_w_override.shape[1]) == int(h.shape[-1]):
                return F.linear(h, fc2_w_override, fc2.bias)
            # Fallback if mismatch
            fc2_w_override = None

        fc2_w = fc2.weight.index_select(1, mlp_keep)  # [dim_out, |S|]
        return F.linear(h, fc2_w, fc2.bias)

    # ---------------------- Full block forward (LayerDrop+Thin) ----------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,  # passthrough
        **kwargs,  # tolerate future args
    ) -> torch.Tensor:
        U = self._get_unet()
        t = U._current_t
        b = self._blk_id

        if U._should_drop(b, t):
            return hidden_states

        residual = hidden_states

        # attn1 (self)
        hidden_states = self._apply_norm(self.norm1, hidden_states, temb)
        attn1_out = self._attn_thin_forward(
            self.attn1,
            hidden_states,
            None,
            None,  # IMPORTANT: do not pass attention_mask
            kind="attn1",
            cross_attention_kwargs=cross_attention_kwargs,
        )
        hidden_states = residual + attn1_out

        # attn2 (cross) if present
        if getattr(self, "attn2", None) is not None:
            residual = hidden_states
            hidden_states = self._apply_norm(self.norm2, hidden_states, temb)
            attn2_out = self._attn_thin_forward(
                self.attn2,
                hidden_states,
                encoder_hidden_states,
                None,  # IMPORTANT: do not pass attention_mask
                kind="attn2",
                cross_attention_kwargs=cross_attention_kwargs,
            )
            hidden_states = residual + attn2_out

        # ff
        residual = hidden_states
        hidden_states = self._apply_norm(self.norm3, hidden_states, temb)
        ff_out = self._mlp_thin_forward(hidden_states)
        hidden_states = residual + ff_out

        return hidden_states

# -----------------------------------------------------------------------------
# Pruned blocks (two front doors reuse the same mixin)
# -----------------------------------------------------------------------------

class PrunedTransformer2DModel(_PruneMixin, Transformer2DModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self._init_prune_caches()


if _BasicBlock is not None:
    class PrunedBasicTransformerBlock(_PruneMixin, _BasicBlock):  # type: ignore[misc]
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # self._init_prune_caches()
else:
    PrunedBasicTransformerBlock = None  # type: ignore


# -----------------------------------------------------------------------------
# Main UNet subclass
# -----------------------------------------------------------------------------

class UNet2DConditionPruned(UNet2DConditionModel):
    """
    UNet subclass that compiles/serves thin schedules and OBS banks to internal
    Transformer blocks (attn1, attn2, MLP), with LayerDrop support.

    Public controls:
      - set_layerdrop(drop_ids: Iterable[int])
      - set_layerdrop_schedule(schedule: Dict[int, Iterable[int]] | Callable[[int], Iterable[int]])
      - set_secondorder_stage_schedule(schedule: dict, stages: Optional[List[Tuple[int,int]]])
      - set_projection_bank(bank: dict, stages: Optional[List[Tuple[int,int]]])
      - apply_weight_masks(mask_map: Dict[str, torch.Tensor])   # optional once-only masks (WANDA/magnitude)
      - clear_all_accel()
    """

    # ---------------------- Construction & install ----------------------

    def __init__(self, *args, diffusion_steps: int = 1000, swap_on_init: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        # Runtime control
        self.diffusion_steps: int = int(diffusion_steps)
        self.drop_block_ids: set[int] = set()
        self.layerdrop_schedule = None  # dict[int→Iterable[int]] or callable(int)->Iterable[int]
        self._drop_by_t: Optional[List[Optional[torch.Tensor]]] = None  # precompiled per-t masks

        # Stage-based second-order (compiled per stage; looked up via t->stage map)
        # _so_stage_schedule[sid] = {"attn1": {blk: drop_heads}, "attn2": {...}, "mlp": {blk: drop_neurons}}
        self._so_stage_schedule: Optional[Dict[int, Dict[str, Dict[int, Iterable[int]]]]] = None
        # Compiled per-stage: list indexed by sid; each item is [per-block dict {"attn1":(...), "attn2":(...), "mlp":(...)}]
        self._so_comp_per_stage: Optional[List[List[Optional[Dict[str, Optional[Tuple]]]]]] = None
        # Stages and t→sid mapping
        self._so_stage_spans: Optional[List[Tuple[int, int]]] = None
        self._so_t2s: Optional[List[int]] = None

        # OBS projection bank (stage-based)
        #   bank[sid]["attn1"][blk] = {"proj_w", "kept_idx", "head_dim"}
        #   bank[sid]["mlp"][blk]   = {"fc2_w", "kept_idx"}
        self._proj_bank: Optional[Dict[int, Dict[str, Dict[int, Dict[str, Any]]]]] = None
        self._proj_bank_stages: Optional[List[Tuple[int, int]]] = None
        self._proj_bank_t2s: Optional[List[int]] = None
        self._pb_by_t: Optional[List[Optional[Tuple[
            List[Optional[Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[int]]]],
            List[Optional[Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[int]]]],
            List[Optional[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]],
        ]]]] = None
        self._pb_depth: Optional[int] = None

        # Current timestep set at forward
        self._current_t: int = 0

        # Flag to ensure weight masks apply once
        self._weight_masks_applied: bool = False

        # Idempotent guard for swaps
        self._is_swapped: bool = False

        # Count blocks (populated on swap)
        self._num_blocks: int = 0

        # Stable flat index → dotted path mapping (recorded at swap)
        self._blk_paths: List[str] = []

        # Per-basic-block properties (heads, head_dim, inner_dim)
        self._blk_props: List[Dict[str, int]] = []

        # Optional: hint to enable thin path in custom forwards (no-op safe)
        self._struct_prune_enabled: bool = False

        # Indices-only caches (per stage, per block)
        self._idx_attn_cache: Optional[List[Optional[List[Optional[Tuple[Optional[_IdxAttnPack], Optional[_IdxAttnPack]]]]]]] = None
        self._idx_mlp_cache: Optional[List[Optional[List[Optional[_IdxMLPPack]]]]] = None

        # Optionally install pruned blocks on init
        if swap_on_init:
            self._install_pruned_transformers()

    # ---------------------- Override forward to set current t and context ----------------------

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # Derive integer t; UNet calls usually broadcast a scalar t
        if isinstance(timestep, torch.Tensor):
            if timestep.numel() == 1:
                t_int = int(timestep.item())
            else:
                t_int = int(timestep.flatten()[0].item())
        else:
            t_int = int(timestep)
        self._current_t = max(0, t_int)

        # sanitize cross-attn kwargs once
        ca_kwargs = _PruneMixin._sanitize_ca_kwargs(cross_attention_kwargs)

        # Set context var for child blocks during this forward
        token = _CTX_UNET.set(self)
        try:
            # IMPORTANT: pass attention_mask=None to avoid NaNs in processors
            return super().forward(
                sample=sample,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                class_labels=class_labels,
                attention_mask=None,  # mask-free inside UNet
                cross_attention_kwargs=ca_kwargs,
                added_cond_kwargs=added_cond_kwargs,
                down_block_additional_residuals=down_block_additional_residuals,
                mid_block_additional_residual=mid_block_additional_residual,
                **kwargs,
            )
        finally:
            _CTX_UNET.reset(token)

    # ---------------------- Optional hint (safe no-op) ----------------------

    def enable_struct_prune_forward(self, enabled: bool = True):
        self._struct_prune_enabled = bool(enabled)

    # ---------------------- Install pruned transformer/basic blocks ----------------------

    def _install_pruned_transformers(self):
        if self._is_swapped:
            return  # idempotent
        blk_id = 0
        self._blk_paths = []   # reset if re-installing
        self._blk_props = []   # reset if re-installing

        def _clone_cfg_from(mod: nn.Module) -> dict:
            """
            Robust constructor cloning (diffusers version tolerant).
            Prefer the block's .config when available to avoid FutureWarnings.
            """
            fields_common = [
                "num_attention_heads",
                "attention_head_dim",
                "in_channels",
                "cross_attention_dim",
                "norm_num_groups",
                "attention_bias",
                "activation_fn",
                "num_layers",
                "use_linear_projection",
                "norm_type",
                "dropout",
                "only_cross_attention",
                "double_self_attention",
                "num_embeds_ada_norm",
                "norm_elementwise_affine",
                "attention_type",
                "eps",
            ]
            out: Dict[str, Any] = {}

            cfg_obj = getattr(mod, "config", None)
            if cfg_obj is not None:
                try:
                    to_dict = cfg_obj.to_dict() if hasattr(cfg_obj, "to_dict") else dict(cfg_obj)
                except Exception:
                    to_dict = {}
                for k in fields_common:
                    v = to_dict.get(k, None)
                    if v is not None:
                        out[k] = v
                # BasicTransformerBlock often needs "dim"
                if "dim" in to_dict and to_dict["dim"] is not None:
                    out["dim"] = int(to_dict["dim"])
            else:
                for k in fields_common:
                    v = getattr(mod, k, None)
                    if v is not None:
                        out[k] = v
                # Try to recover "dim" for BasicTransformerBlock if present
                dim = getattr(mod, "dim", None)
                if dim is None:
                    # infer from attn1.to_q input features (channels)
                    try:
                        dim = int(getattr(mod, "attn1").to_q.in_features)  # type: ignore[attr-defined]
                    except Exception:
                        pass
                if dim is not None:
                    out["dim"] = int(dim)

            # Drop Nones
            return {k: v for k, v in out.items() if v is not None}

        def _read_block_props(inner_block: nn.Module) -> Dict[str, int]:
            # Per-block heads/hdim from SDXL blocks (attn1 has attributes heads/head_dim)
            try:
                H_b = int(getattr(inner_block.attn1, "heads"))
            except Exception:
                H_b = int(getattr(inner_block, "num_attention_heads", 1))

            try:
                d_b = int(getattr(inner_block.attn1, "head_dim"))
            except Exception:
                d_b = int(getattr(inner_block, "attention_head_dim", 64))

            try:
                inner_b = int(inner_block.ff.net[2].in_features)
            except Exception:
                inner_b = int(getattr(inner_block.ff.net[2], "in_features"))

            return {"H": H_b, "d": d_b, "inner": inner_b}

        def _register_pruned(inner_block, path: str):
            nonlocal blk_id
            cfg = _clone_cfg_from(inner_block)
            pruned = PrunedBasicTransformerBlock(**cfg)  # type: ignore
            pruned.load_state_dict(inner_block.state_dict(), strict=True)
            pruned._blk_id = blk_id
            self._blk_paths.append(path)
            self._blk_props.append(_read_block_props(inner_block))
            blk_id += 1
            return pruned

        def swap(container, base_path: str):
            if not hasattr(container, "attentions"):
                return
            for i, mod in enumerate(container.attentions):
                cont_path = f"{base_path}.attentions.{i}"
                # Case A: Transformer2DModel wrapping inner BasicTransformerBlocks
                if isinstance(mod, Transformer2DModel) and hasattr(mod, "transformer_blocks"):
                    for j, inner in enumerate(mod.transformer_blocks):
                        if (_BasicBlock is not None) and isinstance(inner, _BasicBlock):
                            inner_path = f"{cont_path}.transformer_blocks.{j}"
                            pruned = _register_pruned(inner, inner_path)
                            mod.transformer_blocks[j] = pruned
                    continue
                # Case B: BasicTransformerBlock directly
                if (_BasicBlock is not None) and isinstance(mod, _BasicBlock):
                    inner_path = cont_path
                    pruned = _register_pruned(mod, inner_path)
                    container.attentions[i] = pruned

        for di, down in enumerate(self.down_blocks):
            swap(down, f"down_blocks.{di}")
        if hasattr(self.mid_block, "attentions") and getattr(self.mid_block, "attentions", None):
            swap(self.mid_block, "mid_block")
        for ui, up in enumerate(self.up_blocks):
            swap(up, f"up_blocks.{ui}")

        self._num_blocks = blk_id
        self._is_swapped = True

    # ---------------------- Public Controls ----------------------

    def get_block_paths(self) -> List[str]:
        return list(self._blk_paths)

    def get_path_by_block_id(self, idx: int) -> str:
        if idx < 0 or idx >= len(self._blk_paths):
            raise IndexError(f"block id {idx} out of range 0..{len(self._blk_paths)-1}")
        return self._blk_paths[idx]

    def get_block_id_by_path(self, path: str) -> int:
        try:
            return self._blk_paths.index(path)
        except ValueError:
            raise KeyError(f"Block path not found: {path}")

    def debug_print_block_index(self) -> None:
        for i, p in enumerate(self._blk_paths):
            props = self._blk_props[i] if i < len(self._blk_props) else {"H": -1, "d": -1, "inner": -1}
            print(f"{i:3d}: {p} | heads={props['H']} head_dim={props['d']} inner={props['inner']}")

    def set_layerdrop(self, ids: Iterable[int]):
        self.drop_block_ids = set(int(i) for i in ids)

    def _compile_layerdrop_schedule(self):
        """Build dense per-t boolean masks for fast LayerDrop checks."""
        if self.layerdrop_schedule is None or self._num_blocks == 0:
            self._drop_by_t = None
            return
        T = int(max(self.diffusion_steps, max(self.layerdrop_schedule.keys(), default=-1) + 1)) \
            if isinstance(self.layerdrop_schedule, dict) else int(self.diffusion_steps)
        drop_by_t: List[Optional[torch.Tensor]] = [None] * max(T, 1)
        for t in range(len(drop_by_t)):
            if callable(self.layerdrop_schedule):
                ids = self.layerdrop_schedule(t) or ()
            else:
                ids = self.layerdrop_schedule.get(t, ()) or ()
            if ids:
                mask = torch.zeros(self._num_blocks, dtype=torch.bool)
                mask[torch.as_tensor(list(map(int, ids)), dtype=torch.long)] = True
                drop_by_t[t] = mask
        self._drop_by_t = drop_by_t

    def set_layerdrop_schedule(self, schedule, stages: Optional[List[Tuple[int, int]]] = None):
        """
        Accepts EITHER:
        - per-t schedule: {t: Iterable[blk_ids]}  (legacy; unchanged)
        - stage schedule: {sid: Iterable[blk_ids]} with `stages` provided
        """
        # Callables: keep as-is
        if callable(schedule):
            self.layerdrop_schedule = schedule
            self._compile_layerdrop_schedule()
            return

        if not isinstance(schedule, dict):
            # Fallback: treat as empty / disable
            self.layerdrop_schedule = None
            self._compile_layerdrop_schedule()
            return

        # Heuristic: if keys look like stage IDs and we have stage spans, expand to per-t
        if stages and len(stages) > 0:
            try:
                keys = [int(k) for k in schedule.keys()]
                # Stage-like if number of keys <= #stages and max key < (#stages + a little)
                if len(keys) <= len(stages) and (max(keys) if keys else -1) < len(stages) + 3:
                    per_t: Dict[int, Iterable[int]] = {}
                    for sid, (t_lo, t_hi) in enumerate(stages):
                        ids = schedule.get(sid, ()) or ()
                        for t in range(int(t_lo), int(t_hi) + 1):
                            per_t[t] = ids
                    self.layerdrop_schedule = per_t
                    self._compile_layerdrop_schedule()
                    return
            except Exception:
                pass  # fall through to treat it as a per-t dict

        # Assume it's already {t: ids}
        self.layerdrop_schedule = schedule
        self._compile_layerdrop_schedule()

    # ---------------------- Stage-schedule normalization ----------------------

    @staticmethod
    def _normalize_struct_stage_keys(stage_sch: Optional[dict] = None) -> Dict[int, Dict[str, Dict[int, Iterable[int]]]]:
        """
        Accept stage-based schedule where each stage maps to {attn1/attn2/mlp:{blk_id -> drop_list}}.
        Mirrors legacy "attn" → "attn2" if only "attn" provided.
        """
        if not stage_sch:
            return {}
        norm: Dict[int, Dict[str, Dict[int, Iterable[int]]]] = {}
        for sid_k, ent in stage_sch.items():
            sid = int(sid_k)
            out = {"attn1": {}, "attn2": {}, "mlp": {}}
            # legacy "attn"
            if "attn" in ent and "attn1" not in ent and "attn2" not in ent:
                out["attn2"] = dict(ent["attn"])
            # explicit keys override
            for k in ("attn1", "attn2", "mlp"):
                if k in ent and isinstance(ent[k], dict):
                    out[k] = {int(b): v for b, v in ent[k].items()}
            norm[sid] = out
        return norm

    @staticmethod
    def _normalize_bank_keys(bank: Optional[dict]) -> dict:
        # Accept {"attn": {...}} and mirror to attn2 if attn1/attn2 absent.
        if not bank:
            return {}
        norm = {}
        for sid, stage_map in bank.items():
            s = int(sid)
            out = {"attn1": {}, "attn2": {}, "mlp": {}}
            # legacy "attn"
            if "attn" in stage_map and "attn1" not in stage_map and "attn2" not in stage_map:
                out["attn2"] = dict(stage_map["attn"])  # or copy to both
            # explicit keys override
            for k in ("attn1", "attn2", "mlp"):
                if k in stage_map:
                    out[k] = dict(stage_map[k])
            norm[s] = out
        return norm

    # ---------------------- Stage schedule compilation ----------------------

    def set_secondorder_stage_schedule(self, schedule: dict, stages: Optional[List[Tuple[int, int]]] = None):
        """
        STAGE-BASED ONLY API.
        `schedule`: {sid: {"attn1": {blk:[drop_heads]}, "attn2": {...}, "mlp": {blk:[drop_neurons]}}}
        `stages`:   list of (t_lo, t_hi) defining each sid's timestep span. REQUIRED unless there's exactly one sid.
        """
        self._so_stage_schedule = self._normalize_struct_stage_keys(schedule or {})
        self._compile_stage_schedule(stages)
    
    # put this inside UNet2DConditionPruned
    def set_secondorder_struct_schedule(self, schedule: dict, stages: Optional[List[Tuple[int, int]]] = None):
        # shim for older utils: stage-based only
        return self.set_secondorder_stage_schedule(schedule, stages)

    def _compile_stage_schedule(self, stages: Optional[List[Tuple[int, int]]]):
        sch = self._so_stage_schedule or {}
        if not sch:
            self._so_comp_per_stage = None
            self._so_stage_spans = None
            self._so_t2s = None
            # also drop any index caches
            self._idx_attn_cache = None
            self._idx_mlp_cache = None
            return

        if self._num_blocks == 0 or len(self._blk_props) != self._num_blocks:
            # Not swapped yet; nothing to compile against
            self._so_comp_per_stage = None
            self._so_stage_spans = [(0, max(0, self.diffusion_steps - 1))]
            self._so_t2s = [0] * int(max(1, self.diffusion_steps))
            # clear caches for now
            self._idx_attn_cache = None
            self._idx_mlp_cache = None
            return

        depth = int(self._num_blocks)
        dev = next(self.parameters()).device

        # Compile per-stage
        max_sid = max(int(s) for s in sch.keys())
        comp: List[List[Optional[Dict[str, Optional[Tuple]]]]] = [ [None]*depth for _ in range(max_sid + 1) ]

        for sid, ent in sch.items():
            sid_i = int(sid)
            a1map = ent.get("attn1", {}) or {}
            a2map = ent.get("attn2", {}) or {}
            mmap  = ent.get("mlp",   {}) or {}

            row: List[Optional[Dict[str, Optional[Tuple]]]] = [None] * depth
            for b in range(depth):
                props = self._blk_props[b]  # per-block
                H_b   = int(props["H"])
                d_b   = int(props["d"])
                inner_b = int(props["inner"])

                full_heads  = torch.arange(H_b, device=dev, dtype=torch.long)
                full_d      = torch.arange(d_b, device=dev, dtype=torch.long)
                full_hidden = torch.arange(inner_b, device=dev, dtype=torch.long)

                rec: Dict[str, Optional[Tuple]] = {"attn1": None, "attn2": None, "mlp": None}

                for which, amap in (("attn1", a1map), ("attn2", a2map)):
                    dh = amap.get(b, None)
                    if dh:
                        drops = torch.as_tensor(sorted(set(int(x) for x in dh)), device=dev, dtype=torch.long)
                        if drops.numel() > 0:
                            mask = torch.ones(H_b, dtype=torch.bool, device=dev)
                            mask[drops.clamp_(0, H_b - 1)] = False
                            kept_heads = full_heads[mask]                         # [Hk]
                        else:
                            kept_heads = full_heads
                        head_spans = (kept_heads[:, None] * d_b + full_d[None, :]).reshape(-1)  # [Hk*d_b]
                        qkv_rows  = torch.cat([head_spans, head_spans, head_spans], dim=0)
                        proj_cols = head_spans
                        rec[which] = (qkv_rows, proj_cols, d_b)

                dm = mmap.get(b, None)
                if dm:
                    drops = torch.as_tensor(sorted(set(int(x) for x in dm)), device=dev, dtype=torch.long)
                    if drops.numel() > 0:
                        msk = torch.ones(inner_b, dtype=torch.bool, device=dev)
                        msk[drops.clamp_(0, inner_b - 1)] = False
                        mlp_keep = full_hidden[msk]
                    else:
                        mlp_keep = full_hidden
                    rec["mlp"] = (mlp_keep,)

                row[b] = rec
            comp[sid_i] = row

        self._so_comp_per_stage = comp

        # Timesteps → stage mapping
        if stages and len(stages) > 0:
            spans = [(int(lo), int(hi)) for (lo, hi) in stages]
            max_t = max(hi for _, hi in spans)
            t2s = [-1] * (max_t + 1)
            for sid_i, (lo, hi) in enumerate(spans):
                lo_i, hi_i = max(0, int(lo)), int(hi)
                for t in range(lo_i, hi_i + 1):
                    if 0 <= t < len(t2s):
                        t2s[t] = sid_i
            self._so_stage_spans = spans
            self._so_t2s = t2s
        else:
            # Single stage → apply across all t
            self._so_stage_spans = None
            self._so_t2s = [0] * int(max(1, self.diffusion_steps))

        # (Re)build indices-only caches after compiling schedule
        self._rebuild_index_caches()

    # ---------------------- OBS projection bank install (stage-based) ----------------------

    def set_projection_bank(self, bank: dict, stages: Optional[List[Tuple[int, int]]] = None):
        self._install_projection_bank(bank, stages)

    def _install_projection_bank(self, bank: dict, stages: Optional[List[Tuple[int, int]]] = None):
        """
        Install a (stage-based) OBS projection bank.
        This version keeps bank keys as-is (int block ids OR layer path strings) and
        resolves them to flat block indices when constructing the per-t lookup tables.
        """

        # ---- device / dtype anchors ----
        dev = next(self.parameters()).device
        ref_dtype = next(self.parameters()).dtype

        # ---- optional t->stage mapping from caller ----
        self._proj_bank_stages = [(int(lo), int(hi)) for (lo, hi) in (stages or [])] if stages else None
        self._proj_bank_t2s = None
        if self._proj_bank_stages:
            max_hi = max(hi for _, hi in self._proj_bank_stages)
            T = max_hi + 1
            t2s = [-1] * T
            for sid, (lo, hi) in enumerate(self._proj_bank_stages):
                lo_i, hi_i = max(0, int(lo)), int(hi)
                for t in range(lo_i, hi_i + 1):
                    if 0 <= t < T:
                        t2s[t] = sid
            self._proj_bank_t2s = t2s

        # ---- normalize bank: mirror legacy {"attn": {...}} into attn2 if attn1/attn2 absent ----
        bank = self._normalize_bank_keys(bank)

        # ---- store tensors on device/dtype; keep keys as int OR str (do NOT cast to int here) ----
        self._proj_bank = {}
        for sid, stage_map in (bank or {}).items():
            s_out = {"attn1": {}, "attn2": {}, "mlp": {}}

            # attn1/attn2 entries: {"proj_w", "kept_idx", "head_dim"}
            for which in ("attn1", "attn2"):
                for bkey, ent in stage_map.get(which, {}).items():
                    o = {}
                    if ent.get("proj_w") is not None:
                        o["proj_w"] = _to_device_dtype(ent["proj_w"], dev, ref_dtype)
                    if "kept_idx" in ent:
                        o["kept_idx"] = _as_long(ent["kept_idx"], dev)
                    if "head_dim" in ent:
                        o["head_dim"] = int(ent["head_dim"])
                    s_out[which][bkey] = o  # ← keep original key (int or str)

            # mlp entries: {"fc2_w", "kept_idx"}
            for bkey, ent in stage_map.get("mlp", {}).items():
                o = {}
                if ent.get("fc2_w") is not None:
                    o["fc2_w"] = _to_device_dtype(ent["fc2_w"], dev, ref_dtype)
                if "kept_idx" in ent:
                    o["kept_idx"] = _as_long(ent["kept_idx"], dev)
                s_out["mlp"][bkey] = o  # ← keep original key (int or str)

            self._proj_bank[int(sid)] = s_out

        # ---- build per-t lookup lists by resolving keys to flat block indices ----
        depth = int(self._num_blocks)
        self._pb_depth = depth
        self._pb_by_t = None  # will set below

        # helper: resolve a bank key (int or path string) -> flat block index
        def _resolve_bkey_to_idx(bkey) -> Optional[int]:
            # Already an int
            if isinstance(bkey, int):
                return bkey
            # Numeric string
            if isinstance(bkey, str) and bkey.isdigit():
                return int(bkey)

            s = str(bkey)

            # First try: exact block path match recorded during swap()
            try:
                return self.get_block_id_by_path(s)
            except Exception:
                pass

            # If the key is a *layer* path (ends with the specific parameter),
            # strip the suffix to recover the owning block path, then resolve.
            for suf in (".attn1.to_out.0", ".attn2.to_out.0", ".ff.net.2"):
                if s.endswith(suf):
                    s_block = s[: -len(suf)]
                    try:
                        return self.get_block_id_by_path(s_block)
                    except Exception:
                        pass

            # Heuristic fallback: extract an "attentions.<N>" number and try to map,
            # but do not return it directly (it is NOT a flat block id). Give up.
            return None

        # Case 1: we have a stages→t mapping → build a per-t table aligned to T
        if self._proj_bank_t2s is not None:
            T = len(self._proj_bank_t2s)
            by_t: List[Optional[Tuple[
                List[Optional[Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[int]]]],
                List[Optional[Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[int]]]],
                List[Optional[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]],
            ]]] = [None] * T

            for t, sid in enumerate(self._proj_bank_t2s):
                if sid < 0 or sid not in self._proj_bank:
                    by_t[t] = None
                    continue

                a1_list = [None] * depth
                a2_list = [None] * depth
                mlp_list = [None] * depth
                stage_map = self._proj_bank[sid]

                for bkey, ent in stage_map["attn1"].items():
                    idx = _resolve_bkey_to_idx(bkey)
                    if idx is not None and 0 <= idx < depth:
                        a1_list[idx] = (
                            ent.get("proj_w", None),
                            ent.get("kept_idx", None),
                            ent.get("head_dim", None),
                        )
                for bkey, ent in stage_map["attn2"].items():
                    idx = _resolve_bkey_to_idx(bkey)
                    if idx is not None and 0 <= idx < depth:
                        a2_list[idx] = (
                            ent.get("proj_w", None),
                            ent.get("kept_idx", None),
                            ent.get("head_dim", None),
                        )
                for bkey, ent in stage_map["mlp"].items():
                    idx = _resolve_bkey_to_idx(bkey)
                    if idx is not None and 0 <= idx < depth:
                        mlp_list[idx] = (
                            ent.get("fc2_w", None),
                            ent.get("kept_idx", None),
                        )

                by_t[t] = (a1_list, a2_list, mlp_list)

            self._pb_by_t = by_t
            # After bank changes, refresh index caches (they depend on bank kept_idx/head_dim)
            self._rebuild_index_caches()
            return  # done

        # Case 2: no explicit stages mapping
        # If exactly one stage is present, apply it for all timesteps; else leave None (require stages from caller)
        if self._proj_bank and len(self._proj_bank) == 1:
            only_sid = next(iter(self._proj_bank.keys()))
            T = int(max(1, self.diffusion_steps))
            a1_list = [None] * depth
            a2_list = [None] * depth
            mlp_list = [None] * depth
            stage_map = self._proj_bank[only_sid]

            for bkey, ent in stage_map["attn1"].items():
                idx = _resolve_bkey_to_idx(bkey)
                if idx is not None and 0 <= idx < depth:
                    a1_list[idx] = (
                        ent.get("proj_w", None),
                        ent.get("kept_idx", None),
                        ent.get("head_dim", None),
                    )
            for bkey, ent in stage_map["attn2"].items():
                idx = _resolve_bkey_to_idx(bkey)
                if idx is not None and 0 <= idx < depth:
                    a2_list[idx] = (
                        ent.get("proj_w", None),
                        ent.get("kept_idx", None),
                        ent.get("head_dim", None),
                    )
            for bkey, ent in stage_map["mlp"].items():
                idx = _resolve_bkey_to_idx(bkey)
                if idx is not None and 0 <= idx < depth:
                    mlp_list[idx] = (
                        ent.get("fc2_w", None),
                        ent.get("kept_idx", None),
                    )

            self._pb_by_t = [(a1_list, a2_list, mlp_list) for _ in range(T)]
        else:
            # Multiple stages but no mapping → ambiguous; keep None and require `stages` on next call
            self._pb_by_t = None

        # After bank changes, refresh index caches
        self._rebuild_index_caches()

    # ---------------------- Runtime lookups (called by blocks) ----------------------

    def _sid_for_t(self, t: int) -> int:
        # prefer second-order schedule mapping if present; else bank mapping; else 0
        if self._so_t2s is not None and 0 <= t < len(self._so_t2s) and self._so_t2s[t] >= 0:
            return int(self._so_t2s[t])
        if self._proj_bank_t2s is not None and 0 <= t < len(self._proj_bank_t2s) and self._proj_bank_t2s[t] >= 0:
            return int(self._proj_bank_t2s[t])
        return 0  # single-stage default

    def _lookup_attn_thin(self, t: int, blk: int, kind: Literal["attn1", "attn2"]) -> Tuple[
        Optional[torch.Tensor], Optional[torch.Tensor], Optional[int], Optional[torch.Tensor]
    ]:
        qkv_rows = proj_cols = None
        head_dim = None

        # Stage-compiled schedule
        if self._so_comp_per_stage is not None:
            sid = self._sid_for_t(t)
            if 0 <= sid < len(self._so_comp_per_stage):
                row = self._so_comp_per_stage[sid]
                if row is not None and 0 <= blk < len(row) and row[blk] is not None:
                    rec = row[blk].get(kind)
                    if rec is not None:
                        qkv_rows, proj_cols, head_dim = rec  # (qkv_rows, proj_cols, d)

        # Override with bank (kept_idx/head_dim/proj_w)
        proj_w_override = None
        if self._pb_by_t is not None and 0 <= t < len(self._pb_by_t) and self._pb_by_t[t] is not None:
            a1_list, a2_list, _ = self._pb_by_t[t]
            alist = a1_list if kind == "attn1" else a2_list
            if 0 <= blk < len(alist) and alist[blk] is not None:
                p_w, kept_idx, hdim = alist[blk]
                if kept_idx is not None:
                    proj_cols = kept_idx
                if hdim is not None:
                    head_dim = hdim
                if p_w is not None:
                    proj_w_override = p_w

        return qkv_rows, proj_cols, head_dim, proj_w_override

    def _lookup_mlp_thin(self, t: int, blk: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        mlp_keep = None
        if self._so_comp_per_stage is not None:
            sid = self._sid_for_t(t)
            if 0 <= sid < len(self._so_comp_per_stage):
                row = self._so_comp_per_stage[sid]
                if row is not None and 0 <= blk < len(row) and row[blk] is not None:
                    rec = row[blk].get("mlp")
                    if rec is not None:
                        (mlp_keep,) = rec

        fc2_w_override = None
        if self._pb_by_t is not None and 0 <= t < len(self._pb_by_t) and self._pb_by_t[t] is not None:
            _, _, mlp_list = self._pb_by_t[t]
            if 0 <= blk < len(mlp_list) and mlp_list[blk] is not None:
                fc2_w, kept_idx = mlp_list[blk]
                if kept_idx is not None:
                    mlp_keep = kept_idx
                if fc2_w is not None:
                    fc2_w_override = fc2_w

        return mlp_keep, fc2_w_override

    # ---------------------- Optional: apply once-only weight masks ----------------------

    @torch.no_grad()
    def apply_weight_masks(self, mask_map: Dict[str, torch.Tensor]) -> None:
        """
        Apply binary/real masks to named parameters exactly once (first-order methods).
        `mask_map` keys are full module paths to parameters, e.g. "mid_block.attentions.0.attn1.to_out.0.weight".
        """
        if self._weight_masks_applied:
            return
        for n, p in self.named_parameters():
            if n in mask_map:
                m = mask_map[n].to(device=p.device, dtype=p.dtype, non_blocking=True)
                if m.shape == p.data.shape:
                    p.data.mul_(m)
        self._weight_masks_applied = True

    # ---------------------- LayerDrop & Clear all accelerations ----------------------

    def _should_drop(self, blk_id: int, t: int) -> bool:
        # Fast mask if available
        if self._drop_by_t is not None and 0 <= t < len(self._drop_by_t) and self._drop_by_t[t] is not None:
            if self._drop_by_t[t][blk_id]:
                return True
        # Plus static drop ids
        return blk_id in self.drop_block_ids

    def clear_all_accel(self):
        self.drop_block_ids = set()
        self.layerdrop_schedule = None
        self._drop_by_t = None

        # second-order (stage-based)
        self._so_stage_schedule = None
        self._so_comp_per_stage = None
        self._so_stage_spans = None
        self._so_t2s = None

        # OBS bank
        self._proj_bank = None
        self._proj_bank_stages = None
        self._proj_bank_t2s = None
        self._pb_by_t = None
        self._pb_depth = None

        # Indices caches
        self._idx_attn_cache = None
        self._idx_mlp_cache = None

        self._weight_masks_applied = False

    # ---------------------- Indices-only cache builder ----------------------

    def _rebuild_index_caches(self):
        """
        Precompute only index tensors for the thin path (no weight copies).
        Safe to call multiple times. Very small memory footprint.
        """
        depth = self._num_blocks
        if depth == 0:
            self._idx_attn_cache = None
            self._idx_mlp_cache = None
            return

        dev = next(self.parameters()).device
        spans = self._so_stage_spans or self._proj_bank_stages or [(0, max(0, self.diffusion_steps - 1))]
        nstages = len(spans)

        attn_cache: List[Optional[List[Optional[Tuple[Optional[_IdxAttnPack], Optional[_IdxAttnPack]]]]]] = [None] * nstages
        mlp_cache:  List[Optional[List[Optional[_IdxMLPPack]]]] = [None] * nstages

        for sid in range(nstages):
            arow: List[Optional[Tuple[Optional[_IdxAttnPack], Optional[_IdxAttnPack]]]] = [None] * depth
            mrow: List[Optional[_IdxMLPPack]] = [None] * depth

            # choose a representative timestep for this stage (lo)
            if self._so_stage_spans is not None:
                t_rep = int(self._so_stage_spans[sid][0])
            elif self._proj_bank_stages is not None:
                t_rep = int(self._proj_bank_stages[sid][0])
            else:
                t_rep = 0

            for b in range(depth):
                # ---- attention (attn1/attn2)
                pair: List[Optional[_IdxAttnPack]] = [None, None]
                for ki, kind in enumerate(("attn1", "attn2")):
                    _qkv_rows, proj_cols, head_dim, _ = self._lookup_attn_thin(t_rep, b, kind)
                    if proj_cols is None or head_dim is None:
                        pair[ki] = None
                        continue

                    proj_cols = proj_cols.to(device=dev, dtype=torch.long).unique(sorted=True)
                    d = int(head_dim)
                    if proj_cols.numel() == 0 or d <= 0:
                        pair[ki] = None
                        continue

                    kept_heads = torch.div(proj_cols, d, rounding_mode="floor").unique(sorted=True)
                    if kept_heads.numel() == 0:
                        pair[ki] = None
                        continue

                    full_d = torch.arange(d, device=dev, dtype=torch.long)
                    kept_rows = (kept_heads[:, None] * d + full_d[None, :]).reshape(-1).contiguous()

                    pair[ki] = _IdxAttnPack(proj_cols=proj_cols, kept_rows=kept_rows, head_dim=d)
                arow[b] = (pair[0], pair[1])

                # ---- mlp
                mlp_keep, _ = self._lookup_mlp_thin(t_rep, b)
                if mlp_keep is not None:
                    mi = mlp_keep.to(device=dev, dtype=torch.long).unique(sorted=True)
                    if mi.numel() > 0:
                        mrow[b] = _IdxMLPPack(kept_idx=mi)
                    else:
                        mrow[b] = None
                else:
                    mrow[b] = None

            attn_cache[sid] = arow
            mlp_cache[sid]  = mrow

        self._idx_attn_cache = attn_cache
        self._idx_mlp_cache  = mlp_cache

    # -----------------------------------------------------------------------------
    # Classmethod helper to clone weights/config from a pretrained stock UNet
    # -----------------------------------------------------------------------------

    @classmethod
    def from_pretrained_pruned(cls, repo_or_path: str, **kwargs) -> "UNet2DConditionPruned":
        """
        Load a stock UNet2DConditionModel, instantiate this subclass WITHOUT swapping,
        copy the state_dict 1:1, then install pruned transformers (which copy weights
        per-block). This avoids cyclic module graphs during load and fixes recursion issues.
        """
        # Allow caller to pass diffusion_steps via kwargs; keep a local copy
        diffusion_steps = int(kwargs.pop("diffusion_steps", 1000))

        # 1) Load the stock UNet with all its original submodules
        base = UNet2DConditionModel.from_pretrained(repo_or_path, **kwargs)

        # 2) Build the pruned subclass WITHOUT swapping any submodules yet
        try:
            base_cfg = base.config.to_dict() if hasattr(base.config, "to_dict") else dict(base.config)
        except Exception:
            base_cfg = dict(base.config)

        sub = cls(**base_cfg, diffusion_steps=diffusion_steps, swap_on_init=False)

        # 3) Copy weights 1:1 while the graph matches exactly
        sub.load_state_dict(base.state_dict(), strict=True)

        # 4) Now swap transformer/basic blocks to their pruned counterparts and
        #    internally copy each block's weights during the swap.
        sub._install_pruned_transformers()

        # 5) Match dtype/device to 'base' (in case caller moved/cast it)
        try:
            p = next(base.parameters())
            sub.to(device=p.device, dtype=p.dtype)
        except StopIteration:
            pass

        return sub
