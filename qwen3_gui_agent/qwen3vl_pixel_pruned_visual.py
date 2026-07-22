from __future__ import annotations

import time
from typing import Any

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

    class _NNNamespace:
        Module = object

    nn = _NNNamespace()  # type: ignore[assignment]

from qwen3_gui_agent.predictive_patch_pruner import PredictivePatchPruner


def _require_torch() -> None:
    if torch is None or F is None:
        raise ImportError("Qwen3VLPixelPrunedVisualWrapper requires torch.")


class Qwen3VLPixelPrunedVisualWrapper(nn.Module):
    """
    Wrap Qwen3-VL visual forward with PixelPrune-style predictive raw-patch
    pruning before `patch_embed`.

    Flow:
    raw patch pixels
    -> predictive patch pruning plan
    -> patch_embed on kept raw patches only
    -> reconstruct full patch embeddings using linear predictors
    -> regular Qwen3-VL pos/blocks/merger path
    """

    def __init__(
        self,
        visual: Any,
        *,
        pixel_prune_threshold: float = 0.0,
        pixel_prune_predictor_order: str = "pred2d,left,up",
        pixel_temporal_reuse: bool = False,
        pixel_temporal_threshold: float = 0.0,
    ) -> None:
        _require_torch()
        super().__init__()
        object.__setattr__(self, "_original_visual_ref", visual)
        self.config = getattr(visual, "config", None)
        self.spatial_merge_size = int(getattr(visual, "spatial_merge_size", 1) or 1)
        self.num_grid_per_side = int(
            getattr(
                visual,
                "num_grid_per_side",
                int(getattr(getattr(visual, "config", None), "num_position_embeddings", 0) ** 0.5),
            )
            or 0
        )
        self.patch_embed = visual.patch_embed
        self.pos_embed = getattr(visual, "pos_embed", None)
        self.rotary_pos_emb = getattr(visual, "rotary_pos_emb", None)
        self.blocks = visual.blocks
        self.merger = visual.merger
        self.deepstack_merger_list = getattr(visual, "deepstack_merger_list", None)
        self.deepstack_visual_indexes = list(getattr(visual, "deepstack_visual_indexes", []))
        self.fast_pos_embed_interpolate = visual.fast_pos_embed_interpolate
        self.rot_pos_emb = visual.rot_pos_emb
        self.predictive_pruner = PredictivePatchPruner(
            threshold=pixel_prune_threshold,
            predictor_order=pixel_prune_predictor_order,
        )
        self.pixel_prune_threshold = float(pixel_prune_threshold)
        self.pixel_prune_predictor_order = pixel_prune_predictor_order
        self.pixel_temporal_reuse = bool(pixel_temporal_reuse)
        self.pixel_temporal_threshold = float(pixel_temporal_threshold)
        self._temporal_batch_sample_keys: list[str] | None = None
        self._temporal_cache: dict[str, dict[str, Any]] = {}
        self.last_debug_shapes: dict[str, Any] | None = None

    @property
    def dtype(self) -> Any:
        original_dtype = getattr(object.__getattribute__(self, "_original_visual_ref"), "dtype", None)
        if original_dtype is not None:
            return original_dtype
        proj = getattr(self.patch_embed, "proj", None)
        weight = getattr(proj, "weight", None)
        if weight is not None:
            return weight.dtype
        first_param = next(self.parameters(), None)
        return None if first_param is None else first_param.dtype

    @property
    def device(self) -> Any:
        proj = getattr(self.patch_embed, "proj", None)
        weight = getattr(proj, "weight", None)
        if weight is not None:
            return weight.device
        first_param = next(self.parameters(), None)
        return None if first_param is None else first_param.device

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            original_visual = object.__getattribute__(self, "_original_visual_ref")
            return getattr(original_visual, name)

    def set_temporal_batch_context(self, sample_keys: list[str] | None) -> None:
        if sample_keys is None:
            self._temporal_batch_sample_keys = None
            return
        normalized = [str(key) for key in sample_keys]
        self._temporal_batch_sample_keys = normalized
        active_key_set = set(normalized)
        stale_keys = [key for key in self._temporal_cache.keys() if key not in active_key_set]
        for key in stale_keys:
            self._temporal_cache.pop(key, None)

    def _validated_grid_rows(self, hidden_states: Any, grid_thw: Any) -> list[tuple[int, int, int]]:
        if grid_thw is None or not hasattr(grid_thw, "ndim") or int(grid_thw.ndim) != 2:
            shape = None if grid_thw is None else tuple(getattr(grid_thw, "shape", ()))
            raise ValueError(f"Expected grid_thw with shape [num_items, 3], got {shape}.")
        if int(grid_thw.shape[1]) != 3:
            raise ValueError(f"Expected grid_thw with shape [num_items, 3], got {tuple(grid_thw.shape)}.")
        raw_rows = grid_thw.detach().to(device="cpu", dtype=torch.long).tolist()
        if not raw_rows:
            raise ValueError("grid_thw must contain at least one image/video grid row.")
        rows: list[tuple[int, int, int]] = []
        for row_index, row in enumerate(raw_rows):
            t, h, w = (int(row[0]), int(row[1]), int(row[2]))
            if t <= 0 or h <= 0 or w <= 0:
                raise ValueError(f"grid_thw row {row_index} must be positive, got {(t, h, w)}.")
            if h % self.spatial_merge_size != 0 or w % self.spatial_merge_size != 0:
                raise ValueError(
                    "Qwen3-VL visual grid must be divisible by spatial_merge_size before positional indexing: "
                    f"row={row_index}, grid={(t, h, w)}, spatial_merge_size={self.spatial_merge_size}."
                )
            rows.append((t, h, w))
        expected_tokens = sum(t * h * w for t, h, w in rows)
        actual_tokens = int(hidden_states.shape[0])
        if expected_tokens != actual_tokens:
            raise ValueError(
                "Qwen3-VL pixel_values/grid_thw mismatch before visual indexing: "
                f"grid_tokens={expected_tokens}, raw_patch_tokens={actual_tokens}, grids={rows}."
            )
        return rows

    def _safe_pos_embed_interpolate(self, grid_rows: list[tuple[int, int, int]]) -> Any:
        if self.pos_embed is None or self.num_grid_per_side <= 0:
            raise RuntimeError("Qwen3-VL visual position embedding metadata is unavailable.")
        merge_size = self.spatial_merge_size
        device = self.pos_embed.weight.device
        weight_dtype = self.pos_embed.weight.dtype
        output_chunks: list[Any] = []
        for t, h, w in grid_rows:
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h, device=device, dtype=torch.float32)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w, device=device, dtype=torch.float32)
            h_floor = h_idxs.floor().to(dtype=torch.long)
            w_floor = w_idxs.floor().to(dtype=torch.long)
            h_ceil = (h_floor + 1).clamp(max=self.num_grid_per_side - 1)
            w_ceil = (w_floor + 1).clamp(max=self.num_grid_per_side - 1)
            dh = h_idxs - h_floor.to(dtype=h_idxs.dtype)
            dw = w_idxs - w_floor.to(dtype=w_idxs.dtype)
            base_h = h_floor * self.num_grid_per_side
            base_h_ceil = h_ceil * self.num_grid_per_side
            indices = (
                (base_h[:, None] + w_floor[None, :]).reshape(-1),
                (base_h[:, None] + w_ceil[None, :]).reshape(-1),
                (base_h_ceil[:, None] + w_floor[None, :]).reshape(-1),
                (base_h_ceil[:, None] + w_ceil[None, :]).reshape(-1),
            )
            max_index = max(int(index.max().item()) for index in indices)
            embedding_rows = int(self.pos_embed.weight.shape[0])
            if max_index >= embedding_rows:
                raise RuntimeError(
                    f"Visual position index {max_index} exceeds embedding rows {embedding_rows} for grid {(t, h, w)}."
                )
            weights = (
                ((1 - dh)[:, None] * (1 - dw)[None, :]).reshape(-1),
                ((1 - dh)[:, None] * dw[None, :]).reshape(-1),
                (dh[:, None] * (1 - dw)[None, :]).reshape(-1),
                (dh[:, None] * dw[None, :]).reshape(-1),
            )
            pos_embed = sum(
                self.pos_embed(index) * weight.to(dtype=weight_dtype)[:, None]
                for index, weight in zip(indices, weights)
            )
            pos_embed = pos_embed.repeat(t, 1)
            pos_embed = (
                pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            output_chunks.append(pos_embed)
        output = torch.cat(output_chunks, dim=0)
        expected_tokens = sum(t * h * w for t, h, w in grid_rows)
        if int(output.shape[0]) != expected_tokens:
            raise RuntimeError(
                f"Safe visual position interpolation produced {int(output.shape[0])} rows, expected {expected_tokens}."
            )
        return output

    def _safe_rot_pos_emb(self, grid_rows: list[tuple[int, int, int]]) -> Any:
        if self.rotary_pos_emb is None:
            raise RuntimeError("Qwen3-VL visual rotary position embedding module is unavailable.")
        merge_size = self.spatial_merge_size
        max_hw = max(max(h, w) for _, h, w in grid_rows)
        freq_table = self.rotary_pos_emb(max_hw)
        device = freq_table.device
        position_chunks: list[Any] = []
        for t, h, w in grid_rows:
            merged_h = h // merge_size
            merged_w = w // merge_size
            block_rows = torch.arange(merged_h, device=device)
            block_cols = torch.arange(merged_w, device=device)
            intra_row = torch.arange(merge_size, device=device)
            intra_col = torch.arange(merge_size, device=device)
            row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
            col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]
            row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            coords = torch.stack((row_idx, col_idx), dim=-1)
            if t > 1:
                coords = coords.repeat(t, 1)
            expected_row_tokens = t * h * w
            if int(coords.shape[0]) != expected_row_tokens:
                raise RuntimeError(
                    f"Rotary position construction produced {int(coords.shape[0])} rows, "
                    f"expected {expected_row_tokens} for grid {(t, h, w)}."
                )
            position_chunks.append(coords)
        position_ids = torch.cat(position_chunks, dim=0)
        expected_tokens = sum(t * h * w for t, h, w in grid_rows)
        if int(position_ids.shape[0]) != expected_tokens:
            raise RuntimeError(
                f"Rotary position construction produced {int(position_ids.shape[0])} rows, expected {expected_tokens}."
            )
        min_position = int(position_ids.min().item())
        max_position = int(position_ids.max().item())
        if min_position < 0 or max_position >= int(freq_table.shape[0]):
            raise RuntimeError(
                "Rotary position IDs are out of range before CUDA indexing: "
                f"min={min_position}, max={max_position}, table_rows={int(freq_table.shape[0])}, grids={grid_rows}."
            )
        return freq_table[position_ids].flatten(1)

    def forward(self, hidden_states: Any, grid_thw: Any, **kwargs: Any) -> Any:
        raw_hidden_states = hidden_states
        sample_keys = list(self._temporal_batch_sample_keys or [])
        grid_rows = self._validated_grid_rows(raw_hidden_states, grid_thw)
        if self.pixel_temporal_reuse and sample_keys and len(sample_keys) != len(grid_rows):
            raise ValueError(
                "Temporal cache keys must align one-to-one with image_grid_thw rows: "
                f"keys={len(sample_keys)}, grids={len(grid_rows)}."
            )
        previous_hidden_states = None
        previous_patch_embeddings = None
        temporal_cache_hit_count = 0
        temporal_cache_hits_per_sample: list[int] = []
        if self.pixel_temporal_reuse and sample_keys and len(sample_keys) == len(grid_rows):
            previous_hidden_chunks: list[Any] = []
            previous_embedding_chunks: list[Any] = []
            previous_available = False
            offset = 0
            for sample_key, row in zip(sample_keys, grid_rows):
                t, h, w = int(row[0]), int(row[1]), int(row[2])
                sample_tokens = t * h * w
                cache_entry = self._temporal_cache.get(sample_key)
                if (
                    cache_entry is not None
                    and tuple(cache_entry.get("grid_thw", ())) == (t, h, w)
                    and int(cache_entry["raw_patches"].shape[0]) == sample_tokens
                    and int(cache_entry["patch_embeddings"].shape[0]) == sample_tokens
                ):
                    previous_hidden_chunks.append(
                        cache_entry["raw_patches"].to(device=raw_hidden_states.device, dtype=raw_hidden_states.dtype)
                    )
                    previous_embedding_chunks.append(
                        cache_entry["patch_embeddings"].to(device=raw_hidden_states.device, dtype=self.patch_embed.proj.weight.dtype)
                    )
                    previous_available = True
                    current_sample_hidden = raw_hidden_states[offset : offset + sample_tokens]
                    diff = torch.abs(current_sample_hidden - previous_hidden_chunks[-1]).amax(dim=-1)
                    hit_count = int((diff <= self.pixel_temporal_threshold).sum().item())
                    temporal_cache_hits_per_sample.append(hit_count)
                    temporal_cache_hit_count += hit_count
                else:
                    previous_hidden_chunks.append(
                        torch.full_like(raw_hidden_states[offset : offset + sample_tokens], float("nan"))
                    )
                    previous_embedding_chunks.append(
                        torch.zeros(
                            (sample_tokens, int(self.patch_embed.proj.out_channels)),
                            device=raw_hidden_states.device,
                            dtype=self.patch_embed.proj.weight.dtype,
                        )
                    )
                    temporal_cache_hits_per_sample.append(0)
                offset += sample_tokens
            if previous_available:
                previous_hidden_states = torch.cat(previous_hidden_chunks, dim=0)
                previous_patch_embeddings = torch.cat(previous_embedding_chunks, dim=0)

        sync_device = getattr(hidden_states, "device", None)
        if sync_device is not None and hasattr(sync_device, "type") and sync_device.type == "cuda":
            torch.cuda.synchronize(sync_device)
        started_plan = time.perf_counter()
        pruned = self.predictive_pruner(
            hidden_states,
            grid_thw,
            previous_hidden_states=previous_hidden_states,
        )
        if sync_device is not None and hasattr(sync_device, "type") and sync_device.type == "cuda":
            torch.cuda.synchronize(sync_device)
        plan_seconds = time.perf_counter() - started_plan

        kept_patch_embeddings = self.patch_embed(pruned.kept_raw_patches)
        kept_patch_embed_shape = tuple(kept_patch_embeddings.shape)

        if sync_device is not None and hasattr(sync_device, "type") and sync_device.type == "cuda":
            torch.cuda.synchronize(sync_device)
        started_reconstruct = time.perf_counter()
        hidden_states = self.predictive_pruner.reconstruct_patch_embeddings(
            kept_patch_embeddings=kept_patch_embeddings,
            keep_mask=pruned.keep_mask,
            predictor_codes=pruned.predictor_codes,
            grid_thw=grid_thw,
            previous_patch_embeddings=previous_patch_embeddings,
        )
        reconstructed_patch_embeddings = hidden_states
        if sync_device is not None and hasattr(sync_device, "type") and sync_device.type == "cuda":
            torch.cuda.synchronize(sync_device)
        reconstruct_seconds = time.perf_counter() - started_reconstruct

        patch_embed_shape = tuple(reconstructed_patch_embeddings.shape)
        pos_embeds = self._safe_pos_embed_interpolate(grid_rows)
        hidden_states = reconstructed_patch_embeddings + pos_embeds

        rotary_pos_emb = self._safe_rot_pos_emb(grid_rows)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        self.last_debug_shapes = {
            "raw_patch_shape": tuple(pruned.kept_raw_patches.shape),
            "kept_patch_embed_shape": kept_patch_embed_shape,
            "patch_embed_shape": patch_embed_shape,
            "full_patch_token_count": int(pruned.full_patch_count),
            "kept_patch_token_count": int(pruned.kept_patch_count),
            "kept_patch_token_ratio": float(pruned.kept_patch_count) / max(1, int(pruned.full_patch_count)),
            "kept_counts_per_frame": list(pruned.kept_counts),
            "predictor_hit_counts": dict(pruned.predictor_hit_counts),
            "temporal_reused_patch_count": int(pruned.temporal_reused_count),
            "temporal_reused_counts_per_frame": list(pruned.temporal_reused_counts),
            "temporal_cache_hit_count": int(temporal_cache_hit_count),
            "temporal_cache_hits_per_sample": list(temporal_cache_hits_per_sample),
            "temporal_cache_entry_count": int(len(self._temporal_cache)),
            "temporal_cache_patch_count": int(
                sum(int(entry["raw_patches"].shape[0]) for entry in self._temporal_cache.values())
            ),
            "pixel_prune_plan_seconds": plan_seconds,
            "pixel_prune_reconstruct_seconds": reconstruct_seconds,
        }

        deepstack_feature_lists: list[Any] = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            if layer_num in self.deepstack_visual_indexes and self.deepstack_merger_list is not None:
                deepstack_feature = self.deepstack_merger_list[
                    self.deepstack_visual_indexes.index(layer_num)
                ](hidden_states)
                deepstack_feature_lists.append(deepstack_feature)

        if self.last_debug_shapes is not None:
            self.last_debug_shapes["selected_hidden_shape_after_blocks"] = tuple(hidden_states.shape)
        if self.pixel_temporal_reuse and sample_keys and len(sample_keys) == len(grid_rows):
            cache_offset = 0
            for sample_key, row in zip(sample_keys, grid_rows):
                t, h, w = int(row[0]), int(row[1]), int(row[2])
                sample_tokens = t * h * w
                self._temporal_cache[sample_key] = {
                    "grid_thw": (t, h, w),
                    # Keep only the previous active frame on GPU. This avoids CPU<->GPU
                    # copies while still detaching the cache from the current graph.
                    "raw_patches": raw_hidden_states[cache_offset : cache_offset + sample_tokens].detach().clone(),
                    "patch_embeddings": reconstructed_patch_embeddings[
                        cache_offset : cache_offset + sample_tokens
                    ].detach().clone(),
                }
                cache_offset += sample_tokens
        hidden_states = self.merger(hidden_states)
        if self.last_debug_shapes is not None:
            self.last_debug_shapes["merger_output_shape"] = tuple(hidden_states.shape)
        return hidden_states, deepstack_feature_lists
