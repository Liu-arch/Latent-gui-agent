from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

    class _NNNamespace:
        Module = object

    nn = _NNNamespace()  # type: ignore[assignment]


KEEP_CODE = 0
LEFT_CODE = 1
UP_CODE = 2
PRED2D_CODE = 3
TEMPORAL_CODE = 4
_VALID_PREDICTORS = ("pred2d", "left", "up")


def _require_torch() -> None:
    if torch is None:
        raise ImportError("PredictivePatchPruner requires torch.")


def _validated_grid_rows(grid_thw: Any) -> list[tuple[int, int, int]]:
    if grid_thw is None:
        raise ValueError("grid_thw is required for predictive patch pruning.")
    if hasattr(grid_thw, "ndim") and int(grid_thw.ndim) != 2:
        raise ValueError(f"Expected grid_thw with shape [num_items, 3], got {tuple(grid_thw.shape)}.")
    rows = grid_thw.tolist() if hasattr(grid_thw, "tolist") else grid_thw
    if not isinstance(rows, (list, tuple)) or not rows:
        raise ValueError("grid_thw must contain at least one image/video grid row.")
    normalized: list[tuple[int, int, int]] = []
    for row_index, row in enumerate(rows):
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            raise ValueError(f"grid_thw row {row_index} must contain exactly (t, h, w), got {row!r}.")
        t, h, w = (int(row[0]), int(row[1]), int(row[2]))
        if t <= 0 or h <= 0 or w <= 0:
            raise ValueError(f"grid_thw row {row_index} must be positive, got {(t, h, w)}.")
        normalized.append((t, h, w))
    return normalized


@dataclass
class PredictivePatchPruneOutput:
    kept_raw_patches: Any
    keep_mask: Any
    predictor_codes: Any
    kept_counts: list[int]
    full_patch_count: int
    kept_patch_count: int
    predictor_hit_counts: dict[str, int]
    temporal_reused_count: int
    temporal_reused_counts: list[int]


class PredictivePatchPruner(nn.Module):
    """
    PixelPrune-style predictive raw-patch pruning before `patch_embed`.

    This minimal implementation starts from the strict setting `threshold=0`:
    - if a patch can be predicted exactly from its raster-order neighbors,
      we skip patch_embed for that patch
    - later we reconstruct the skipped patch embedding from already available
      neighbor embeddings using the same linear predictor

    Because Qwen3-VL patch embedding is linear, the following reuse is exact:
    - left:    E(x) = E(left)
    - up:      E(x) = E(up)
    - pred2d:  E(x) = E(left) + E(up) - E(up_left)
    """

    def __init__(
        self,
        *,
        threshold: float = 0.0,
        predictor_order: str = "pred2d,left,up",
    ) -> None:
        _require_torch()
        super().__init__()
        self.threshold = float(threshold)
        parsed_predictors = [item.strip().lower() for item in predictor_order.split(",") if item.strip()]
        if not parsed_predictors:
            parsed_predictors = list(_VALID_PREDICTORS)
        invalid = [item for item in parsed_predictors if item not in _VALID_PREDICTORS]
        if invalid:
            raise ValueError(
                f"Unsupported predictor(s) {invalid}. Expected subset of {_VALID_PREDICTORS}."
            )
        self.predictor_order = parsed_predictors

    def forward(
        self,
        hidden_states: Any,
        grid_thw: Any,
        previous_hidden_states: Any | None = None,
    ) -> PredictivePatchPruneOutput:
        if hidden_states.ndim != 2:
            raise ValueError(
                f"Expected hidden_states with shape [seq, raw_patch_dim], got {tuple(hidden_states.shape)}"
            )
        grid_rows = _validated_grid_rows(grid_thw)
        expected_patch_count = sum(t * h * w for t, h, w in grid_rows)
        actual_patch_count = int(hidden_states.shape[0])
        if expected_patch_count != actual_patch_count:
            raise ValueError(
                "grid_thw does not match the raw patch sequence before pruning: "
                f"grid_tokens={expected_patch_count}, hidden_tokens={actual_patch_count}, grids={grid_rows}."
            )
        if previous_hidden_states is not None:
            if previous_hidden_states.ndim != 2:
                raise ValueError(
                    "Expected previous_hidden_states with shape [seq, raw_patch_dim], "
                    f"got {tuple(previous_hidden_states.shape)}"
                )
            if previous_hidden_states.shape != hidden_states.shape:
                raise ValueError(
                    "previous_hidden_states must match hidden_states shape, "
                    f"got {tuple(previous_hidden_states.shape)} vs {tuple(hidden_states.shape)}."
                )

        keep_chunks: list[Any] = []
        code_chunks: list[Any] = []
        kept_counts: list[int] = []
        temporal_reused_counts: list[int] = []
        predictor_hit_counts = {"pred2d": 0, "left": 0, "up": 0, "temporal": 0}
        offset = 0

        for t, h, w in grid_rows:
            frame_tokens = h * w
            for _ in range(t):
                start = offset
                end = offset + frame_tokens
                frame_hidden = hidden_states[start:end]
                previous_frame_hidden = None
                if previous_hidden_states is not None:
                    previous_frame_hidden = previous_hidden_states[start:end]
                frame_keep_mask, frame_codes, frame_hits = self._build_frame_plan(
                    frame_hidden=frame_hidden,
                    previous_frame_hidden=previous_frame_hidden,
                    h=h,
                    w=w,
                )
                keep_chunks.append(frame_keep_mask)
                code_chunks.append(frame_codes)
                kept_counts.append(int(frame_keep_mask.sum().item()))
                temporal_reused_counts.append(int(frame_hits.get("temporal", 0)))
                for key, value in frame_hits.items():
                    predictor_hit_counts[key] += int(value)
                offset = end

        if offset != expected_patch_count:
            raise RuntimeError(
                f"Patch-pruning plan consumed {offset} patches, expected {expected_patch_count}."
            )

        keep_mask = torch.cat(keep_chunks, dim=0)
        predictor_codes = torch.cat(code_chunks, dim=0)
        if int(keep_mask.numel()) != actual_patch_count or int(predictor_codes.numel()) != actual_patch_count:
            raise RuntimeError(
                "Patch-pruning plan length mismatch: "
                f"keep_mask={int(keep_mask.numel())}, predictor_codes={int(predictor_codes.numel())}, "
                f"hidden_tokens={actual_patch_count}."
            )
        kept_indices = torch.nonzero(keep_mask, as_tuple=False).view(-1)
        kept_raw_patches = hidden_states.index_select(0, kept_indices)
        return PredictivePatchPruneOutput(
            kept_raw_patches=kept_raw_patches,
            keep_mask=keep_mask,
            predictor_codes=predictor_codes,
            kept_counts=kept_counts,
            full_patch_count=int(hidden_states.shape[0]),
            kept_patch_count=int(kept_raw_patches.shape[0]),
            predictor_hit_counts=predictor_hit_counts,
            temporal_reused_count=int(sum(temporal_reused_counts)),
            temporal_reused_counts=temporal_reused_counts,
        )

    def reconstruct_patch_embeddings(
        self,
        *,
        kept_patch_embeddings: Any,
        keep_mask: Any,
        predictor_codes: Any,
        grid_thw: Any,
        previous_patch_embeddings: Any | None = None,
    ) -> Any:
        if kept_patch_embeddings.ndim != 2:
            raise ValueError(
                "Expected kept_patch_embeddings with shape [kept_seq, embed_dim], "
                f"got {tuple(kept_patch_embeddings.shape)}"
            )
        if keep_mask.ndim != 1 or predictor_codes.ndim != 1:
            raise ValueError("keep_mask and predictor_codes must be flat tensors.")
        grid_rows = _validated_grid_rows(grid_thw)
        expected_patch_count = sum(t * h * w for t, h, w in grid_rows)
        if int(keep_mask.shape[0]) != expected_patch_count:
            raise ValueError(
                "grid_thw does not match keep_mask during reconstruction: "
                f"grid_tokens={expected_patch_count}, keep_mask={int(keep_mask.shape[0])}, grids={grid_rows}."
            )
        if int(predictor_codes.shape[0]) != expected_patch_count:
            raise ValueError(
                f"predictor_codes length {int(predictor_codes.shape[0])} does not match {expected_patch_count}."
            )
        expected_kept_count = int(keep_mask.sum().item())
        if int(kept_patch_embeddings.shape[0]) != expected_kept_count:
            raise ValueError(
                "Kept patch embedding count does not match keep_mask: "
                f"embeddings={int(kept_patch_embeddings.shape[0])}, mask_sum={expected_kept_count}."
            )
        if previous_patch_embeddings is not None:
            if previous_patch_embeddings.ndim != 2:
                raise ValueError(
                    "Expected previous_patch_embeddings with shape [seq, embed_dim], "
                    f"got {tuple(previous_patch_embeddings.shape)}"
                )
            if int(previous_patch_embeddings.shape[0]) != int(keep_mask.shape[0]):
                raise ValueError(
                    "previous_patch_embeddings length must match keep_mask, "
                    f"got {int(previous_patch_embeddings.shape[0])} vs {int(keep_mask.shape[0])}."
                )

        embed_dim = int(kept_patch_embeddings.shape[-1])
        full_hidden = torch.empty(
            (int(keep_mask.shape[0]), embed_dim),
            device=kept_patch_embeddings.device,
            dtype=kept_patch_embeddings.dtype,
        )
        keep_ptr = 0
        offset = 0

        for t, h, w in grid_rows:
            frame_tokens = h * w
            for _ in range(t):
                start = offset
                end = offset + frame_tokens
                frame_keep = keep_mask[start:end].view(h, w)
                frame_codes = predictor_codes[start:end].view(h, w)
                frame_keep_cpu = frame_keep.to(device="cpu")
                frame_codes_cpu = frame_codes.to(device="cpu")
                frame_hidden = torch.empty(
                    (h, w, embed_dim),
                    device=kept_patch_embeddings.device,
                    dtype=kept_patch_embeddings.dtype,
                )
                previous_frame_hidden = None
                if previous_patch_embeddings is not None:
                    previous_frame_hidden = previous_patch_embeddings[start:end].view(h, w, embed_dim).to(
                        device=kept_patch_embeddings.device,
                        dtype=kept_patch_embeddings.dtype,
                    )
                for row_idx in range(h):
                    for col_idx in range(w):
                        if bool(frame_keep_cpu[row_idx, col_idx]):
                            if keep_ptr >= int(kept_patch_embeddings.shape[0]):
                                raise RuntimeError(
                                    "Predictive reconstruction attempted to read beyond kept patch embeddings."
                                )
                            frame_hidden[row_idx, col_idx] = kept_patch_embeddings[keep_ptr]
                            keep_ptr += 1
                            continue
                        code = int(frame_codes_cpu[row_idx, col_idx])
                        if code == LEFT_CODE:
                            frame_hidden[row_idx, col_idx] = frame_hidden[row_idx, col_idx - 1]
                        elif code == UP_CODE:
                            frame_hidden[row_idx, col_idx] = frame_hidden[row_idx - 1, col_idx]
                        elif code == PRED2D_CODE:
                            frame_hidden[row_idx, col_idx] = (
                                frame_hidden[row_idx, col_idx - 1]
                                + frame_hidden[row_idx - 1, col_idx]
                                - frame_hidden[row_idx - 1, col_idx - 1]
                            )
                        elif code == TEMPORAL_CODE:
                            if previous_frame_hidden is None:
                                raise RuntimeError(
                                    "Encountered TEMPORAL_CODE without previous_patch_embeddings."
                                )
                            frame_hidden[row_idx, col_idx] = previous_frame_hidden[row_idx, col_idx]
                        else:
                            raise RuntimeError(
                                f"Unexpected predictor code={code} at frame offset {start}, "
                                f"position=({row_idx}, {col_idx})."
                            )
                full_hidden[start:end] = frame_hidden.view(frame_tokens, embed_dim)
                offset = end

        if offset != expected_patch_count:
            raise RuntimeError(
                f"Predictive reconstruction wrote {offset} patches, expected {expected_patch_count}."
            )
        if keep_ptr != int(kept_patch_embeddings.shape[0]):
            raise RuntimeError(
                f"Predictive reconstruction consumed {keep_ptr} kept patch embeddings, "
                f"expected {int(kept_patch_embeddings.shape[0])}."
            )
        return full_hidden

    def _build_frame_plan(
        self,
        *,
        frame_hidden: Any,
        previous_frame_hidden: Any | None,
        h: int,
        w: int,
    ) -> tuple[Any, Any, dict[str, int]]:
        frame_grid = frame_hidden.view(h, w, -1)
        keep_mask = torch.ones((h, w), device=frame_hidden.device, dtype=torch.bool)
        predictor_codes = torch.zeros((h, w), device=frame_hidden.device, dtype=torch.uint8)
        predictor_hit_counts = {"pred2d": 0, "left": 0, "up": 0, "temporal": 0}

        if previous_frame_hidden is not None:
            previous_grid = previous_frame_hidden.view(h, w, -1)
            temporal_matches = self._matches(frame_grid, previous_grid)
            temporal_chosen = temporal_matches & keep_mask
            keep_mask[temporal_chosen] = False
            predictor_codes[temporal_chosen] = TEMPORAL_CODE
            predictor_hit_counts["temporal"] += int(temporal_chosen.sum().item())

        for predictor_name in self.predictor_order:
            if predictor_name == "left":
                target = frame_grid[:, 1:, :]
                predictor = frame_grid[:, :-1, :]
                matches = self._matches(target, predictor)
                available = keep_mask[:, 1:]
                chosen = matches & available
                keep_mask[:, 1:][chosen] = False
                predictor_codes[:, 1:][chosen] = LEFT_CODE
                predictor_hit_counts["left"] += int(chosen.sum().item())
            elif predictor_name == "up":
                target = frame_grid[1:, :, :]
                predictor = frame_grid[:-1, :, :]
                matches = self._matches(target, predictor)
                available = keep_mask[1:, :]
                chosen = matches & available
                keep_mask[1:, :][chosen] = False
                predictor_codes[1:, :][chosen] = UP_CODE
                predictor_hit_counts["up"] += int(chosen.sum().item())
            elif predictor_name == "pred2d":
                target = frame_grid[1:, 1:, :]
                predictor = (
                    frame_grid[1:, :-1, :]
                    + frame_grid[:-1, 1:, :]
                    - frame_grid[:-1, :-1, :]
                )
                matches = self._matches(target, predictor)
                available = keep_mask[1:, 1:]
                chosen = matches & available
                keep_mask[1:, 1:][chosen] = False
                predictor_codes[1:, 1:][chosen] = PRED2D_CODE
                predictor_hit_counts["pred2d"] += int(chosen.sum().item())

        return keep_mask.view(-1), predictor_codes.view(-1), predictor_hit_counts

    def _matches(self, target: Any, predictor: Any) -> Any:
        if self.threshold <= 0.0:
            return torch.eq(target, predictor).all(dim=-1)
        max_abs_error = torch.abs(target - predictor).amax(dim=-1)
        return max_abs_error <= self.threshold
