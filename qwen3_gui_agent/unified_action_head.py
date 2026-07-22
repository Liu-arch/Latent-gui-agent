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
        Linear = object
        Sequential = object
        GELU = object
        LayerNorm = object
        Dropout = object

    nn = _NNNamespace()  # type: ignore[assignment]


@dataclass
class UnifiedActionHeadOutput:
    action_type_logits: Any
    region_logits: Any
    pointer_pred: Any
    scroll_pred: Any
    terminate_logits: Any
    confidence_pred: Any
    target_patch_logits: Any
    target_patch_probs: Any
    target_null_logits: Any
    target_with_null_probs: Any
    actor_query: Any
    target_patch_grid_sizes: list[tuple[int, int]]


def _fallback_grid_size(token_count: int) -> tuple[int, int]:
    if token_count <= 0:
        return 1, 1
    width = max(1, int(round(token_count ** 0.5)))
    while width > 1 and token_count % width != 0:
        width -= 1
    height = max(1, token_count // width)
    if height * width != token_count:
        height = token_count
        width = 1
    return height, width


def resolve_grid_sizes_from_image_grid(
    image_grid_thw: Any,
    *,
    spatial_merge_size: int,
    token_count: int,
) -> list[tuple[int, int]]:
    if image_grid_thw is None or not hasattr(image_grid_thw, "tolist"):
        return [_fallback_grid_size(token_count)]
    rows = image_grid_thw.tolist()
    grid_sizes: list[tuple[int, int]] = []
    for row in rows:
        if len(row) < 3:
            grid_sizes.append(_fallback_grid_size(token_count))
            continue
        height = max(1, int(row[1]) // max(1, int(spatial_merge_size)))
        width = max(1, int(row[2]) // max(1, int(spatial_merge_size)))
        if height * width != token_count:
            grid_sizes.append(_fallback_grid_size(token_count))
        else:
            grid_sizes.append((height, width))
    return grid_sizes


def target_patch_index_from_point(
    *,
    x_norm: float,
    y_norm: float,
    grid_height: int,
    grid_width: int,
) -> int:
    x_idx = min(max(int(float(x_norm) * grid_width), 0), max(0, grid_width - 1))
    y_idx = min(max(int(float(y_norm) * grid_height), 0), max(0, grid_height - 1))
    return int(y_idx * grid_width + x_idx)


class UnifiedActionHead(nn.Module):
    def __init__(
        self,
        *,
        fused_dim: int,
        visual_dim: int,
        action_type_count: int,
        terminate_count: int,
        region_count: int = 9,
    ) -> None:
        super().__init__()
        hidden_dim = max(visual_dim, fused_dim // 2)
        self.actor_query_proj = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, visual_dim),
        )
        self.visual_context_proj = nn.Sequential(
            nn.Linear(visual_dim, visual_dim),
            nn.GELU(),
            nn.Linear(visual_dim, visual_dim),
        )
        self.visual_norm = nn.LayerNorm(visual_dim)
        self.target_key_proj = nn.Linear(visual_dim, visual_dim)
        self.target_null_head = nn.Linear(fused_dim, 1)

        self.action_type_head = nn.Linear(fused_dim, action_type_count)
        self.scroll_head = nn.Sequential(
            nn.Linear(fused_dim, max(1, fused_dim // 2)),
            nn.GELU(),
            nn.Linear(max(1, fused_dim // 2), 1),
        )
        self.terminate_head = nn.Linear(fused_dim, terminate_count)
        self.confidence_head = nn.Linear(fused_dim, 1)
        self.region_count = int(region_count)

    def forward(
        self,
        *,
        fused: Any,
        current_visual_tokens: Any,
        current_visual_token_mask: Any | None,
        target_patch_grid_sizes: list[tuple[int, int]],
    ) -> UnifiedActionHeadOutput:
        actor_query = self.actor_query_proj(fused)
        visual_ctx = self.visual_norm(current_visual_tokens + self.visual_context_proj(current_visual_tokens))
        keys = self.target_key_proj(visual_ctx)
        scale = float(keys.shape[-1]) ** -0.5
        target_patch_logits = torch.matmul(keys, actor_query.unsqueeze(-1)).squeeze(-1) * scale
        if current_visual_token_mask is not None:
            valid_mask = current_visual_token_mask.to(device=target_patch_logits.device, dtype=torch.bool)
            target_patch_logits = target_patch_logits.masked_fill(~valid_mask, -1e9)
        target_null_logits = self.target_null_head(fused)
        target_with_null_logits = torch.cat([target_patch_logits, target_null_logits], dim=-1)
        target_with_null_probs = torch.softmax(target_with_null_logits, dim=-1)
        target_patch_probs = target_with_null_probs[:, :-1]

        pointer_pred = self._decode_pointer_from_patch_probs(
            target_patch_probs=target_patch_probs,
            target_patch_grid_sizes=target_patch_grid_sizes,
        )
        region_logits = self._decode_region_logits_from_patch_probs(
            target_patch_probs=target_patch_probs,
            target_patch_grid_sizes=target_patch_grid_sizes,
        )
        return UnifiedActionHeadOutput(
            action_type_logits=self.action_type_head(fused),
            region_logits=region_logits,
            pointer_pred=pointer_pred,
            scroll_pred=self.scroll_head(fused),
            terminate_logits=self.terminate_head(fused),
            confidence_pred=torch.sigmoid(self.confidence_head(fused)),
            target_patch_logits=target_patch_logits,
            target_patch_probs=target_patch_probs,
            target_null_logits=target_null_logits,
            target_with_null_probs=target_with_null_probs,
            actor_query=actor_query,
            target_patch_grid_sizes=target_patch_grid_sizes,
        )

    def _decode_pointer_from_patch_probs(
        self,
        *,
        target_patch_probs: Any,
        target_patch_grid_sizes: list[tuple[int, int]],
    ) -> Any:
        batch_centers: list[Any] = []
        for batch_index, probs in enumerate(target_patch_probs):
            token_count = int(probs.shape[0])
            grid_height, grid_width = (
                target_patch_grid_sizes[batch_index]
                if batch_index < len(target_patch_grid_sizes)
                else _fallback_grid_size(token_count)
            )
            if grid_height * grid_width != token_count:
                grid_height, grid_width = _fallback_grid_size(token_count)
            x_coords = []
            y_coords = []
            for patch_index in range(token_count):
                row = patch_index // grid_width
                col = patch_index % grid_width
                x_coords.append((col + 0.5) / grid_width)
                y_coords.append((row + 0.5) / grid_height)
            x_tensor = probs.new_tensor(x_coords)
            y_tensor = probs.new_tensor(y_coords)
            total = probs.sum().clamp_min(1e-6)
            center_x = (probs * x_tensor).sum() / total
            center_y = (probs * y_tensor).sum() / total
            batch_centers.append(torch.stack([center_x, center_y], dim=0))
        return torch.stack(batch_centers, dim=0)

    def _decode_region_logits_from_patch_probs(
        self,
        *,
        target_patch_probs: Any,
        target_patch_grid_sizes: list[tuple[int, int]],
    ) -> Any:
        batch_region_scores: list[Any] = []
        for batch_index, probs in enumerate(target_patch_probs):
            token_count = int(probs.shape[0])
            grid_height, grid_width = (
                target_patch_grid_sizes[batch_index]
                if batch_index < len(target_patch_grid_sizes)
                else _fallback_grid_size(token_count)
            )
            if grid_height * grid_width != token_count:
                grid_height, grid_width = _fallback_grid_size(token_count)
            region_scores = probs.new_zeros(self.region_count)
            for patch_index in range(token_count):
                row = patch_index // grid_width
                col = patch_index % grid_width
                row_region = min(2, max(0, int((row + 0.5) / grid_height * 3)))
                col_region = min(2, max(0, int((col + 0.5) / grid_width * 3)))
                region_index = row_region * 3 + col_region
                region_scores[region_index] += probs[patch_index]
            batch_region_scores.append(region_scores.clamp_min(1e-6).log())
        return torch.stack(batch_region_scores, dim=0)
