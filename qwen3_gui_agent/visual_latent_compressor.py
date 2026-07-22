from __future__ import annotations

import math
from dataclasses import dataclass
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
        Parameter = object
        Linear = object
        LayerNorm = object
        Sequential = object
        GELU = object
        Embedding = object

    nn = _NNNamespace()  # type: ignore[assignment]


def _require_torch() -> None:
    if torch is None or F is None:
        raise ImportError("VisualLatentCompressor requires torch.")


@dataclass
class VisualLatentCompressionOutput:
    slot_vectors: Any
    logits: Any


class VisualLatentCompressor(nn.Module):
    """
    Trainable visual latent compressor.

    visual tokens
    -> learnable slot queries attend over tokens
    -> K slot vectors
    -> compare against a learnable codebook
    -> K x codebook_size logits
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_slots: int,
        codebook_size: int,
        slot_dim: int | None = None,
    ) -> None:
        _require_torch()
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_slots = int(num_slots)
        self.codebook_size = int(codebook_size)
        self.slot_dim = int(slot_dim or hidden_dim)

        self.slot_queries = nn.Parameter(torch.randn(self.num_slots, self.slot_dim) * 0.02)
        self.register_buffer(
            "slot_identity_encoding",
            build_slot_identity_encoding(self.num_slots, self.slot_dim),
            persistent=False,
        )
        self.key_proj = nn.Linear(self.hidden_dim, self.slot_dim)
        self.value_proj = nn.Linear(self.hidden_dim, self.slot_dim)
        self.slot_out = nn.Sequential(
            nn.Linear(self.slot_dim, self.slot_dim),
            nn.GELU(),
            nn.Linear(self.slot_dim, self.slot_dim),
        )
        self.slot_norm = nn.LayerNorm(self.slot_dim)
        self.codebook = nn.Embedding(self.codebook_size, self.slot_dim)
        nn.init.normal_(self.codebook.weight, mean=0.0, std=0.02)

    def forward(self, visual_tokens: Any) -> VisualLatentCompressionOutput:
        if visual_tokens.ndim != 3:
            raise ValueError(
                f"Expected visual tokens with shape [batch, num_tokens, hidden_dim], got {tuple(visual_tokens.shape)}"
            )
        batch_size = int(visual_tokens.shape[0])
        keys = self.key_proj(visual_tokens)
        values = self.value_proj(visual_tokens)
        identity = self.slot_identity_encoding.to(device=visual_tokens.device, dtype=visual_tokens.dtype)
        queries = (self.slot_queries + identity).unsqueeze(0).expand(batch_size, -1, -1)

        scale = 1.0 / math.sqrt(float(self.slot_dim))
        attn_scores = torch.matmul(queries, keys.transpose(1, 2)) * scale
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attended = torch.matmul(attn_weights, values)
        slot_vectors = self.slot_norm(attended + self.slot_out(attended))

        normalized_slots = F.normalize(slot_vectors, dim=-1)
        normalized_codebook = F.normalize(self.codebook.weight, dim=-1)
        logits = torch.matmul(normalized_slots, normalized_codebook.transpose(0, 1))
        return VisualLatentCompressionOutput(slot_vectors=slot_vectors, logits=logits)

    def align_to_tensor(self, tensor: Any) -> None:
        first_param = next(self.parameters(), None)
        if first_param is not None and (
            first_param.device != tensor.device or first_param.dtype != tensor.dtype
        ):
            self.to(device=tensor.device, dtype=tensor.dtype)

    def update_ema_from_(self, student: "VisualLatentCompressor", momentum: float) -> None:
        _require_torch()
        momentum = float(momentum)
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"EMA momentum must be in [0, 1), got {momentum}")
        with torch.no_grad():
            student_state = dict(student.named_parameters())
            for name, teacher_param in self.named_parameters():
                student_param = student_state[name]
                teacher_param.data.mul_(momentum).add_(student_param.data, alpha=1.0 - momentum)
            student_buffers = dict(student.named_buffers())
            for name, teacher_buffer in self.named_buffers():
                student_buffer = student_buffers[name]
                teacher_buffer.data.copy_(student_buffer.data)


def build_slot_identity_encoding(num_slots: int, slot_dim: int) -> Any:
    _require_torch()
    positions = torch.arange(num_slots, dtype=torch.float32).unsqueeze(1)
    if slot_dim <= 0:
        raise ValueError(f"slot_dim must be positive, got {slot_dim}")
    div_terms = torch.exp(
        torch.arange(0, slot_dim, 2, dtype=torch.float32) * (-math.log(10000.0) / max(1, slot_dim))
    )
    encoding = torch.zeros(num_slots, slot_dim, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(positions * div_terms)
    if slot_dim > 1:
        cos_width = encoding[:, 1::2].shape[1]
        encoding[:, 1::2] = torch.cos(positions * div_terms[:cos_width])
    return encoding


def extract_visual_token_sequence_batch(
    *,
    hidden_states: Any,
    image_grid_thw: Any,
    spatial_merge_size: int = 2,
) -> Any:
    visual_tokens, _ = extract_visual_token_sequence_batch_with_mask(
        hidden_states=hidden_states,
        image_grid_thw=image_grid_thw,
        spatial_merge_size=spatial_merge_size,
    )
    return visual_tokens


def extract_visual_token_sequence_batch_with_mask(
    *,
    hidden_states: Any,
    image_grid_thw: Any,
    spatial_merge_size: int = 2,
    image_counts_per_sample: list[int] | None = None,
    select_last_image_per_sample: bool = False,
) -> tuple[Any, Any]:
    _require_torch()
    if hidden_states.ndim != 3:
        raise ValueError(f"Expected [batch, seq, hidden], got {tuple(hidden_states.shape)}")
    batch_size = int(hidden_states.shape[0])
    visual_token_spans = extract_visual_token_spans_per_sample(
        image_grid_thw=image_grid_thw,
        batch_size=batch_size,
        spatial_merge_size=spatial_merge_size,
        image_counts_per_sample=image_counts_per_sample,
        select_last_image_per_sample=select_last_image_per_sample,
    )
    visual_token_counts = [count for _, count in visual_token_spans]
    if not visual_token_counts:
        empty = hidden_states[:, :0, :]
        return empty, torch.zeros(empty.shape[:2], device=hidden_states.device, dtype=torch.bool)
    if max(visual_token_counts) <= 0:
        empty = hidden_states[:, :0, :]
        return empty, torch.zeros(empty.shape[:2], device=hidden_states.device, dtype=torch.bool)
    visual_token_count = min(max(int(count) for count in visual_token_counts), int(hidden_states.shape[1]))
    visual_tokens = hidden_states.new_zeros((batch_size, visual_token_count, hidden_states.shape[-1]))
    visual_token_mask = torch.zeros(
        (batch_size, visual_token_count),
        device=hidden_states.device,
        dtype=torch.bool,
    )
    # Image content occupies the sequence prefix before the text prompt tokens.
    for batch_index, (start, count) in enumerate(visual_token_spans):
        sample_count = min(max(0, int(count)), int(hidden_states.shape[1]))
        if sample_count <= 0:
            continue
        sample_start = min(max(0, int(start)), int(hidden_states.shape[1]))
        sample_end = min(sample_start + sample_count, int(hidden_states.shape[1]))
        actual_count = max(0, sample_end - sample_start)
        if actual_count <= 0:
            continue
        visual_tokens[batch_index, :actual_count, :] = hidden_states[batch_index, sample_start:sample_end, :]
        visual_token_mask[batch_index, :actual_count] = True
    return visual_tokens, visual_token_mask


def extract_visual_token_spans_per_sample(
    *,
    image_grid_thw: Any,
    batch_size: int,
    spatial_merge_size: int,
    image_counts_per_sample: list[int] | None = None,
    select_last_image_per_sample: bool = False,
) -> list[tuple[int, int]]:
    if image_grid_thw is None:
        return []
    rows = image_grid_thw.tolist() if hasattr(image_grid_thw, "tolist") else image_grid_thw
    merge_divisor = max(1, int(spatial_merge_size) ** 2)

    def row_count(row: Any) -> int:
        if isinstance(row, (list, tuple)) and len(row) >= 3:
            try:
                t, h, w = int(row[0]), int(row[1]), int(row[2])
                return max(0, (t * h * w) // merge_divisor)
            except Exception:
                return 0
        return 0

    if image_counts_per_sample is not None:
        counts = [max(0, int(value)) for value in image_counts_per_sample]
        if len(counts) != int(batch_size):
            raise ValueError(
                f"Expected image_counts_per_sample length {batch_size}, got {len(counts)}."
            )
        if sum(counts) != len(rows):
            raise ValueError(
                "image_counts_per_sample does not match image_grid_thw rows. "
                f"sum(image_counts_per_sample)={sum(counts)}, grid_rows={len(rows)}."
            )
        spans: list[tuple[int, int]] = []
        row_offset = 0
        for image_count in counts:
            sample_rows = rows[row_offset : row_offset + image_count]
            token_counts = [row_count(row) for row in sample_rows]
            if select_last_image_per_sample and token_counts:
                start = sum(token_counts[:-1])
                count = token_counts[-1]
            else:
                start = 0
                count = sum(token_counts)
            spans.append((start, count))
            row_offset += image_count
        return spans

    visual_token_counts = extract_visual_token_counts_per_sample(
        image_grid_thw=image_grid_thw,
        batch_size=batch_size,
        spatial_merge_size=spatial_merge_size,
    )
    return [(0, int(count)) for count in visual_token_counts]


def extract_visual_token_counts_per_sample(
    *,
    image_grid_thw: Any,
    batch_size: int,
    spatial_merge_size: int,
) -> list[int]:
    if image_grid_thw is None:
        return []
    rows = image_grid_thw.tolist() if hasattr(image_grid_thw, "tolist") else image_grid_thw
    merge_divisor = max(1, int(spatial_merge_size) ** 2)
    if batch_size <= 1:
        return [extract_visual_token_count(rows, spatial_merge_size=spatial_merge_size)]
    if len(rows) != batch_size:
        raise ValueError(
            "Batched visual token extraction currently expects one image grid row per sample. "
            f"Got {len(rows)} grid rows for batch_size={batch_size}."
        )
    counts: list[int] = []
    for row in rows:
        if isinstance(row, (list, tuple)) and len(row) >= 3:
            try:
                t, h, w = int(row[0]), int(row[1]), int(row[2])
                counts.append(max(0, (t * h * w) // merge_divisor))
            except Exception:
                counts.append(0)
        else:
            counts.append(0)
    return counts


def extract_visual_token_count(image_grid_thw: Any, *, spatial_merge_size: int = 2) -> int:
    if image_grid_thw is None:
        return 0
    rows = image_grid_thw.tolist() if hasattr(image_grid_thw, "tolist") else image_grid_thw
    total = 0
    merge_divisor = max(1, int(spatial_merge_size) ** 2)
    for row in rows:
        if isinstance(row, (list, tuple)) and len(row) >= 3:
            try:
                t, h, w = int(row[0]), int(row[1]), int(row[2])
                total += max(0, (t * h * w) // merge_divisor)
            except Exception:
                continue
    return total


def masked_visual_token_mean(
    visual_tokens: Any,
    visual_token_mask: Any | None = None,
) -> Any:
    _require_torch()
    if visual_tokens.ndim != 3:
        raise ValueError(f"Expected visual tokens with shape [batch, num_tokens, hidden_dim], got {tuple(visual_tokens.shape)}")
    if visual_token_mask is None:
        return visual_tokens.mean(dim=1)
    if visual_token_mask.ndim != 2:
        raise ValueError(f"Expected visual token mask with shape [batch, num_tokens], got {tuple(visual_token_mask.shape)}")
    mask = visual_token_mask.to(device=visual_tokens.device, dtype=visual_tokens.dtype).unsqueeze(-1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return (visual_tokens * mask).sum(dim=1) / denom
