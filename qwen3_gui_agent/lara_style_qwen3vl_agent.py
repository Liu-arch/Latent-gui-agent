from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from torch import Tensor, nn
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    Image = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[misc,assignment]

    class _NNNamespace:
        Module = object
        Linear = object
        Sequential = object
        LayerNorm = object
        GELU = object

    nn = _NNNamespace()  # type: ignore[assignment]
    AutoProcessor = None  # type: ignore[assignment]
    Qwen3VLForConditionalGeneration = None  # type: ignore[assignment]

from qwen3_gui_agent.gui_action_tokenizer import GUIActionTokenizer
from qwen3_gui_agent.flow_matching_action_head import (
    FlowMatchingActionHeadOutput,
    GUIFlowMatchingActionHead,
)
from qwen3_gui_agent.latent_two_way_action_head import LatentTwoWayActionHead
from qwen3_gui_agent.qwen3vl_pixel_pruned_visual import Qwen3VLPixelPrunedVisualWrapper
from qwen3_gui_agent.unified_action_head import (
    UnifiedActionHead,
    UnifiedActionHeadOutput,
    resolve_grid_sizes_from_image_grid,
    target_patch_index_from_point,
)
from qwen3_gui_agent.visual_latent_compressor import extract_visual_token_sequence_batch_with_mask


def _require_runtime() -> None:
    if torch is None or AutoProcessor is None or Qwen3VLForConditionalGeneration is None or Image is None:
        raise ImportError("LaRA-style Qwen3-VL agent requires torch, pillow, and transformers.")


ACTION_TYPES = [
    "click",
    "double_click",
    "right_click",
    "type",
    "hotkey",
    "scroll",
    "wait",
    "terminate",
]

REGION_LABELS = [
    "top_left",
    "top_center",
    "top_right",
    "middle_left",
    "middle_center",
    "middle_right",
    "bottom_left",
    "bottom_center",
    "bottom_right",
]

TERMINATE_STATUSES = ["success", "failure"]
POINTER_ACTION_TYPES = {"click", "double_click", "right_click"}
REASONING_FIELD_NAMES = ("actual_task", "thought", "reflection")
REASONING_ALIGNMENT_MODES = {"aggregate", "field_aligned"}


def resolve_reasoning_field_slot_counts(
    value: str | list[int] | tuple[int, ...] | None,
    *,
    latent_slot_count: int,
) -> tuple[int, int, int]:
    """Resolve three positive field budgets whose sum is the latent slot count."""
    total = int(latent_slot_count)
    if total < len(REASONING_FIELD_NAMES):
        raise ValueError(
            f"latent_slot_count must be at least {len(REASONING_FIELD_NAMES)} for field alignment; got {total}."
        )
    if value is None or (isinstance(value, str) and value.strip().lower() in {"", "auto"}):
        base, remainder = divmod(total, len(REASONING_FIELD_NAMES))
        counts = tuple(base + (1 if index < remainder else 0) for index in range(len(REASONING_FIELD_NAMES)))
    elif isinstance(value, str):
        try:
            counts = tuple(int(part.strip()) for part in value.split(","))
        except ValueError as exc:
            raise ValueError(
                "reasoning_field_slot_counts must be 'auto' or three comma-separated integers."
            ) from exc
    else:
        counts = tuple(int(part) for part in value)
    if len(counts) != len(REASONING_FIELD_NAMES):
        raise ValueError(
            "reasoning_field_slot_counts must contain exactly three values for "
            "actual_task, thought, and reflection."
        )
    if any(count <= 0 for count in counts):
        raise ValueError(f"reasoning_field_slot_counts must be positive; got {counts}.")
    if sum(counts) != total:
        raise ValueError(
            f"reasoning_field_slot_counts must sum to latent_slot_count={total}; got {counts}."
        )
    return counts  # type: ignore[return-value]


@dataclass
class LaRAForwardOutput:
    loss: Tensor | None
    hidden_states: Tensor
    latent_reasoning_states: Tensor
    latent_reasoning_summary: Tensor
    img_next_state: Tensor
    predicted_future_frame: Tensor
    target_future_frame: Tensor | None
    reasoning_teacher_embedding: Tensor | None
    latent_reasoning_field_summaries: dict[str, Tensor] | None = None
    reasoning_teacher_field_embeddings: dict[str, Tensor] | None = None
    action_head_output: UnifiedActionHeadOutput | None = None
    flow_action_head_output: FlowMatchingActionHeadOutput | None = None
    gold_action: dict[str, Any] | None = None
    action_text: str | None = None
    debug_info: dict[str, Any] | None = None


class LaRAStyleQwen3VLAgent(nn.Module):
    """
    LaRA-style GUI agent with optional latent-state action decoders:
    - keep official Qwen3-VL backbone
    - keep pixel prune / temporal reuse in visual tower
    - learn explicit-to-latent reasoning slots across Stage 1/2
    - support native generation, legacy action heads, and a latent two-way
      decoder that grounds Stage-2 reasoning states in current visual patches
    """

    def __init__(
        self,
        *,
        model: Qwen3VLForConditionalGeneration,
        processor: AutoProcessor,
        latent_slot_count: int = 8,
        pixel_prune_threshold: float = 0.0,
        pixel_prune_predictor_order: str = "pred2d,left,up",
        pixel_temporal_reuse: bool = False,
        pixel_temporal_threshold: float = 0.0,
        action_coord_bins: int = 1000,
        action_model: str = "unified",
        flow_action_sample_steps: int = 8,
        flow_head_hidden_dim: int | None = None,
        flow_head_depth: int = 2,
        two_way_hidden_dim: int = 512,
        two_way_depth: int = 2,
        two_way_num_heads: int = 8,
        two_way_location_queries: int = 3,
        two_way_dropout: float = 0.0,
        two_way_query_mode: str = "semantic_pool",
        image_min_pixels: int | None = None,
        image_max_pixels: int | None = None,
        include_current_subtask_in_prompt: bool = True,
        include_expected_next_screen_in_prompt: bool = True,
        latent_scaffolds_in_prompt: bool = True,
        reasoning_alignment_mode: str = "aggregate",
        reasoning_field_slot_counts: str | list[int] | tuple[int, ...] | None = None,
        use_lora: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
    ) -> None:
        _require_runtime()
        super().__init__()
        self.model = model
        self.processor = processor
        self.latent_slot_count = int(latent_slot_count)
        self.hidden_size = int(model.config.text_config.hidden_size)

        self.bot_token = "<BOT_LATENT>"
        self.eot_token = "<EOT_LATENT>"
        self.public_future_slot_token = "<img next>"
        self.public_thinking_start_token = "<|startofthinking|>"
        self.public_thinking_token = "<|thinking|>"
        self.public_thinking_end_token = "<|endofthinking|>"
        self.latent_slot_tokens = [f"<LATENT_{idx}>" for idx in range(self.latent_slot_count)]
        self.future_slot_tokens = [f"<IMG_NEXT_{idx}>" for idx in range(self.latent_slot_count)]
        self.gui_action_tokenizer = GUIActionTokenizer(coord_bins=action_coord_bins)
        self.action_coord_bins = int(action_coord_bins)
        self.special_tokens = (
            [
                self.bot_token,
                self.eot_token,
                self.public_thinking_start_token,
                self.public_thinking_end_token,
            ]
            + list(self.latent_slot_tokens)
            + list(self.future_slot_tokens)
            + list(self.gui_action_tokenizer.special_tokens)
        )
        self._install_special_tokens()
        self._install_trainable_special_token_embeddings()

        original_visual = self.model.model.visual
        self.model.model.visual = Qwen3VLPixelPrunedVisualWrapper(
            original_visual,
            pixel_prune_threshold=pixel_prune_threshold,
            pixel_prune_predictor_order=pixel_prune_predictor_order,
            pixel_temporal_reuse=pixel_temporal_reuse,
            pixel_temporal_threshold=pixel_temporal_threshold,
        )
        self.pixel_prune_threshold = float(pixel_prune_threshold)
        self.pixel_prune_predictor_order = str(pixel_prune_predictor_order)
        self.pixel_temporal_reuse = bool(pixel_temporal_reuse)
        self.pixel_temporal_threshold = float(pixel_temporal_threshold)

        self.reasoning_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.future_frame_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.reasoning_norm = nn.LayerNorm(self.hidden_size)
        self.action_types = list(ACTION_TYPES)
        self.region_labels = list(REGION_LABELS)
        self.terminate_statuses = list(TERMINATE_STATUSES)
        self.action_model = str(action_model or "unified")
        self.flow_action_sample_steps = int(flow_action_sample_steps)
        self.flow_head_hidden_dim = int(flow_head_hidden_dim) if flow_head_hidden_dim else None
        self.flow_head_depth = max(1, int(flow_head_depth))
        self.two_way_hidden_dim = max(64, int(two_way_hidden_dim))
        self.two_way_depth = max(1, int(two_way_depth))
        self.two_way_num_heads = max(1, int(two_way_num_heads))
        self.two_way_location_queries = max(1, int(two_way_location_queries))
        self.two_way_dropout = max(0.0, float(two_way_dropout))
        self.two_way_query_mode = str(two_way_query_mode or "semantic_pool")
        self.two_way_candidate_coord_loss_weight = 1.0
        self.two_way_candidate_confidence_loss_weight = 0.25
        self.flow_continuous_source = "sample"
        self.flow_action_loss_weight = 1.0
        self.flow_coord_loss_weight = 0.0
        self.flow_coord_loss_scale = 1.0
        self.flow_coord_loss_space = "logit"
        self.flow_patch_loss_mode = "ce"
        self.flow_patch_gaussian_sigma = 0.05
        self.flow_pointer_coord_source = "patch_residual"
        self.flow_patch_logit_temperature = 1.0
        self.flow_patch_residual_scale = 1.0
        self.flow_coord_loss_log_var = nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.action_hidden_source = "summary"
        self.action_prompt_query = nn.Parameter(torch.empty(self.hidden_size, dtype=torch.float32))
        self.action_slot_query = nn.Parameter(torch.empty(self.hidden_size, dtype=torch.float32))
        nn.init.normal_(self.action_prompt_query, mean=0.0, std=0.02)
        nn.init.normal_(self.action_slot_query, mean=0.0, std=0.02)
        self.include_flow_alternatives = False
        self.include_flow_training_sample_metrics = False
        self.image_min_pixels = int(image_min_pixels) if image_min_pixels else None
        self.image_max_pixels = int(image_max_pixels) if image_max_pixels else None
        self.include_current_subtask_in_prompt = bool(include_current_subtask_in_prompt)
        self.include_expected_next_screen_in_prompt = bool(include_expected_next_screen_in_prompt)
        self.latent_scaffolds_in_prompt = bool(latent_scaffolds_in_prompt)
        self.set_reasoning_alignment_config(
            mode=reasoning_alignment_mode,
            field_slot_counts=reasoning_field_slot_counts,
        )
        self.use_lora = False
        self.lora_r = max(1, int(lora_r))
        self.lora_alpha = max(1, int(lora_alpha))
        self.lora_dropout = max(0.0, float(lora_dropout))
        self._init_action_head()
        if use_lora:
            self.enable_lora(
                r=self.lora_r,
                alpha=self.lora_alpha,
                dropout=self.lora_dropout,
            )
        self.action_only_output = False
        self.use_action_head = False
        self.action_format = "text"
        self.lm_action_target = "include"
        self.stage2_target_format = "mixed_reasoning_action"
        self.training_stage = "stage1"

    def set_reasoning_alignment_config(
        self,
        *,
        mode: str,
        field_slot_counts: str | list[int] | tuple[int, ...] | None = None,
    ) -> None:
        normalized_mode = str(mode or "aggregate").strip().lower()
        if normalized_mode not in REASONING_ALIGNMENT_MODES:
            raise ValueError(
                f"Unsupported reasoning alignment mode {mode!r}; expected one of {sorted(REASONING_ALIGNMENT_MODES)}."
            )
        self.reasoning_alignment_mode = normalized_mode
        if normalized_mode == "aggregate" and self.latent_slot_count < len(REASONING_FIELD_NAMES):
            # Legacy aggregate runs may use one or two slots. Field partitions
            # are inactive in that mode, so preserve those configurations.
            self.reasoning_field_slot_counts = (
                self.latent_slot_count,
                0,
                0,
            )
        else:
            self.reasoning_field_slot_counts = resolve_reasoning_field_slot_counts(
                field_slot_counts,
                latent_slot_count=self.latent_slot_count,
            )

    def _reasoning_field_slot_ranges(self) -> dict[str, tuple[int, int]]:
        ranges: dict[str, tuple[int, int]] = {}
        offset = 0
        for field_name, count in zip(REASONING_FIELD_NAMES, self.reasoning_field_slot_counts):
            ranges[field_name] = (offset, offset + int(count))
            offset += int(count)
        return ranges

    def _init_action_head(self) -> None:
        fused_dim = self.hidden_size * 3
        self.action_state_norm = nn.LayerNorm(fused_dim)
        self.action_head = UnifiedActionHead(
            fused_dim=fused_dim,
            visual_dim=self.hidden_size,
            action_type_count=len(self.action_types),
            terminate_count=len(self.terminate_statuses),
            region_count=len(self.region_labels),
        )
        self.flow_action_head = GUIFlowMatchingActionHead(
            fused_dim=fused_dim,
            action_type_count=len(self.action_types),
            terminate_count=len(self.terminate_statuses),
            visual_dim=self.hidden_size,
            hidden_dim=self.flow_head_hidden_dim,
            head_depth=self.flow_head_depth,
        )
        self.latent_two_way_action_head = LatentTwoWayActionHead(
            input_dim=self.hidden_size,
            action_type_count=len(self.action_types),
            terminate_count=len(self.terminate_statuses),
            hidden_dim=self.two_way_hidden_dim,
            depth=self.two_way_depth,
            num_heads=self.two_way_num_heads,
            location_query_count=self.two_way_location_queries,
            max_latent_tokens=self.latent_slot_count,
            dropout=self.two_way_dropout,
            query_mode=self.two_way_query_mode,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        *,
        device_map: str | dict[str, Any] | None = None,
        torch_dtype: str | torch.dtype | None = None,
        attn_implementation: str | None = None,
        latent_slot_count: int = 8,
        pixel_prune_threshold: float = 0.0,
        pixel_prune_predictor_order: str = "pred2d,left,up",
        pixel_temporal_reuse: bool = False,
        pixel_temporal_threshold: float = 0.0,
        action_coord_bins: int = 1000,
        action_model: str = "unified",
        flow_action_sample_steps: int = 8,
        flow_head_hidden_dim: int | None = None,
        flow_head_depth: int = 2,
        two_way_hidden_dim: int = 512,
        two_way_depth: int = 2,
        two_way_num_heads: int = 8,
        two_way_location_queries: int = 3,
        two_way_dropout: float = 0.0,
        two_way_query_mode: str = "semantic_pool",
        image_min_pixels: int | None = None,
        image_max_pixels: int | None = None,
        include_current_subtask_in_prompt: bool = True,
        include_expected_next_screen_in_prompt: bool = True,
        latent_scaffolds_in_prompt: bool = True,
        reasoning_alignment_mode: str = "aggregate",
        reasoning_field_slot_counts: str | list[int] | tuple[int, ...] | None = None,
        use_lora: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        **kwargs: Any,
    ) -> "LaRAStyleQwen3VLAgent":
        _require_runtime()
        load_kwargs: dict[str, Any] = dict(kwargs)
        if device_map not in (None, "", "none"):
            load_kwargs["device_map"] = device_map
        if torch_dtype is not None:
            load_kwargs["dtype"] = torch_dtype
        resolved_attn_implementation = (
            attn_implementation
            or os.environ.get("LARA_ATTN_IMPLEMENTATION", "sdpa")
        ).strip()
        if resolved_attn_implementation:
            load_kwargs["attn_implementation"] = resolved_attn_implementation
        model = Qwen3VLForConditionalGeneration.from_pretrained(model_name_or_path, **load_kwargs)
        processor_kwargs: dict[str, Any] = {}
        if image_min_pixels and int(image_min_pixels) > 0:
            processor_kwargs["min_pixels"] = int(image_min_pixels)
        if image_max_pixels and int(image_max_pixels) > 0:
            processor_kwargs["max_pixels"] = int(image_max_pixels)
        try:
            processor = AutoProcessor.from_pretrained(model_name_or_path, **processor_kwargs)
        except TypeError:
            processor = AutoProcessor.from_pretrained(model_name_or_path)
        cls._configure_processor_pixel_budget(
            processor,
            image_min_pixels=int(image_min_pixels) if image_min_pixels else None,
            image_max_pixels=int(image_max_pixels) if image_max_pixels else None,
        )
        return cls(
            model=model,
            processor=processor,
            latent_slot_count=latent_slot_count,
            pixel_prune_threshold=pixel_prune_threshold,
            pixel_prune_predictor_order=pixel_prune_predictor_order,
            pixel_temporal_reuse=pixel_temporal_reuse,
            pixel_temporal_threshold=pixel_temporal_threshold,
            action_coord_bins=action_coord_bins,
            action_model=action_model,
            flow_action_sample_steps=flow_action_sample_steps,
            flow_head_hidden_dim=flow_head_hidden_dim,
            flow_head_depth=flow_head_depth,
            two_way_hidden_dim=two_way_hidden_dim,
            two_way_depth=two_way_depth,
            two_way_num_heads=two_way_num_heads,
            two_way_location_queries=two_way_location_queries,
            two_way_dropout=two_way_dropout,
            two_way_query_mode=two_way_query_mode,
            image_min_pixels=image_min_pixels,
            image_max_pixels=image_max_pixels,
            include_current_subtask_in_prompt=include_current_subtask_in_prompt,
            include_expected_next_screen_in_prompt=include_expected_next_screen_in_prompt,
            latent_scaffolds_in_prompt=latent_scaffolds_in_prompt,
            reasoning_alignment_mode=reasoning_alignment_mode,
            reasoning_field_slot_counts=reasoning_field_slot_counts,
            use_lora=use_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )

    def enable_lora(self, *, r: int, alpha: int, dropout: float) -> None:
        if any("lora_" in name for name, _ in self.model.named_parameters()):
            self.use_lora = True
            return
        try:
            from peft import LoraConfig
            try:
                from peft import inject_adapter_in_model
            except ImportError:  # Older PEFT versions expose it from peft.mapping.
                from peft.mapping import inject_adapter_in_model
        except ImportError as exc:  # pragma: no cover - depends on the cluster runtime
            raise ImportError("--use-lora requires the peft package.") from exc

        config = LoraConfig(
            r=max(1, int(r)),
            lora_alpha=max(1, int(alpha)),
            lora_dropout=max(0.0, float(dropout)),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        # Injection preserves Qwen3VLForConditionalGeneration's module layout,
        # unlike wrapping it in PeftModel, so the custom visual forward path can
        # continue to call self.model.model directly.
        self.model = inject_adapter_in_model(config, self.model)
        self.use_lora = True
        self.lora_r = max(1, int(r))
        self.lora_alpha = max(1, int(alpha))
        self.lora_dropout = max(0.0, float(dropout))

    @staticmethod
    def _configure_processor_pixel_budget(
        processor: Any,
        *,
        image_min_pixels: int | None,
        image_max_pixels: int | None,
    ) -> None:
        image_processor = getattr(processor, "image_processor", None)
        targets = [processor]
        if image_processor is not None:
            targets.append(image_processor)
        for target in targets:
            if image_min_pixels and int(image_min_pixels) > 0:
                try:
                    setattr(target, "min_pixels", int(image_min_pixels))
                except Exception:
                    pass
            if image_max_pixels and int(image_max_pixels) > 0:
                try:
                    setattr(target, "max_pixels", int(image_max_pixels))
                except Exception:
                    pass

    @property
    def device(self) -> torch.device:
        embedding = self.model.get_input_embeddings()
        weight = getattr(embedding, "weight", None)
        if weight is not None:
            return weight.device
        return next(self.parameters()).device

    def _align_auxiliary_modules(self, reference: Tensor) -> None:
        target_device = reference.device
        target_dtype = reference.dtype
        for module in (
            self.reasoning_proj,
            self.future_frame_head,
            self.reasoning_norm,
            self.action_state_norm,
            self.action_head,
            self.flow_action_head,
            self.latent_two_way_action_head,
        ):
            first_param = next(module.parameters(), None)
            if first_param is None:
                continue
            if first_param.device != target_device or first_param.dtype != target_dtype:
                module.to(device=target_device, dtype=target_dtype)
        for attr_name in ("action_prompt_query", "action_slot_query"):
            parameter = getattr(self, attr_name, None)
            if parameter is not None and (parameter.device != target_device or parameter.dtype != target_dtype):
                parameter.data = parameter.data.to(device=target_device, dtype=target_dtype)
        if self.flow_coord_loss_log_var.device != target_device:
            self.flow_coord_loss_log_var.data = self.flow_coord_loss_log_var.data.to(device=target_device)
        self.flow_action_head.pointer_coord_source = str(
            getattr(self, "flow_pointer_coord_source", "patch_residual") or "patch_residual"
        )
        self.flow_action_head.patch_logit_temperature = float(
            getattr(self, "flow_patch_logit_temperature", 1.0) or 1.0
        )
        self.flow_action_head.patch_residual_scale = float(
            getattr(self, "flow_patch_residual_scale", 1.0) or 1.0
        )
        self.latent_two_way_action_head.patch_logit_temperature = float(
            getattr(self, "flow_patch_logit_temperature", 1.0) or 1.0
        )
        self.latent_two_way_action_head.patch_residual_scale = float(
            getattr(self, "flow_patch_residual_scale", 1.0) or 1.0
        )

    def _install_special_tokens(self) -> None:
        tokenizer = self.processor.tokenizer
        added = tokenizer.add_special_tokens({"additional_special_tokens": self.special_tokens})
        if added > 0:
            self.model.resize_token_embeddings(len(tokenizer))
        token_ids = tokenizer.convert_tokens_to_ids(self.special_tokens)
        self.special_token_to_id = dict(zip(self.special_tokens, token_ids))

    def _install_trainable_special_token_embeddings(self) -> None:
        """Learn only the newly added token rows instead of the full vocabulary table."""
        embedding = self.model.get_input_embeddings()
        token_ids = torch.tensor(
            [self.special_token_to_id[token] for token in self.special_tokens],
            device=embedding.weight.device,
            dtype=torch.long,
        )
        initial_rows = embedding.weight.detach().index_select(0, token_ids).clone()
        self.special_token_embeddings = nn.Parameter(initial_rows)

        lookup = torch.full(
            (int(embedding.weight.shape[0]),),
            -1,
            device=embedding.weight.device,
            dtype=torch.long,
        )
        lookup[token_ids] = torch.arange(len(self.special_tokens), device=lookup.device, dtype=torch.long)
        self.register_buffer("_special_token_row_lookup", lookup, persistent=False)
        self._special_embedding_hook_handle = embedding.register_forward_hook(
            self._replace_special_token_embeddings
        )

    def _replace_special_token_embeddings(
        self,
        _module: nn.Module,
        module_inputs: tuple[Any, ...],
        output: Tensor,
    ) -> Tensor:
        if not module_inputs or not torch.is_tensor(module_inputs[0]):
            return output
        input_ids = module_inputs[0]
        lookup = self._special_token_row_lookup
        if lookup.device != input_ids.device:
            lookup = lookup.to(input_ids.device)
        row_indices = lookup[input_ids]
        special_mask = row_indices.ge(0)
        if not bool(special_mask.any()):
            return output
        rows = self.special_token_embeddings
        if rows.device != output.device or rows.dtype != output.dtype:
            rows = rows.to(device=output.device, dtype=output.dtype)
        replacements = F.embedding(row_indices.clamp_min(0), rows)
        return torch.where(special_mask.unsqueeze(-1), replacements, output)

    @staticmethod
    def _normalized_state_key(key: str) -> str:
        """Strip wrappers introduced by DDP or torch.compile."""
        normalized = str(key)
        prefixes = ("module.", "_orig_mod.")
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if normalized.startswith(prefix):
                    normalized = normalized[len(prefix) :]
                    changed = True
        return normalized

    @classmethod
    def _is_compact_adapter_key(cls, key: str) -> bool:
        """Keep auxiliary heads and LoRA tensors, never frozen base-VLM weights."""
        normalized = cls._normalized_state_key(key)
        if "lora_" in normalized:
            return True
        return not normalized.startswith("model.")

    def adapter_state_dict(self) -> dict[str, Tensor]:
        return {
            key: value
            for key, value in self.state_dict().items()
            if self._is_compact_adapter_key(key)
        }

    def save_adapter(self, path: str | Path, *, extra_metadata: dict[str, Any] | None = None) -> Path:
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "adapter_state_dict": self.adapter_state_dict(),
            "extra_metadata": extra_metadata or {},
            "special_tokens": list(self.special_tokens),
            "latent_slot_count": int(self.latent_slot_count),
            "action_coord_bins": int(self.action_coord_bins),
            "action_model": str(self.action_model),
            "lm_action_target": str(self.lm_action_target),
            "flow_action_sample_steps": int(self.flow_action_sample_steps),
            "flow_head_hidden_dim": self.flow_head_hidden_dim,
            "flow_head_depth": int(self.flow_head_depth),
            "two_way_hidden_dim": int(self.two_way_hidden_dim),
            "two_way_depth": int(self.two_way_depth),
            "two_way_num_heads": int(self.two_way_num_heads),
            "two_way_location_queries": int(self.two_way_location_queries),
            "two_way_dropout": float(self.two_way_dropout),
            "two_way_query_mode": str(self.two_way_query_mode),
            "two_way_candidate_coord_loss_weight": float(self.two_way_candidate_coord_loss_weight),
            "two_way_candidate_confidence_loss_weight": float(
                self.two_way_candidate_confidence_loss_weight
            ),
            "flow_pointer_coord_source": str(self.flow_pointer_coord_source),
            "flow_patch_logit_temperature": float(self.flow_patch_logit_temperature),
            "flow_patch_residual_scale": float(self.flow_patch_residual_scale),
            "action_hidden_source": str(self.action_hidden_source),
            "image_min_pixels": self.image_min_pixels,
            "image_max_pixels": self.image_max_pixels,
            "include_current_subtask_in_prompt": bool(self.include_current_subtask_in_prompt),
            "include_expected_next_screen_in_prompt": bool(self.include_expected_next_screen_in_prompt),
            "latent_scaffolds_in_prompt": bool(self.latent_scaffolds_in_prompt),
            "reasoning_alignment_mode": str(self.reasoning_alignment_mode),
            "reasoning_field_slot_counts": list(self.reasoning_field_slot_counts),
            "use_lora": bool(self.use_lora),
            "lora_r": int(self.lora_r),
            "lora_alpha": int(self.lora_alpha),
            "lora_dropout": float(self.lora_dropout),
        }
        temp_save_path = save_path.with_name(save_path.name + ".tmp")
        torch.save(payload, temp_save_path)
        os.replace(temp_save_path, save_path)
        meta_path = save_path.with_suffix(save_path.suffix + ".json")
        temp_meta_path = meta_path.with_name(meta_path.name + ".tmp")
        temp_meta_path.write_text(
            json.dumps(
                {
                    "latent_slot_count": int(self.latent_slot_count),
                    "action_coord_bins": int(self.action_coord_bins),
                    "action_model": str(self.action_model),
                    "lm_action_target": str(self.lm_action_target),
                    "flow_action_sample_steps": int(self.flow_action_sample_steps),
                    "flow_head_hidden_dim": self.flow_head_hidden_dim,
                    "flow_head_depth": int(self.flow_head_depth),
                    "two_way_hidden_dim": int(self.two_way_hidden_dim),
                    "two_way_depth": int(self.two_way_depth),
                    "two_way_num_heads": int(self.two_way_num_heads),
                    "two_way_location_queries": int(self.two_way_location_queries),
                    "two_way_dropout": float(self.two_way_dropout),
                    "two_way_query_mode": str(self.two_way_query_mode),
                    "two_way_candidate_coord_loss_weight": float(
                        self.two_way_candidate_coord_loss_weight
                    ),
                    "two_way_candidate_confidence_loss_weight": float(
                        self.two_way_candidate_confidence_loss_weight
                    ),
                    "flow_pointer_coord_source": str(self.flow_pointer_coord_source),
                    "flow_patch_logit_temperature": float(self.flow_patch_logit_temperature),
                    "flow_patch_residual_scale": float(self.flow_patch_residual_scale),
                    "action_hidden_source": str(self.action_hidden_source),
                    "image_min_pixels": self.image_min_pixels,
                    "image_max_pixels": self.image_max_pixels,
                    "include_current_subtask_in_prompt": bool(self.include_current_subtask_in_prompt),
                    "include_expected_next_screen_in_prompt": bool(self.include_expected_next_screen_in_prompt),
                    "latent_scaffolds_in_prompt": bool(self.latent_scaffolds_in_prompt),
                    "reasoning_alignment_mode": str(self.reasoning_alignment_mode),
                    "reasoning_field_slot_counts": list(self.reasoning_field_slot_counts),
                    "use_lora": bool(self.use_lora),
                    "lora_r": int(self.lora_r),
                    "lora_alpha": int(self.lora_alpha),
                    "lora_dropout": float(self.lora_dropout),
                    "special_tokens": list(self.special_tokens),
                    "extra_metadata": extra_metadata or {},
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        os.replace(temp_meta_path, meta_path)
        return save_path

    def load_adapter(self, path: str | Path, *, strict: bool = True) -> dict[str, Any]:
        payload = torch.load(Path(path), map_location="cpu")
        loaded_state = payload.get("adapter_state_dict")
        if loaded_state is None:
            loaded_state = payload.get("model_state_dict")
        if loaded_state is None:
            raise KeyError("Adapter payload must contain 'adapter_state_dict' or checkpoint 'model_state_dict'.")
        current_state = self.state_dict()
        compatible_state: dict[str, Tensor] = {}
        skipped_shape_mismatch: list[dict[str, Any]] = []
        skipped_base_model_keys: list[str] = []
        normalized_loaded_state = {
            self._normalized_state_key(key): value for key, value in loaded_state.items()
        }

        # Legacy checkpoints sometimes stored the complete resized embedding
        # table. Recover only the compact rows for our special tokens. Loading
        # the full embedding and tied lm-head into a device-mapped model is both
        # unnecessary and can trigger CUDA illegal-memory errors.
        if "special_token_embeddings" not in normalized_loaded_state:
            full_embedding = next(
                (
                    value
                    for key, value in normalized_loaded_state.items()
                    if key.endswith("language_model.embed_tokens.weight")
                ),
                None,
            )
            if full_embedding is not None:
                special_ids = torch.tensor(
                    [self.special_token_to_id[token] for token in self.special_tokens],
                    device="cpu",
                    dtype=torch.long,
                )
                if int(special_ids.max().item()) < int(full_embedding.shape[0]):
                    compatible_state["special_token_embeddings"] = (
                        full_embedding.detach().to("cpu").index_select(0, special_ids)
                    )

        for key, value in normalized_loaded_state.items():
            if not self._is_compact_adapter_key(key):
                skipped_base_model_keys.append(str(key))
                continue
            if key in current_state and tuple(current_state[key].shape) != tuple(value.shape):
                skipped_shape_mismatch.append(
                    {
                        "key": key,
                        "loaded_shape": list(value.shape),
                        "current_shape": list(current_state[key].shape),
                    }
                )
                continue
            compatible_state[key] = value
        result = self.load_state_dict(compatible_state, strict=False)
        if "special_token_embeddings" not in compatible_state:
            # Backward compatibility for older checkpoints that stored the
            # complete resized embedding table instead of compact token rows.
            embedding_weight = self.model.get_input_embeddings().weight
            special_ids = torch.tensor(
                [self.special_token_to_id[token] for token in self.special_tokens],
                device=embedding_weight.device,
                dtype=torch.long,
            )
            with torch.no_grad():
                self.special_token_embeddings.copy_(
                    embedding_weight.detach().index_select(0, special_ids).to(
                        device=self.special_token_embeddings.device,
                        dtype=self.special_token_embeddings.dtype,
                    )
                )
        extra_metadata = payload.get("extra_metadata", {}) or {}
        if not extra_metadata and isinstance(payload.get("args"), dict):
            checkpoint_args = dict(payload.get("args") or {})
            extra_metadata = {
                key: checkpoint_args[key]
                for key in (
                    "history_n",
                    "latent_slot_count",
                    "pixel_prune_threshold",
                    "pixel_prune_predictor_order",
                    "pixel_temporal_reuse",
                    "pixel_temporal_threshold",
                    "image_min_pixels",
                    "image_max_pixels",
                    "training_stage",
                    "stage2_target_format",
                    "action_format",
                    "action_model",
                    "lm_action_target",
                    "flow_action_sample_steps",
                    "flow_head_hidden_dim",
                    "flow_head_depth",
                    "two_way_hidden_dim",
                    "two_way_depth",
                    "two_way_num_heads",
                    "two_way_location_queries",
                    "two_way_dropout",
                    "two_way_query_mode",
                    "two_way_candidate_coord_loss_weight",
                    "two_way_candidate_confidence_loss_weight",
                    "flow_coord_loss_scale",
                    "flow_coord_loss_space",
                    "flow_patch_loss_mode",
                    "flow_patch_gaussian_sigma",
                    "flow_pointer_coord_source",
                    "flow_patch_logit_temperature",
                    "flow_patch_residual_scale",
                    "action_hidden_source",
                    "learnable_flow_coord_weight",
                    "clean_observable_prompt",
                    "include_current_subtask_in_prompt",
                    "include_expected_next_screen_in_prompt",
                    "latent_scaffolds_in_prompt",
                    "reasoning_alignment_mode",
                    "reasoning_field_slot_counts",
                    "use_lora",
                    "lora_r",
                    "lora_alpha",
                    "lora_dropout",
                )
                if key in checkpoint_args
            }
        self.training_stage = str(extra_metadata.get("training_stage", "stage1"))
        if bool(extra_metadata.get("clean_observable_prompt", False)):
            self.include_current_subtask_in_prompt = False
            self.include_expected_next_screen_in_prompt = False
            self.latent_scaffolds_in_prompt = False
        self.include_current_subtask_in_prompt = bool(
            extra_metadata.get("include_current_subtask_in_prompt", self.include_current_subtask_in_prompt)
        )
        self.include_expected_next_screen_in_prompt = bool(
            extra_metadata.get("include_expected_next_screen_in_prompt", self.include_expected_next_screen_in_prompt)
        )
        self.latent_scaffolds_in_prompt = bool(
            extra_metadata.get("latent_scaffolds_in_prompt", self.latent_scaffolds_in_prompt)
        )
        self.set_reasoning_alignment_config(
            mode=str(
                extra_metadata.get(
                    "reasoning_alignment_mode",
                    payload.get("reasoning_alignment_mode", self.reasoning_alignment_mode),
                )
            ),
            field_slot_counts=extra_metadata.get(
                "reasoning_field_slot_counts",
                payload.get("reasoning_field_slot_counts", self.reasoning_field_slot_counts),
            ),
        )
        self.use_lora = bool(extra_metadata.get("use_lora", self.use_lora))
        self.lora_r = int(extra_metadata.get("lora_r", self.lora_r) or self.lora_r)
        self.lora_alpha = int(extra_metadata.get("lora_alpha", self.lora_alpha) or self.lora_alpha)
        self.lora_dropout = float(extra_metadata.get("lora_dropout", self.lora_dropout) or self.lora_dropout)
        self.stage2_target_format = str(extra_metadata.get("stage2_target_format", "mixed_reasoning_action"))
        self.action_only_output = bool(extra_metadata.get("action_only_output", False))
        self.use_action_head = bool(extra_metadata.get("use_action_head", False))
        self.action_format = str(extra_metadata.get("action_format", "text") or "text")
        self.action_model = str(extra_metadata.get("action_model", self.action_model) or self.action_model)
        self.lm_action_target = str(extra_metadata.get("lm_action_target", self.lm_action_target) or self.lm_action_target)
        self.flow_action_sample_steps = int(
            extra_metadata.get("flow_action_sample_steps", self.flow_action_sample_steps)
            or self.flow_action_sample_steps
        )
        self.flow_head_hidden_dim = (
            int(extra_metadata["flow_head_hidden_dim"])
            if extra_metadata.get("flow_head_hidden_dim")
            else self.flow_head_hidden_dim
        )
        self.flow_head_depth = max(
            1,
            int(extra_metadata.get("flow_head_depth", self.flow_head_depth) or self.flow_head_depth),
        )
        self.two_way_hidden_dim = max(
            64,
            int(extra_metadata.get("two_way_hidden_dim", self.two_way_hidden_dim) or self.two_way_hidden_dim),
        )
        self.two_way_depth = max(
            1,
            int(extra_metadata.get("two_way_depth", self.two_way_depth) or self.two_way_depth),
        )
        self.two_way_num_heads = max(
            1,
            int(extra_metadata.get("two_way_num_heads", self.two_way_num_heads) or self.two_way_num_heads),
        )
        self.two_way_location_queries = max(
            1,
            int(
                extra_metadata.get("two_way_location_queries", self.two_way_location_queries)
                or self.two_way_location_queries
            ),
        )
        self.two_way_dropout = max(
            0.0,
            float(extra_metadata.get("two_way_dropout", self.two_way_dropout) or self.two_way_dropout),
        )
        self.two_way_query_mode = str(
            extra_metadata.get(
                "two_way_query_mode",
                payload.get("two_way_query_mode", self.two_way_query_mode),
            )
            or self.two_way_query_mode
        )
        self.latent_two_way_action_head.query_mode = self.two_way_query_mode
        self.two_way_candidate_coord_loss_weight = max(
            0.0,
            float(
                extra_metadata.get(
                    "two_way_candidate_coord_loss_weight",
                    self.two_way_candidate_coord_loss_weight,
                )
            ),
        )
        self.two_way_candidate_confidence_loss_weight = max(
            0.0,
            float(
                extra_metadata.get(
                    "two_way_candidate_confidence_loss_weight",
                    self.two_way_candidate_confidence_loss_weight,
                )
            ),
        )
        self.flow_coord_loss_scale = max(
            1.0,
            float(extra_metadata.get("flow_coord_loss_scale", self.flow_coord_loss_scale) or self.flow_coord_loss_scale),
        )
        self.flow_coord_loss_space = str(
            extra_metadata.get("flow_coord_loss_space", self.flow_coord_loss_space) or self.flow_coord_loss_space
        )
        self.flow_patch_loss_mode = str(
            extra_metadata.get("flow_patch_loss_mode", self.flow_patch_loss_mode) or self.flow_patch_loss_mode
        )
        self.flow_patch_gaussian_sigma = max(
            1e-4,
            float(
                extra_metadata.get("flow_patch_gaussian_sigma", self.flow_patch_gaussian_sigma)
                or self.flow_patch_gaussian_sigma
            ),
        )
        self.flow_pointer_coord_source = str(
            extra_metadata.get(
                "flow_pointer_coord_source",
                payload.get("flow_pointer_coord_source", self.flow_pointer_coord_source),
            )
            or self.flow_pointer_coord_source
        )
        self.flow_patch_logit_temperature = max(
            1e-4,
            float(
                extra_metadata.get(
                    "flow_patch_logit_temperature",
                    payload.get("flow_patch_logit_temperature", self.flow_patch_logit_temperature),
                )
                or self.flow_patch_logit_temperature
            ),
        )
        self.flow_patch_residual_scale = max(
            0.0,
            float(
                extra_metadata.get(
                    "flow_patch_residual_scale",
                    payload.get("flow_patch_residual_scale", self.flow_patch_residual_scale),
                )
                or self.flow_patch_residual_scale
            ),
        )
        self.action_hidden_source = str(
            extra_metadata.get(
                "action_hidden_source",
                payload.get("action_hidden_source", self.action_hidden_source),
            )
            or self.action_hidden_source
        )
        image_min_pixels = extra_metadata.get("image_min_pixels", payload.get("image_min_pixels", self.image_min_pixels))
        image_max_pixels = extra_metadata.get("image_max_pixels", payload.get("image_max_pixels", self.image_max_pixels))
        self.image_min_pixels = int(image_min_pixels) if image_min_pixels else None
        self.image_max_pixels = int(image_max_pixels) if image_max_pixels else None
        self._configure_processor_pixel_budget(
            self.processor,
            image_min_pixels=self.image_min_pixels,
            image_max_pixels=self.image_max_pixels,
        )
        if strict:
            filtered_missing = [name for name in result.missing_keys if not name.startswith("model.")]
            filtered_unexpected = [name for name in result.unexpected_keys if not name.startswith("model.")]
            if filtered_missing or filtered_unexpected:
                raise RuntimeError(
                    f"Adapter load mismatch. Missing={filtered_missing} Unexpected={filtered_unexpected}"
                )
        return {
            "missing_keys": result.missing_keys,
            "unexpected_keys": result.unexpected_keys,
            "skipped_shape_mismatch": skipped_shape_mismatch,
            "skipped_base_model_key_count": len(skipped_base_model_keys),
            "skipped_base_model_keys_preview": skipped_base_model_keys[:8],
            "extra_metadata": extra_metadata,
        }

    def _set_temporal_context(self, sample_keys: list[str] | None) -> None:
        visual_module = getattr(self.model.model, "visual", None)
        if visual_module is not None and hasattr(visual_module, "set_temporal_batch_context"):
            visual_module.set_temporal_batch_context(sample_keys)

    def _move_inputs_to_device(self, inputs: dict[str, Any]) -> dict[str, Any]:
        moved: dict[str, Any] = {}
        for key, value in inputs.items():
            if hasattr(value, "to"):
                moved[key] = value.to(self.device)
            else:
                moved[key] = value
        return moved

    @staticmethod
    def _load_image(screenshot: str | Path | Image.Image) -> Image.Image:
        if isinstance(screenshot, Image.Image):
            return screenshot
        return Image.open(screenshot).convert("RGB")

    def latent_scaffold_text(self) -> str:
        slots = " ".join(self.latent_slot_tokens)
        return f"{self.bot_token} {slots} {self.eot_token}"

    def thinking_scaffold_text(self, token_count: int) -> str:
        count = max(1, min(int(token_count), self.latent_slot_count))
        body = " ".join([self.public_thinking_token] * count)
        return f"{self.public_thinking_start_token} {body} {self.public_thinking_end_token}"

    def future_scaffold_text(self) -> str:
        return " ".join([self.public_future_slot_token] * self.latent_slot_count)

    def action_head_assistant_prefix(self) -> str | None:
        if str(getattr(self, "training_stage", "stage1")) != "stage2":
            return None
        return "\n".join(
            [
                "Reasoning:",
                self.thinking_scaffold_text(self.latent_slot_count),
                self.future_scaffold_text(),
            ]
        )

    def _expand_public_future_slots(self, text: str) -> str:
        if not text or self.public_future_slot_token not in text:
            return text
        expanded = str(text)
        for private_token in self.future_slot_tokens:
            expanded = expanded.replace(self.public_future_slot_token, private_token, 1)
        return expanded

    def _expand_public_thinking_slots(self, text: str) -> tuple[str, list[str]]:
        if not text or self.public_thinking_token not in text:
            return text, []
        expanded = str(text)
        used_latent_tokens: list[str] = []
        for private_token in self.latent_slot_tokens:
            if self.public_thinking_token not in expanded:
                break
            expanded = expanded.replace(self.public_thinking_token, private_token, 1)
            used_latent_tokens.append(private_token)
        return expanded, used_latent_tokens

    def build_user_prompt(
        self,
        *,
        task: str,
        history_frame_count: int,
        current_subtask: str | None,
        expected_next_screen: str | None,
        action_only_output: bool = False,
        action_format: str = "text",
        include_action_in_lm: bool = True,
    ) -> str:
        lines = [
            "You are a GUI agent.",
            f"instruction: {task}",
        ]
        if history_frame_count > 0:
            lines.append(
                f"There are {history_frame_count} history screenshots first, and the last screenshot is the current screenshot."
            )
        else:
            lines.append("The screenshot is the current screenshot.")
        visible_current_subtask = current_subtask if self.include_current_subtask_in_prompt else None
        visible_expected_next_screen = (
            expected_next_screen if self.include_expected_next_screen_in_prompt else None
        )
        if visible_current_subtask:
            lines.append(f"actual_task: {visible_current_subtask}")
        if visible_expected_next_screen:
            lines.append(f"reflection_hint: {visible_expected_next_screen}")
        if action_only_output and self.latent_scaffolds_in_prompt:
            lines.append(f"Internal future-image latent slots: {self.future_scaffold_text()}")
            lines.append(f"Internal latent thinking slots: {self.latent_scaffold_text()}")
        elif self.latent_scaffolds_in_prompt:
            if visible_expected_next_screen:
                lines.append("Predict the next action while internally modeling the next GUI state.")
            lines.append(f"Internal future-image latent slots: {self.future_scaffold_text()}")
            if "stage2" in str(getattr(self, "_active_training_stage", "")):
                lines.append(f"Internal latent thinking slots: {self.latent_scaffold_text()}")
        use_action_tokens = str(action_format) == "action_tokens"
        if not include_action_in_lm:
            lines.extend(
                [
                    "Use the screenshots and task to build only the reasoning state.",
                    "The next GUI action is predicted by a separate action expert, so do not output any action fields.",
                    "You must output plain text only. Do not output JSON. Do not output markdown. Do not output code fences.",
                    "Do not output Action:, Point:, Text:, Keys:, Amount:, Status:, or any action-token sequence.",
                    "Do not add any prefix such as Assistant:, Answer:, Response:, or explanations before the first line.",
                    "The first line must be exactly: Reasoning:",
                    "Then output the reasoning block in this exact order:",
                    "actual_task: ...",
                    "thought: ...",
                    "reflection: ...",
                    f"{self.public_future_slot_token} repeated exactly {self.latent_slot_count} times on one line",
                    "Do not output extra trailing commentary.",
                ]
            )
        elif action_only_output and use_action_tokens:
            lines.extend(
                [
                    "Use the screenshots and task to decide the next GUI action.",
                    "Return the action using GUI action tokens only. Do not output JSON, markdown, or explanations.",
                    "Valid action-token examples:",
                    "<ACT_CLICK> <X_512> <Y_288>",
                    "<ACT_DOUBLE_CLICK> <X_512> <Y_288>",
                    "<ACT_RIGHT_CLICK> <X_512> <Y_288>",
                    "<ACT_TYPE> <TEXT_START> hello world <TEXT_END>",
                    "<ACT_HOTKEY> <KEY_CTRL> <KEY_SHIFT> <KEY_S>",
                    "<ACT_SCROLL> <SCROLL_DOWN>",
                    "<ACT_WAIT> <STATUS_SUCCESS>",
                    "<ACT_TERMINATE> <STATUS_SUCCESS>",
                ]
            )
        elif action_only_output:
            lines.extend(
                [
                    "Use the screenshots and task to decide the next GUI action.",
                    "The internal latent slots are private scaffolds. Do not explain them or output them unless required.",
                    "Return plain text only with these fields.",
                    "Do not output JSON. Do not output markdown.",
                    "Valid action format examples:",
                    "Action: click",
                    "Point: [512 288]    # pointer coordinates use Qwen GUI scale: integers in [0, 1000]",
                    "",
                    "Action: type",
                    'Text: "hello world"',
                    "",
                    "Action: hotkey",
                    "Keys: [ctrl, shift, s]",
                    "",
                    "Action: scroll",
                    "Amount: -512",
                    "",
                    "Action: terminate",
                    "Status: success",
                ]
            )
        elif use_action_tokens:
            lines.extend(
                [
                    "Use the screenshots and task to decide the next GUI action.",
                    "The internal future-image slots are private scaffolds for future visual prediction.",
                    "If latent thinking tokens appear in the reasoning field, keep them exactly as given.",
                    "You must output plain text only. Do not output JSON. Do not output markdown. Do not output code fences.",
                    "Do not add any prefix such as Assistant:, Answer:, Response:, or explanations before the first line.",
                    "Do not omit required reasoning fields.",
                    "The first line must be exactly: Reasoning:",
                    f"When latent reasoning is used, output it as: {self.public_thinking_start_token} {self.public_thinking_token} ... {self.public_thinking_end_token}",
                    "Then output the reasoning block in this exact order:",
                    "actual_task: ...",
                    "thought: ...",
                    "reflection: ...",
                    f"{self.public_future_slot_token} repeated exactly {self.latent_slot_count} times on one line",
                    "After the reasoning block, output exactly one GUI action-token line.",
                    "Valid action-token examples:",
                    "<ACT_CLICK> <X_512> <Y_288>",
                    "<ACT_DOUBLE_CLICK> <X_512> <Y_288>",
                    "<ACT_RIGHT_CLICK> <X_512> <Y_288>",
                    "<ACT_TYPE> <TEXT_START> hello world <TEXT_END>",
                    "<ACT_HOTKEY> <KEY_CTRL> <KEY_SHIFT> <KEY_S>",
                    "<ACT_SCROLL> <SCROLL_DOWN>",
                    "<ACT_WAIT> <STATUS_SUCCESS>",
                    "<ACT_TERMINATE> <STATUS_SUCCESS>",
                    "Do not output extra trailing commentary.",
                ]
            )
        else:
            lines.extend(
                [
                    "Use the screenshots and task to decide the next GUI action.",
                    "The internal future-image slots are private scaffolds for future visual prediction.",
                    "If latent thinking tokens appear in the reasoning field, keep them exactly as given.",
                    "You must output plain text only. Do not output JSON. Do not output markdown. Do not output code fences.",
                    "Do not add any prefix such as Assistant:, Answer:, Response:, or explanations before the first line.",
                    "Do not omit required reasoning fields.",
                    "The first line must be exactly: Reasoning:",
                    f"When latent reasoning is used, output it as: {self.public_thinking_start_token} {self.public_thinking_token} ... {self.public_thinking_end_token}",
                    "Then output the reasoning block in this exact order:",
                    "actual_task: ...",
                    "thought: ...",
                    "reflection: ...",
                    f"{self.public_future_slot_token} repeated exactly {self.latent_slot_count} times on one line",
                    "After the reasoning block, output the action block.",
                    "The first action line must be exactly one of:",
                    "Action: click",
                    "Action: double_click",
                    "Action: right_click",
                    "Action: type",
                    "Action: hotkey",
                    "Action: scroll",
                    "Action: terminate",
                    "Action: wait",
                    "Then output exactly one corresponding parameter line when needed:",
                    "Point: [x y]         # only for pointer actions; x and y are integers in [0, 1000]",
                    'Text: "..."          # only for type',
                    "Keys: [ctrl, c]      # only for hotkey",
                    "Amount: -512         # only for scroll",
                    "Status: success      # only for terminate or wait",
                    "Do not output extra trailing commentary.",
                ]
            )
        return "\n".join(lines)

    @staticmethod
    def _norm_to_qwen_coord(value: Any) -> int:
        try:
            return int(round(max(0.0, min(1.0, float(value))) * 1000.0))
        except Exception:
            return 500

    @staticmethod
    def _coord_to_qwen_coord(value: Any) -> int:
        try:
            coord = float(value)
        except Exception:
            return 500
        if abs(coord) <= 1.0:
            coord *= 1000.0
        return int(round(max(0.0, min(1000.0, coord))))

    @staticmethod
    def _qwen_coord_to_norm(value: Any) -> float:
        try:
            coord = float(value)
        except Exception:
            return 0.5
        if abs(coord) > 1.0:
            coord = coord / 1000.0
        return round(max(0.0, min(1.0, coord)), 4)

    @staticmethod
    def _bbox_text_to_qwen(bbox: Any) -> str:
        text = str(bbox or "").strip()
        if not text or text == "[]":
            return "[]"
        numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
        if len(numbers) < 4:
            return text
        coords = [LaRAStyleQwen3VLAgent._coord_to_qwen_coord(value) for value in numbers[:4]]
        return f"[{coords[0]} {coords[1]} {coords[2]} {coords[3]}]"

    @staticmethod
    def _normalize_reasoning_bbox_to_qwen(reasoning: str) -> str:
        def replace_bbox(match: re.Match[str]) -> str:
            return f"{match.group(1)}{LaRAStyleQwen3VLAgent._bbox_text_to_qwen(match.group(2))}"

        return re.sub(
            r"(?m)^(bbox:\s*)(\[[^\n]*\])",
            replace_bbox,
            str(reasoning or ""),
        )

    @staticmethod
    def _remove_reasoning_bbox(reasoning: str) -> str:
        return "\n".join(
            line
            for line in str(reasoning or "").splitlines()
            if not line.strip().lower().startswith("bbox:")
        ).strip()

    @staticmethod
    def build_action_text_target(action: dict[str, Any]) -> str:
        action_type = str(action.get("type", "wait")).strip() or "wait"
        lines = [f"Action: {action_type}"]
        if action_type in {"click", "double_click", "right_click"}:
            if action.get("x_norm") is not None and action.get("y_norm") is not None:
                x_coord = LaRAStyleQwen3VLAgent._norm_to_qwen_coord(action["x_norm"])
                y_coord = LaRAStyleQwen3VLAgent._norm_to_qwen_coord(action["y_norm"])
                lines.append(f"Point: [{x_coord} {y_coord}]")
            elif action.get("x") is not None and action.get("y") is not None:
                lines.append(f"PointPx: [{int(action['x'])} {int(action['y'])}]")
        elif action_type == "type":
            lines.append(f'Text: "{str(action.get("text", ""))}"')
        elif action_type == "hotkey":
            keys = list(action.get("keys") or [])
            lines.append(f"Keys: [{', '.join(str(key) for key in keys)}]")
        elif action_type == "scroll":
            lines.append(f"Amount: {int(action.get('amount', 0) or 0)}")
        elif action_type in {"terminate", "wait"}:
            lines.append(f"Status: {str(action.get('status', 'success') or 'success')}")
        return "\n".join(lines)

    def build_action_target(self, action: dict[str, Any], *, action_format: str = "text") -> str:
        if str(action_format) == "action_tokens":
            return self.gui_action_tokenizer.encode(action)
        return self.build_action_text_target(action)

    @staticmethod
    def _region_from_point(x_norm: float | None, y_norm: float | None) -> str:
        if x_norm is None or y_norm is None:
            return "middle_center"
        col = min(2, max(0, int(float(x_norm) * 3.0)))
        row = min(2, max(0, int(float(y_norm) * 3.0)))
        return REGION_LABELS[row * 3 + col]

    @staticmethod
    def _coerce_norm(value: Any) -> float | None:
        if value is None:
            return None
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        if parsed != parsed:
            return None
        if abs(parsed) > 1.0:
            parsed = parsed / 1000.0
        return max(0.0, min(1.0, parsed))

    def _build_action_fused_state(
        self,
        *,
        hidden_states: Tensor,
        latent_summary: Tensor,
        img_next_state: Tensor,
        sequence_summary: Tensor | None = None,
    ) -> Tensor:
        if sequence_summary is None:
            sequence_summary = hidden_states[:, -1, :]
        return self.action_state_norm(torch.cat([sequence_summary, latent_summary, img_next_state], dim=-1))

    def _prompt_sequence_summary(
        self,
        hidden_states: Tensor,
        prompt_lengths: int | list[int],
    ) -> Tensor:
        if isinstance(prompt_lengths, int):
            prompt_lengths = [prompt_lengths]
        if len(prompt_lengths) != int(hidden_states.shape[0]):
            raise ValueError(
                f"prompt_lengths size {len(prompt_lengths)} does not match batch size {int(hidden_states.shape[0])}."
            )
        seq_len = int(hidden_states.shape[1])
        indices = [
            max(0, min(seq_len - 1, int(prompt_length) - 1))
            for prompt_length in prompt_lengths
        ]
        gather_index = torch.tensor(indices, device=hidden_states.device, dtype=torch.long)
        return hidden_states[torch.arange(hidden_states.shape[0], device=hidden_states.device), gather_index, :]

    def _normalised_action_hidden_source(self) -> str:
        source = str(getattr(self, "action_hidden_source", "summary") or "summary").strip().lower()
        aliases = {
            "baseline": "summary",
            "last": "summary",
            "prompt": "prompt_attn",
            "query": "prompt_attn",
            "learned_query": "prompt_attn",
            "slots": "slot_attn",
            "prompt_slots": "prompt_slot_attn",
        }
        source = aliases.get(source, source)
        if source not in {"summary", "prompt_attn", "slot_attn", "prompt_slot_attn"}:
            source = "summary"
        return source

    def _uses_prompt_attention_summary(self) -> bool:
        return self._normalised_action_hidden_source() in {"prompt_attn", "prompt_slot_attn"}

    def _uses_slot_attention_summary(self) -> bool:
        return self._normalised_action_hidden_source() in {"slot_attn", "prompt_slot_attn"}

    def _attention_pool_token_states(self, token_states: Tensor, query: Tensor) -> Tensor:
        if int(token_states.shape[0]) <= 1:
            return token_states.mean(dim=0)
        query = query.to(device=token_states.device, dtype=token_states.dtype)
        scores = torch.matmul(token_states.float(), query.float())
        weights = torch.softmax(scores, dim=0).to(dtype=token_states.dtype)
        return torch.sum(token_states * weights.unsqueeze(-1), dim=0)

    def _prompt_attention_summary(
        self,
        hidden_states: Tensor,
        prompt_lengths: int | list[int],
    ) -> Tensor:
        if isinstance(prompt_lengths, int):
            prompt_lengths = [prompt_lengths]
        rows: list[Tensor] = []
        seq_len = int(hidden_states.shape[1])
        for row_index, prompt_length in enumerate(prompt_lengths):
            end_at = max(1, min(seq_len, int(prompt_length)))
            prompt_states = hidden_states[row_index, :end_at, :]
            rows.append(self._attention_pool_token_states(prompt_states, self.action_prompt_query))
        return torch.stack(rows, dim=0)

    def _action_sequence_summary(
        self,
        hidden_states: Tensor,
        prompt_lengths: int | list[int],
    ) -> Tensor:
        if self._uses_prompt_attention_summary():
            return self._prompt_attention_summary(hidden_states, prompt_lengths)
        return self._prompt_sequence_summary(hidden_states, prompt_lengths)

    def _pool_positions(
        self,
        hidden_states: Tensor,
        *,
        row_index: int,
        positions: list[int],
        use_attention: bool,
        query: Tensor,
    ) -> Tensor:
        token_states = hidden_states[int(row_index), positions, :]
        if use_attention:
            return self._attention_pool_token_states(token_states, query)
        return token_states.mean(dim=0)

    def _prompt_action_latent_summary(
        self,
        *,
        hidden_states: Tensor,
        input_ids: Tensor,
        prompt_lengths: int | list[int],
        future_slot_ids: list[int],
    ) -> Tensor:
        if isinstance(prompt_lengths, int):
            prompt_lengths = [prompt_lengths]
        rows: list[Tensor] = []
        latent_slot_ids = [self.special_token_to_id[token] for token in self.latent_slot_tokens]
        for row_index, prompt_length in enumerate(prompt_lengths):
            prompt_length_int = int(prompt_length)
            latent_positions = self._find_token_positions_before(
                input_ids,
                latent_slot_ids,
                prompt_length_int,
                row_index=row_index,
            )
            if not latent_positions:
                latent_positions = self._find_token_positions_before(
                    input_ids,
                    future_slot_ids,
                    prompt_length_int,
                    row_index=row_index,
                )
            if not latent_positions:
                fallback_position = max(0, min(int(hidden_states.shape[1]) - 1, prompt_length_int - 1))
                latent_positions = [fallback_position]
            rows.append(
                self._pool_positions(
                    hidden_states,
                    row_index=row_index,
                    positions=latent_positions,
                    use_attention=self._uses_slot_attention_summary(),
                    query=self.action_slot_query,
                )
            )
        latent_pre_summary = torch.stack(rows, dim=0)
        return self.reasoning_norm(self.reasoning_proj(latent_pre_summary))

    def _current_visual_tokens(
        self,
        *,
        hidden_states: Tensor,
        inputs: dict[str, Any],
        image_counts_per_sample: list[int] | None = None,
    ) -> tuple[Tensor, Tensor, list[tuple[int, int]]]:
        visual_module = getattr(self.model.model, "visual", None)
        spatial_merge_size = int(getattr(visual_module, "spatial_merge_size", 2) or 2)
        if image_counts_per_sample is None and int(hidden_states.shape[0]) == 1:
            grid_rows = inputs.get("image_grid_thw")
            if grid_rows is not None:
                rows = grid_rows.tolist() if hasattr(grid_rows, "tolist") else grid_rows
                image_counts_per_sample = [len(rows)]
        visual_tokens, visual_mask = extract_visual_token_sequence_batch_with_mask(
            hidden_states=hidden_states,
            image_grid_thw=inputs.get("image_grid_thw"),
            spatial_merge_size=spatial_merge_size,
            image_counts_per_sample=image_counts_per_sample,
            select_last_image_per_sample=True,
        )
        grid_sizes = self._current_image_grid_sizes(
            inputs.get("image_grid_thw"),
            image_counts_per_sample=image_counts_per_sample,
            spatial_merge_size=spatial_merge_size,
            token_count=int(visual_tokens.shape[1]) if visual_tokens.ndim == 3 else 0,
        )
        return visual_tokens, visual_mask, grid_sizes

    @staticmethod
    def _current_image_grid_sizes(
        image_grid_thw: Any,
        *,
        image_counts_per_sample: list[int] | None,
        spatial_merge_size: int,
        token_count: int,
    ) -> list[tuple[int, int]]:
        if image_grid_thw is None:
            return resolve_grid_sizes_from_image_grid(
                image_grid_thw,
                spatial_merge_size=spatial_merge_size,
                token_count=token_count,
            )
        rows = image_grid_thw.tolist() if hasattr(image_grid_thw, "tolist") else image_grid_thw
        merge = max(1, int(spatial_merge_size))

        def grid_size(row: Any) -> tuple[int, int]:
            if isinstance(row, (list, tuple)) and len(row) >= 3:
                try:
                    return max(1, int(row[1]) // merge), max(1, int(row[2]) // merge)
                except Exception:
                    pass
            fallback = resolve_grid_sizes_from_image_grid(
                None,
                spatial_merge_size=spatial_merge_size,
                token_count=token_count,
            )
            return fallback[0]

        if image_counts_per_sample is not None and sum(image_counts_per_sample) == len(rows):
            output: list[tuple[int, int]] = []
            row_offset = 0
            for image_count in image_counts_per_sample:
                count = max(0, int(image_count))
                sample_rows = rows[row_offset : row_offset + count]
                output.append(grid_size(sample_rows[-1]) if sample_rows else grid_size(None))
                row_offset += count
            return output

        return resolve_grid_sizes_from_image_grid(
            image_grid_thw,
            spatial_merge_size=spatial_merge_size,
            token_count=token_count,
        )

    def _run_action_head(
        self,
        *,
        hidden_states: Tensor,
        latent_summary: Tensor,
        img_next_state: Tensor,
        inputs: dict[str, Any],
        image_counts_per_sample: list[int] | None = None,
        sequence_summary: Tensor | None = None,
    ) -> UnifiedActionHeadOutput | None:
        visual_tokens, visual_mask, grid_sizes = self._current_visual_tokens(
            hidden_states=hidden_states,
            inputs=inputs,
            image_counts_per_sample=image_counts_per_sample,
        )
        if visual_tokens.shape[1] <= 0:
            return None
        fused = self._build_action_fused_state(
            hidden_states=hidden_states,
            latent_summary=latent_summary,
            img_next_state=img_next_state,
            sequence_summary=sequence_summary,
        )
        return self.action_head(
            fused=fused,
            current_visual_tokens=visual_tokens,
            current_visual_token_mask=visual_mask,
            target_patch_grid_sizes=grid_sizes,
        )

    def _continuous_action_target_and_mask(
        self,
        gold_action: dict[str, Any],
        *,
        reference: Tensor,
    ) -> tuple[Tensor, Tensor]:
        action_type = str(gold_action.get("type", "wait") or "wait")
        target = reference.new_tensor([[0.5, 0.5, 0.0]])
        mask = reference.new_zeros((1, 3))
        x_norm = self._coerce_norm(gold_action.get("x_norm"))
        y_norm = self._coerce_norm(gold_action.get("y_norm"))
        if action_type in POINTER_ACTION_TYPES and x_norm is not None and y_norm is not None:
            target[:, 0] = float(x_norm)
            target[:, 1] = float(y_norm)
            mask[:, 0:2] = 1.0
        if action_type == "scroll" and gold_action.get("amount") is not None:
            try:
                amount = float(gold_action.get("amount", 0) or 0)
            except (TypeError, ValueError):
                amount = 0.0
            target[:, 2] = max(-1.0, min(1.0, amount / 1000.0))
            mask[:, 2] = 1.0
        return target, mask

    def _continuous_action_target_and_mask_batch(
        self,
        gold_actions: list[dict[str, Any]],
        *,
        reference: Tensor,
    ) -> tuple[Tensor, Tensor]:
        targets: list[Tensor] = []
        masks: list[Tensor] = []
        for gold_action in gold_actions:
            target, mask = self._continuous_action_target_and_mask(gold_action, reference=reference)
            targets.append(target)
            masks.append(mask)
        return torch.cat(targets, dim=0), torch.cat(masks, dim=0)

    def _run_flow_action_head(
        self,
        *,
        hidden_states: Tensor,
        latent_summary: Tensor,
        img_next_state: Tensor,
        gold_action: dict[str, Any] | list[dict[str, Any]] | None = None,
        sequence_summary: Tensor | None = None,
        inputs: dict[str, Any] | None = None,
        image_counts_per_sample: list[int] | None = None,
    ) -> FlowMatchingActionHeadOutput:
        fused = self._build_action_fused_state(
            hidden_states=hidden_states,
            latent_summary=latent_summary,
            img_next_state=img_next_state,
            sequence_summary=sequence_summary,
        )
        visual_tokens = None
        visual_mask = None
        grid_sizes = None
        if inputs is not None:
            visual_tokens, visual_mask, grid_sizes = self._current_visual_tokens(
                hidden_states=hidden_states,
                inputs=inputs,
                image_counts_per_sample=image_counts_per_sample,
            )
        target = None
        if gold_action is not None:
            if isinstance(gold_action, list):
                target, _ = self._continuous_action_target_and_mask_batch(gold_action, reference=fused)
            else:
                target, _ = self._continuous_action_target_and_mask(gold_action, reference=fused)
        return self.flow_action_head(
            fused=fused,
            target_continuous_action=target,
            current_visual_tokens=visual_tokens,
            current_visual_token_mask=visual_mask,
            target_patch_grid_sizes=grid_sizes,
        )

    def _pad_latent_state_rows(
        self,
        rows: list[Tensor],
    ) -> tuple[Tensor, Tensor]:
        if not rows:
            raise ValueError("At least one latent-state row is required.")
        max_count = min(
            int(self.latent_slot_count),
            max(1, max(int(row.shape[0]) for row in rows)),
        )
        padded = rows[0].new_zeros((len(rows), max_count, int(rows[0].shape[-1])))
        valid_mask = torch.zeros(
            (len(rows), max_count),
            device=rows[0].device,
            dtype=torch.bool,
        )
        for row_index, row in enumerate(rows):
            count = min(max_count, int(row.shape[0]))
            if count <= 0:
                continue
            padded[row_index, :count, :] = row[:count]
            valid_mask[row_index, :count] = True
        return padded, valid_mask

    def _run_latent_two_way_action_head(
        self,
        *,
        hidden_states: Tensor,
        latent_states: Tensor,
        latent_valid_mask: Tensor,
        img_next_state: Tensor,
        sequence_summary: Tensor,
        inputs: dict[str, Any],
        image_counts_per_sample: list[int] | None = None,
    ) -> FlowMatchingActionHeadOutput:
        visual_tokens, visual_mask, grid_sizes = self._current_visual_tokens(
            hidden_states=hidden_states,
            inputs=inputs,
            image_counts_per_sample=image_counts_per_sample,
        )
        if int(visual_tokens.shape[1]) <= 0:
            raise RuntimeError("No current-image visual tokens found for latent two-way action head.")
        return self.latent_two_way_action_head(
            latent_states=latent_states,
            latent_valid_mask=latent_valid_mask,
            current_visual_tokens=visual_tokens,
            current_visual_token_mask=visual_mask,
            target_patch_grid_sizes=grid_sizes,
            sequence_summary=sequence_summary,
            img_next_state=img_next_state,
        )

    def build_stage1_teacher_response(
        self,
        *,
        explicit_reasoning: str,
        gold_action: dict[str, Any],
        action_format: str = "text",
        include_action: bool = True,
    ) -> str:
        reasoning = self._remove_reasoning_bbox(str(explicit_reasoning or "").strip())
        if not include_action:
            return f"Reasoning: {reasoning}".strip()
        action_text = self.build_action_target(gold_action, action_format=action_format)
        return f"Reasoning: {reasoning}\n{action_text}".strip()

    def build_stage1_teacher_response_with_subtask(
        self,
        *,
        current_subtask: str | None,
        explicit_reasoning: str,
        gold_action: dict[str, Any],
        action_format: str = "text",
        include_action: bool = True,
    ) -> str:
        parts: list[str] = []
        if str(explicit_reasoning or "").strip():
            parts.append("Reasoning:")
            parts.append(self._remove_reasoning_bbox(str(explicit_reasoning or "").strip()))
        elif str(current_subtask or "").strip():
            parts.append(f"actual_task: {str(current_subtask).strip()}")
        if include_action:
            parts.append(self.build_action_target(gold_action, action_format=action_format))
        return "\n".join(parts).strip()

    @staticmethod
    def _parse_explicit_reasoning_fields(explicit_reasoning: str) -> dict[str, str]:
        fields = {
            "actual_task": "",
            "thought": "",
            "reflection": "",
            "img_next": "",
        }
        for raw_line in str(explicit_reasoning or "").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("actual_task:"):
                fields["actual_task"] = line[len("actual_task:") :].strip()
            elif line.startswith("thought:"):
                fields["thought"] = line[len("thought:") :].strip()
            elif line.startswith("reflection:"):
                fields["reflection"] = line[len("reflection:") :].strip()
            elif "<img next>" in line or "<IMG_NEXT_" in line:
                fields["img_next"] = line.strip()
        return fields

    def _build_stage2_field_schedule(
        self,
        *,
        explicit_keep_ratio: float,
        max_thinking_tokens: int,
    ) -> tuple[list[str], int]:
        ordered_fields = ["actual_task", "thought", "reflection"]
        keep_ratio = min(1.0, max(0.0, float(explicit_keep_ratio)))
        keep_count = int(round(len(ordered_fields) * keep_ratio))
        keep_count = max(0, min(len(ordered_fields), keep_count))
        kept_fields = ordered_fields[-keep_count:] if keep_count > 0 else []
        dropped_count = len(ordered_fields) - keep_count
        max_tokens = max(1, min(int(max_thinking_tokens), self.latent_slot_count))
        if dropped_count <= 0:
            thinking_token_count = 0
        elif self.reasoning_alignment_mode == "field_aligned":
            # Prefix replacement preserves the semantic partition:
            # task -> task+thought -> task+thought+reflection.
            thinking_token_count = min(
                max_tokens,
                sum(self.reasoning_field_slot_counts[:dropped_count]),
            )
        else:
            thinking_token_count = min(
                max_tokens,
                max(1, int(math.ceil(max_tokens * dropped_count / len(ordered_fields)))),
            )
        return kept_fields, thinking_token_count

    def build_stage2_teacher_response(
        self,
        *,
        current_subtask: str | None,
        explicit_reasoning: str,
        gold_action: dict[str, Any],
        explicit_keep_ratio: float,
        min_explicit_tokens: int,
        max_thinking_tokens: int,
        action_format: str = "text",
        include_action: bool = True,
    ) -> tuple[str, list[str]]:
        del min_explicit_tokens
        fields = self._parse_explicit_reasoning_fields(explicit_reasoning)
        kept_fields, thinking_token_count = self._build_stage2_field_schedule(
            explicit_keep_ratio=explicit_keep_ratio,
            max_thinking_tokens=max_thinking_tokens,
        )
        used_latent_tokens = self.latent_slot_tokens[:thinking_token_count]
        reasoning_lines: list[str] = []
        if thinking_token_count > 0:
            reasoning_lines.append(self.thinking_scaffold_text(thinking_token_count))
        if "actual_task" in kept_fields and fields["actual_task"]:
            reasoning_lines.append(f"actual_task: {fields['actual_task']}")
        if "thought" in kept_fields and fields["thought"]:
            reasoning_lines.append(f"thought: {fields['thought']}")
        if "reflection" in kept_fields and fields["reflection"]:
            reasoning_lines.append(f"reflection: {fields['reflection']}")
        if fields["img_next"]:
            reasoning_lines.append(fields["img_next"])
        parts: list[str] = []
        if reasoning_lines:
            parts.append("Reasoning:")
            parts.extend(reasoning_lines)
        elif str(current_subtask or "").strip():
            parts.append(f"actual_task: {str(current_subtask).strip()}")
        if include_action:
            parts.append(self.build_action_target(gold_action, action_format=action_format))
        return "\n".join(parts).strip(), used_latent_tokens

    def build_stage2_action_only_teacher_response(
        self,
        *,
        gold_action: dict[str, Any],
        action_format: str = "text",
    ) -> str:
        return self.build_action_target(gold_action, action_format=action_format)

    def _build_messages(
        self,
        *,
        image_paths: list[str | Path | Image.Image],
        user_prompt: str,
        assistant_target: str | None = None,
    ) -> list[dict[str, Any]]:
        images = [self._load_image(path) for path in image_paths]
        user_content: list[dict[str, Any]] = []
        for image in images:
            user_content.append({"type": "image", "image": image})
        user_content.append({"type": "text", "text": user_prompt})
        messages = [{"role": "user", "content": user_content}]
        if assistant_target is not None:
            messages.append({"role": "assistant", "content": [{"type": "text", "text": assistant_target}]})
        return messages

    def prepare_conversation_inputs(
        self,
        *,
        image_paths: list[str | Path | Image.Image],
        user_prompt: str,
        assistant_target: str | None = None,
        continue_assistant: bool = False,
    ) -> tuple[dict[str, Any], int]:
        user_prompt = self._expand_public_future_slots(user_prompt)
        user_prompt, _ = self._expand_public_thinking_slots(user_prompt)
        if assistant_target is not None:
            assistant_target = self._expand_public_future_slots(assistant_target)
            assistant_target, _ = self._expand_public_thinking_slots(assistant_target)
        messages = self._build_messages(
            image_paths=image_paths,
            user_prompt=user_prompt,
            assistant_target=assistant_target,
        )
        full_template_kwargs: dict[str, Any] = {
            "tokenize": True,
            "add_generation_prompt": assistant_target is None,
            "return_dict": True,
            "return_tensors": "pt",
        }
        if continue_assistant and assistant_target is not None:
            full_template_kwargs["continue_final_message"] = True
        full_inputs = self.processor.apply_chat_template(messages, **full_template_kwargs)
        full_inputs.pop("token_type_ids", None)

        prompt_messages = self._build_messages(
            image_paths=image_paths,
            user_prompt=user_prompt,
            assistant_target=None,
        )
        prompt_inputs = self.processor.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        prompt_length = int(prompt_inputs["input_ids"].shape[1])
        return self._move_inputs_to_device(full_inputs), prompt_length

    def prepare_conversation_inputs_batch(
        self,
        *,
        batch_image_paths: list[list[str | Path | Image.Image]],
        user_prompts: list[str],
        assistant_targets: list[str | None],
        padding_side: str = "right",
        continue_assistant: bool = False,
    ) -> tuple[dict[str, Any], list[int]]:
        tokenizer = self.processor.tokenizer
        original_padding_side = getattr(tokenizer, "padding_side", "right")
        tokenizer.padding_side = str(padding_side)
        try:
            batch_messages = []
            prompt_messages = []
            for image_paths, user_prompt, assistant_target in zip(batch_image_paths, user_prompts, assistant_targets):
                expanded_prompt = self._expand_public_future_slots(user_prompt)
                expanded_prompt, _ = self._expand_public_thinking_slots(expanded_prompt)
                expanded_target = assistant_target
                if expanded_target is not None:
                    expanded_target = self._expand_public_future_slots(expanded_target)
                    expanded_target, _ = self._expand_public_thinking_slots(expanded_target)
                batch_messages.append(
                    self._build_messages(
                        image_paths=image_paths,
                        user_prompt=expanded_prompt,
                        assistant_target=expanded_target,
                    )
                )
                prompt_messages.append(
                    self._build_messages(
                        image_paths=image_paths,
                        user_prompt=expanded_prompt,
                        assistant_target=None,
                    )
                )
            full_template_kwargs: dict[str, Any] = {
                "tokenize": True,
                "padding": True,
                "add_generation_prompt": all(target is None for target in assistant_targets),
                "return_dict": True,
                "return_tensors": "pt",
            }
            if continue_assistant and all(target is not None for target in assistant_targets):
                full_template_kwargs["continue_final_message"] = True
            full_inputs = self.processor.apply_chat_template(batch_messages, **full_template_kwargs)
            full_inputs.pop("token_type_ids", None)
            prompt_lengths: list[int] = []
            for prompt_message, assistant_target in zip(prompt_messages, assistant_targets):
                prompt_inputs = self.processor.apply_chat_template(
                    prompt_message,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                prompt_lengths.append(int(prompt_inputs["input_ids"].shape[1]))
            return self._move_inputs_to_device(full_inputs), prompt_lengths
        finally:
            tokenizer.padding_side = original_padding_side

    def _find_token_positions(self, input_ids: Tensor, token_ids: list[int], *, row_index: int = 0) -> list[int]:
        positions: list[int] = []
        flat = input_ids[int(row_index)].detach().to("cpu").tolist()
        for token_id in token_ids:
            try:
                positions.append(flat.index(int(token_id)))
            except ValueError:
                continue
        return positions

    def _find_token_positions_after(
        self, input_ids: Tensor, token_ids: list[int], start_index: int, *, row_index: int = 0
    ) -> list[int]:
        positions: list[int] = []
        flat = input_ids[int(row_index)].detach().to("cpu").tolist()
        start_at = max(0, int(start_index))
        for token_id in token_ids:
            try:
                relative_index = flat[start_at:].index(int(token_id))
            except ValueError:
                continue
            positions.append(start_at + relative_index)
        return positions

    def _find_token_positions_before(
        self, input_ids: Tensor, token_ids: list[int], end_index: int, *, row_index: int = 0
    ) -> list[int]:
        positions: list[int] = []
        flat = input_ids[int(row_index)].detach().to("cpu").tolist()
        end_at = min(len(flat), max(0, int(end_index)))
        prefix = flat[:end_at]
        for token_id in token_ids:
            try:
                positions.append(prefix.index(int(token_id)))
            except ValueError:
                continue
        return positions

    @staticmethod
    def _find_subsequence(haystack: list[int], needle: list[int], *, start_index: int = 0) -> int:
        if not needle:
            return -1
        upper = len(haystack) - len(needle) + 1
        for index in range(max(0, int(start_index)), max(0, upper)):
            if haystack[index : index + len(needle)] == needle:
                return index
        return -1

    def _find_reasoning_token_positions_in_target(
        self,
        input_ids: Tensor,
        assistant_target: str,
        prompt_length: int,
        *,
        row_index: int = 0,
    ) -> list[int]:
        tokenizer = self.processor.tokenizer
        flat = input_ids[int(row_index)].detach().to("cpu").tolist()
        start_index = max(0, int(prompt_length))
        for marker in ("Reasoning:\n", "Reasoning:"):
            marker_ids = tokenizer(marker, add_special_tokens=False)["input_ids"]
            marker_start = self._find_subsequence(flat, marker_ids, start_index=start_index)
            if marker_start >= 0:
                start_index = marker_start + len(marker_ids)
                break

        end_candidates = [len(flat)]
        action_ids = tokenizer("\nAction:", add_special_tokens=False)["input_ids"]
        action_start = self._find_subsequence(flat, action_ids, start_index=start_index)
        if action_start >= 0:
            end_candidates.append(action_start)
        future_ids = {self.special_token_to_id[token] for token in self.future_slot_tokens}
        end_candidates.extend(
            index for index in range(start_index, len(flat)) if int(flat[index]) in future_ids
        )
        end_index = max(start_index, min(end_candidates))
        return list(range(start_index, end_index))

    def encode_reasoning_teacher_text(self, text: str, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        tokenizer = self.processor.tokenizer
        # Future-frame slots have their own visual objective and must not leak
        # into the explicit reasoning teacher embedding.
        teacher_text = re.sub(r"<img next>|<IMG_NEXT_\d+>", " ", str(text or ""), flags=re.I)
        teacher_text = self._remove_reasoning_bbox(teacher_text)
        teacher_text = "\n".join(line.strip() for line in teacher_text.splitlines() if line.strip())
        encoded = tokenizer(
            [teacher_text if teacher_text else "none"],
            padding=True,
            truncation=True,
            max_length=192,
            return_tensors="pt",
            add_special_tokens=True,
        )
        embed_module = self.model.get_input_embeddings()
        input_ids = encoded["input_ids"].to(embed_module.weight.device)
        attention_mask = encoded["attention_mask"].to(embed_module.weight.device)
        token_embeds = embed_module(input_ids)
        mask = attention_mask.unsqueeze(-1).to(dtype=token_embeds.dtype)
        pooled = (token_embeds * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return pooled.detach().to(device=device, dtype=dtype)

    def _field_aligned_reasoning_pairs(
        self,
        *,
        hidden_states: Tensor,
        row_index: int,
        latent_positions: list[int],
        explicit_reasoning: str,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """Pair each complete latent slot group with its own explicit field teacher."""
        parsed_fields = self._parse_explicit_reasoning_fields(explicit_reasoning)
        summaries: dict[str, Tensor] = {}
        teachers: dict[str, Tensor] = {}
        for field_name, (start, end) in self._reasoning_field_slot_ranges().items():
            field_text = str(parsed_fields.get(field_name, "") or "").strip()
            if not field_text or len(latent_positions) < end:
                continue
            field_positions = latent_positions[start:end]
            field_hidden = hidden_states[row_index, field_positions, :].mean(dim=0, keepdim=True)
            field_summary = self.reasoning_norm(self.reasoning_proj(field_hidden))
            summaries[field_name] = field_summary
            teachers[field_name] = self.encode_reasoning_teacher_text(
                f"{field_name}: {field_text}",
                device=field_summary.device,
                dtype=field_summary.dtype,
            )
        return summaries, teachers

    def encode_next_frame_target(
        self,
        *,
        image_path: str | Path | Image.Image,
        task: str,
        current_subtask: str | None,
        expected_next_screen: str | None,
    ) -> Tensor:
        with torch.no_grad():
            inputs, _ = self.prepare_conversation_inputs(
                image_paths=[image_path],
                user_prompt=f"Encode this next GUI frame for instruction context: {task}",
                assistant_target=None,
            )
            self._set_temporal_context(None)
            outputs = self.model.model(
                **inputs,
                output_hidden_states=False,
                return_dict=True,
            )
            visual_module = getattr(self.model.model, "visual", None)
            spatial_merge_size = int(getattr(visual_module, "spatial_merge_size", 2) or 2)
            visual_tokens, visual_mask = extract_visual_token_sequence_batch_with_mask(
                hidden_states=outputs.last_hidden_state,
                image_grid_thw=inputs.get("image_grid_thw"),
                spatial_merge_size=spatial_merge_size,
            )
            mask = visual_mask.unsqueeze(-1).to(dtype=visual_tokens.dtype)
            pooled = (visual_tokens * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            return pooled.detach()

    def encode_next_frame_targets_batch(self, samples: list[dict[str, Any]]) -> Tensor:
        if not samples:
            raise ValueError("encode_next_frame_targets_batch requires at least one sample.")
        image_batches = [[sample["next_image_path"]] for sample in samples]
        with torch.no_grad():
            inputs, _ = self.prepare_conversation_inputs_batch(
                batch_image_paths=image_batches,
                user_prompts=["Encode the visual state of this GUI frame." for _ in samples],
                assistant_targets=[None for _ in samples],
            )
            self._set_temporal_context(None)
            visual_module = self.model.model.visual
            was_training = visual_module.training
            visual_module.eval()
            try:
                visual_output = visual_module(
                    inputs["pixel_values"].to(device=visual_module.device, dtype=visual_module.dtype),
                    grid_thw=inputs["image_grid_thw"].to(visual_module.device),
                )
            finally:
                visual_module.train(was_training)
            visual_tokens = visual_output[0] if isinstance(visual_output, tuple) else visual_output
            spatial_merge_size = int(getattr(visual_module, "spatial_merge_size", 2) or 2)
            grid_rows = inputs["image_grid_thw"].detach().to("cpu").tolist()
            token_counts = [
                max(1, int(row[0]) * int(row[1]) * int(row[2]) // (spatial_merge_size ** 2))
                for row in grid_rows
            ]
            pooled_rows: list[Tensor] = []
            offset = 0
            for token_count in token_counts:
                pooled_rows.append(visual_tokens[offset : offset + token_count].mean(dim=0))
                offset += token_count
            if len(pooled_rows) != len(samples):
                raise RuntimeError(
                    f"Future visual teacher produced {len(pooled_rows)} rows for {len(samples)} samples."
                )
            return torch.stack(pooled_rows, dim=0).detach()

    def forward_train(
        self,
        *,
        image_paths: list[str | Path | Image.Image],
        task: str,
        history_frame_count: int,
        current_subtask: str | None,
        expected_next_screen: str | None,
        explicit_reasoning: str,
        gold_action: dict[str, Any],
        next_image_path: str | Path | Image.Image,
        training_stage: str,
        stage2_target_format: str = "mixed_reasoning_action",
        stage2_explicit_keep_ratio: float = 0.5,
        stage2_min_explicit_tokens: int = 4,
        stage2_max_thinking_tokens: int = 4,
        future_frame_enabled: bool = True,
        temporal_sample_key: str | None = None,
        action_format: str = "text",
        include_action_in_lm: bool = True,
        action_head_enabled: bool = True,
    ) -> LaRAForwardOutput:
        self._active_training_stage = str(training_stage)
        self.action_format = str(action_format or "text")
        self.lm_action_target = "include" if include_action_in_lm else "omit"
        user_prompt = self.build_user_prompt(
            task=task,
            history_frame_count=history_frame_count,
            current_subtask=current_subtask,
            expected_next_screen=expected_next_screen,
            action_only_output=(training_stage == "stage2" and stage2_target_format == "action_only"),
            action_format=self.action_format,
            include_action_in_lm=include_action_in_lm,
        )
        assistant_latent_tokens: list[str] = []
        if training_stage == "stage1":
            assistant_target = self.build_stage1_teacher_response_with_subtask(
                current_subtask=current_subtask,
                explicit_reasoning=explicit_reasoning,
                gold_action=gold_action,
                action_format=self.action_format,
                include_action=include_action_in_lm,
            )
        else:
            if stage2_target_format == "action_only" and include_action_in_lm:
                assistant_target = self.build_stage2_action_only_teacher_response(
                    gold_action=gold_action,
                    action_format=self.action_format,
                )
            elif stage2_target_format == "action_only":
                assistant_target = ""
            else:
                assistant_target, assistant_latent_tokens = self.build_stage2_teacher_response(
                    current_subtask=current_subtask,
                    explicit_reasoning=explicit_reasoning,
                    gold_action=gold_action,
                    explicit_keep_ratio=stage2_explicit_keep_ratio,
                    min_explicit_tokens=stage2_min_explicit_tokens,
                    max_thinking_tokens=stage2_max_thinking_tokens,
                    action_format=self.action_format,
                    include_action=include_action_in_lm,
                )

        inputs, prompt_length = self.prepare_conversation_inputs(
            image_paths=image_paths,
            user_prompt=user_prompt,
            assistant_target=assistant_target,
        )
        labels = inputs["input_ids"].clone()
        labels[:, :prompt_length] = -100
        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100
        future_slot_ids = [self.special_token_to_id[token] for token in self.future_slot_tokens]
        assistant_future_positions = self._find_token_positions_after(
            inputs["input_ids"],
            future_slot_ids,
            prompt_length,
        )
        for position in assistant_future_positions:
            labels[:, position] = -100
        if assistant_latent_tokens:
            assistant_latent_positions = self._find_token_positions_after(
                inputs["input_ids"],
                [self.special_token_to_id[token] for token in assistant_latent_tokens],
                prompt_length,
            )
            for position in assistant_latent_positions:
                labels[:, position] = -100
        sample_keys = [str(temporal_sample_key)] if temporal_sample_key else None
        self._set_temporal_context(sample_keys)
        model_inputs = dict(inputs)
        model_inputs.pop("labels", None)
        outputs = self.model.model(
            **model_inputs,
            output_hidden_states=False,
            return_dict=True,
        )
        self._set_temporal_context(None)

        hidden_states = outputs.last_hidden_state
        self._align_auxiliary_modules(hidden_states)
        shift_hidden_states = hidden_states[:, :-1, :]
        shift_labels = labels[:, 1:]
        supervised_mask = shift_labels.ne(-100)
        if supervised_mask.any():
            supervised_hidden_states = shift_hidden_states[supervised_mask]
            supervised_labels = shift_labels[supervised_mask]
            lm_head = self.model.get_output_embeddings()
            supervised_logits = lm_head(supervised_hidden_states)
            lm_loss = F.cross_entropy(
                supervised_logits.float(),
                supervised_labels.to(device=supervised_logits.device),
            )
        else:
            lm_loss = hidden_states.new_zeros(())
        if training_stage == "stage1":
            latent_positions = self._find_reasoning_token_positions_in_target(
                inputs["input_ids"],
                assistant_target,
                prompt_length,
            )
        elif training_stage == "stage2" and assistant_latent_tokens:
            latent_positions = self._find_token_positions_after(
                inputs["input_ids"],
                [self.special_token_to_id[token] for token in assistant_latent_tokens],
                prompt_length,
            )
        else:
            latent_positions = self._find_token_positions(
                inputs["input_ids"],
                [self.special_token_to_id[token] for token in self.latent_slot_tokens],
            )
        if not latent_positions:
            raise RuntimeError("Latent slot tokens not found in input_ids.")
        latent_states = hidden_states[:, latent_positions, :]
        latent_summary = self.reasoning_norm(self.reasoning_proj(latent_states.mean(dim=1)))
        field_summaries: dict[str, Tensor] | None = None
        field_teachers: dict[str, Tensor] | None = None
        if (
            self.reasoning_alignment_mode == "field_aligned"
            and training_stage == "stage2"
            and assistant_latent_tokens
        ):
            resolved_summaries, resolved_teachers = self._field_aligned_reasoning_pairs(
                hidden_states=hidden_states,
                row_index=0,
                latent_positions=latent_positions,
                explicit_reasoning=explicit_reasoning,
            )
            if resolved_summaries:
                field_summaries = resolved_summaries
                field_teachers = resolved_teachers

        img_next_positions = []
        if not self.latent_scaffolds_in_prompt:
            img_next_positions = self._find_token_positions_after(
                inputs["input_ids"],
                future_slot_ids,
                prompt_length,
            )
        if not img_next_positions:
            img_next_positions = self._find_token_positions_before(
                inputs["input_ids"],
                future_slot_ids,
                prompt_length,
            )
        if not img_next_positions:
            img_next_positions = self._find_token_positions(
                inputs["input_ids"],
                future_slot_ids,
            )
        if not img_next_positions:
            raise RuntimeError("IMG_NEXT future slot tokens not found in input_ids.")
        img_next_state = self._pool_positions(
            hidden_states,
            row_index=0,
            positions=img_next_positions,
            use_attention=self._uses_slot_attention_summary(),
            query=self.action_slot_query,
        ).unsqueeze(0)
        action_sequence_summary = self._action_sequence_summary(hidden_states, prompt_length)
        action_latent_summary = latent_summary
        if self.latent_scaffolds_in_prompt:
            action_latent_summary = self._prompt_action_latent_summary(
                hidden_states=hidden_states,
                input_ids=inputs["input_ids"],
                prompt_lengths=prompt_length,
                future_slot_ids=future_slot_ids,
            )
        predicted_future = self.future_frame_head(img_next_state)
        future_target = None
        if future_frame_enabled:
            future_target = self.encode_next_frame_target(
                image_path=next_image_path,
                task=task,
                current_subtask=current_subtask,
                expected_next_screen=expected_next_screen,
            )
        reasoning_teacher = None
        if field_summaries is None:
            reasoning_teacher = self.encode_reasoning_teacher_text(
                explicit_reasoning,
                device=latent_summary.device,
                dtype=latent_summary.dtype,
            )
        visual_debug = getattr(getattr(self.model, "model", None).visual, "last_debug_shapes", None)
        action_head_output = None
        flow_action_head_output = None
        if action_head_enabled and str(self.action_model) == "latent_two_way":
            flow_action_head_output = self._run_latent_two_way_action_head(
                hidden_states=hidden_states,
                latent_states=latent_states,
                latent_valid_mask=torch.ones(
                    latent_states.shape[:2],
                    device=latent_states.device,
                    dtype=torch.bool,
                ),
                img_next_state=img_next_state,
                sequence_summary=action_sequence_summary,
                inputs=inputs,
            )
        elif action_head_enabled and str(self.action_model) == "flow_matching":
            flow_action_head_output = self._run_flow_action_head(
                hidden_states=hidden_states,
                latent_summary=action_latent_summary,
                img_next_state=img_next_state,
                gold_action=gold_action,
                sequence_summary=action_sequence_summary,
                inputs=inputs,
            )
        elif action_head_enabled:
            action_head_output = self._run_action_head(
                hidden_states=hidden_states,
                latent_summary=action_latent_summary,
                img_next_state=img_next_state,
                inputs=inputs,
                sequence_summary=action_sequence_summary,
            )
        return LaRAForwardOutput(
            loss=lm_loss,
            hidden_states=hidden_states,
            latent_reasoning_states=latent_states,
            latent_reasoning_summary=latent_summary,
            img_next_state=img_next_state,
            predicted_future_frame=predicted_future,
            target_future_frame=future_target,
            reasoning_teacher_embedding=reasoning_teacher,
            latent_reasoning_field_summaries=field_summaries,
            reasoning_teacher_field_embeddings=field_teachers,
            action_head_output=action_head_output,
            flow_action_head_output=flow_action_head_output,
            gold_action=dict(gold_action),
            action_text=assistant_target,
            debug_info={
                "prompt_length": prompt_length,
                "visual_debug": visual_debug,
                "training_stage": training_stage,
                "stage2_target_format": stage2_target_format,
                "assistant_latent_tokens": list(assistant_latent_tokens),
                "assistant_future_positions_count": len(assistant_future_positions),
                "assistant_target": assistant_target,
                "assistant_target_preview": assistant_target[:400],
                "assistant_target_char_count": len(assistant_target),
                "stage2_explicit_keep_ratio": float(stage2_explicit_keep_ratio),
                "reasoning_alignment_mode": str(self.reasoning_alignment_mode),
                "reasoning_field_slot_counts": list(self.reasoning_field_slot_counts),
                "reasoning_aligned_fields": list(field_summaries or {}),
                "supervised_token_count": int(supervised_mask.sum().item()),
                "future_frame_enabled": bool(future_frame_enabled),
                "action_format": self.action_format,
                "action_model": self.action_model,
                "action_head_hidden_source": self._normalised_action_hidden_source(),
                "lm_action_target": self.lm_action_target,
            },
        )

    def forward_train_batch(
        self,
        samples: list[dict[str, Any]],
        *,
        training_stage: str,
        stage2_target_format: str = "mixed_reasoning_action",
        stage2_explicit_keep_ratio: float = 0.5,
        stage2_min_explicit_tokens: int = 4,
        stage2_max_thinking_tokens: int = 4,
        future_frame_enabled: bool = True,
        action_format: str = "text",
        include_action_in_lm: bool = True,
        reasoning_teacher_enabled: bool = True,
        action_head_enabled: bool = True,
    ) -> LaRAForwardOutput:
        if not samples:
            raise ValueError("forward_train_batch requires at least one sample.")
        self._active_training_stage = str(training_stage)
        self.action_format = str(action_format or "text")
        self.lm_action_target = "include" if include_action_in_lm else "omit"

        batch_image_paths: list[list[str | Path | Image.Image]] = []
        user_prompts: list[str] = []
        assistant_targets: list[str | None] = []
        assistant_latent_tokens_batch: list[list[str]] = []
        gold_actions: list[dict[str, Any]] = []
        temporal_keys: list[str] = []

        for sample in samples:
            image_paths = list(sample["image_paths"])
            explicit_reasoning = str(sample.get("explicit_reasoning", "") or "")
            gold_action = dict(sample.get("gold_action", {}) or {})
            user_prompt = self.build_user_prompt(
                task=str(sample.get("task", "") or ""),
                history_frame_count=int(sample.get("history_frame_count", max(0, len(image_paths) - 1))),
                current_subtask=sample.get("current_subtask"),
                expected_next_screen=sample.get("expected_next_screen"),
                action_only_output=(training_stage == "stage2" and stage2_target_format == "action_only"),
                action_format=self.action_format,
                include_action_in_lm=include_action_in_lm,
            )
            assistant_latent_tokens: list[str] = []
            if training_stage == "stage1":
                assistant_target = self.build_stage1_teacher_response_with_subtask(
                    current_subtask=sample.get("current_subtask"),
                    explicit_reasoning=explicit_reasoning,
                    gold_action=gold_action,
                    action_format=self.action_format,
                    include_action=include_action_in_lm,
                )
            else:
                if stage2_target_format == "action_only" and include_action_in_lm:
                    assistant_target = self.build_stage2_action_only_teacher_response(
                        gold_action=gold_action,
                        action_format=self.action_format,
                    )
                elif stage2_target_format == "action_only":
                    assistant_target = ""
                else:
                    assistant_target, assistant_latent_tokens = self.build_stage2_teacher_response(
                        current_subtask=sample.get("current_subtask"),
                        explicit_reasoning=explicit_reasoning,
                        gold_action=gold_action,
                        explicit_keep_ratio=stage2_explicit_keep_ratio,
                        min_explicit_tokens=stage2_min_explicit_tokens,
                        max_thinking_tokens=stage2_max_thinking_tokens,
                        action_format=self.action_format,
                        include_action=include_action_in_lm,
                    )
            batch_image_paths.append(image_paths)
            user_prompts.append(user_prompt)
            assistant_targets.append(assistant_target)
            assistant_latent_tokens_batch.append(list(assistant_latent_tokens))
            gold_actions.append(gold_action)
            temporal_key = str(sample.get("temporal_sample_key", "") or "")
            for image_index in range(len(image_paths)):
                temporal_keys.append(f"{temporal_key}|img_{image_index}" if temporal_key else "")

        inputs, prompt_lengths = self.prepare_conversation_inputs_batch(
            batch_image_paths=batch_image_paths,
            user_prompts=user_prompts,
            assistant_targets=assistant_targets,
        )
        future_target = self.encode_next_frame_targets_batch(samples) if future_frame_enabled else None
        labels = inputs["input_ids"].clone()
        for row_index, prompt_length in enumerate(prompt_lengths):
            labels[row_index, : int(prompt_length)] = -100
        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100

        future_slot_ids = [self.special_token_to_id[token] for token in self.future_slot_tokens]
        assistant_future_position_counts: list[int] = []
        for row_index, prompt_length in enumerate(prompt_lengths):
            assistant_future_positions = self._find_token_positions_after(
                inputs["input_ids"],
                future_slot_ids,
                int(prompt_length),
                row_index=row_index,
            )
            assistant_future_position_counts.append(len(assistant_future_positions))
            for position in assistant_future_positions:
                labels[row_index, position] = -100
            assistant_latent_tokens = assistant_latent_tokens_batch[row_index]
            if assistant_latent_tokens:
                assistant_latent_positions = self._find_token_positions_after(
                    inputs["input_ids"],
                    [self.special_token_to_id[token] for token in assistant_latent_tokens],
                    int(prompt_length),
                    row_index=row_index,
                )
                for position in assistant_latent_positions:
                    labels[row_index, position] = -100

        sample_keys = temporal_keys if temporal_keys and all(temporal_keys) else None
        self._set_temporal_context(sample_keys)
        model_inputs = dict(inputs)
        model_inputs.pop("labels", None)
        outputs = self.model.model(
            **model_inputs,
            output_hidden_states=False,
            return_dict=True,
        )
        self._set_temporal_context(None)

        hidden_states = outputs.last_hidden_state
        self._align_auxiliary_modules(hidden_states)
        shift_hidden_states = hidden_states[:, :-1, :]
        shift_labels = labels[:, 1:]
        supervised_mask = shift_labels.ne(-100)
        if supervised_mask.any():
            supervised_hidden_states = shift_hidden_states[supervised_mask]
            supervised_labels = shift_labels[supervised_mask]
            lm_head = self.model.get_output_embeddings()
            supervised_logits = lm_head(supervised_hidden_states)
            lm_loss = F.cross_entropy(
                supervised_logits.float(),
                supervised_labels.to(device=supervised_logits.device),
            )
        else:
            lm_loss = hidden_states.new_zeros(())

        latent_state_rows: list[Tensor] = []
        latent_token_rows: list[Tensor] = []
        latent_summary_rows: list[Tensor] = []
        img_next_rows: list[Tensor] = []
        field_summary_rows: dict[str, list[Tensor]] = {
            field_name: [] for field_name in REASONING_FIELD_NAMES
        }
        field_teacher_rows: dict[str, list[Tensor]] = {
            field_name: [] for field_name in REASONING_FIELD_NAMES
        }
        for row_index, (assistant_target, prompt_length, assistant_latent_tokens) in enumerate(
            zip(assistant_targets, prompt_lengths, assistant_latent_tokens_batch)
        ):
            if training_stage == "stage1":
                latent_positions = self._find_reasoning_token_positions_in_target(
                    inputs["input_ids"],
                    str(assistant_target or ""),
                    int(prompt_length),
                    row_index=row_index,
                )
            elif training_stage == "stage2" and assistant_latent_tokens:
                latent_positions = self._find_token_positions_after(
                    inputs["input_ids"],
                    [self.special_token_to_id[token] for token in assistant_latent_tokens],
                    int(prompt_length),
                    row_index=row_index,
                )
            else:
                latent_positions = self._find_token_positions(
                    inputs["input_ids"],
                    [self.special_token_to_id[token] for token in self.latent_slot_tokens],
                    row_index=row_index,
                )
            if not latent_positions:
                raise RuntimeError("Latent slot tokens not found in batched input_ids.")
            latent_row = hidden_states[row_index, latent_positions, :]
            latent_token_rows.append(latent_row)
            latent_state_rows.append(latent_row.mean(dim=0, keepdim=True))
            latent_summary_rows.append(latent_row.mean(dim=0))
            if (
                reasoning_teacher_enabled
                and self.reasoning_alignment_mode == "field_aligned"
                and training_stage == "stage2"
                and assistant_latent_tokens
            ):
                row_summaries, row_teachers = self._field_aligned_reasoning_pairs(
                    hidden_states=hidden_states,
                    row_index=row_index,
                    latent_positions=latent_positions,
                    explicit_reasoning=str(samples[row_index].get("explicit_reasoning", "") or ""),
                )
                for field_name in row_summaries:
                    field_summary_rows[field_name].append(row_summaries[field_name])
                    field_teacher_rows[field_name].append(row_teachers[field_name])

            img_next_positions = []
            if not self.latent_scaffolds_in_prompt:
                img_next_positions = self._find_token_positions_after(
                    inputs["input_ids"],
                    future_slot_ids,
                    int(prompt_length),
                    row_index=row_index,
                )
            if not img_next_positions:
                img_next_positions = self._find_token_positions_before(
                    inputs["input_ids"],
                    future_slot_ids,
                    int(prompt_length),
                    row_index=row_index,
                )
            if not img_next_positions:
                img_next_positions = self._find_token_positions(
                    inputs["input_ids"],
                    future_slot_ids,
                    row_index=row_index,
                )
            if not img_next_positions:
                raise RuntimeError("IMG_NEXT future slot tokens not found in batched input_ids.")
            img_next_rows.append(
                self._pool_positions(
                    hidden_states,
                    row_index=row_index,
                    positions=img_next_positions,
                    use_attention=self._uses_slot_attention_summary(),
                    query=self.action_slot_query,
                )
            )

        action_latent_states, action_latent_valid_mask = self._pad_latent_state_rows(
            latent_token_rows
        )
        latent_states = (
            action_latent_states
            if self.reasoning_alignment_mode == "field_aligned" and training_stage == "stage2"
            else torch.stack(latent_state_rows, dim=0)
        )
        latent_pre_summary = torch.stack(latent_summary_rows, dim=0)
        latent_summary = self.reasoning_norm(self.reasoning_proj(latent_pre_summary))
        img_next_state = torch.stack(img_next_rows, dim=0)
        action_sequence_summary = self._action_sequence_summary(hidden_states, prompt_lengths)
        action_latent_summary = latent_summary
        if self.latent_scaffolds_in_prompt:
            action_latent_summary = self._prompt_action_latent_summary(
                hidden_states=hidden_states,
                input_ids=inputs["input_ids"],
                prompt_lengths=prompt_lengths,
                future_slot_ids=future_slot_ids,
            )
        predicted_future = self.future_frame_head(img_next_state) if future_frame_enabled else torch.zeros_like(img_next_state)

        reasoning_teacher = None
        field_summaries: dict[str, Tensor] | None = None
        field_teachers: dict[str, Tensor] | None = None
        if reasoning_teacher_enabled:
            available_fields = [
                field_name for field_name in REASONING_FIELD_NAMES if field_summary_rows[field_name]
            ]
            if available_fields:
                field_summaries = {
                    field_name: torch.cat(field_summary_rows[field_name], dim=0)
                    for field_name in available_fields
                }
                field_teachers = {
                    field_name: torch.cat(field_teacher_rows[field_name], dim=0)
                    for field_name in available_fields
                }
            else:
                teachers = [
                    self.encode_reasoning_teacher_text(
                        str(sample.get("explicit_reasoning", "") or ""),
                        device=latent_summary.device,
                        dtype=latent_summary.dtype,
                    )
                    for sample in samples
                ]
                reasoning_teacher = torch.cat(teachers, dim=0)

        visual_debug = getattr(getattr(self.model, "model", None).visual, "last_debug_shapes", None)
        action_head_output = None
        flow_action_head_output = None
        if action_head_enabled and str(self.action_model) == "latent_two_way":
            flow_action_head_output = self._run_latent_two_way_action_head(
                hidden_states=hidden_states,
                latent_states=action_latent_states,
                latent_valid_mask=action_latent_valid_mask,
                img_next_state=img_next_state,
                sequence_summary=action_sequence_summary,
                inputs=inputs,
                image_counts_per_sample=[len(paths) for paths in batch_image_paths],
            )
        elif action_head_enabled and str(self.action_model) == "flow_matching":
            flow_action_head_output = self._run_flow_action_head(
                hidden_states=hidden_states,
                latent_summary=action_latent_summary,
                img_next_state=img_next_state,
                gold_action=gold_actions,
                sequence_summary=action_sequence_summary,
                inputs=inputs,
                image_counts_per_sample=[len(paths) for paths in batch_image_paths],
            )
        elif action_head_enabled:
            action_head_output = self._run_action_head(
                hidden_states=hidden_states,
                latent_summary=action_latent_summary,
                img_next_state=img_next_state,
                inputs=inputs,
                image_counts_per_sample=[len(paths) for paths in batch_image_paths],
                sequence_summary=action_sequence_summary,
            )

        return LaRAForwardOutput(
            loss=lm_loss,
            hidden_states=hidden_states,
            latent_reasoning_states=latent_states,
            latent_reasoning_summary=latent_summary,
            img_next_state=img_next_state,
            predicted_future_frame=predicted_future,
            target_future_frame=future_target,
            reasoning_teacher_embedding=reasoning_teacher,
            latent_reasoning_field_summaries=field_summaries,
            reasoning_teacher_field_embeddings=field_teachers,
            action_head_output=action_head_output,
            flow_action_head_output=flow_action_head_output,
            gold_action=gold_actions,
            action_text="\n\n".join(str(target or "") for target in assistant_targets),
            debug_info={
                "batch_size": len(samples),
                "prompt_lengths": [int(value) for value in prompt_lengths],
                "prompt_length": int(max(prompt_lengths) if prompt_lengths else 0),
                "visual_debug": visual_debug,
                "training_stage": training_stage,
                "stage2_target_format": stage2_target_format,
                "assistant_future_positions_count": int(sum(assistant_future_position_counts)),
                "assistant_target_preview": str(assistant_targets[0] or "")[:400],
                "assistant_target_char_count": int(sum(len(str(target or "")) for target in assistant_targets)),
                "stage2_explicit_keep_ratio": float(stage2_explicit_keep_ratio),
                "reasoning_alignment_mode": str(self.reasoning_alignment_mode),
                "reasoning_field_slot_counts": list(self.reasoning_field_slot_counts),
                "reasoning_aligned_fields": list(field_summaries or {}),
                "supervised_token_count": int(supervised_mask.sum().item()),
                "future_frame_enabled": bool(future_frame_enabled),
                "action_format": self.action_format,
                "action_model": self.action_model,
                "action_head_hidden_source": self._normalised_action_hidden_source(),
                "lm_action_target": self.lm_action_target,
            },
        )

    def forward(
        self,
        samples: list[dict[str, Any]],
        **kwargs: Any,
    ) -> LaRAForwardOutput:
        """Thin DDP-friendly entry point for batched training."""
        return self.forward_train_batch(samples, **kwargs)

    def compute_auxiliary_losses(
        self,
        output: LaRAForwardOutput,
        *,
        training_stage: str = "stage1",
    ) -> tuple[dict[str, Tensor], dict[str, float]]:
        if F is None:
            raise RuntimeError("Torch functional runtime is required.")
        losses: dict[str, Tensor] = {}
        metrics: dict[str, float] = {}
        if output.latent_reasoning_field_summaries and output.reasoning_teacher_field_embeddings:
            field_losses: list[Tensor] = []
            field_cosines: list[Tensor] = []
            for field_name in REASONING_FIELD_NAMES:
                if field_name not in output.latent_reasoning_field_summaries:
                    continue
                student = output.latent_reasoning_field_summaries[field_name]
                teacher = output.reasoning_teacher_field_embeddings[field_name].to(
                    device=student.device,
                    dtype=student.dtype,
                )
                cosine = F.cosine_similarity(student, teacher, dim=-1).mean()
                mse = F.mse_loss(student, teacher)
                field_loss = (1.0 - cosine) + 0.5 * mse
                field_losses.append(field_loss)
                field_cosines.append(cosine)
                losses[f"reasoning_alignment_{field_name}_loss"] = field_loss
                metrics[f"reasoning_alignment_{field_name}_loss"] = float(
                    field_loss.detach().item()
                )
                metrics[f"reasoning_{field_name}_cosine_similarity"] = float(
                    cosine.detach().item()
                )
                metrics[f"reasoning_{field_name}_aligned_samples"] = float(student.shape[0])
            if field_losses:
                losses["reasoning_alignment_loss"] = torch.stack(field_losses).mean()
                mean_cosine = torch.stack(field_cosines).mean()
                metrics["reasoning_cosine_similarity"] = float(mean_cosine.detach().item())
                metrics["reasoning_aligned_field_count"] = float(len(field_losses))
        elif output.reasoning_teacher_embedding is not None:
            cosine = F.cosine_similarity(
                output.latent_reasoning_summary,
                output.reasoning_teacher_embedding,
                dim=-1,
            ).mean()
            mse = F.mse_loss(output.latent_reasoning_summary, output.reasoning_teacher_embedding)
            reasoning_loss = (1.0 - cosine) + 0.5 * mse
            losses["reasoning_alignment_loss"] = reasoning_loss
            metrics["reasoning_cosine_similarity"] = float(cosine.detach().item())
        if output.target_future_frame is not None:
            cosine = F.cosine_similarity(
                output.predicted_future_frame,
                output.target_future_frame.to(
                    device=output.predicted_future_frame.device,
                    dtype=output.predicted_future_frame.dtype,
                ),
                dim=-1,
            ).mean()
            mse = F.mse_loss(
                output.predicted_future_frame,
                output.target_future_frame.to(
                    device=output.predicted_future_frame.device,
                    dtype=output.predicted_future_frame.dtype,
                ),
            )
            future_loss = (1.0 - cosine) + 0.5 * mse
            losses["future_frame_loss"] = future_loss
            metrics["future_frame_cosine_similarity"] = float(cosine.detach().item())
        if str(training_stage) != "stage1":
            normalized = F.normalize(output.latent_reasoning_states, dim=-1)
            sims = torch.matmul(normalized, normalized.transpose(1, 2))
            eye = torch.eye(int(sims.shape[-1]), device=sims.device, dtype=sims.dtype).unsqueeze(0)
            diversity_loss = (sims * (1.0 - eye)).pow(2).mean()
            losses["latent_diversity_loss"] = diversity_loss
            metrics["latent_diversity_loss"] = float(diversity_loss.detach().item())
        return losses, metrics

    def compute_action_head_losses(
        self,
        output: LaRAForwardOutput,
    ) -> tuple[dict[str, Tensor], dict[str, float]]:
        if F is None:
            raise RuntimeError("Torch functional runtime is required.")
        if str(self.action_model) in {"flow_matching", "latent_two_way"}:
            return self.compute_flow_action_head_losses(output)
        if output.action_head_output is None or output.gold_action is None:
            return {}, {}
        head = output.action_head_output
        gold_action = output.gold_action
        device = head.action_type_logits.device
        action_type = str(gold_action.get("type", "wait") or "wait")
        action_index = self.action_types.index(action_type) if action_type in self.action_types else self.action_types.index("wait")
        action_target = torch.tensor([action_index], device=device, dtype=torch.long)
        losses: dict[str, Tensor] = {
            "action_head_type_loss": F.cross_entropy(head.action_type_logits.float(), action_target),
        }
        metrics: dict[str, float] = {}

        x_norm = self._coerce_norm(gold_action.get("x_norm"))
        y_norm = self._coerce_norm(gold_action.get("y_norm"))
        is_pointer = action_type in POINTER_ACTION_TYPES and x_norm is not None and y_norm is not None
        null_index = int(head.target_with_null_probs.shape[-1]) - 1
        target_class = null_index
        if is_pointer:
            grid_height, grid_width = head.target_patch_grid_sizes[0]
            target_class = target_patch_index_from_point(
                x_norm=float(x_norm),
                y_norm=float(y_norm),
                grid_height=grid_height,
                grid_width=grid_width,
            )
            pointer_target = head.pointer_pred.new_tensor([[float(x_norm), float(y_norm)]])
            losses["action_head_pointer_reg_loss"] = F.smooth_l1_loss(head.pointer_pred, pointer_target)
            region_name = self._region_from_point(x_norm, y_norm)
            region_target = torch.tensor([self.region_labels.index(region_name)], device=device, dtype=torch.long)
            losses["action_head_region_loss"] = F.cross_entropy(head.region_logits.float(), region_target)
            metrics["action_head_pointer_l1"] = float((head.pointer_pred.detach() - pointer_target).abs().sum(dim=-1).mean().item())
        target_tensor = torch.tensor([int(target_class)], device=device, dtype=torch.long)
        losses["action_head_target_loss"] = F.cross_entropy(head.target_with_null_logits.float(), target_tensor)

        if action_type in {"terminate", "wait"}:
            status = str(gold_action.get("status", "success") or "success")
            status_index = self.terminate_statuses.index(status) if status in self.terminate_statuses else 0
            terminate_target = torch.tensor([status_index], device=device, dtype=torch.long)
            losses["action_head_terminate_loss"] = F.cross_entropy(head.terminate_logits.float(), terminate_target)
        if action_type == "scroll" and gold_action.get("amount") is not None:
            amount = max(-1.0, min(1.0, float(gold_action.get("amount", 0)) / 1000.0))
            amount_target = head.scroll_pred.new_tensor([[amount]])
            losses["action_head_scroll_loss"] = F.smooth_l1_loss(head.scroll_pred, amount_target)

        pred_type_index = int(head.action_type_logits.detach().argmax(dim=-1)[0].item())
        metrics["action_head_type_accuracy"] = float(pred_type_index == action_index)
        if is_pointer:
            pred_region_index = int(head.region_logits.detach().argmax(dim=-1)[0].item())
            metrics["action_head_region_accuracy"] = float(pred_region_index == int(region_target[0].item()))
        return losses, metrics

    def _soft_patch_target_from_point(
        self,
        *,
        x_norm: float,
        y_norm: float,
        grid_height: int,
        grid_width: int,
        token_count: int,
        reference: Tensor,
    ) -> Tensor:
        grid_height = max(1, int(grid_height))
        grid_width = max(1, int(grid_width))
        token_count = max(1, int(token_count))
        valid_token_count = grid_height * grid_width
        if valid_token_count > token_count:
            grid_height, grid_width = self.flow_action_head._fallback_grid_size(token_count)
            valid_token_count = grid_height * grid_width
        patch_indices = torch.arange(token_count, device=reference.device, dtype=torch.long)
        valid_mask = patch_indices < valid_token_count
        safe_indices = patch_indices.clamp(max=max(0, valid_token_count - 1))
        rows = torch.div(safe_indices, grid_width, rounding_mode="floor")
        cols = safe_indices % grid_width
        x_centers = (cols.to(dtype=torch.float32) + 0.5) / float(grid_width)
        y_centers = (rows.to(dtype=torch.float32) + 0.5) / float(grid_height)
        sigma = max(1e-4, float(getattr(self, "flow_patch_gaussian_sigma", 0.05) or 0.05))
        target_x = reference.new_tensor(float(x_norm), dtype=torch.float32)
        target_y = reference.new_tensor(float(y_norm), dtype=torch.float32)
        dist_sq = (x_centers - target_x).pow(2) + (y_centers - target_y).pow(2)
        scores = -dist_sq / (2.0 * sigma * sigma)
        scores = scores.masked_fill(~valid_mask, -1e9)
        return torch.softmax(scores, dim=-1).to(device=reference.device, dtype=torch.float32)

    def compute_flow_action_head_losses(
        self,
        output: LaRAForwardOutput,
    ) -> tuple[dict[str, Tensor], dict[str, Any]]:
        if F is None:
            raise RuntimeError("Torch functional runtime is required.")
        if output.flow_action_head_output is None or output.gold_action is None:
            return {}, {}
        head = output.flow_action_head_output
        gold_actions = output.gold_action if isinstance(output.gold_action, list) else [output.gold_action]
        device = head.action_type_logits.device
        action_types = [str(gold_action.get("type", "wait") or "wait") for gold_action in gold_actions]
        action_indices = [
            self.action_types.index(action_type) if action_type in self.action_types else self.action_types.index("wait")
            for action_type in action_types
        ]
        action_target = torch.tensor(action_indices, device=device, dtype=torch.long)
        losses: dict[str, Tensor] = {
            "action_head_type_loss": F.cross_entropy(head.action_type_logits.float(), action_target),
        }
        metrics: dict[str, Any] = {}
        pos_entropy = getattr(head, "pos_latent_attention_entropy", None)
        pos_attention_max = getattr(head, "pos_latent_attention_max", None)
        if pos_entropy is not None:
            metrics["two_way_pos_latent_attention_entropy"] = float(
                pos_entropy.detach().float().mean().item()
            )
        if pos_attention_max is not None:
            metrics["two_way_pos_latent_attention_max"] = float(
                pos_attention_max.detach().float().mean().item()
            )
        metrics["two_way_query_mode"] = str(
            getattr(head, "two_way_query_mode", "semantic_pool")
        )

        target, mask = self._continuous_action_target_and_mask_batch(gold_actions, reference=head.flow_velocity)
        target = target.to(device=head.flow_velocity.device, dtype=head.flow_velocity.dtype)
        mask = mask.to(device=head.flow_velocity.device, dtype=head.flow_velocity.dtype)
        flow_loss_enabled = float(getattr(self, "flow_action_loss_weight", 1.0) or 0.0) > 0.0
        coord_loss_enabled = float(getattr(self, "flow_coord_loss_weight", 0.0) or 0.0) > 0.0
        patch_loss_enabled = float(getattr(self, "flow_patch_loss_weight", 0.0) or 0.0) > 0.0
        predicted_target = None
        direct_target = getattr(head, "direct_continuous_action", None)
        direct_raw = getattr(head, "direct_continuous_raw", None)
        target_patch_logits = getattr(head, "target_patch_logits", None)
        target_patch_grid_sizes = getattr(head, "target_patch_grid_sizes", None)
        patch_continuous = getattr(head, "patch_continuous_action", None)
        patch_argmax = getattr(head, "patch_argmax_action", None)
        metrics["action_head_pointer_coord_source"] = str(
            getattr(head, "pointer_coord_source", getattr(self, "flow_pointer_coord_source", "patch_residual"))
        )
        if direct_target is not None:
            direct_target = self.flow_action_head._clamp_continuous(
                direct_target.to(device=target.device, dtype=target.dtype)
            )
        if direct_raw is not None:
            direct_raw = direct_raw.to(device=target.device, dtype=target.dtype)
        if patch_continuous is not None:
            patch_continuous = patch_continuous.to(device=target.device, dtype=target.dtype)
        if patch_argmax is not None:
            patch_argmax = patch_argmax.to(device=target.device, dtype=target.dtype)
        sampled_target = None
        pointer_rows_with_coords = [
            idx
            for idx, action_type in enumerate(action_types)
            if action_type in POINTER_ACTION_TYPES
            and self._coerce_norm(gold_actions[idx].get("x_norm")) is not None
            and self._coerce_norm(gold_actions[idx].get("y_norm")) is not None
        ]
        candidate_actions = getattr(head, "candidate_continuous_action", None)
        candidate_confidence_logits = getattr(head, "candidate_confidence_logits", None)
        if (
            str(self.action_model) == "latent_two_way"
            and pointer_rows_with_coords
            and candidate_actions is not None
            and candidate_confidence_logits is not None
        ):
            candidate_rows = candidate_actions[pointer_rows_with_coords, :, :2].float()
            candidate_targets = target[pointer_rows_with_coords, :2].float()
            candidate_errors = (
                candidate_rows - candidate_targets.unsqueeze(1)
            ).abs().sum(dim=-1)
            winning_candidates = candidate_errors.detach().argmin(dim=-1)
            winning_xy = candidate_rows.gather(
                1,
                winning_candidates.view(-1, 1, 1).expand(-1, 1, 2),
            ).squeeze(1)
            losses["two_way_candidate_coord_loss"] = F.smooth_l1_loss(
                winning_xy,
                candidate_targets,
            )
            losses["two_way_candidate_confidence_loss"] = F.cross_entropy(
                candidate_confidence_logits[pointer_rows_with_coords].float(),
                winning_candidates,
            )
            metrics["two_way_best_candidate_pointer_l1"] = float(
                candidate_errors.detach().min(dim=-1).values.mean().item()
            )
            metrics["two_way_location_confidence"] = float(
                torch.softmax(
                    candidate_confidence_logits[pointer_rows_with_coords].detach().float(),
                    dim=-1,
                ).max(dim=-1).values.mean().item()
            )
        if patch_loss_enabled and target_patch_logits is not None and target_patch_grid_sizes is not None:
            patch_loss_mode = str(getattr(self, "flow_patch_loss_mode", "ce") or "ce").lower()
            patch_target_indices: list[int] = []
            patch_rows: list[int] = []
            soft_patch_targets: list[Tensor] = []
            for row_index in pointer_rows_with_coords:
                if row_index >= len(target_patch_grid_sizes):
                    continue
                grid_height, grid_width = target_patch_grid_sizes[row_index]
                x_norm = self._coerce_norm(gold_actions[row_index].get("x_norm"))
                y_norm = self._coerce_norm(gold_actions[row_index].get("y_norm"))
                if x_norm is None or y_norm is None:
                    continue
                patch_target_indices.append(
                    target_patch_index_from_point(
                        x_norm=float(x_norm),
                        y_norm=float(y_norm),
                        grid_height=int(grid_height),
                        grid_width=int(grid_width),
                    )
                )
                if patch_loss_mode == "gaussian":
                    soft_patch_targets.append(
                        self._soft_patch_target_from_point(
                            x_norm=float(x_norm),
                            y_norm=float(y_norm),
                            grid_height=int(grid_height),
                            grid_width=int(grid_width),
                            token_count=int(target_patch_logits.shape[1]),
                            reference=target_patch_logits,
                        )
                    )
                patch_rows.append(row_index)
            if patch_rows:
                patch_targets = torch.tensor(patch_target_indices, device=device, dtype=torch.long)
                patch_logits = target_patch_logits[patch_rows]
                if patch_loss_mode == "gaussian" and len(soft_patch_targets) == len(patch_rows):
                    soft_targets = torch.stack(soft_patch_targets, dim=0).to(device=device, dtype=torch.float32)
                    log_probs = F.log_softmax(patch_logits.float(), dim=-1)
                    losses["flow_action_patch_loss"] = -(soft_targets * log_probs).sum(dim=-1).mean()
                    metrics["action_head_patch_target_entropy"] = float(
                        (-(soft_targets * soft_targets.clamp_min(1e-12).log()).sum(dim=-1).mean()).detach().item()
                    )
                else:
                    losses["flow_action_patch_loss"] = F.cross_entropy(patch_logits.float(), patch_targets)
                patch_preds = patch_logits.detach().argmax(dim=-1)
                metrics["action_head_patch_accuracy"] = float(
                    (patch_preds == patch_targets).float().mean().item()
                )
                metrics["action_head_patch_loss_mode"] = patch_loss_mode
                metrics["action_head_patch_target_prob"] = float(
                    torch.softmax(patch_logits.detach().float(), dim=-1)
                    .gather(dim=1, index=patch_targets.unsqueeze(-1))
                    .mean()
                    .item()
                )
        if bool(mask.gt(0).any().item()):
            flow_velocity = torch.nan_to_num(head.flow_velocity, nan=0.0, posinf=4.0, neginf=-4.0).clamp(-4.0, 4.0)
            if flow_loss_enabled:
                flow_target_velocity = torch.nan_to_num(
                    head.flow_target_velocity,
                    nan=0.0,
                    posinf=4.0,
                    neginf=-4.0,
                ).clamp(-4.0, 4.0)
                flow_diff = (flow_velocity - flow_target_velocity) * mask
                losses["flow_action_loss"] = flow_diff.float().pow(2).sum() / mask.float().sum().clamp_min(1.0)
            else:
                losses["flow_action_loss"] = head.flow_velocity.new_zeros(())
            predicted_target = head.flow_noisy_action + flow_velocity * (1.0 - head.flow_t)
            predicted_target = self.flow_action_head._clamp_continuous(predicted_target)
            if coord_loss_enabled:
                coord_scale_value = max(1.0, float(getattr(self, "flow_coord_loss_scale", 1.0) or 1.0))
                coord_loss_space = str(getattr(self, "flow_coord_loss_space", "logit") or "logit").lower()
                if coord_loss_space == "logit" and direct_raw is not None and target.shape[-1] >= 2:
                    eps = target.new_tensor(1e-4)
                    target_xy = target[:, :2].clamp(min=eps, max=1.0 - eps)
                    target_xy_logit = torch.logit(target_xy.float()).to(dtype=target.dtype)
                    coord_mask = mask[:, :2]
                    coord_diff = (direct_raw[:, :2] - target_xy_logit) * coord_mask
                    coord_denominator = coord_mask.float().sum().clamp_min(1.0)
                else:
                    coord_prediction = direct_target if direct_target is not None else predicted_target
                    coord_scale = torch.ones_like(target)
                    if coord_loss_space == "scaled":
                        if coord_scale.shape[-1] >= 1:
                            coord_scale[:, 0:1] = coord_scale_value
                        if coord_scale.shape[-1] >= 2:
                            coord_scale[:, 1:2] = coord_scale_value
                    coord_diff = (coord_prediction - target) * mask * coord_scale
                    coord_denominator = mask.float().sum().clamp_min(1.0)
                losses["flow_action_coord_loss"] = F.smooth_l1_loss(
                    coord_diff.float(),
                    torch.zeros_like(coord_diff).float(),
                    reduction="sum",
                ) / coord_denominator
                metrics["action_head_coord_loss"] = float(losses["flow_action_coord_loss"].detach().item())
                metrics["action_head_coord_loss_scale"] = coord_scale_value
                metrics["action_head_coord_loss_space"] = coord_loss_space
            else:
                losses["flow_action_coord_loss"] = head.flow_velocity.new_zeros(())
            source_for_metrics = (
                direct_target
                if (
                    str(self.action_model) == "latent_two_way"
                    or str(getattr(self, "flow_continuous_source", "sample") or "sample") == "direct"
                )
                and direct_target is not None
                else predicted_target
            )
            teacher_pointer_l1 = float(
                ((source_for_metrics[:, :2].detach() - target[:, :2]).abs() * mask[:, :2]).sum().item()
                / max(1.0, float(mask[:, :2].sum().item()))
            )
            metrics["action_head_pointer_l1"] = teacher_pointer_l1
            metrics["action_head_teacher_pointer_l1"] = teacher_pointer_l1
            if patch_continuous is not None:
                metrics["action_head_patch_pointer_l1"] = float(
                    ((patch_continuous[:, :2].detach() - target[:, :2]).abs() * mask[:, :2]).sum().item()
                    / max(1.0, float(mask[:, :2].sum().item()))
                )
            if patch_argmax is not None:
                metrics["action_head_patch_argmax_pointer_l1"] = float(
                    ((patch_argmax[:, :2].detach() - target[:, :2]).abs() * mask[:, :2]).sum().item()
                    / max(1.0, float(mask[:, :2].sum().item()))
                )
            patch_residual = getattr(head, "patch_residual", None)
            if patch_residual is not None:
                metrics["action_head_patch_residual_abs"] = float(
                    patch_residual.detach().float().abs().mean().item()
                )
        if bool(getattr(self, "include_flow_training_sample_metrics", False)) and getattr(head, "fused", None) is not None:
                with torch.no_grad():
                    sampled_target = self.flow_action_head.sample(
                        fused=head.fused.detach(),
                        steps=max(1, int(self.flow_action_sample_steps)),
                    ).to(device=target.device, dtype=target.dtype)
                    sampled_target = self.flow_action_head._clamp_continuous(sampled_target)
                metrics["action_head_sampled_pointer_l1"] = float(
                    ((sampled_target[:, :2] - target[:, :2]).abs() * mask[:, :2]).sum().item()
                    / max(1.0, float(mask[:, :2].sum().item()))
                )
        else:
            losses["flow_action_loss"] = head.flow_velocity.new_zeros(())
            losses["flow_action_coord_loss"] = head.flow_velocity.new_zeros(())

        terminate_rows = [idx for idx, action_type in enumerate(action_types) if action_type in {"terminate", "wait"}]
        if terminate_rows:
            status_indices = []
            for row_index in terminate_rows:
                status = str(gold_actions[row_index].get("status", "success") or "success")
                status_indices.append(self.terminate_statuses.index(status) if status in self.terminate_statuses else 0)
            terminate_target = torch.tensor(status_indices, device=device, dtype=torch.long)
            losses["action_head_terminate_loss"] = F.cross_entropy(
                head.terminate_logits[terminate_rows].float(),
                terminate_target,
            )

        if bool(getattr(self, "ddp_zero_unused_branch_anchors", False)):
            # DDP's unused-parameter search starts from all tensors returned by
            # forward, not from the subset later selected by the label-dependent
            # losses. Connect every executed action branch to the scalar loss
            # with an exact zero so click-only/terminate-only batches reduce the
            # same parameter set without changing the optimization objective.
            branch_tensors = [
                head.action_type_logits,
                head.terminate_logits,
                head.flow_velocity,
                direct_raw,
                target_patch_logits,
                getattr(head, "patch_residual", None),
                candidate_confidence_logits,
                candidate_actions,
            ]
            zero_anchor = head.action_type_logits.new_zeros((), dtype=torch.float32)
            for branch_tensor in branch_tensors:
                if isinstance(branch_tensor, torch.Tensor) and branch_tensor.numel() > 0:
                    branch_scalar = torch.nan_to_num(
                        branch_tensor.reshape(-1)[0].float(),
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0,
                    )
                    zero_anchor = zero_anchor + branch_scalar * 0.0
            losses["action_head_ddp_zero_anchor"] = zero_anchor

        pred_type_indices = head.action_type_logits.detach().argmax(dim=-1)
        metrics["action_head_type_accuracy"] = float((pred_type_indices == action_target).float().mean().item())
        pointer_rows = [idx for idx, action_type in enumerate(action_types) if action_type in POINTER_ACTION_TYPES]
        if pointer_rows:
            source_name = (
                "direct"
                if str(self.action_model) == "latent_two_way"
                else str(getattr(self, "flow_continuous_source", "sample") or "sample")
            )
            if source_name == "direct" and direct_target is not None:
                primary_prediction = direct_target
            elif predicted_target is not None:
                primary_prediction = predicted_target
            else:
                primary_prediction = head.flow_noisy_action + torch.nan_to_num(
                    head.flow_velocity,
                    nan=0.0,
                    posinf=4.0,
                    neginf=-4.0,
                ).clamp(-4.0, 4.0) * (1.0 - head.flow_t)
                primary_prediction = self.flow_action_head._clamp_continuous(primary_prediction)
            correct = 0
            sampled_correct = 0
            direct_correct = 0
            total = 0
            for row_index in pointer_rows:
                primary_x = max(0.0, min(1.0, float(primary_prediction.detach()[row_index, 0].float().cpu().item())))
                primary_y = max(0.0, min(1.0, float(primary_prediction.detach()[row_index, 1].float().cpu().item())))
                pred_region = self._region_from_point(primary_x, primary_y)
                if sampled_target is not None:
                    multi_x = max(0.0, min(1.0, float(sampled_target.detach()[row_index, 0].float().cpu().item())))
                    multi_y = max(0.0, min(1.0, float(sampled_target.detach()[row_index, 1].float().cpu().item())))
                    sampled_region = self._region_from_point(multi_x, multi_y)
                else:
                    sampled_region = pred_region
                if direct_target is not None:
                    direct_x = float(direct_target.detach()[row_index, 0].float().cpu().item())
                    direct_y = float(direct_target.detach()[row_index, 1].float().cpu().item())
                    direct_region = self._region_from_point(
                        max(0.0, min(1.0, direct_x)),
                        max(0.0, min(1.0, direct_y)),
                    )
                else:
                    direct_region = pred_region
                gold_region = self._region_from_point(
                    self._coerce_norm(gold_actions[row_index].get("x_norm")),
                    self._coerce_norm(gold_actions[row_index].get("y_norm")),
                )
                correct += int(pred_region == gold_region)
                sampled_correct += int(sampled_region == gold_region)
                direct_correct += int(direct_region == gold_region)
                total += 1
            metrics["action_head_region_accuracy"] = float(correct / max(1, total))
            if sampled_target is not None:
                metrics["action_head_sampled_region_accuracy"] = float(sampled_correct / max(1, total))
            metrics["action_head_direct_region_accuracy"] = float(direct_correct / max(1, total))
            coord_preview = []
            for row_index in pointer_rows[:3]:
                gold_x = self._coerce_norm(gold_actions[row_index].get("x_norm"))
                gold_y = self._coerce_norm(gold_actions[row_index].get("y_norm"))
                teacher_x = (
                    float(primary_prediction.detach()[row_index, 0].float().cpu().item())
                    if primary_prediction is not None
                    else None
                )
                teacher_y = (
                    float(primary_prediction.detach()[row_index, 1].float().cpu().item())
                    if primary_prediction is not None
                    else None
                )
                if sampled_target is not None:
                    final_x = float(sampled_target.detach()[row_index, 0].float().cpu().item())
                    final_y = float(sampled_target.detach()[row_index, 1].float().cpu().item())
                else:
                    final_x = None
                    final_y = None
                if direct_target is not None:
                    direct_x = float(direct_target.detach()[row_index, 0].float().cpu().item())
                    direct_y = float(direct_target.detach()[row_index, 1].float().cpu().item())
                else:
                    direct_x = None
                    direct_y = None
                coord_preview.append(
                    {
                        "row_index": int(row_index),
                        "action_type": action_types[row_index],
                        "gt_x": round(float(gold_x), 4),
                        "gt_y": round(float(gold_y), 4),
                        "teacher_x": round(float(max(0.0, min(1.0, teacher_x))), 4)
                        if teacher_x is not None
                        else None,
                        "teacher_y": round(float(max(0.0, min(1.0, teacher_y))), 4)
                        if teacher_y is not None
                        else None,
                        "sampled_x": round(float(max(0.0, min(1.0, final_x))), 4)
                        if final_x is not None
                        else None,
                        "sampled_y": round(float(max(0.0, min(1.0, final_y))), 4)
                        if final_y is not None
                        else None,
                        "direct_x": round(float(max(0.0, min(1.0, direct_x))), 4)
                        if direct_x is not None
                        else None,
                        "direct_y": round(float(max(0.0, min(1.0, direct_y))), 4)
                        if direct_y is not None
                        else None,
                    }
                )
            metrics["action_head_coord_preview"] = coord_preview
        return losses, metrics

    def action_head_output_to_action(self, head: UnifiedActionHeadOutput) -> dict[str, Any]:
        action_type_index = int(head.action_type_logits.detach().argmax(dim=-1)[0].item())
        action_type = self.action_types[action_type_index]
        action: dict[str, Any] = {"type": action_type}
        if action_type in POINTER_ACTION_TYPES:
            pointer = head.pointer_pred.detach()[0].float().cpu().tolist()
            action["x_norm"] = round(max(0.0, min(1.0, float(pointer[0]))), 4)
            action["y_norm"] = round(max(0.0, min(1.0, float(pointer[1]))), 4)
            action["region"] = self._region_from_point(action["x_norm"], action["y_norm"])
        elif action_type == "scroll":
            amount = float(head.scroll_pred.detach()[0, 0].float().cpu().item())
            action["amount"] = int(max(-1.0, min(1.0, amount)) * 1000.0)
        elif action_type in {"terminate", "wait"}:
            status_index = int(head.terminate_logits.detach().argmax(dim=-1)[0].item())
            action["status"] = self.terminate_statuses[status_index]
        return action

    def flow_action_output_to_action(
        self,
        *,
        fused: Tensor,
        head: FlowMatchingActionHeadOutput,
    ) -> dict[str, Any]:
        action_type_index = int(head.action_type_logits.detach().argmax(dim=-1)[0].item())
        action_type = self.action_types[action_type_index]
        continuous = self._select_flow_continuous_action(fused=fused, head=head)
        vector = continuous.detach()[0].float().cpu().tolist()
        terminate_index = int(head.terminate_logits.detach().argmax(dim=-1)[0].item())
        return self._flow_vector_to_action(
            action_type=action_type,
            vector=vector,
            terminate_index=terminate_index,
        )

    def _flow_vector_to_action(
        self,
        *,
        action_type: str,
        vector: list[float],
        terminate_index: int,
    ) -> dict[str, Any]:
        action: dict[str, Any] = {"type": action_type}
        if action_type in POINTER_ACTION_TYPES:
            x_norm = round(max(0.0, min(1.0, float(vector[0]))), 4)
            y_norm = round(max(0.0, min(1.0, float(vector[1]))), 4)
            action["x_norm"] = x_norm
            action["y_norm"] = y_norm
            action["region"] = self._region_from_point(x_norm, y_norm)
        elif action_type == "scroll":
            amount = float(vector[2]) if len(vector) > 2 else 0.0
            action["amount"] = int(max(-1.0, min(1.0, amount)) * 1000.0)
        elif action_type in {"terminate", "wait"}:
            action["status"] = self.terminate_statuses[int(terminate_index)]
        return action

    def _select_flow_continuous_action(
        self,
        *,
        fused: Tensor,
        head: FlowMatchingActionHeadOutput,
    ) -> Tensor:
        source = str(getattr(self, "flow_continuous_source", "sample") or "sample")
        if (
            str(self.action_model) == "latent_two_way"
            or source == "direct"
        ) and getattr(head, "direct_continuous_action", None) is not None:
            return self.flow_action_head._clamp_continuous(head.direct_continuous_action)
        return self.flow_action_head.sample(
            fused=fused,
            steps=max(1, int(self.flow_action_sample_steps)),
        )

    def predict_action_with_head(
        self,
        *,
        image_paths: list[str | Path | Image.Image],
        task: str,
        history_frame_count: int,
        current_subtask: str | None,
        expected_next_screen: str | None,
        temporal_sample_key: str | None = None,
    ) -> dict[str, Any]:
        self._active_training_stage = str(getattr(self, "training_stage", "stage1"))
        user_prompt = self.build_user_prompt(
            task=task,
            history_frame_count=history_frame_count,
            current_subtask=current_subtask,
            expected_next_screen=expected_next_screen,
            action_only_output=self.action_only_output,
            action_format=self.action_format,
            include_action_in_lm=False,
        )
        assistant_prefix = None if self.latent_scaffolds_in_prompt else self.action_head_assistant_prefix()
        inputs, prompt_length = self.prepare_conversation_inputs(
            image_paths=image_paths,
            user_prompt=user_prompt,
            assistant_target=assistant_prefix,
            continue_assistant=bool(assistant_prefix),
        )
        sample_keys = [str(temporal_sample_key)] if temporal_sample_key else None
        self._set_temporal_context(sample_keys)
        with torch.no_grad():
            outputs = self.model.model(
                **inputs,
                output_hidden_states=False,
                return_dict=True,
            )
        self._set_temporal_context(None)
        hidden_states = outputs.last_hidden_state
        self._align_auxiliary_modules(hidden_states)
        future_slot_ids = [self.special_token_to_id[token] for token in self.future_slot_tokens]
        img_next_positions = []
        if assistant_prefix:
            img_next_positions = self._find_token_positions_after(
                inputs["input_ids"], future_slot_ids, prompt_length
            )
        if not img_next_positions:
            img_next_positions = self._find_token_positions_before(inputs["input_ids"], future_slot_ids, prompt_length)
        if not img_next_positions:
            img_next_positions = self._find_token_positions(inputs["input_ids"], future_slot_ids)
        if not img_next_positions:
            return {"_parse_error": "IMG_NEXT future slot tokens not found", "action": {"type": "wait", "status": "success"}}
        img_next_state = self._pool_positions(
            hidden_states,
            row_index=0,
            positions=img_next_positions,
            use_attention=self._uses_slot_attention_summary(),
            query=self.action_slot_query,
        ).unsqueeze(0)
        latent_token_ids = [self.special_token_to_id[token] for token in self.latent_slot_tokens]
        latent_positions = (
            self._find_token_positions_after(inputs["input_ids"], latent_token_ids, prompt_length)
            if assistant_prefix
            else self._find_token_positions(inputs["input_ids"], latent_token_ids)
        )
        if not latent_positions:
            latent_positions = img_next_positions
        latent_raw_summary = self._pool_positions(
            hidden_states,
            row_index=0,
            positions=latent_positions,
            use_attention=self._uses_slot_attention_summary(),
            query=self.action_slot_query,
        ).unsqueeze(0)
        latent_summary = self.reasoning_norm(self.reasoning_proj(latent_raw_summary))
        if str(self.action_model) == "latent_two_way":
            sequence_summary = self._action_sequence_summary(hidden_states, prompt_length)
            latent_token_states = hidden_states[:, latent_positions, :]
            head = self._run_latent_two_way_action_head(
                hidden_states=hidden_states,
                latent_states=latent_token_states,
                latent_valid_mask=torch.ones(
                    latent_token_states.shape[:2],
                    device=latent_token_states.device,
                    dtype=torch.bool,
                ),
                img_next_state=img_next_state,
                sequence_summary=sequence_summary,
                inputs=inputs,
            )
            action = self.flow_action_output_to_action(fused=head.fused, head=head)
            pointer_grounding_required = str(action.get("type", "")) in POINTER_ACTION_TYPES
            pos_attention = getattr(head, "pos_latent_attention", None)
            pos_entropy = getattr(head, "pos_latent_attention_entropy", None)
            pos_attention_max = getattr(head, "pos_latent_attention_max", None)
            return {
                "action": action,
                "raw_text": (
                    "<|POS|>" if pointer_grounding_required else "<latent_two_way_action_head>"
                ),
                "two_way_query_mode": str(getattr(head, "two_way_query_mode", "semantic_pool")),
                "pointer_grounding_required": pointer_grounding_required,
                "two_way_pos_latent_attention": (
                    pos_attention.detach().float().cpu().tolist()[0]
                    if pos_attention is not None
                    else None
                ),
                "two_way_pos_latent_attention_entropy": (
                    float(pos_entropy.detach().float().cpu().item())
                    if pos_entropy is not None
                    else None
                ),
                "two_way_pos_latent_attention_max": (
                    float(pos_attention_max.detach().float().cpu().item())
                    if pos_attention_max is not None
                    else None
                ),
            }
        if str(self.action_model) == "flow_matching":
            sequence_summary = self._action_sequence_summary(hidden_states, prompt_length)
            head = self._run_flow_action_head(
                hidden_states=hidden_states,
                latent_summary=latent_summary,
                img_next_state=img_next_state,
                inputs=inputs,
                sequence_summary=sequence_summary,
            )
            fused = head.fused
            return {
                "action": self.flow_action_output_to_action(fused=fused, head=head),
                "raw_text": "<flow_matching_action_head>",
            }
        head = self._run_action_head(
            hidden_states=hidden_states,
            latent_summary=latent_summary,
            img_next_state=img_next_state,
            inputs=inputs,
            sequence_summary=self._action_sequence_summary(hidden_states, prompt_length),
        )
        if head is None:
            return {"_parse_error": "No visual tokens found for action head", "action": {"type": "wait", "status": "success"}}
        return {"action": self.action_head_output_to_action(head), "raw_text": "<action_head>"}

    def predict_actions_with_head_batch(
        self,
        samples: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not samples:
            return []
        self._active_training_stage = str(getattr(self, "training_stage", "stage1"))
        batch_image_paths: list[list[str | Path | Image.Image]] = []
        user_prompts: list[str] = []
        assistant_prefixes: list[str | None] = []
        temporal_keys: list[str] = []
        for sample in samples:
            image_paths = list(sample["image_paths"])
            batch_image_paths.append(image_paths)
            user_prompts.append(
                self.build_user_prompt(
                    task=str(sample.get("task", "") or ""),
                    history_frame_count=int(sample.get("history_frame_count", max(0, len(image_paths) - 1))),
                    current_subtask=sample.get("current_subtask"),
                    expected_next_screen=sample.get("expected_next_screen"),
                    action_only_output=self.action_only_output,
                    action_format=self.action_format,
                    include_action_in_lm=False,
                )
            )
            assistant_prefixes.append(
                None if self.latent_scaffolds_in_prompt else self.action_head_assistant_prefix()
            )
            temporal_sample_key = str(sample.get("temporal_sample_key", "") or "")
            for image_index in range(len(image_paths)):
                temporal_keys.append(f"{temporal_sample_key}|img_{image_index}" if temporal_sample_key else "")

        inputs, prompt_lengths = self.prepare_conversation_inputs_batch(
            batch_image_paths=batch_image_paths,
            user_prompts=user_prompts,
            assistant_targets=assistant_prefixes,
            continue_assistant=all(prefix is not None for prefix in assistant_prefixes),
        )
        sample_keys = temporal_keys if temporal_keys and all(temporal_keys) else None
        self._set_temporal_context(sample_keys)
        with torch.no_grad():
            outputs = self.model.model(
                **inputs,
                output_hidden_states=False,
                return_dict=True,
            )
        self._set_temporal_context(None)

        hidden_states = outputs.last_hidden_state
        self._align_auxiliary_modules(hidden_states)
        future_slot_ids = [self.special_token_to_id[token] for token in self.future_slot_tokens]
        img_next_rows: list[Tensor] = []
        latent_summary_rows: list[Tensor] = []
        latent_token_rows: list[Tensor] = []
        for row_index, prompt_length in enumerate(prompt_lengths):
            assistant_prefix = assistant_prefixes[row_index]
            img_next_positions = []
            if assistant_prefix:
                img_next_positions = self._find_token_positions_after(
                    inputs["input_ids"],
                    future_slot_ids,
                    int(prompt_length),
                    row_index=row_index,
                )
            if not img_next_positions:
                img_next_positions = self._find_token_positions_before(
                    inputs["input_ids"],
                    future_slot_ids,
                    int(prompt_length),
                    row_index=row_index,
                )
            if not img_next_positions:
                img_next_positions = self._find_token_positions(
                    inputs["input_ids"],
                    future_slot_ids,
                    row_index=row_index,
                )
            if not img_next_positions:
                return [
                    {
                        "_parse_error": "IMG_NEXT future slot tokens not found",
                        "action": {"type": "wait", "status": "success"},
                    }
                    for _ in samples
                ]
            img_next_rows.append(
                self._pool_positions(
                    hidden_states,
                    row_index=row_index,
                    positions=img_next_positions,
                    use_attention=self._uses_slot_attention_summary(),
                    query=self.action_slot_query,
                )
            )

            latent_token_ids = [self.special_token_to_id[token] for token in self.latent_slot_tokens]
            latent_positions = (
                self._find_token_positions_after(
                    inputs["input_ids"],
                    latent_token_ids,
                    int(prompt_length),
                    row_index=row_index,
                )
                if assistant_prefix
                else self._find_token_positions(
                    inputs["input_ids"],
                    latent_token_ids,
                    row_index=row_index,
                )
            )
            if not latent_positions:
                latent_positions = img_next_positions
            latent_token_rows.append(hidden_states[row_index, latent_positions, :])
            latent_summary_rows.append(
                self._pool_positions(
                    hidden_states,
                    row_index=row_index,
                    positions=latent_positions,
                    use_attention=self._uses_slot_attention_summary(),
                    query=self.action_slot_query,
                )
            )

        img_next_state = torch.stack(img_next_rows, dim=0)
        latent_pre_summary = torch.stack(latent_summary_rows, dim=0)
        latent_summary = self.reasoning_norm(self.reasoning_proj(latent_pre_summary))
        action_latent_states, action_latent_valid_mask = self._pad_latent_state_rows(
            latent_token_rows
        )

        if str(self.action_model) in {"flow_matching", "latent_two_way"}:
            sequence_summary = self._action_sequence_summary(hidden_states, prompt_lengths)
            if str(self.action_model) == "latent_two_way":
                head = self._run_latent_two_way_action_head(
                    hidden_states=hidden_states,
                    latent_states=action_latent_states,
                    latent_valid_mask=action_latent_valid_mask,
                    img_next_state=img_next_state,
                    sequence_summary=sequence_summary,
                    inputs=inputs,
                    image_counts_per_sample=[len(paths) for paths in batch_image_paths],
                )
            else:
                head = self._run_flow_action_head(
                    hidden_states=hidden_states,
                    latent_summary=latent_summary,
                    img_next_state=img_next_state,
                    inputs=inputs,
                    image_counts_per_sample=[len(paths) for paths in batch_image_paths],
                    sequence_summary=sequence_summary,
                )
            fused = head.fused
            selected_continuous = self._select_flow_continuous_action(fused=fused, head=head)
            direct_continuous = getattr(head, "direct_continuous_action", None)
            direct_continuous = (
                self.flow_action_head._clamp_continuous(direct_continuous)
                if direct_continuous is not None
                else None
            )
            sample_continuous = None
            include_alternatives = bool(getattr(self, "include_flow_alternatives", False))
            source_name = (
                "direct"
                if str(self.action_model) == "latent_two_way"
                else str(getattr(self, "flow_continuous_source", "sample") or "sample")
            )
            if source_name == "sample":
                sample_continuous = selected_continuous
            elif include_alternatives and str(self.action_model) == "flow_matching":
                sample_continuous = self.flow_action_head.sample(
                    fused=fused,
                    steps=max(1, int(self.flow_action_sample_steps)),
                )
            action_type_indices = head.action_type_logits.detach().argmax(dim=-1).cpu().tolist()
            terminate_indices = head.terminate_logits.detach().argmax(dim=-1).cpu().tolist()
            vectors = selected_continuous.detach().float().cpu().tolist()
            direct_vectors = (
                direct_continuous.detach().float().cpu().tolist() if direct_continuous is not None else None
            )
            sample_vectors = (
                sample_continuous.detach().float().cpu().tolist() if sample_continuous is not None else None
            )
            patch_continuous = getattr(head, "patch_continuous_action", None)
            patch_argmax_continuous = getattr(head, "patch_argmax_action", None)
            patch_vectors = (
                patch_continuous.detach().float().cpu().tolist() if patch_continuous is not None else None
            )
            patch_argmax_vectors = (
                patch_argmax_continuous.detach().float().cpu().tolist()
                if patch_argmax_continuous is not None
                else None
            )
            pos_attention = getattr(head, "pos_latent_attention", None)
            pos_attention_vectors = (
                pos_attention.detach().float().cpu().tolist() if pos_attention is not None else None
            )
            pos_entropy = getattr(head, "pos_latent_attention_entropy", None)
            pos_entropy_values = (
                pos_entropy.detach().float().cpu().tolist() if pos_entropy is not None else None
            )
            pos_attention_max = getattr(head, "pos_latent_attention_max", None)
            pos_attention_max_values = (
                pos_attention_max.detach().float().cpu().tolist()
                if pos_attention_max is not None
                else None
            )
            results: list[dict[str, Any]] = []
            for row_index, action_type_index in enumerate(action_type_indices):
                action_type = self.action_types[int(action_type_index)]
                terminate_index = int(terminate_indices[row_index])
                action = self._flow_vector_to_action(
                    action_type=action_type,
                    vector=vectors[row_index],
                    terminate_index=terminate_index,
                )
                row_result: dict[str, Any] = {
                    "action": action,
                    "raw_text": (
                        (
                            "<|POS|>"
                            if action_type in POINTER_ACTION_TYPES
                            else "<latent_two_way_action_head_batch>"
                        )
                        if str(self.action_model) == "latent_two_way"
                        else "<flow_matching_action_head_batch>"
                    ),
                    "flow_continuous_source": source_name,
                }
                if str(self.action_model) == "latent_two_way":
                    row_result["two_way_query_mode"] = str(
                        getattr(head, "two_way_query_mode", "semantic_pool")
                    )
                    row_result["pointer_grounding_required"] = action_type in POINTER_ACTION_TYPES
                    row_result["two_way_pos_latent_attention"] = (
                        pos_attention_vectors[row_index] if pos_attention_vectors is not None else None
                    )
                    row_result["two_way_pos_latent_attention_entropy"] = (
                        float(pos_entropy_values[row_index]) if pos_entropy_values is not None else None
                    )
                    row_result["two_way_pos_latent_attention_max"] = (
                        float(pos_attention_max_values[row_index])
                        if pos_attention_max_values is not None
                        else None
                    )
                if include_alternatives:
                    if direct_vectors is not None:
                        row_result["flow_direct_action"] = self._flow_vector_to_action(
                            action_type=action_type,
                            vector=direct_vectors[row_index],
                            terminate_index=terminate_index,
                        )
                    if sample_vectors is not None:
                        row_result["flow_sample_action"] = self._flow_vector_to_action(
                            action_type=action_type,
                            vector=sample_vectors[row_index],
                            terminate_index=terminate_index,
                        )
                    if patch_vectors is not None:
                        row_result["flow_patch_action"] = self._flow_vector_to_action(
                            action_type=action_type,
                            vector=patch_vectors[row_index],
                            terminate_index=terminate_index,
                        )
                    if patch_argmax_vectors is not None:
                        row_result["flow_patch_argmax_action"] = self._flow_vector_to_action(
                            action_type=action_type,
                            vector=patch_argmax_vectors[row_index],
                            terminate_index=terminate_index,
                        )
                results.append(row_result)
            return results

        head = self._run_action_head(
            hidden_states=hidden_states,
            latent_summary=latent_summary,
            img_next_state=img_next_state,
            inputs=inputs,
            image_counts_per_sample=[len(paths) for paths in batch_image_paths],
            sequence_summary=self._action_sequence_summary(hidden_states, prompt_lengths),
        )
        if head is None:
            return [
                {"_parse_error": "No visual tokens found for action head", "action": {"type": "wait", "status": "success"}}
                for _ in samples
            ]
        action_type_indices = head.action_type_logits.detach().argmax(dim=-1).cpu().tolist()
        terminate_indices = head.terminate_logits.detach().argmax(dim=-1).cpu().tolist()
        pointers = head.pointer_pred.detach().float().cpu().tolist()
        scrolls = head.scroll_pred.detach().float().cpu().view(-1).tolist()
        results = []
        for row_index, action_type_index in enumerate(action_type_indices):
            action_type = self.action_types[int(action_type_index)]
            action = {"type": action_type}
            if action_type in POINTER_ACTION_TYPES:
                x_norm = round(max(0.0, min(1.0, float(pointers[row_index][0]))), 4)
                y_norm = round(max(0.0, min(1.0, float(pointers[row_index][1]))), 4)
                action["x_norm"] = x_norm
                action["y_norm"] = y_norm
                action["region"] = self._region_from_point(x_norm, y_norm)
            elif action_type == "scroll":
                action["amount"] = int(max(-1.0, min(1.0, float(scrolls[row_index]))) * 1000.0)
            elif action_type in {"terminate", "wait"}:
                action["status"] = self.terminate_statuses[int(terminate_indices[row_index])]
            results.append({"action": action, "raw_text": "<action_head_batch>"})
        return results

    def generate_action(
        self,
        *,
        image_paths: list[str | Path | Image.Image],
        task: str,
        history_frame_count: int,
        current_subtask: str | None,
        expected_next_screen: str | None,
        max_new_tokens: int = 192,
        temporal_sample_key: str | None = None,
    ) -> dict[str, Any]:
        self._active_training_stage = str(getattr(self, "training_stage", "stage1"))
        user_prompt = self.build_user_prompt(
            task=task,
            history_frame_count=history_frame_count,
            current_subtask=current_subtask,
            expected_next_screen=expected_next_screen,
            action_only_output=self.action_only_output,
            action_format=self.action_format,
        )
        assistant_prefix = None if self.latent_scaffolds_in_prompt else self.action_head_assistant_prefix()
        inputs, _ = self.prepare_conversation_inputs(
            image_paths=image_paths,
            user_prompt=user_prompt,
            assistant_target=assistant_prefix,
            continue_assistant=bool(assistant_prefix),
        )
        sample_keys = [str(temporal_sample_key)] if temporal_sample_key else None
        self._set_temporal_context(sample_keys)
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        self._set_temporal_context(None)
        trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        generated_text = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=(
                self.action_format != "action_tokens"
                and str(getattr(self, "training_stage", "stage1")) != "stage2"
            ),
            clean_up_tokenization_spaces=False,
        )[0].strip()
        text = f"{assistant_prefix}\n{generated_text}".strip() if assistant_prefix else generated_text
        token_payload = self.gui_action_tokenizer.decode(text)
        if token_payload is not None:
            return {"action": token_payload.action, "raw_text": token_payload.raw_text}
        payload = self.safe_extract_json(text)
        payload["raw_text"] = text
        return payload

    def generate_actions_batch(
        self,
        samples: list[dict[str, Any]],
        *,
        max_new_tokens: int = 192,
    ) -> list[dict[str, Any]]:
        if not samples:
            return []
        self._active_training_stage = str(getattr(self, "training_stage", "stage1"))
        batch_image_paths: list[list[str | Path | Image.Image]] = []
        user_prompts: list[str] = []
        assistant_prefixes: list[str | None] = []
        temporal_keys: list[str] = []
        for sample in samples:
            image_paths = list(sample["image_paths"])
            batch_image_paths.append(image_paths)
            user_prompts.append(
                self.build_user_prompt(
                    task=str(sample.get("task", "") or ""),
                    history_frame_count=int(sample.get("history_frame_count", max(0, len(image_paths) - 1))),
                    current_subtask=sample.get("current_subtask"),
                    expected_next_screen=sample.get("expected_next_screen"),
                    action_only_output=self.action_only_output,
                    action_format=self.action_format,
                )
            )
            assistant_prefixes.append(
                None if self.latent_scaffolds_in_prompt else self.action_head_assistant_prefix()
            )
            temporal_sample_key = str(sample.get("temporal_sample_key", "") or "")
            for image_index in range(len(image_paths)):
                temporal_keys.append(f"{temporal_sample_key}|img_{image_index}" if temporal_sample_key else "")

        inputs, _ = self.prepare_conversation_inputs_batch(
            batch_image_paths=batch_image_paths,
            user_prompts=user_prompts,
            assistant_targets=assistant_prefixes,
            padding_side="left",
            continue_assistant=all(prefix is not None for prefix in assistant_prefixes),
        )
        sample_keys = temporal_keys if temporal_keys and all(temporal_keys) else None
        self._set_temporal_context(sample_keys)
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        self._set_temporal_context(None)
        trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        texts = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=(
                self.action_format != "action_tokens"
                and str(getattr(self, "training_stage", "stage1")) != "stage2"
            ),
            clean_up_tokenization_spaces=False,
        )
        results: list[dict[str, Any]] = []
        for row_index, text in enumerate(texts):
            generated_text = str(text or "").strip()
            assistant_prefix = assistant_prefixes[row_index]
            cleaned = (
                f"{assistant_prefix}\n{generated_text}".strip()
                if assistant_prefix
                else generated_text
            )
            token_payload = self.gui_action_tokenizer.decode(cleaned)
            if token_payload is not None:
                results.append({"action": token_payload.action, "raw_text": token_payload.raw_text})
                continue
            payload = self.safe_extract_json(cleaned)
            payload["raw_text"] = cleaned
            results.append(payload)
        return results

    def generate_action_parameters_batch(
        self,
        samples: list[dict[str, Any]],
        *,
        action_types: list[str],
        max_new_tokens: int = 96,
    ) -> list[dict[str, Any]]:
        """Generate only variable-length parameters for preselected action types.

        This is intentionally separate from normal reasoning generation. The
        action head has already selected the type, so the LM may only emit the
        text payload for ``type`` or the key sequence for ``hotkey``.
        """

        if not samples:
            return []
        if len(samples) != len(action_types):
            raise ValueError("samples and action_types must have the same length")

        self._active_training_stage = str(getattr(self, "training_stage", "stage1"))
        batch_image_paths: list[list[str | Path | Image.Image]] = []
        user_prompts: list[str] = []
        temporal_keys: list[str] = []
        normalized_types: list[str] = []
        for sample, raw_action_type in zip(samples, action_types):
            action_type = str(raw_action_type or "").strip().lower()
            if action_type not in {"type", "hotkey"}:
                raise ValueError(f"Unsupported parameter-generation action type: {action_type!r}")
            normalized_types.append(action_type)
            image_paths = list(sample["image_paths"])
            batch_image_paths.append(image_paths)
            prompt = self.build_user_prompt(
                task=str(sample.get("task", "") or ""),
                history_frame_count=int(sample.get("history_frame_count", max(0, len(image_paths) - 1))),
                current_subtask=sample.get("current_subtask"),
                expected_next_screen=sample.get("expected_next_screen"),
                action_only_output=True,
                action_format="text",
                include_action_in_lm=True,
            )
            if action_type == "type":
                prompt += (
                    "\nThe action expert has already fixed the action type to type. "
                    "Do not reconsider or change the action type. Do not output reasoning. "
                    "Return exactly two lines:\nAction: type\nText: \"text to enter\""
                )
            else:
                prompt += (
                    "\nThe action expert has already fixed the action type to hotkey. "
                    "Do not reconsider or change the action type. Do not output reasoning. "
                    "Return exactly two lines:\nAction: hotkey\nKeys: [key1, key2]"
                )
            user_prompts.append(prompt)
            temporal_sample_key = str(sample.get("temporal_sample_key", "") or "")
            for image_index in range(len(image_paths)):
                temporal_keys.append(f"{temporal_sample_key}|img_{image_index}" if temporal_sample_key else "")

        inputs, _ = self.prepare_conversation_inputs_batch(
            batch_image_paths=batch_image_paths,
            user_prompts=user_prompts,
            assistant_targets=[None] * len(samples),
            padding_side="left",
        )
        sample_keys = temporal_keys if temporal_keys and all(temporal_keys) else None
        self._set_temporal_context(sample_keys)
        try:
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=max(1, int(max_new_tokens)))
        finally:
            self._set_temporal_context(None)
        trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        texts = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        results: list[dict[str, Any]] = []
        for action_type, text in zip(normalized_types, texts):
            generated_text = str(text or "").strip()
            payload = self.safe_extract_json(generated_text)
            payload["raw_text"] = generated_text
            action = payload.get("action")
            if not isinstance(action, dict) or str(action.get("type", "") or "").strip().lower() != action_type:
                payload["_parse_error"] = f"parameter_decoder_type_mismatch:{action_type}"
            results.append(payload)
        return results

    @staticmethod
    def safe_extract_json(text: str) -> dict[str, Any]:
        cleaned = text.strip()
        token_payload = GUIActionTokenizer().decode(cleaned)
        if token_payload is not None:
            return {"action": token_payload.action, "raw_text": token_payload.raw_text}
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
            cleaned = re.sub(r"```$", "", cleaned).strip()
        try:
            return LaRAStyleQwen3VLAgent._ensure_generated_payload(json.loads(cleaned), cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, flags=re.S)
            if not match:
                parsed_text = LaRAStyleQwen3VLAgent.safe_extract_text_action(cleaned)
                if parsed_text is not None:
                    return parsed_text
                return {"_parse_error": "JSON object not found", "action": {"type": "wait", "status": "success"}}
            candidate = match.group(0)
            try:
                return LaRAStyleQwen3VLAgent._ensure_generated_payload(json.loads(candidate), cleaned)
            except json.JSONDecodeError as exc:
                parsed_text = LaRAStyleQwen3VLAgent.safe_extract_text_action(cleaned)
                if parsed_text is not None:
                    return parsed_text
                return {
                    "_parse_error": f"{type(exc).__name__}: {exc}",
                    "action": {"type": "wait", "status": "success"},
                }

    @staticmethod
    def _ensure_generated_payload(payload: Any, raw_text: str = "") -> dict[str, Any]:
        if isinstance(payload, dict):
            return LaRAStyleQwen3VLAgent._normalize_generated_action_coords(payload)
        parsed_text = LaRAStyleQwen3VLAgent.safe_extract_text_action(str(payload))
        if parsed_text is not None:
            return parsed_text
        return {
            "_parse_error": f"Generated JSON payload is {type(payload).__name__}, expected object",
            "action": {"type": "wait", "status": "success"},
            "raw_text": raw_text,
        }

    @staticmethod
    def _normalize_generated_action_coords(payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict):
            return {
                "_parse_error": f"Generated payload is {type(payload).__name__}, expected object",
                "action": {"type": "wait", "status": "success"},
            }
        action = payload.get("action")
        if not isinstance(action, dict):
            return payload
        if action.get("x_norm") is not None:
            action["x_norm"] = LaRAStyleQwen3VLAgent._qwen_coord_to_norm(action.get("x_norm"))
        if action.get("y_norm") is not None:
            action["y_norm"] = LaRAStyleQwen3VLAgent._qwen_coord_to_norm(action.get("y_norm"))
        return payload

    @staticmethod
    def safe_extract_text_action(text: str) -> dict[str, Any] | None:
        if not text.strip():
            return None
        reasoning_match = re.search(r"Reasoning:\s*(.*?)(?:\nAction:|\Z)", text, flags=re.S)
        action_match = re.search(r"Action:\s*([A-Za-z_]+)", text)
        if not action_match:
            return None
        action_type = action_match.group(1).strip().lower()
        action: dict[str, Any] = {"type": action_type}
        point_match = re.search(r"Point:\s*\[\s*([^\s,\]]+)\s*,?\s+([^\s,\]]+)\s*\]", text)
        if point_match:
            try:
                action["x_norm"] = LaRAStyleQwen3VLAgent._qwen_coord_to_norm(point_match.group(1))
                action["y_norm"] = LaRAStyleQwen3VLAgent._qwen_coord_to_norm(point_match.group(2))
            except Exception:
                pass
        point_px_match = re.search(r"PointPx:\s*\[\s*([^\s,\]]+)\s*,?\s+([^\s,\]]+)\s*\]", text)
        if point_px_match:
            try:
                action["x"] = int(float(point_px_match.group(1)))
                action["y"] = int(float(point_px_match.group(2)))
            except Exception:
                pass
        text_match = re.search(r'Text:\s*"(.*?)"', text, flags=re.S)
        if text_match:
            action["text"] = text_match.group(1)
        keys_match = re.search(r"Keys:\s*\[(.*?)\]", text, flags=re.S)
        if keys_match:
            action["keys"] = [item.strip().strip("'\"") for item in keys_match.group(1).split(",") if item.strip()]
        amount_match = re.search(r"Amount:\s*(-?\d+)", text)
        if amount_match:
            action["amount"] = int(amount_match.group(1))
        status_match = re.search(r"Status:\s*([A-Za-z_]+)", text)
        if status_match:
            action["status"] = status_match.group(1).strip().lower()
        payload: dict[str, Any] = {"action": action}
        if reasoning_match:
            payload["reasoning"] = reasoning_match.group(1).strip()
        payload["raw_text"] = text
        return payload
