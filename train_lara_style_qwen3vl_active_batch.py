from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import torch

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore[assignment]

from qwen3_gui_agent.lara_style_qwen3vl_agent import (
    LaRAStyleQwen3VLAgent,
    resolve_reasoning_field_slot_counts,
)
from qwen3_gui_agent.rl.schema import read_jsonl
from qwen3_gui_agent.training_checkpoint import (
    align_optimizer_state_with_params,
    force_safe_adamw_runtime_flags,
    load_training_checkpoint,
    save_training_checkpoint,
)
from qwen3_gui_agent.training_utils import build_scheduler, current_lr, resolve_device_map, resolve_torch_dtype
from train_lara_style_qwen3vl import (
    build_trajectories,
    compact_debug_info,
    configure_trainable_params,
    truncate_for_log,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Active-pool batched trajectory trainer for LaRA-style Qwen3-VL."
    )
    parser.add_argument("--steps", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter-out", required=True)
    parser.add_argument("--report-out", default=None)
    parser.add_argument("--checkpoint-out", default=None)
    parser.add_argument("--best-checkpoint-out", default=None)
    parser.add_argument("--init-adapter", default=None)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lr-scheduler", choices=["constant", "cosine"], default="cosine")
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--history-n", type=int, default=1)
    parser.add_argument("--latent-slot-count", type=int, default=16)
    parser.add_argument(
        "--reasoning-alignment-mode",
        choices=["aggregate", "field_aligned"],
        default="aggregate",
        help="aggregate preserves the legacy pooled teacher; field_aligned assigns latent slots to actual_task/thought/reflection.",
    )
    parser.add_argument(
        "--reasoning-field-slot-counts",
        default="auto",
        help="Three comma-separated slot counts for actual_task,thought,reflection. auto gives 6,5,5 for 16 slots.",
    )
    parser.add_argument("--pixel-prune-threshold", type=float, default=0.0)
    parser.add_argument("--pixel-prune-predictor-order", default="pred2d,left,up")
    parser.add_argument("--pixel-temporal-reuse", action="store_true")
    parser.add_argument("--pixel-temporal-threshold", type=float, default=0.0)
    parser.add_argument("--image-min-pixels", type=int, default=0)
    parser.add_argument("--image-max-pixels", type=int, default=0)
    parser.add_argument("--training-stage", choices=["stage1", "stage2"], default="stage1")
    parser.add_argument("--action-format", choices=["text", "action_tokens"], default="text")
    parser.add_argument(
        "--action-model",
        choices=["unified", "flow_matching", "latent_two_way"],
        default="flow_matching",
    )
    parser.add_argument("--flow-action-sample-steps", type=int, default=8)
    parser.add_argument(
        "--flow-head-hidden-dim",
        type=int,
        default=0,
        help="Optional hidden width for the flow/action head MLPs. 0 keeps the previous lightweight default.",
    )
    parser.add_argument(
        "--flow-head-depth",
        type=int,
        default=2,
        help="Optional depth for the flow/action head MLPs. 2 keeps the previous lightweight default.",
    )
    parser.add_argument("--two-way-hidden-dim", type=int, default=512)
    parser.add_argument("--two-way-depth", type=int, default=2)
    parser.add_argument("--two-way-num-heads", type=int, default=8)
    parser.add_argument("--two-way-location-queries", type=int, default=3)
    parser.add_argument("--two-way-dropout", type=float, default=0.0)
    parser.add_argument(
        "--two-way-query-mode",
        choices=["semantic_pool", "latent_pos"],
        default="semantic_pool",
        help=(
            "semantic_pool keeps the legacy pooled semantic readout; latent_pos uses an internal "
            "<|POS|> router for action type while Stage-2 latent states directly drive grounding."
        ),
    )
    parser.add_argument("--two-way-candidate-coord-loss-weight", type=float, default=1.0)
    parser.add_argument("--two-way-candidate-confidence-loss-weight", type=float, default=0.25)
    parser.add_argument("--lm-action-target", choices=["auto", "include", "omit"], default="auto")
    parser.add_argument("--stage2-target-format", choices=["mixed_reasoning_action", "action_only"], default="mixed_reasoning_action")
    parser.add_argument("--stage1-max-reasoning-chars", type=int, default=0)
    parser.add_argument("--stage2-explicit-keep-start", type=float, default=0.7)
    parser.add_argument("--stage2-explicit-keep-end", type=float, default=0.0)
    parser.add_argument("--stage2-min-explicit-tokens", type=int, default=0)
    parser.add_argument("--stage2-max-thinking-tokens", type=int, default=4)
    parser.add_argument("--lm-loss-weight", type=float, default=1.0)
    parser.add_argument("--reasoning-align-weight", type=float, default=0.0)
    parser.add_argument("--future-frame-loss-weight", type=float, default=0.0)
    parser.add_argument("--latent-diversity-weight", type=float, default=0.0)
    parser.add_argument("--action-head-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--flow-action-loss-weight",
        type=float,
        default=1.0,
        help="Weight for flow velocity matching loss inside the flow-matching action head.",
    )
    parser.add_argument(
        "--flow-coord-loss-weight",
        type=float,
        default=1.0,
        help="Weight for direct coordinate regression auxiliary loss inside the flow-matching action head.",
    )
    parser.add_argument(
        "--learnable-flow-coord-weight",
        action="store_true",
        help=(
            "Learn the direct coordinate loss weight with uncertainty weighting: "
            "exp(-s) * coord_loss + s. --flow-coord-loss-weight is used as the initial weight."
        ),
    )
    parser.add_argument(
        "--flow-coord-loss-scale",
        type=float,
        default=1.0,
        help="Scale x/y residuals inside the direct coordinate loss. Use 1000 to train coordinates on a 0-1000 scale while keeping predictions normalized.",
    )
    parser.add_argument(
        "--flow-coord-loss-space",
        choices=["logit", "scaled", "normalized"],
        default="logit",
        help="Coordinate auxiliary loss space. logit supervises raw x/y logits against logit(gt); scaled uses normalized residuals multiplied by --flow-coord-loss-scale.",
    )
    parser.add_argument(
        "--flow-patch-loss-weight",
        type=float,
        default=0.0,
        help="Optional CE weight for supervising the attended visual patch from GT x/y. Useful for GUI grounding overfit tests.",
    )
    parser.add_argument(
        "--flow-patch-loss-mode",
        choices=["ce", "gaussian"],
        default="ce",
        help="Patch grounding loss mode. ce uses the nearest GT patch; gaussian uses a soft heatmap around the GT point.",
    )
    parser.add_argument(
        "--flow-patch-gaussian-sigma",
        type=float,
        default=0.05,
        help="Gaussian patch target sigma in normalized image coordinates when --flow-patch-loss-mode gaussian.",
    )
    parser.add_argument(
        "--flow-pointer-coord-source",
        choices=["mlp", "patch", "argmax_patch", "patch_residual"],
        default="patch_residual",
        help=(
            "Coordinate source for the direct pointer branch. "
            "patch/argmax_patch force visual grounding; patch_residual adds a learned local offset; mlp ignores patch attention."
        ),
    )
    parser.add_argument(
        "--flow-patch-logit-temperature",
        type=float,
        default=1.0,
        help="Temperature for converting patch attention logits to coordinate softargmax probabilities.",
    )
    parser.add_argument(
        "--flow-patch-residual-scale",
        type=float,
        default=1.0,
        help="Multiplier for the learned local offset added to patch softargmax coordinates.",
    )
    parser.add_argument(
        "--action-hidden-source",
        choices=["summary", "prompt_attn", "slot_attn", "prompt_slot_attn"],
        default="summary",
        help=(
            "Which hidden states feed the action head. summary is the old prompt/slot mean baseline; "
            "prompt_attn attention-pools all prompt hidden states; slot_attn attention-pools latent/img-next slots; "
            "prompt_slot_attn enables both."
        ),
    )
    parser.add_argument("--train-backbone", action="store_true")
    parser.add_argument("--train-embeddings", action="store_true")
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--clean-observable-prompt",
        action="store_true",
        help=(
            "Remove current-step actual_task/expected-next-screen oracle fields and keep latent/img-next "
            "scaffolds in the assistant sequence instead of the user prompt."
        ),
    )
    parser.add_argument(
        "--train-action-head-only",
        action="store_true",
        help="Freeze reasoning/future modules and train only action_state_norm plus action head modules.",
    )
    parser.add_argument("--shuffle-trajectories", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--prep-log-every", type=int, default=10)
    parser.add_argument("--checkpoint-every-steps", type=int, default=0)
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=0.0,
        help="Clip trainable gradient norm before optimizer.step; 0 disables clipping.",
    )
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    parser.add_argument("--early-stop-min-epochs", type=int, default=1)
    parser.add_argument(
        "--early-stop-monitor",
        choices=["loss", "lm_loss", "action_head_loss"],
        default="loss",
    )
    parser.add_argument("--profile-stages", action="store_true")
    parser.add_argument("--minimal-logging", action="store_true")
    parser.add_argument("--no-progress-bar", action="store_true")
    parser.add_argument("--log-gt", action="store_true")
    parser.add_argument("--log-gt-max-chars", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.max_samples = max(1, int(args.max_samples))
    args.batch_size = max(1, int(args.batch_size))
    args.grad_accum_steps = max(1, int(args.grad_accum_steps))
    args.history_n = max(1, int(args.history_n))
    args.log_every = max(1, int(args.log_every))
    args.prep_log_every = max(1, int(args.prep_log_every))
    args.checkpoint_every_steps = max(0, int(args.checkpoint_every_steps))
    args.max_grad_norm = max(0.0, float(args.max_grad_norm))
    args.log_gt_max_chars = max(0, int(args.log_gt_max_chars))
    args.stage2_max_thinking_tokens = max(1, int(args.stage2_max_thinking_tokens))
    if args.reasoning_alignment_mode == "aggregate" and args.latent_slot_count < 3:
        args.resolved_reasoning_field_slot_counts = [args.latent_slot_count, 0, 0]
    else:
        args.resolved_reasoning_field_slot_counts = list(
            resolve_reasoning_field_slot_counts(
                args.reasoning_field_slot_counts,
                latent_slot_count=args.latent_slot_count,
            )
        )
    if (
        args.reasoning_alignment_mode == "field_aligned"
        and args.training_stage == "stage2"
        and args.stage2_target_format == "mixed_reasoning_action"
        and args.stage2_max_thinking_tokens != args.latent_slot_count
    ):
        raise ValueError(
            "Field-aligned Stage 2 requires --stage2-max-thinking-tokens to equal --latent-slot-count "
            "so field groups are never truncated."
        )
    args.lora_r = max(1, int(args.lora_r))
    args.lora_alpha = max(1, int(args.lora_alpha))
    args.lora_dropout = max(0.0, float(args.lora_dropout))
    args.early_stop_patience = max(0, int(args.early_stop_patience))
    args.early_stop_min_delta = max(0.0, float(args.early_stop_min_delta))
    args.early_stop_min_epochs = max(1, int(args.early_stop_min_epochs))
    args.flow_action_sample_steps = max(1, int(args.flow_action_sample_steps))
    args.flow_head_hidden_dim = max(0, int(args.flow_head_hidden_dim))
    args.flow_head_depth = max(1, int(args.flow_head_depth))
    args.two_way_hidden_dim = max(64, int(args.two_way_hidden_dim))
    args.two_way_depth = max(1, int(args.two_way_depth))
    args.two_way_num_heads = max(1, int(args.two_way_num_heads))
    args.two_way_location_queries = max(1, int(args.two_way_location_queries))
    args.two_way_dropout = max(0.0, float(args.two_way_dropout))
    args.two_way_candidate_coord_loss_weight = max(
        0.0, float(args.two_way_candidate_coord_loss_weight)
    )
    args.two_way_candidate_confidence_loss_weight = max(
        0.0, float(args.two_way_candidate_confidence_loss_weight)
    )
    if args.two_way_hidden_dim % args.two_way_num_heads != 0:
        raise ValueError("--two-way-hidden-dim must be divisible by --two-way-num-heads.")
    args.image_min_pixels = max(0, int(args.image_min_pixels))
    args.image_max_pixels = max(0, int(args.image_max_pixels))
    args.flow_action_loss_weight = max(0.0, float(args.flow_action_loss_weight))
    args.flow_coord_loss_weight = max(0.0, float(args.flow_coord_loss_weight))
    args.flow_coord_loss_scale = max(1.0, float(args.flow_coord_loss_scale))
    args.flow_coord_loss_space = str(args.flow_coord_loss_space)
    args.flow_patch_loss_weight = max(0.0, float(args.flow_patch_loss_weight))
    args.flow_patch_loss_mode = str(args.flow_patch_loss_mode)
    args.flow_patch_gaussian_sigma = max(1e-4, float(args.flow_patch_gaussian_sigma))
    args.flow_pointer_coord_source = str(args.flow_pointer_coord_source)
    args.flow_patch_logit_temperature = max(1e-4, float(args.flow_patch_logit_temperature))
    args.flow_patch_residual_scale = max(0.0, float(args.flow_patch_residual_scale))
    if args.lm_action_target == "auto":
        args.resolved_lm_action_target = "omit" if float(args.action_head_loss_weight) > 0.0 else "include"
    else:
        args.resolved_lm_action_target = str(args.lm_action_target)
    args.flow_continuous_source = (
        "direct"
        if args.action_model == "latent_two_way" or args.flow_coord_loss_weight > 0.0
        else "sample"
    )
    if args.action_model not in {"flow_matching", "latent_two_way"}:
        raise ValueError(
            "The active-batch trainer supports --action-model flow_matching or latent_two_way."
        )
    return args


def make_state(trajectory: dict[str, Any]) -> dict[str, Any]:
    return {"trajectory": trajectory, "row_index": 0, "history_image_paths": []}


def sample_from_state(state: dict[str, Any], *, history_slots: int, explicit_reasoning: str) -> dict[str, Any]:
    row = state["trajectory"]["rows"][state["row_index"]]
    image_paths = list(state["history_image_paths"][-history_slots:]) + [row["image_path"]]
    return {
        "image_paths": image_paths,
        "task": row["instruction"],
        "history_frame_count": len(image_paths) - 1,
        "current_subtask": row["actual_task"],
        "expected_next_screen": None,
        "explicit_reasoning": explicit_reasoning,
        "gold_action": row["gold_action"],
        "next_image_path": row["after_image_path"],
        "temporal_sample_key": str(state["trajectory"]["trajectory_key"]),
    }


def advance_state(state: dict[str, Any]) -> None:
    row = state["trajectory"]["rows"][state["row_index"]]
    state["history_image_paths"].append(row["image_path"])
    state["row_index"] += 1


def compact_sample_summary(
    *,
    states: list[dict[str, Any]],
    samples: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for state, sample in zip(states, samples):
        row = state["trajectory"]["rows"][state["row_index"]]
        image_paths = list(sample["image_paths"])
        summary.append(
            {
                "sample_id": row["sample_id"],
                "trajectory_key": row["trajectory_key"],
                "trajectory_step_index": int(state["row_index"]) + 1,
                "history_frame_count": int(sample["history_frame_count"]),
                "image_count": len(image_paths),
                "current_image": Path(row["image_path"]).name,
                "gold_action": row["gold_action"],
            }
        )
    return summary


def advance_active_pool(
    *,
    active_states: list[dict[str, Any]],
    epoch_trajectories: list[dict[str, Any]],
    next_trajectory_index: int,
    batch_size: int,
) -> tuple[list[dict[str, Any]], int]:
    for state in list(active_states):
        advance_state(state)
    active_states = [state for state in active_states if state["row_index"] < len(state["trajectory"]["rows"])]
    while next_trajectory_index < len(epoch_trajectories) and len(active_states) < batch_size:
        active_states.append(make_state(epoch_trajectories[next_trajectory_index]))
        next_trajectory_index += 1
    return active_states, next_trajectory_index


def skip_active_rows(
    *,
    active_states: list[dict[str, Any]],
    epoch_trajectories: list[dict[str, Any]],
    next_trajectory_index: int,
    rows_to_skip: int,
    batch_size: int,
) -> tuple[list[dict[str, Any]], int, int]:
    skipped = 0
    while active_states and skipped < rows_to_skip:
        batch_states = [state for state in active_states if state["row_index"] < len(state["trajectory"]["rows"])]
        if not batch_states:
            break
        skipped += len(batch_states)
        active_states, next_trajectory_index = advance_active_pool(
            active_states=batch_states,
            epoch_trajectories=epoch_trajectories,
            next_trajectory_index=next_trajectory_index,
            batch_size=batch_size,
        )
    return active_states, next_trajectory_index, skipped


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dtype = resolve_torch_dtype(args.torch_dtype)
    device_map = resolve_device_map(args.device_map)
    trajectories = build_trajectories(
        steps=read_jsonl(Path(args.steps)),
        dataset_root=Path(args.dataset_root),
        max_samples=args.max_samples,
        prep_log_every=args.prep_log_every,
    )
    if not trajectories:
        raise RuntimeError("No valid LaRA-style training samples found.")

    agent = LaRAStyleQwen3VLAgent.from_pretrained(
        args.model,
        device_map=device_map,
        torch_dtype=dtype,
        latent_slot_count=args.latent_slot_count,
        pixel_prune_threshold=args.pixel_prune_threshold,
        pixel_prune_predictor_order=args.pixel_prune_predictor_order,
        pixel_temporal_reuse=args.pixel_temporal_reuse,
        pixel_temporal_threshold=args.pixel_temporal_threshold,
        action_model=args.action_model,
        flow_action_sample_steps=args.flow_action_sample_steps,
        flow_head_hidden_dim=args.flow_head_hidden_dim or None,
        flow_head_depth=args.flow_head_depth,
        two_way_hidden_dim=args.two_way_hidden_dim,
        two_way_depth=args.two_way_depth,
        two_way_num_heads=args.two_way_num_heads,
        two_way_location_queries=args.two_way_location_queries,
        two_way_dropout=args.two_way_dropout,
        two_way_query_mode=args.two_way_query_mode,
        image_min_pixels=args.image_min_pixels or None,
        image_max_pixels=args.image_max_pixels or None,
        include_current_subtask_in_prompt=not args.clean_observable_prompt,
        include_expected_next_screen_in_prompt=not args.clean_observable_prompt,
        latent_scaffolds_in_prompt=not args.clean_observable_prompt,
        reasoning_alignment_mode=args.reasoning_alignment_mode,
        reasoning_field_slot_counts=args.resolved_reasoning_field_slot_counts,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    hf_device_map = getattr(agent.model, "hf_device_map", None)
    if isinstance(hf_device_map, dict):
        device_module_counts: dict[str, int] = {}
        for mapped_device in hf_device_map.values():
            device_name = str(mapped_device)
            device_module_counts[device_name] = device_module_counts.get(device_name, 0) + 1
        print(
            json.dumps(
                {
                    "stage": "model_device_map",
                    "requested_device_map": args.device_map,
                    "device_module_counts": device_module_counts,
                },
                ensure_ascii=False,
            )
        )
    if args.init_adapter:
        load_info = agent.load_adapter(args.init_adapter, strict=False)
        print(json.dumps({"stage": "init_adapter", "init_adapter": args.init_adapter, "load_info": load_info}, ensure_ascii=False))
    if args.gradient_checkpointing:
        try:
            agent.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError as exc:
            raise RuntimeError(
                "This trainer requires non-reentrant gradient checkpointing. "
                "Upgrade transformers or remove --gradient-checkpointing."
            ) from exc
        if hasattr(agent.model, "config"):
            agent.model.config.use_cache = False
        print(
            json.dumps(
                {
                    "stage": "gradient_checkpointing",
                    "enabled": True,
                    "implementation": "non_reentrant",
                },
                ensure_ascii=False,
            )
        )
    # Re-apply explicit runtime settings after loading adapter metadata.
    agent.set_reasoning_alignment_config(
        mode=args.reasoning_alignment_mode,
        field_slot_counts=args.resolved_reasoning_field_slot_counts,
    )
    agent.action_model = str(args.action_model)
    agent.two_way_hidden_dim = int(args.two_way_hidden_dim)
    agent.two_way_depth = int(args.two_way_depth)
    agent.two_way_num_heads = int(args.two_way_num_heads)
    agent.two_way_location_queries = int(args.two_way_location_queries)
    agent.two_way_dropout = float(args.two_way_dropout)
    agent.two_way_query_mode = str(args.two_way_query_mode)
    agent.latent_two_way_action_head.query_mode = str(args.two_way_query_mode)
    agent.flow_action_sample_steps = max(1, int(args.flow_action_sample_steps))
    agent.flow_continuous_source = str(args.flow_continuous_source)
    agent.flow_action_loss_weight = float(args.flow_action_loss_weight)
    agent.flow_coord_loss_weight = float(args.flow_coord_loss_weight)
    agent.flow_coord_loss_scale = float(args.flow_coord_loss_scale)
    agent.flow_coord_loss_space = str(args.flow_coord_loss_space)
    agent.flow_patch_loss_weight = float(args.flow_patch_loss_weight)
    agent.flow_patch_loss_mode = str(args.flow_patch_loss_mode)
    agent.flow_patch_gaussian_sigma = float(args.flow_patch_gaussian_sigma)
    agent.flow_pointer_coord_source = str(args.flow_pointer_coord_source)
    agent.flow_patch_logit_temperature = float(args.flow_patch_logit_temperature)
    agent.flow_patch_residual_scale = float(args.flow_patch_residual_scale)
    agent.action_hidden_source = str(args.action_hidden_source)
    agent.two_way_candidate_coord_loss_weight = float(
        args.two_way_candidate_coord_loss_weight
    )
    agent.two_way_candidate_confidence_loss_weight = float(
        args.two_way_candidate_confidence_loss_weight
    )
    agent.learnable_flow_coord_weight = bool(args.learnable_flow_coord_weight)
    if args.learnable_flow_coord_weight:
        init_weight = max(1e-6, float(args.flow_coord_loss_weight))
        agent.flow_coord_loss_log_var.data.fill_(-math.log(init_weight))

    configure_trainable_params(agent, train_backbone=args.train_backbone, train_embeddings=args.train_embeddings)
    if args.action_head_loss_weight <= 0.0:
        for module in [
            agent.action_state_norm,
            agent.action_head,
            agent.flow_action_head,
            agent.latent_two_way_action_head,
        ]:
            for parameter in module.parameters():
                parameter.requires_grad = False
        agent.action_prompt_query.requires_grad = False
        agent.action_slot_query.requires_grad = False
        agent.flow_coord_loss_log_var.requires_grad = False
    if args.future_frame_loss_weight <= 0.0:
        for parameter in agent.future_frame_head.parameters():
            parameter.requires_grad = False
    if args.train_action_head_only:
        for parameter in agent.parameters():
            parameter.requires_grad = False
        action_modules = [agent.action_state_norm]
        if args.action_model == "flow_matching":
            action_modules.append(agent.flow_action_head)
        elif args.action_model == "latent_two_way":
            action_modules.append(agent.latent_two_way_action_head)
        else:
            action_modules.append(agent.action_head)
        for module in action_modules:
            for parameter in module.parameters():
                parameter.requires_grad = True
        agent.action_prompt_query.requires_grad = True
        agent.action_slot_query.requires_grad = True
        if args.learnable_flow_coord_weight:
            agent.flow_coord_loss_log_var.requires_grad = True
    agent.train()
    trainable_params = [parameter for parameter in agent.parameters() if parameter.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters were selected for this stage.")
    trainable_param_count = sum(int(parameter.numel()) for parameter in trainable_params)
    print(
        json.dumps(
            {
                "stage": "trainable_parameters",
                "training_stage": args.training_stage,
                "trainable_parameter_count": trainable_param_count,
                "use_lora": bool(args.use_lora),
                "train_special_token_embeddings": bool(args.train_embeddings),
                "train_action_head_only": bool(args.train_action_head_only),
            },
            ensure_ascii=False,
        )
    )
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, foreach=False)
    force_safe_adamw_runtime_flags(optimizer)

    total_rows = sum(len(trajectory["rows"]) for trajectory in trajectories)
    total_optimizer_steps = max(1, (total_rows * max(1, args.epochs)) // max(1, args.batch_size * args.grad_accum_steps))
    scheduler = build_scheduler(
        optimizer=optimizer,
        scheduler_name=args.lr_scheduler,
        total_steps=total_optimizer_steps,
        warmup_ratio=args.warmup_ratio,
        min_lr_ratio=args.min_lr_ratio,
    )

    started = time.time()
    history: list[dict[str, Any]] = []
    global_step = 0
    optimizer_steps = 0
    processed_rows = 0
    resume_epoch = 1
    resume_rows_in_epoch = 0
    best_monitor_value = float("inf")
    best_epoch = 0
    early_stop_bad_epochs = 0
    stopped_early = False
    if args.resume_from:
        payload = load_training_checkpoint(
            checkpoint_path=args.resume_from,
            agent_model=agent,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        align_optimizer_state_with_params(optimizer)
        force_safe_adamw_runtime_flags(optimizer)
        global_step = int(payload.get("global_step", 0))
        extra_state = payload.get("extra_state", {}) or {}
        resume_epoch = int(payload.get("epoch", 1))
        optimizer_steps = int(extra_state.get("optimizer_steps", 0))
        processed_rows = int(extra_state.get("processed_rows", 0))
        resume_rows_in_epoch = int(extra_state.get("processed_rows_in_epoch", 0))
        best_monitor_value = float(extra_state.get("best_monitor_value", float("inf")))
        best_epoch = int(extra_state.get("best_epoch", 0))
        early_stop_bad_epochs = int(extra_state.get("early_stop_bad_epochs", 0))
        if bool(extra_state.get("epoch_complete", False)):
            resume_epoch += 1
            resume_rows_in_epoch = 0
        print(
            json.dumps(
                {
                    "stage": "resume_active_batch",
                    "resume_from": args.resume_from,
                    "resume_epoch": resume_epoch,
                    "resume_global_step": global_step,
                    "resume_processed_rows": processed_rows,
                    "resume_rows_in_epoch": resume_rows_in_epoch,
                    "optimizer_steps": optimizer_steps,
                },
                ensure_ascii=False,
            )
        )
    history_slots = max(0, int(args.history_n) - 1)

    for epoch in range(resume_epoch, int(args.epochs) + 1):
        epoch_trajectories = list(trajectories)
        if args.shuffle_trajectories:
            random.Random(args.seed + epoch).shuffle(epoch_trajectories)
        next_trajectory_index = 0
        active_states: list[dict[str, Any]] = []
        while next_trajectory_index < len(epoch_trajectories) and len(active_states) < args.batch_size:
            active_states.append(make_state(epoch_trajectories[next_trajectory_index]))
            next_trajectory_index += 1

        skipped_rows_this_epoch = 0
        if epoch == resume_epoch and resume_rows_in_epoch > 0:
            active_states, next_trajectory_index, skipped_rows_this_epoch = skip_active_rows(
                active_states=active_states,
                epoch_trajectories=epoch_trajectories,
                next_trajectory_index=next_trajectory_index,
                rows_to_skip=resume_rows_in_epoch,
                batch_size=args.batch_size,
            )
            print(
                json.dumps(
                    {
                        "stage": "resume_active_batch_fast_forward",
                        "epoch": epoch,
                        "requested_skip_rows": int(resume_rows_in_epoch),
                        "actual_skipped_rows": int(skipped_rows_this_epoch),
                        "active_batch_size": len(active_states),
                        "next_trajectory_index": int(next_trajectory_index),
                    },
                    ensure_ascii=False,
                )
            )

        progress = (
            tqdm(total=total_rows, desc=f"train_lara_active_epoch{epoch}", dynamic_ncols=True)
            if tqdm is not None and not args.no_progress_bar
            else None
        )
        if progress is not None and skipped_rows_this_epoch > 0:
            progress.update(skipped_rows_this_epoch)
        epoch_loss_total = 0.0
        epoch_lm_total = 0.0
        epoch_action_total = 0.0
        epoch_rows = skipped_rows_this_epoch
        epoch_trained_rows = 0
        optimizer.zero_grad(set_to_none=True)
        micro_step = 0

        while active_states:
            batch_states = [state for state in active_states if state["row_index"] < len(state["trajectory"]["rows"])]
            if not batch_states:
                break
            samples = []
            for state in batch_states:
                row = state["trajectory"]["rows"][state["row_index"]]
                explicit_reasoning = str(row["explicit_supervision"])
                if args.training_stage == "stage1" and args.stage1_max_reasoning_chars > 0:
                    explicit_reasoning = explicit_reasoning[: int(args.stage1_max_reasoning_chars)].strip()
                samples.append(sample_from_state(state, history_slots=history_slots, explicit_reasoning=explicit_reasoning))

            if args.training_stage == "stage2":
                total_planned_rows = max(1, total_rows * max(1, int(args.epochs)))
                progress_ratio = min(
                    1.0,
                    max(0.0, float(processed_rows) / float(max(1, total_planned_rows - 1))),
                )
                current_stage2_keep_ratio = (
                    float(args.stage2_explicit_keep_start)
                    + (float(args.stage2_explicit_keep_end) - float(args.stage2_explicit_keep_start)) * progress_ratio
                )
            else:
                current_stage2_keep_ratio = 1.0

            batch_started_at = time.perf_counter()
            forward_started_at = time.perf_counter()
            output = agent.forward_train_batch(
                samples,
                training_stage=args.training_stage,
                stage2_target_format=args.stage2_target_format,
                stage2_explicit_keep_ratio=current_stage2_keep_ratio,
                stage2_min_explicit_tokens=args.stage2_min_explicit_tokens,
                stage2_max_thinking_tokens=args.stage2_max_thinking_tokens,
                future_frame_enabled=args.future_frame_loss_weight > 0.0,
                action_format=args.action_format,
                include_action_in_lm=(args.resolved_lm_action_target == "include"),
                reasoning_teacher_enabled=args.reasoning_align_weight > 0.0,
                action_head_enabled=args.action_head_loss_weight > 0.0,
            )
            forward_done_at = time.perf_counter()
            aux_losses, aux_metrics = agent.compute_auxiliary_losses(output, training_stage=args.training_stage)
            action_head_losses, action_head_metrics = agent.compute_action_head_losses(output)
            if "flow_action_loss" in action_head_losses:
                action_head_losses["flow_action_loss"] = (
                    args.flow_action_loss_weight * action_head_losses["flow_action_loss"]
                )
            if "flow_action_coord_loss" in action_head_losses:
                raw_coord_loss = action_head_losses["flow_action_coord_loss"]
                if args.learnable_flow_coord_weight:
                    coord_log_var = agent.flow_coord_loss_log_var.to(device=raw_coord_loss.device, dtype=torch.float32)
                    action_head_losses["flow_action_coord_loss"] = (
                        torch.exp(-coord_log_var) * raw_coord_loss.float() + coord_log_var
                    ).to(dtype=raw_coord_loss.dtype)
                else:
                    action_head_losses["flow_action_coord_loss"] = args.flow_coord_loss_weight * raw_coord_loss
            if "flow_action_patch_loss" in action_head_losses:
                action_head_losses["flow_action_patch_loss"] = (
                    args.flow_patch_loss_weight * action_head_losses["flow_action_patch_loss"]
                )
            if "two_way_candidate_coord_loss" in action_head_losses:
                action_head_losses["two_way_candidate_coord_loss"] = (
                    args.two_way_candidate_coord_loss_weight
                    * action_head_losses["two_way_candidate_coord_loss"]
                )
            if "two_way_candidate_confidence_loss" in action_head_losses:
                action_head_losses["two_way_candidate_confidence_loss"] = (
                    args.two_way_candidate_confidence_loss_weight
                    * action_head_losses["two_way_candidate_confidence_loss"]
                )
            loss_done_at = time.perf_counter()
            lm_loss = output.loss if output.loss is not None else output.latent_reasoning_summary.new_zeros(())
            reasoning_loss = aux_losses.get("reasoning_alignment_loss", lm_loss.new_zeros(()))
            future_loss = aux_losses.get("future_frame_loss", lm_loss.new_zeros(()))
            diversity_loss = aux_losses.get("latent_diversity_loss", lm_loss.new_zeros(()))
            action_head_loss = sum(action_head_losses.values(), start=lm_loss.new_zeros(())) if action_head_losses else lm_loss.new_zeros(())
            total_loss = (
                args.lm_loss_weight * lm_loss
                + args.reasoning_align_weight * reasoning_loss
                + args.future_frame_loss_weight * future_loss
                + args.latent_diversity_weight * diversity_loss
                + args.action_head_loss_weight * action_head_loss
            )
            if not bool(torch.isfinite(total_loss.detach()).all().item()):
                raise FloatingPointError(
                    f"Non-finite training loss at epoch={epoch}, global_step={global_step + 1}: "
                    f"total={float(total_loss.detach().item())}, "
                    f"action_head={float(action_head_loss.detach().item())}"
                )
            backward_started_at = time.perf_counter()
            (total_loss / float(args.grad_accum_steps)).backward()
            backward_done_at = time.perf_counter()

            global_step += 1
            micro_step += 1
            batch_rows = len(samples)
            processed_rows += batch_rows
            epoch_rows += batch_rows
            epoch_trained_rows += batch_rows
            epoch_loss_total += float(total_loss.detach().item()) * batch_rows
            epoch_lm_total += float(lm_loss.detach().item()) * batch_rows
            epoch_action_total += float(action_head_loss.detach().item()) * batch_rows

            should_step = micro_step % args.grad_accum_steps == 0
            optimizer_started_at = time.perf_counter()
            if should_step:
                if args.max_grad_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        trainable_params,
                        max_norm=args.max_grad_norm,
                        error_if_nonfinite=True,
                    )
                align_optimizer_state_with_params(optimizer)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
            optimizer_done_at = time.perf_counter()

            batch_summary = compact_sample_summary(states=batch_states, samples=samples)
            active_states, next_trajectory_index = advance_active_pool(
                active_states=batch_states,
                epoch_trajectories=epoch_trajectories,
                next_trajectory_index=next_trajectory_index,
                batch_size=args.batch_size,
            )

            if progress is not None:
                progress.update(batch_rows)
                progress_text = str(progress).replace("\r", "")
                tqdm.write(
                    (
                        f"{progress_text}, gstep={global_step}, opt={optimizer_steps}, batch={batch_rows}, "
                        f"loss={float(total_loss.detach().item()):.3f}, "
                        f"lm={float(lm_loss.detach().item()):.3f}, "
                        f"ah={float(action_head_loss.detach().item()):.3f}, "
                        f"cw={float(torch.exp(-agent.flow_coord_loss_log_var.detach()).cpu().item()):.3f}, "
                        f"ptr_l1={float(action_head_metrics.get('action_head_teacher_pointer_l1', 0.0)):.3f}, "
                        f"samp_l1={float(action_head_metrics.get('action_head_sampled_pointer_l1', 0.0)):.3f}, "
                        f"patch_l1={float(action_head_metrics.get('action_head_patch_pointer_l1', 0.0)):.3f}, "
                        f"patch_acc={float(action_head_metrics.get('action_head_patch_accuracy', 0.0)):.3f}, "
                        f"patch_p={float(action_head_metrics.get('action_head_patch_target_prob', 0.0)):.3f}, "
                        f"fwd={forward_done_at - forward_started_at:.1f}s, "
                        f"step_t={optimizer_done_at - batch_started_at:.1f}s"
                    )
                )

            if global_step % args.log_every == 0:
                debug_info = compact_debug_info(output.debug_info, keep_full_debug=False)
                timing = {
                    "forward_seconds": forward_done_at - forward_started_at,
                    "loss_seconds": loss_done_at - forward_done_at,
                    "backward_seconds": backward_done_at - backward_started_at,
                    "optimizer_seconds": optimizer_done_at - optimizer_started_at,
                    "step_seconds": optimizer_done_at - batch_started_at,
                }
                visual_debug = {}
                if output.debug_info and isinstance(output.debug_info.get("visual_debug"), dict):
                    visual_debug = output.debug_info["visual_debug"]
                timing["visual_pixel_prune_reconstruct_seconds"] = float(
                    visual_debug.get("pixel_prune_reconstruct_seconds", 0.0)
                )
                timing["visual_pixel_prune_plan_seconds"] = float(visual_debug.get("pixel_prune_plan_seconds", 0.0))
                sample_log = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "optimizer_steps": optimizer_steps,
                    "processed_rows": processed_rows,
                    "processed_rows_in_epoch": epoch_rows,
                    "batch_size": batch_rows,
                    "loss": float(total_loss.detach().item()),
                    "lm_loss": float(lm_loss.detach().item()),
                    "action_head_loss": float(action_head_loss.detach().item()),
                    "reasoning_alignment_loss": float(reasoning_loss.detach().item()),
                    "future_frame_loss": float(future_loss.detach().item()),
                    "latent_diversity_loss": float(diversity_loss.detach().item()),
                    "action_head_type_accuracy": float(action_head_metrics.get("action_head_type_accuracy", 0.0)),
                    "action_head_region_accuracy": float(action_head_metrics.get("action_head_region_accuracy", 0.0)),
                    "action_head_sampled_region_accuracy": float(
                        action_head_metrics.get("action_head_sampled_region_accuracy", 0.0)
                    ),
                    "action_head_coord_loss": float(action_head_metrics.get("action_head_coord_loss", 0.0)),
                    "action_head_patch_accuracy": float(action_head_metrics.get("action_head_patch_accuracy", 0.0)),
                    "action_head_patch_target_prob": float(
                        action_head_metrics.get("action_head_patch_target_prob", 0.0)
                    ),
                    "action_head_patch_target_entropy": float(
                        action_head_metrics.get("action_head_patch_target_entropy", 0.0)
                    ),
                    "two_way_best_candidate_pointer_l1": float(
                        action_head_metrics.get("two_way_best_candidate_pointer_l1", 0.0)
                    ),
                    "two_way_location_confidence": float(
                        action_head_metrics.get("two_way_location_confidence", 0.0)
                    ),
                    "two_way_pos_latent_attention_entropy": float(
                        action_head_metrics.get("two_way_pos_latent_attention_entropy", 0.0)
                    ),
                    "two_way_pos_latent_attention_max": float(
                        action_head_metrics.get("two_way_pos_latent_attention_max", 0.0)
                    ),
                    "two_way_query_mode": str(
                        action_head_metrics.get("two_way_query_mode", args.two_way_query_mode)
                    ),
                    "learned_flow_coord_weight": float(torch.exp(-agent.flow_coord_loss_log_var.detach()).cpu().item()),
                    "flow_coord_loss_log_var": float(agent.flow_coord_loss_log_var.detach().cpu().item()),
                    "action_head_teacher_pointer_l1": float(
                        action_head_metrics.get("action_head_teacher_pointer_l1", 0.0)
                    ),
                    "action_head_sampled_pointer_l1": float(
                        action_head_metrics.get("action_head_sampled_pointer_l1", 0.0)
                    ),
                    "action_head_patch_pointer_l1": float(
                        action_head_metrics.get("action_head_patch_pointer_l1", 0.0)
                    ),
                    "action_head_patch_argmax_pointer_l1": float(
                        action_head_metrics.get("action_head_patch_argmax_pointer_l1", 0.0)
                    ),
                    "action_head_patch_residual_abs": float(
                        action_head_metrics.get("action_head_patch_residual_abs", 0.0)
                    ),
                    "action_head_pointer_coord_source": str(
                        action_head_metrics.get("action_head_pointer_coord_source", args.flow_pointer_coord_source)
                    ),
                    "action_head_coord_preview": action_head_metrics.get("action_head_coord_preview", []),
                    "lr": current_lr(optimizer),
                    "prompt_lengths": (debug_info or {}).get("prompt_lengths"),
                    "prompt_length": (debug_info or {}).get("prompt_length"),
                    "supervised_token_count": (debug_info or {}).get("supervised_token_count"),
                    "visual_debug": (debug_info or {}).get("visual_debug"),
                    "timing": timing if args.profile_stages else None,
                    "batch_samples": batch_summary,
                }
                sample_log.update(
                    {
                        key: float(value)
                        for key, value in aux_metrics.items()
                        if key.startswith("reasoning_")
                    }
                )
                if not args.minimal_logging:
                    sample_log["debug_info"] = debug_info
                history.append(sample_log)
                print(json.dumps(sample_log, ensure_ascii=False))
                if args.log_gt:
                    gt_log = {
                        "stage": "train_active_batch_gt",
                        "epoch": epoch,
                        "global_step": global_step,
                        "batch_samples": [
                            {
                                **sample_info,
                                "image_paths": [Path(path).name for path in sample["image_paths"]],
                                "explicit_supervision": sample["explicit_reasoning"],
                            }
                            for sample_info, sample in zip(batch_summary, samples)
                        ],
                        "lm_action_target": str(args.resolved_lm_action_target),
                        "include_action_in_lm": bool(args.resolved_lm_action_target == "include"),
                    }
                    print(json.dumps(truncate_for_log(gt_log, args.log_gt_max_chars), ensure_ascii=False))

            if args.checkpoint_out and args.checkpoint_every_steps > 0 and global_step % args.checkpoint_every_steps == 0:
                save_training_checkpoint(
                    checkpoint_path=args.checkpoint_out,
                    agent_model=agent,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    global_step=global_step,
                    args=args,
                    extra_state={
                        "training_stage": args.training_stage,
                        "optimizer_steps": optimizer_steps,
                        "processed_rows": processed_rows,
                        "processed_rows_in_epoch": epoch_rows,
                        "active_batch_training": True,
                        "best_monitor_value": best_monitor_value,
                        "best_epoch": best_epoch,
                        "early_stop_bad_epochs": early_stop_bad_epochs,
                    },
                )

        if micro_step % args.grad_accum_steps != 0:
            if args.max_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    trainable_params,
                    max_norm=args.max_grad_norm,
                    error_if_nonfinite=True,
                )
            align_optimizer_state_with_params(optimizer)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
        if progress is not None:
            progress.close()
        avg_loss = epoch_loss_total / max(1, epoch_trained_rows)
        avg_lm_loss = epoch_lm_total / max(1, epoch_trained_rows)
        avg_action_head_loss = epoch_action_total / max(1, epoch_trained_rows)
        epoch_report = {
            "stage": "train_epoch",
            "epoch": epoch,
            "avg_loss": avg_loss,
            "avg_lm_loss": avg_lm_loss,
            "avg_action_head_loss": avg_action_head_loss,
            "processed_rows": epoch_rows,
            "trained_rows": epoch_trained_rows,
            "skipped_rows": skipped_rows_this_epoch,
            "optimizer_steps": optimizer_steps,
            "elapsed_seconds": time.time() - started,
        }
        monitor_values = {
            "loss": avg_loss,
            "lm_loss": avg_lm_loss,
            "action_head_loss": avg_action_head_loss,
        }
        monitor_value = float(monitor_values[args.early_stop_monitor])
        improved = (
            epoch_trained_rows > 0
            and monitor_value < best_monitor_value - float(args.early_stop_min_delta)
        )
        if improved:
            best_monitor_value = monitor_value
            best_epoch = epoch
            early_stop_bad_epochs = 0
        elif epoch_trained_rows > 0:
            early_stop_bad_epochs += 1
        epoch_report.update(
            {
                "early_stop_monitor": str(args.early_stop_monitor),
                "monitor_value": monitor_value,
                "best_monitor_value": best_monitor_value,
                "best_epoch": best_epoch,
                "early_stop_bad_epochs": early_stop_bad_epochs,
                "improved": bool(improved),
            }
        )
        history.append(epoch_report)
        print(json.dumps(epoch_report, ensure_ascii=False))

        checkpoint_extra_state = {
            "training_stage": args.training_stage,
            "optimizer_steps": optimizer_steps,
            "processed_rows": processed_rows,
            "processed_rows_in_epoch": 0,
            "active_batch_training": True,
            "epoch_complete": True,
            "best_monitor_value": best_monitor_value,
            "best_epoch": best_epoch,
            "early_stop_bad_epochs": early_stop_bad_epochs,
        }
        if args.checkpoint_out:
            save_training_checkpoint(
                checkpoint_path=args.checkpoint_out,
                agent_model=agent,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                global_step=global_step,
                args=args,
                extra_state=checkpoint_extra_state,
            )
        if improved and args.best_checkpoint_out:
            save_training_checkpoint(
                checkpoint_path=args.best_checkpoint_out,
                agent_model=agent,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                global_step=global_step,
                args=args,
                extra_state={**checkpoint_extra_state, "best_checkpoint": True},
            )
        if (
            args.early_stop_patience > 0
            and epoch >= args.early_stop_min_epochs
            and early_stop_bad_epochs >= args.early_stop_patience
        ):
            stopped_early = True
            print(
                json.dumps(
                    {
                        "stage": "early_stop",
                        "epoch": epoch,
                        "monitor": args.early_stop_monitor,
                        "best_epoch": best_epoch,
                        "best_monitor_value": best_monitor_value,
                        "bad_epochs": early_stop_bad_epochs,
                    },
                    ensure_ascii=False,
                )
            )
            break

    extra_metadata = {
        "latent_slot_count": int(args.latent_slot_count),
        "reasoning_alignment_mode": str(args.reasoning_alignment_mode),
        "reasoning_field_slot_counts": list(args.resolved_reasoning_field_slot_counts),
        "history_n": int(args.history_n),
        "active_batch_training": True,
        "batch_size": int(args.batch_size),
        "grad_accum_steps": int(args.grad_accum_steps),
        "pixel_pruned_visual": True,
        "pixel_prune_threshold": float(args.pixel_prune_threshold),
        "pixel_prune_predictor_order": str(args.pixel_prune_predictor_order),
        "pixel_temporal_reuse": bool(args.pixel_temporal_reuse),
        "pixel_temporal_threshold": float(args.pixel_temporal_threshold),
        "image_min_pixels": int(args.image_min_pixels),
        "image_max_pixels": int(args.image_max_pixels),
        "include_current_subtask_in_prompt": not bool(args.clean_observable_prompt),
        "include_expected_next_screen_in_prompt": not bool(args.clean_observable_prompt),
        "latent_scaffolds_in_prompt": not bool(args.clean_observable_prompt),
        "clean_observable_prompt": bool(args.clean_observable_prompt),
        "use_lora": bool(args.use_lora),
        "lora_r": int(args.lora_r),
        "lora_alpha": int(args.lora_alpha),
        "lora_dropout": float(args.lora_dropout),
        "training_stage": str(args.training_stage),
        "stage2_target_format": str(args.stage2_target_format),
        "action_format": str(args.action_format),
        "action_model": str(args.action_model),
        "lm_action_target": str(args.resolved_lm_action_target),
        "flow_action_sample_steps": int(args.flow_action_sample_steps),
        "flow_head_hidden_dim": int(args.flow_head_hidden_dim),
        "flow_head_depth": int(args.flow_head_depth),
        "two_way_hidden_dim": int(args.two_way_hidden_dim),
        "two_way_depth": int(args.two_way_depth),
        "two_way_num_heads": int(args.two_way_num_heads),
        "two_way_location_queries": int(args.two_way_location_queries),
        "two_way_dropout": float(args.two_way_dropout),
        "two_way_query_mode": str(args.two_way_query_mode),
        "two_way_candidate_coord_loss_weight": float(args.two_way_candidate_coord_loss_weight),
        "two_way_candidate_confidence_loss_weight": float(
            args.two_way_candidate_confidence_loss_weight
        ),
        "flow_continuous_source": str(args.flow_continuous_source),
        "action_head_loss_weight": float(args.action_head_loss_weight),
        "flow_action_loss_weight": float(args.flow_action_loss_weight),
        "flow_coord_loss_weight": float(args.flow_coord_loss_weight),
        "learnable_flow_coord_weight": bool(args.learnable_flow_coord_weight),
        "learned_flow_coord_weight": float(torch.exp(-agent.flow_coord_loss_log_var.detach()).cpu().item()),
        "flow_coord_loss_log_var": float(agent.flow_coord_loss_log_var.detach().cpu().item()),
        "flow_coord_loss_scale": float(args.flow_coord_loss_scale),
        "flow_coord_loss_space": str(args.flow_coord_loss_space),
        "flow_patch_loss_weight": float(args.flow_patch_loss_weight),
        "flow_patch_loss_mode": str(args.flow_patch_loss_mode),
        "flow_patch_gaussian_sigma": float(args.flow_patch_gaussian_sigma),
        "flow_pointer_coord_source": str(args.flow_pointer_coord_source),
        "flow_patch_logit_temperature": float(args.flow_patch_logit_temperature),
        "flow_patch_residual_scale": float(args.flow_patch_residual_scale),
        "action_hidden_source": str(args.action_hidden_source),
        "use_action_head": bool(args.action_head_loss_weight > 0.0),
        "train_action_head_only": bool(args.train_action_head_only),
        "checkpoint_out": str(args.checkpoint_out) if args.checkpoint_out else None,
        "best_checkpoint_out": str(args.best_checkpoint_out) if args.best_checkpoint_out else None,
        "resumed_from_checkpoint": str(args.resume_from) if args.resume_from else None,
        "checkpoint_every_steps": int(args.checkpoint_every_steps),
        "profile_stages": bool(args.profile_stages),
        "minimal_logging": bool(args.minimal_logging),
        "early_stop_monitor": str(args.early_stop_monitor),
        "early_stop_patience": int(args.early_stop_patience),
        "early_stop_min_delta": float(args.early_stop_min_delta),
        "best_epoch": int(best_epoch),
        "best_monitor_value": float(best_monitor_value),
        "stopped_early": bool(stopped_early),
    }
    agent.save_adapter(args.adapter_out, extra_metadata=extra_metadata)
    report = {
        "adapter_path": str(args.adapter_out),
        "elapsed_seconds": time.time() - started,
        "sample_count": int(total_rows),
        "trajectory_count": int(len(trajectories)),
        "epochs": int(args.epochs),
        "active_batch_training": True,
        "batch_size": int(args.batch_size),
        "grad_accum_steps": int(args.grad_accum_steps),
        "flow_continuous_source": str(args.flow_continuous_source),
        "flow_action_loss_weight": float(args.flow_action_loss_weight),
        "flow_coord_loss_weight": float(args.flow_coord_loss_weight),
        "learnable_flow_coord_weight": bool(args.learnable_flow_coord_weight),
        "learned_flow_coord_weight": float(torch.exp(-agent.flow_coord_loss_log_var.detach()).cpu().item()),
        "flow_coord_loss_log_var": float(agent.flow_coord_loss_log_var.detach().cpu().item()),
        "flow_coord_loss_scale": float(args.flow_coord_loss_scale),
        "flow_coord_loss_space": str(args.flow_coord_loss_space),
        "flow_patch_loss_weight": float(args.flow_patch_loss_weight),
        "flow_patch_loss_mode": str(args.flow_patch_loss_mode),
        "flow_patch_gaussian_sigma": float(args.flow_patch_gaussian_sigma),
        "processed_rows": int(processed_rows),
        "optimizer_steps": int(optimizer_steps),
        "best_epoch": int(best_epoch),
        "best_monitor_value": float(best_monitor_value),
        "stopped_early": bool(stopped_early),
        "history": history,
        **extra_metadata,
    }
    if args.report_out:
        Path(args.report_out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
