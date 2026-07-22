from __future__ import annotations

import json
import math
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore[assignment]

from qwen3_gui_agent.distributed_training import (
    cleanup_distributed,
    epoch_ordered_trajectories,
    gather_objects,
    init_distributed,
    maybe_disable_device_map_for_ddp,
    maybe_wrap_ddp,
    seed_everything,
    shard_ordered_items,
    sum_across_ranks,
)
from qwen3_gui_agent.lara_style_qwen3vl_agent import LaRAStyleQwen3VLAgent
from qwen3_gui_agent.rl.schema import read_jsonl
from qwen3_gui_agent.training_checkpoint import (
    align_optimizer_state_with_params,
    force_safe_adamw_runtime_flags,
    load_training_checkpoint,
    restore_gradient_state,
    save_training_checkpoint,
)
from qwen3_gui_agent.training_utils import build_scheduler, current_lr, resolve_device_map, resolve_torch_dtype
from train_lara_style_qwen3vl import build_trajectories, compact_debug_info
from train_lara_style_qwen3vl_active_batch import (
    compact_sample_summary,
    make_state,
    parse_args,
    sample_from_state,
)


_RESUME_COMPATIBILITY_FIELDS = (
    "steps",
    "dataset_root",
    "model",
    "torch_dtype",
    "lr",
    "lr_scheduler",
    "warmup_ratio",
    "min_lr_ratio",
    "max_grad_norm",
    "max_samples",
    "batch_size",
    "grad_accum_steps",
    "history_n",
    "latent_slot_count",
    "reasoning_alignment_mode",
    "reasoning_field_slot_counts",
    "pixel_prune_threshold",
    "pixel_prune_predictor_order",
    "pixel_temporal_reuse",
    "pixel_temporal_threshold",
    "image_min_pixels",
    "image_max_pixels",
    "training_stage",
    "action_format",
    "action_model",
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
    "flow_continuous_source",
    "resolved_lm_action_target",
    "stage2_target_format",
    "stage1_max_reasoning_chars",
    "stage2_explicit_keep_start",
    "stage2_explicit_keep_end",
    "stage2_min_explicit_tokens",
    "stage2_max_thinking_tokens",
    "lm_loss_weight",
    "reasoning_align_weight",
    "future_frame_loss_weight",
    "latent_diversity_weight",
    "action_head_loss_weight",
    "flow_action_loss_weight",
    "flow_coord_loss_weight",
    "learnable_flow_coord_weight",
    "flow_coord_loss_scale",
    "flow_coord_loss_space",
    "flow_patch_loss_weight",
    "flow_patch_loss_mode",
    "flow_patch_gaussian_sigma",
    "flow_pointer_coord_source",
    "flow_patch_logit_temperature",
    "flow_patch_residual_scale",
    "action_hidden_source",
    "train_backbone",
    "train_embeddings",
    "use_lora",
    "lora_r",
    "lora_alpha",
    "lora_dropout",
    "gradient_checkpointing",
    "clean_observable_prompt",
    "train_action_head_only",
    "shuffle_trajectories",
    "seed",
)

# Early stopping only controls when a run exits and which epoch is retained as
# best. It is safe to enable or tune when extending an otherwise compatible
# checkpoint, so these fields intentionally are not resume-compatibility keys.

# These options were added after the first DDP action-head checkpoints were
# written. A legacy checkpoint may omit them, but it is safe to resume only
# when the new command keeps the behavior those checkpoints used implicitly.
_RESUME_LEGACY_ARG_DEFAULTS = {
    "max_grad_norm": 0.0,
    "use_lora": False,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "gradient_checkpointing": False,
    "clean_observable_prompt": False,
    "reasoning_alignment_mode": "aggregate",
    "reasoning_field_slot_counts": "auto",
    "two_way_query_mode": "semantic_pool",
    "early_stop_patience": 0,
    "early_stop_min_delta": 0.0,
    "early_stop_min_epochs": 1,
    "early_stop_monitor": "loss",
}


def _chunk_trajectories(trajectories: list[dict[str, Any]], wave_size: int) -> list[list[dict[str, Any]]]:
    return [trajectories[index : index + wave_size] for index in range(0, len(trajectories), wave_size)]


def _wave_micro_batch_count(wave: list[dict[str, Any]]) -> int:
    if not wave:
        return 0
    return max(len(trajectory["rows"]) for trajectory in wave)


def _global_wave_plan(local_waves: list[list[dict[str, Any]]], runtime: Any) -> list[int]:
    local_counts = [_wave_micro_batch_count(wave) for wave in local_waves]
    gathered_counts = gather_objects(local_counts, runtime)
    max_wave_count = max((len(counts) for counts in gathered_counts), default=0)
    global_counts: list[int] = []
    for wave_index in range(max_wave_count):
        global_counts.append(
            max(int(counts[wave_index]) if wave_index < len(counts) else 0 for counts in gathered_counts)
        )
    return global_counts


def _advance_wave_states(active_states: list[dict[str, Any]]) -> None:
    for state in active_states:
        if state["row_index"] >= len(state["trajectory"]["rows"]):
            continue
        row = state["trajectory"]["rows"][state["row_index"]]
        state["history_image_paths"].append(row["image_path"])
        state["row_index"] += 1


def _build_samples_from_states(
    *,
    states: list[dict[str, Any]],
    history_slots: int,
    args: Any,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for state in states:
        row = state["trajectory"]["rows"][state["row_index"]]
        explicit_reasoning = str(row["explicit_supervision"])
        if args.training_stage == "stage1" and args.stage1_max_reasoning_chars > 0:
            explicit_reasoning = explicit_reasoning[: int(args.stage1_max_reasoning_chars)].strip()
        samples.append(sample_from_state(state, history_slots=history_slots, explicit_reasoning=explicit_reasoning))
    return samples


def _load_or_init_agent(args: Any, runtime: Any) -> tuple[LaRAStyleQwen3VLAgent, torch.dtype]:
    dtype = resolve_torch_dtype(args.torch_dtype)
    device_map = maybe_disable_device_map_for_ddp(resolve_device_map(args.device_map), runtime)
    agent = LaRAStyleQwen3VLAgent.from_pretrained(
        args.model,
        device_map=device_map,
        torch_dtype=dtype,
        latent_slot_count=args.latent_slot_count,
        reasoning_alignment_mode=args.reasoning_alignment_mode,
        reasoning_field_slot_counts=args.resolved_reasoning_field_slot_counts,
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
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    if args.init_adapter:
        load_info = agent.load_adapter(args.init_adapter, strict=False)
        allowed_shape_skip_prefixes = (
            "action_head.",
            "flow_action_head.",
            "latent_two_way_action_head.",
        )
        unexpected_shape_skips = [
            item
            for item in load_info.get("skipped_shape_mismatch", [])
            if not str(item.get("key", "")).startswith(allowed_shape_skip_prefixes)
        ]
        if unexpected_shape_skips:
            raise RuntimeError(
                "Initialization adapter has incompatible non-action tensors: "
                + json.dumps(unexpected_shape_skips[:8], ensure_ascii=False)
            )
        if runtime.is_main:
            print(json.dumps({"stage": "init_adapter", "init_adapter": args.init_adapter, "load_info": load_info}, ensure_ascii=False))
    if args.gradient_checkpointing:
        try:
            agent.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError as exc:
            raise RuntimeError(
                "This DDP trainer requires non-reentrant gradient checkpointing. "
                "Upgrade transformers or remove --gradient-checkpointing."
            ) from exc
        if hasattr(agent.model, "config"):
            agent.model.config.use_cache = False
        if runtime.is_main:
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
    # Explicit CLI runtime settings must win over metadata inherited from the
    # initialization adapter. Otherwise a requested 32-step flow run can be
    # silently reset to the old adapter's default of 8 steps.
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
    # load_adapter also restores source-stage metadata. The current command is
    # authoritative for this new stage and may intentionally instantiate a
    # different action-head width/depth.
    agent.training_stage = str(args.training_stage)
    agent.stage2_target_format = str(args.stage2_target_format)
    agent.action_format = str(args.action_format)
    agent.lm_action_target = str(args.resolved_lm_action_target)
    agent.flow_head_hidden_dim = int(args.flow_head_hidden_dim) if int(args.flow_head_hidden_dim) > 0 else None
    agent.flow_head_depth = int(args.flow_head_depth)
    agent.include_current_subtask_in_prompt = not bool(args.clean_observable_prompt)
    agent.include_expected_next_screen_in_prompt = not bool(args.clean_observable_prompt)
    agent.latent_scaffolds_in_prompt = not bool(args.clean_observable_prompt)
    # DDP inspects every tensor returned by forward, while action-head losses
    # intentionally use only the branches supported by the current batch.
    agent.ddp_zero_unused_branch_anchors = bool(runtime.enabled)
    if args.learnable_flow_coord_weight:
        init_weight = max(1e-6, float(args.flow_coord_loss_weight))
        agent.flow_coord_loss_log_var.data.fill_(-math.log(init_weight))
    return agent, dtype


def _configure_trainable_params(agent: LaRAStyleQwen3VLAgent, args: Any) -> None:
    from train_lara_style_qwen3vl import configure_trainable_params

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


def _validate_training_configuration(args: Any) -> None:
    if args.init_adapter and args.resume_from:
        raise ValueError("Use only one of --init-adapter or --resume-from.")
    positive_integer_fields = {
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "grad_accum_steps": int(args.grad_accum_steps),
        "history_n": int(args.history_n),
        "latent_slot_count": int(args.latent_slot_count),
        "log_every": int(args.log_every),
        "prep_log_every": int(args.prep_log_every),
    }
    invalid_positive_fields = {
        name: value for name, value in positive_integer_fields.items() if value <= 0
    }
    if invalid_positive_fields:
        raise ValueError(
            "These integer options must be positive: "
            + json.dumps(invalid_positive_fields, sort_keys=True)
        )
    if int(args.checkpoint_every_steps) < 0:
        raise ValueError("--checkpoint-every-steps must be >= 0.")
    if float(args.lr) <= 0.0:
        raise ValueError("--lr must be > 0.")
    if float(args.max_grad_norm) < 0.0:
        raise ValueError("--max-grad-norm must be >= 0.")
    if int(args.image_min_pixels) < 0 or int(args.image_max_pixels) < 0:
        raise ValueError("Image pixel limits must be >= 0.")
    if (
        int(args.image_min_pixels) > 0
        and int(args.image_max_pixels) > 0
        and int(args.image_min_pixels) > int(args.image_max_pixels)
    ):
        raise ValueError("--image-min-pixels cannot exceed --image-max-pixels.")
    objective_weights = {
        "lm_loss_weight": float(args.lm_loss_weight),
        "reasoning_align_weight": float(args.reasoning_align_weight),
        "future_frame_loss_weight": float(args.future_frame_loss_weight),
        "latent_diversity_weight": float(args.latent_diversity_weight),
        "action_head_loss_weight": float(args.action_head_loss_weight),
    }
    negative_objective_weights = {
        name: weight for name, weight in objective_weights.items() if weight < 0.0
    }
    if negative_objective_weights:
        raise ValueError(
            "Top-level loss weights must be non-negative: "
            + json.dumps(negative_objective_weights, sort_keys=True)
        )
    flow_loss_weights = {
        "flow_action_loss_weight": float(args.flow_action_loss_weight),
        "flow_coord_loss_weight": float(args.flow_coord_loss_weight),
        "flow_patch_loss_weight": float(args.flow_patch_loss_weight),
    }
    negative_flow_weights = {
        name: weight for name, weight in flow_loss_weights.items() if weight < 0.0
    }
    if negative_flow_weights:
        raise ValueError(
            "Flow-head loss weights must be non-negative: "
            + json.dumps(negative_flow_weights, sort_keys=True)
        )
    if int(args.flow_action_sample_steps) <= 0:
        raise ValueError("--flow-action-sample-steps must be > 0.")
    if int(args.flow_head_hidden_dim) < 0 or int(args.flow_head_depth) <= 0:
        raise ValueError("Flow-head hidden dimension must be >= 0 and depth must be > 0.")
    if float(args.flow_patch_logit_temperature) <= 0.0:
        raise ValueError("--flow-patch-logit-temperature must be > 0.")
    if float(args.flow_patch_residual_scale) < 0.0:
        raise ValueError("--flow-patch-residual-scale must be >= 0.")
    if args.flow_patch_loss_mode == "gaussian" and float(args.flow_patch_gaussian_sigma) <= 0.0:
        raise ValueError("Gaussian patch loss requires --flow-patch-gaussian-sigma > 0.")
    if not any(weight > 0.0 for weight in objective_weights.values()):
        raise ValueError(
            "At least one top-level training loss weight must be positive: "
            + json.dumps(objective_weights, sort_keys=True)
        )
    if args.train_action_head_only and float(args.action_head_loss_weight) <= 0.0:
        raise ValueError("--train-action-head-only requires --action-head-loss-weight > 0.")


def _compute_total_scheduler_steps(
    *,
    trajectories: list[dict[str, Any]],
    args: Any,
    runtime: Any,
) -> int:
    total_scheduler_steps = 0
    for epoch in range(1, max(1, int(args.epochs)) + 1):
        ordered = epoch_ordered_trajectories(
            trajectories,
            shuffle=bool(args.shuffle_trajectories),
            seed=int(args.seed),
            epoch=epoch,
        )
        local_trajectories = shard_ordered_items(ordered, runtime)
        local_waves = _chunk_trajectories(local_trajectories, int(args.batch_size))
        global_wave_plan = _global_wave_plan(local_waves, runtime)
        total_scheduler_steps += max(1, math.ceil(sum(global_wave_plan) / max(1, int(args.grad_accum_steps))))
    return max(1, total_scheduler_steps)


def main() -> None:
    args = parse_args()
    _validate_training_configuration(args)
    runtime = init_distributed()
    completed_without_error = False
    try:
        seed_everything(int(args.seed))
        if runtime.is_main:
            print(
                json.dumps(
                    {
                        "stage": "ddp_runtime",
                        "enabled": bool(runtime.enabled),
                        "rank": int(runtime.rank),
                        "local_rank": int(runtime.local_rank),
                        "world_size": int(runtime.world_size),
                        "batch_size_per_rank": int(args.batch_size),
                    },
                    ensure_ascii=False,
                )
            )

        # build_trajectories expects prep_log_every > 0 because it uses modulo
        # logging internally. Non-main ranks still build the same metadata, but
        # use a huge interval so they stay quiet without tripping division by 0.
        quiet_prep_log_every = int(args.max_samples) + 1 if int(args.max_samples) > 0 else 10**18
        trajectories = build_trajectories(
            steps=read_jsonl(Path(args.steps)),
            dataset_root=Path(args.dataset_root),
            max_samples=args.max_samples,
            prep_log_every=args.prep_log_every if runtime.is_main else quiet_prep_log_every,
        )
        if not trajectories:
            raise RuntimeError("No valid LaRA-style training samples found.")

        local_row_count_preview = sum(len(trajectory["rows"]) for trajectory in shard_ordered_items(trajectories, runtime))
        global_row_count = int(sum_across_ranks(local_row_count_preview, runtime))
        if runtime.is_main:
            print(
                json.dumps(
                    {
                        "stage": "prepare_lara_style_trajectories_done",
                        "global_samples": int(global_row_count),
                        "global_trajectory_count": int(len(trajectories)),
                        "world_size": int(runtime.world_size),
                    },
                    ensure_ascii=False,
                )
            )

        agent, _ = _load_or_init_agent(args, runtime)
        _configure_trainable_params(agent, args)
        # DDP creates gradient buckets from the parameter dtype/device present
        # at construction time. Auxiliary heads are initialized in fp32 while
        # Qwen runs in bf16, so align them before wrapping; changing parameter
        # dtype inside the first forward corrupts DDP's bucket expectations.
        embedding_weight = agent.model.get_input_embeddings().weight
        agent._align_auxiliary_modules(embedding_weight)
        train_model = maybe_wrap_ddp(agent, runtime)
        agent_model = train_model.module if hasattr(train_model, "module") else train_model
        agent_model.train()
        if args.train_action_head_only:
            # Keep the frozen Stage-2 feature extractor deterministic so the
            # action head sees the same latent distribution at train and eval.
            agent_model.model.eval()

        trainable_params = [parameter for parameter in agent_model.parameters() if parameter.requires_grad]
        if not trainable_params:
            raise RuntimeError("No trainable parameters found.")
        if runtime.is_main:
            trainable_dtype_counts: dict[str, int] = {}
            for parameter in trainable_params:
                dtype_name = str(parameter.dtype)
                trainable_dtype_counts[dtype_name] = (
                    trainable_dtype_counts.get(dtype_name, 0) + int(parameter.numel())
                )
            print(
                json.dumps(
                    {
                        "stage": "ddp_trainable_parameters",
                        "world_size": int(runtime.world_size),
                        "trainable_params_per_replica": int(
                            sum(parameter.numel() for parameter in trainable_params)
                        ),
                        "total_params_per_replica": int(
                            sum(parameter.numel() for parameter in agent_model.parameters())
                        ),
                        "trainable_param_dtype_counts": trainable_dtype_counts,
                    },
                    ensure_ascii=False,
                )
            )
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, foreach=False)
        force_safe_adamw_runtime_flags(optimizer)
        trainable_signatures = {
            name: (id(parameter), str(parameter.device), str(parameter.dtype))
            for name, parameter in agent_model.named_parameters()
            if parameter.requires_grad
        }
        runtime_alignment_checked = False

        total_scheduler_steps = _compute_total_scheduler_steps(trajectories=trajectories, args=args, runtime=runtime)
        scheduler = build_scheduler(
            optimizer=optimizer,
            scheduler_name=args.lr_scheduler,
            total_steps=total_scheduler_steps,
            warmup_ratio=args.warmup_ratio,
            min_lr_ratio=args.min_lr_ratio,
        )

        start_epoch = 1
        resume_wave_index = 1
        resume_wave_micro_index = 1
        resume_epoch_global_micro_index = 0
        resume_accum_micro_batches = 0
        resume_epoch_loss_total = 0.0
        resume_epoch_lm_total = 0.0
        resume_epoch_action_total = 0.0
        resume_epoch_local_rows = 0
        global_step = 0
        optimizer_steps = 0
        local_rows_total = 0
        best_monitor_value = float("inf")
        best_epoch = 0
        early_stop_bad_epochs = 0
        stopped_early = False
        resumed_from_checkpoint = None
        history: list[dict[str, Any]] = []
        if args.resume_from:
            payload = load_training_checkpoint(
                checkpoint_path=args.resume_from,
                agent_model=agent_model,
                optimizer=optimizer,
                scheduler=scheduler,
                current_args=args,
                compatibility_fields=_RESUME_COMPATIBILITY_FIELDS,
                expected_extra_state={
                    "ddp_active_batch_training": True,
                    "ddp_world_size": int(runtime.world_size),
                },
                legacy_arg_defaults=_RESUME_LEGACY_ARG_DEFAULTS,
            )
            restore_gradient_state(agent_model=agent_model, gradient_state_dict=payload.get("gradient_state_dict"))
            align_optimizer_state_with_params(optimizer)
            force_safe_adamw_runtime_flags(optimizer)
            extra_state = payload.get("extra_state", {}) or {}
            checkpoint_epoch = int(payload.get("epoch", 0))
            saved_args = payload.get("args", {}) or {}
            saved_total_epochs = int(saved_args.get("epochs", checkpoint_epoch))
            if int(args.epochs) < checkpoint_epoch:
                raise RuntimeError(
                    "--epochs is a total epoch target when resuming and cannot be lower than "
                    f"the checkpoint epoch: requested={int(args.epochs)}, checkpoint={checkpoint_epoch}."
                )
            start_epoch = int(extra_state.get("next_epoch", checkpoint_epoch + 1))
            resume_wave_index = int(extra_state.get("next_wave_index", 1))
            resume_wave_micro_index = int(extra_state.get("next_wave_micro_index", 1))
            resume_epoch_global_micro_index = int(extra_state.get("epoch_global_micro_index", 0))
            resume_accum_micro_batches = int(extra_state.get("accum_micro_batches", 0))
            global_step = int(payload.get("global_step", 0))
            optimizer_steps = int(extra_state.get("optimizer_steps", 0))
            local_rows_total_per_rank = extra_state.get("local_rows_total_per_rank")
            if isinstance(local_rows_total_per_rank, list) and runtime.rank < len(local_rows_total_per_rank):
                local_rows_total = int(local_rows_total_per_rank[runtime.rank])
            else:
                # Backward compatibility with checkpoints written before
                # rank-specific counters were recorded.
                local_rows_total = int(extra_state.get("local_rows_total", 0))
            epoch_accumulators_per_rank = extra_state.get("epoch_accumulators_per_rank")
            if (
                isinstance(epoch_accumulators_per_rank, list)
                and runtime.rank < len(epoch_accumulators_per_rank)
                and isinstance(epoch_accumulators_per_rank[runtime.rank], dict)
            ):
                rank_accumulators = epoch_accumulators_per_rank[runtime.rank]
                resume_epoch_loss_total = float(rank_accumulators.get("loss_total", 0.0))
                resume_epoch_lm_total = float(rank_accumulators.get("lm_total", 0.0))
                resume_epoch_action_total = float(rank_accumulators.get("action_total", 0.0))
                resume_epoch_local_rows = int(rank_accumulators.get("local_rows", 0))
            best_monitor_value = float(extra_state.get("best_monitor_value", float("inf")))
            best_epoch = int(extra_state.get("best_epoch", 0))
            early_stop_bad_epochs = int(extra_state.get("early_stop_bad_epochs", 0))
            saved_history = extra_state.get("history", [])
            if isinstance(saved_history, list):
                history = [dict(item) for item in saved_history if isinstance(item, dict)]
            if (
                bool(extra_state.get("epoch_complete", False))
                and int(args.early_stop_patience) > 0
                and checkpoint_epoch >= int(args.early_stop_min_epochs)
                and early_stop_bad_epochs >= int(args.early_stop_patience)
            ):
                stopped_early = True
                start_epoch = int(args.epochs) + 1
            resumed_from_checkpoint = str(args.resume_from)
            if runtime.is_main:
                if int(args.epochs) != saved_total_epochs:
                    print(
                        json.dumps(
                            {
                                "stage": "resume_ddp_epoch_target_override",
                                "checkpoint_total_epochs": saved_total_epochs,
                                "requested_total_epochs": int(args.epochs),
                                "checkpoint_epoch": checkpoint_epoch,
                            },
                            ensure_ascii=False,
                        )
                    )
                print(
                    json.dumps(
                        {
                            "stage": "resume_ddp_checkpoint",
                            "resume_from": resumed_from_checkpoint,
                            "resume_epoch": start_epoch,
                            "resume_wave_index": resume_wave_index,
                            "resume_wave_micro_index": resume_wave_micro_index,
                            "resume_global_step": global_step,
                            "optimizer_steps": optimizer_steps,
                        },
                        ensure_ascii=False,
                    )
                )
            best_path = Path(args.best_checkpoint_out) if args.best_checkpoint_out else None
            best_missing = best_path is not None and (
                not best_path.is_file() or best_path.stat().st_size <= 0
            )
            if best_missing and best_epoch == checkpoint_epoch:
                # The job may have died after writing latest.ckpt.pt but before
                # writing best.ckpt.pt. Recreate that exact best state before
                # any rank enters the next DDP forward.
                if runtime.enabled:
                    import torch.distributed as dist

                    dist.barrier()
                if runtime.is_main:
                    repaired_extra_state = dict(extra_state)
                    repaired_extra_state["best_checkpoint"] = True
                    save_training_checkpoint(
                        checkpoint_path=best_path,
                        agent_model=agent_model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=checkpoint_epoch,
                        global_step=global_step,
                        args=args,
                        extra_state=repaired_extra_state,
                    )
                    print(
                        json.dumps(
                            {
                                "stage": "repair_missing_best_checkpoint",
                                "best_checkpoint_path": str(best_path),
                                "best_epoch": int(best_epoch),
                            },
                            ensure_ascii=False,
                        )
                    )
                if runtime.enabled:
                    dist.barrier()

        started = time.time()
        history_slots = max(0, int(args.history_n) - 1)

        for epoch in range(start_epoch, int(args.epochs) + 1):
            ordered_trajectories = epoch_ordered_trajectories(
                trajectories,
                shuffle=bool(args.shuffle_trajectories),
                seed=int(args.seed),
                epoch=epoch,
            )
            local_trajectories = shard_ordered_items(ordered_trajectories, runtime)
            local_waves = _chunk_trajectories(local_trajectories, int(args.batch_size))
            global_wave_plan = _global_wave_plan(local_waves, runtime)
            epoch_global_micro_batches = sum(global_wave_plan)
            if runtime.is_main:
                print(
                    json.dumps(
                        {
                            "stage": "epoch_wave_plan",
                            "epoch": int(epoch),
                            "global_wave_count": int(len(global_wave_plan)),
                            "epoch_global_micro_batches": int(epoch_global_micro_batches),
                            "batch_size_per_rank": int(args.batch_size),
                            "effective_max_batch": int(args.batch_size) * int(runtime.world_size),
                        },
                        ensure_ascii=False,
                    )
                )

            if epoch == start_epoch:
                epoch_loss_total = float(resume_epoch_loss_total)
                epoch_lm_total = float(resume_epoch_lm_total)
                epoch_action_total = float(resume_epoch_action_total)
                epoch_local_rows = int(resume_epoch_local_rows)
            else:
                epoch_loss_total = 0.0
                epoch_lm_total = 0.0
                epoch_action_total = 0.0
                epoch_local_rows = 0
            epoch_global_micro_index = resume_epoch_global_micro_index if epoch == start_epoch else 0
            accum_micro_batches = resume_accum_micro_batches if epoch == start_epoch else 0
            if accum_micro_batches == 0:
                optimizer.zero_grad(set_to_none=True)

            progress = (
                tqdm(
                    total=epoch_global_micro_batches,
                    desc=f"train_lara_ddp_epoch{epoch}",
                    dynamic_ncols=True,
                    leave=True,
                    mininterval=1.0,
                    initial=epoch_global_micro_index,
                    disable=not runtime.is_main,
                )
                if tqdm is not None and not args.no_progress_bar
                else None
            )

            wave_start_index = resume_wave_index if epoch == start_epoch else 1
            for wave_index in range(wave_start_index, len(global_wave_plan) + 1):
                global_wave_micro_batches = global_wave_plan[wave_index - 1]
                local_wave = local_waves[wave_index - 1] if wave_index - 1 < len(local_waves) else []
                active_states = [make_state(trajectory) for trajectory in local_wave]
                wave_micro_start = 1
                if epoch == start_epoch and wave_index == resume_wave_index:
                    wave_micro_start = max(1, int(resume_wave_micro_index))
                    for _ in range(1, wave_micro_start):
                        _advance_wave_states(active_states)
                if runtime.is_main:
                    print(
                        json.dumps(
                            {
                                "stage": "ddp_wave_start",
                                "epoch": int(epoch),
                                "wave_index": int(wave_index),
                                "local_wave_sizes": [int(value) for value in gather_objects(len(local_wave), runtime)],
                                "global_wave_micro_batches": int(global_wave_micro_batches),
                            },
                            ensure_ascii=False,
                        )
                    )
                else:
                    gather_objects(len(local_wave), runtime)

                for wave_micro_index in range(wave_micro_start, global_wave_micro_batches + 1):
                    epoch_global_micro_index += 1
                    batch_states = [
                        state
                        for state in active_states
                        if state["row_index"] < len(state["trajectory"]["rows"])
                    ]

                    output = None
                    action_head_metrics: dict[str, Any] = {}
                    aux_metrics: dict[str, float] = {}
                    lm_loss = None
                    action_head_loss = None
                    total_loss = None
                    batch_rows = 0
                    batch_summary: list[dict[str, Any]] = []
                    batch_started_at = time.perf_counter()

                    if args.training_stage == "stage2":
                        progress_ratio = min(
                            1.0,
                            max(0.0, float(global_step) / float(max(1, total_scheduler_steps - 1))),
                        )
                        current_stage2_keep_ratio = (
                            float(args.stage2_explicit_keep_start)
                            + (
                                float(args.stage2_explicit_keep_end)
                                - float(args.stage2_explicit_keep_start)
                            )
                            * progress_ratio
                        )
                    else:
                        current_stage2_keep_ratio = 1.0

                    if batch_states:
                        samples = _build_samples_from_states(
                            states=batch_states,
                            history_slots=history_slots,
                            args=args,
                        )
                        output = train_model(
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
                        if not runtime_alignment_checked:
                            changed_parameters = []
                            for name, parameter in agent_model.named_parameters():
                                if not parameter.requires_grad or name not in trainable_signatures:
                                    continue
                                current = (id(parameter), str(parameter.device), str(parameter.dtype))
                                if current != trainable_signatures[name]:
                                    changed_parameters.append(
                                        {
                                            "name": name,
                                            "before": trainable_signatures[name],
                                            "after": current,
                                        }
                                    )
                            if changed_parameters:
                                raise RuntimeError(
                                    "Trainable parameters changed dtype/device after DDP construction: "
                                    + json.dumps(changed_parameters[:8], ensure_ascii=False)
                                )
                            runtime_alignment_checked = True
                        aux_losses, aux_metrics = agent_model.compute_auxiliary_losses(
                            output,
                            training_stage=args.training_stage,
                        )
                        action_head_losses, action_head_metrics = agent_model.compute_action_head_losses(output)
                        if "flow_action_loss" in action_head_losses:
                            action_head_losses["flow_action_loss"] = (
                                args.flow_action_loss_weight * action_head_losses["flow_action_loss"]
                            )
                        if "flow_action_coord_loss" in action_head_losses:
                            raw_coord_loss = action_head_losses["flow_action_coord_loss"]
                            if args.learnable_flow_coord_weight:
                                coord_log_var = agent_model.flow_coord_loss_log_var.to(
                                    device=raw_coord_loss.device,
                                    dtype=torch.float32,
                                )
                                action_head_losses["flow_action_coord_loss"] = (
                                    torch.exp(-coord_log_var) * raw_coord_loss.float() + coord_log_var
                                ).to(dtype=raw_coord_loss.dtype)
                            else:
                                action_head_losses["flow_action_coord_loss"] = (
                                    args.flow_coord_loss_weight * raw_coord_loss
                                )
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

                        lm_loss = output.loss if output.loss is not None else output.latent_reasoning_summary.new_zeros(())
                        reasoning_loss = aux_losses.get("reasoning_alignment_loss", lm_loss.new_zeros(()))
                        future_loss = aux_losses.get("future_frame_loss", lm_loss.new_zeros(()))
                        diversity_loss = aux_losses.get("latent_diversity_loss", lm_loss.new_zeros(()))
                        action_head_loss = (
                            sum(action_head_losses.values(), start=lm_loss.new_zeros(()))
                            if action_head_losses
                            else lm_loss.new_zeros(())
                        )
                        total_loss = (
                            args.lm_loss_weight * lm_loss
                            + args.reasoning_align_weight * reasoning_loss
                            + args.future_frame_loss_weight * future_loss
                            + args.latent_diversity_weight * diversity_loss
                            + args.action_head_loss_weight * action_head_loss
                        )
                        batch_rows = len(samples)
                        batch_summary = compact_sample_summary(states=batch_states, samples=samples)
                    else:
                        # Every rank must execute DDP.forward for every global
                        # micro-batch. Skipping it on an empty rank leaves the
                        # reducer in a different state and eventually causes a
                        # collective mismatch/NCCL timeout. Run one real graph
                        # with zero weight, but do not advance data or metrics.
                        dummy_state = make_state(ordered_trajectories[0])
                        dummy_samples = _build_samples_from_states(
                            states=[dummy_state],
                            history_slots=history_slots,
                            args=args,
                        )
                        output = train_model(
                            dummy_samples,
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
                        aux_losses, _ = agent_model.compute_auxiliary_losses(
                            output,
                            training_stage=args.training_stage,
                        )
                        action_head_losses, _ = agent_model.compute_action_head_losses(output)
                        graph_terms = [
                            output.loss,
                            output.latent_reasoning_summary.sum(),
                            *aux_losses.values(),
                            *action_head_losses.values(),
                        ]
                        total_loss = output.latent_reasoning_summary.new_zeros(())
                        for graph_term in graph_terms:
                            if graph_term is not None:
                                total_loss = total_loss + graph_term * 0.0
                        lm_loss = total_loss.detach().new_zeros(())
                        action_head_loss = total_loss.detach().new_zeros(())

                    accum_micro_batches += 1
                    should_step = (
                        accum_micro_batches >= int(args.grad_accum_steps)
                        or epoch_global_micro_index >= epoch_global_micro_batches
                    )
                    global_batch_rows = int(sum_across_ranks(batch_rows, runtime))
                    if global_batch_rows <= 0:
                        raise RuntimeError(
                            f"Global DDP micro-batch is empty at epoch={epoch}, "
                            f"wave={wave_index}, micro={wave_micro_index}."
                        )
                    local_loss_finite = int(
                        bool(torch.isfinite(total_loss.detach()).all().item())
                    )
                    finite_rank_count = int(sum_across_ranks(local_loss_finite, runtime))
                    if finite_rank_count != int(runtime.world_size):
                        nonfinite_details = gather_objects(
                            {
                                "rank": int(runtime.rank),
                                "finite": bool(local_loss_finite),
                                "total_loss": float(total_loss.detach().item()),
                                "action_head_loss": float(action_head_loss.detach().item()),
                                "epoch": int(epoch),
                                "wave_index": int(wave_index),
                                "wave_micro_index": int(wave_micro_index),
                            },
                            runtime,
                        )
                        optimizer.zero_grad(set_to_none=True)
                        raise FloatingPointError(
                            "Non-finite DDP loss detected before backward: "
                            + json.dumps(nonfinite_details, ensure_ascii=False)
                        )
                    # DDP averages rank gradients equally. Active trajectory
                    # batches can have different local sizes, so weight each
                    # rank mean by its sample share to recover the true global
                    # sample mean instead of over-weighting a short rank.
                    ddp_sample_weight = (
                        float(runtime.world_size) * float(batch_rows) / float(global_batch_rows)
                    )
                    sync_context = nullcontext()
                    if runtime.enabled and hasattr(train_model, "no_sync") and not should_step:
                        sync_context = train_model.no_sync()
                    with sync_context:
                        (
                            total_loss
                            * ddp_sample_weight
                            / float(args.grad_accum_steps)
                        ).backward()

                    if batch_rows > 0:
                        local_rows_total += batch_rows
                        epoch_local_rows += batch_rows
                        epoch_loss_total += float(total_loss.detach().item()) * batch_rows
                        epoch_lm_total += float(lm_loss.detach().item()) * batch_rows
                        epoch_action_total += float(action_head_loss.detach().item()) * batch_rows
                        _advance_wave_states(batch_states)

                    if should_step:
                        if args.action_model == "flow_matching" and args.flow_action_loss_weight <= 0.0:
                            # Keep the parameter group stable for checkpoint
                            # compatibility, but do not apply AdamW decay or
                            # create optimizer state for the zero-weight flow
                            # velocity branch.
                            for module in [
                                agent_model.flow_action_head.condition_proj,
                                agent_model.flow_action_head.flow_net,
                            ]:
                                for parameter in module.parameters():
                                    parameter.grad = None
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
                        global_step += 1
                        optimizer_steps += 1
                        accum_micro_batches = 0

                    step_seconds = time.perf_counter() - batch_started_at
                    processed_rows_global = int(sum_across_ranks(local_rows_total, runtime)) if should_step else None

                    if progress is not None:
                        progress.update(1)
                        progress.set_postfix(
                            gstep=global_step,
                            rows=processed_rows_global if processed_rows_global is not None else "-",
                            local_batch=batch_rows,
                            global_batch=global_batch_rows,
                            loss=(f"{float(total_loss.detach().item()):.3f}" if batch_rows > 0 else "-"),
                            ah=(f"{float(action_head_loss.detach().item()):.3f}" if batch_rows > 0 else "-"),
                            l1=(f"{float(action_head_metrics.get('action_head_teacher_pointer_l1', 0.0)):.3f}" if batch_rows > 0 else "-"),
                            patch_acc=(
                                f"{float(action_head_metrics.get('action_head_patch_accuracy', 0.0)):.3f}"
                                if batch_rows > 0
                                else "-"
                            ),
                            refresh=False,
                        )

                    if runtime.is_main and should_step and global_step % int(args.log_every) == 0:
                        debug_info = compact_debug_info(output.debug_info, keep_full_debug=False) if output is not None else {}
                        sample_log = {
                            "stage": "train_lara_ddp_step",
                            "epoch": int(epoch),
                            "wave_index": int(wave_index),
                            "wave_micro_index": int(wave_micro_index),
                            "global_step": int(global_step),
                            "optimizer_steps": int(optimizer_steps),
                            "local_batch_size": int(batch_rows),
                            "global_batch_size": int(global_batch_rows),
                            "processed_rows_global": int(processed_rows_global or 0),
                            "loss": float(total_loss.detach().item()) if batch_rows > 0 else None,
                            "lm_loss": float(lm_loss.detach().item()) if batch_rows > 0 else None,
                            "action_head_loss": float(action_head_loss.detach().item()) if batch_rows > 0 else None,
                            "action_head_teacher_pointer_l1": float(action_head_metrics.get("action_head_teacher_pointer_l1", 0.0)),
                            "action_head_coord_loss": float(action_head_metrics.get("action_head_coord_loss", 0.0)),
                            "action_head_patch_accuracy": float(
                                action_head_metrics.get("action_head_patch_accuracy", 0.0)
                            ),
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
                            "learned_flow_coord_weight": float(
                                torch.exp(-agent_model.flow_coord_loss_log_var.detach()).cpu().item()
                            ),
                            "lr": current_lr(optimizer),
                            "step_seconds": float(step_seconds),
                            "prompt_lengths": (debug_info or {}).get("prompt_lengths"),
                            "batch_samples": batch_summary[:2],
                        }
                        sample_log.update(
                            {
                                key: float(value)
                                for key, value in aux_metrics.items()
                                if key.startswith("reasoning_")
                            }
                        )
                        history.append(sample_log)
                        print(json.dumps(sample_log, ensure_ascii=False))

                    is_epoch_final_micro = (
                        wave_index >= len(global_wave_plan)
                        and wave_micro_index >= global_wave_micro_batches
                    )
                    if (
                        args.checkpoint_out
                        and int(args.checkpoint_every_steps) > 0
                        and should_step
                        and not is_epoch_final_micro
                        and global_step % int(args.checkpoint_every_steps) == 0
                    ):
                        local_rows_total_per_rank = [
                            int(value) for value in gather_objects(local_rows_total, runtime)
                        ]
                        epoch_accumulators_per_rank = gather_objects(
                            {
                                "loss_total": float(epoch_loss_total),
                                "lm_total": float(epoch_lm_total),
                                "action_total": float(epoch_action_total),
                                "local_rows": int(epoch_local_rows),
                            },
                            runtime,
                        )
                        if wave_micro_index >= global_wave_micro_batches:
                            next_wave_index = int(wave_index + 1)
                            next_wave_micro_index = 1
                        else:
                            next_wave_index = int(wave_index)
                            next_wave_micro_index = int(wave_micro_index + 1)
                        if runtime.enabled:
                            import torch.distributed as dist

                            dist.barrier()
                        if runtime.is_main:
                            checkpoint_path = save_training_checkpoint(
                                checkpoint_path=args.checkpoint_out,
                                agent_model=agent_model,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                epoch=epoch,
                                global_step=global_step,
                                args=args,
                                extra_state={
                                    "ddp_active_batch_training": True,
                                    "ddp_world_size": int(runtime.world_size),
                                    "next_epoch": int(epoch),
                                    "next_wave_index": next_wave_index,
                                    "next_wave_micro_index": next_wave_micro_index,
                                    "epoch_global_micro_index": int(epoch_global_micro_index),
                                    "accum_micro_batches": int(accum_micro_batches),
                                    "optimizer_steps": int(optimizer_steps),
                                    "local_rows_total": int(local_rows_total),
                                    "local_rows_total_per_rank": local_rows_total_per_rank,
                                    "epoch_accumulators_per_rank": epoch_accumulators_per_rank,
                                    "processed_rows_global": int(processed_rows_global or 0),
                                    "best_monitor_value": float(best_monitor_value),
                                    "best_epoch": int(best_epoch),
                                    "early_stop_bad_epochs": int(early_stop_bad_epochs),
                                    "history": history,
                                },
                            )
                            print(
                                json.dumps(
                                    {
                                        "stage": "save_ddp_checkpoint",
                                        "checkpoint_path": str(checkpoint_path),
                                        "global_step": int(global_step),
                                        "next_wave_index": next_wave_index,
                                        "next_wave_micro_index": next_wave_micro_index,
                                    },
                                    ensure_ascii=False,
                                )
                            )
                        if runtime.enabled:
                            import torch.distributed as dist

                            dist.barrier()

            if progress is not None:
                progress.close()

            epoch_global_rows = max(1.0, sum_across_ranks(epoch_local_rows, runtime))
            avg_loss = sum_across_ranks(epoch_loss_total, runtime) / epoch_global_rows
            avg_lm_loss = sum_across_ranks(epoch_lm_total, runtime) / epoch_global_rows
            avg_action_head_loss = sum_across_ranks(epoch_action_total, runtime) / epoch_global_rows
            monitor_values = {
                "loss": avg_loss,
                "lm_loss": avg_lm_loss,
                "action_head_loss": avg_action_head_loss,
            }
            monitor_value = float(monitor_values[args.early_stop_monitor])
            improved = monitor_value < best_monitor_value - float(args.early_stop_min_delta)
            if improved:
                best_monitor_value = monitor_value
                best_epoch = int(epoch)
                early_stop_bad_epochs = 0
            else:
                early_stop_bad_epochs += 1
            epoch_report = {
                "stage": "train_epoch",
                "distributed": True,
                "epoch": int(epoch),
                "avg_loss": avg_loss,
                "avg_lm_loss": avg_lm_loss,
                "avg_action_head_loss": avg_action_head_loss,
                "processed_rows_global": int(epoch_global_rows),
                "optimizer_steps": int(optimizer_steps),
                "elapsed_seconds": time.time() - started,
                "early_stop_monitor": str(args.early_stop_monitor),
                "monitor_value": monitor_value,
                "best_monitor_value": float(best_monitor_value),
                "best_epoch": int(best_epoch),
                "early_stop_bad_epochs": int(early_stop_bad_epochs),
                "improved": bool(improved),
            }
            if runtime.is_main:
                history.append(epoch_report)
                print(json.dumps(epoch_report, ensure_ascii=False))

            # Compute this on every rank before any rank-0-only save. The same
            # value is also needed when only --best-checkpoint-out is set.
            processed_rows_global_total = int(sum_across_ranks(local_rows_total, runtime))
            local_rows_total_per_rank = [int(value) for value in gather_objects(local_rows_total, runtime)]
            if args.checkpoint_out:
                if runtime.enabled:
                    import torch.distributed as dist

                    dist.barrier()
                if runtime.is_main:
                    save_training_checkpoint(
                        checkpoint_path=args.checkpoint_out,
                        agent_model=agent_model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        global_step=global_step,
                        args=args,
                        extra_state={
                            "ddp_active_batch_training": True,
                            "ddp_world_size": int(runtime.world_size),
                            "next_epoch": int(epoch + 1),
                            "next_wave_index": 1,
                            "next_wave_micro_index": 1,
                            "epoch_global_micro_index": 0,
                            "accum_micro_batches": 0,
                            "optimizer_steps": int(optimizer_steps),
                            "local_rows_total": int(local_rows_total),
                            "local_rows_total_per_rank": local_rows_total_per_rank,
                            "processed_rows_global": int(processed_rows_global_total),
                            "epoch_complete": True,
                            "best_monitor_value": float(best_monitor_value),
                            "best_epoch": int(best_epoch),
                            "early_stop_bad_epochs": int(early_stop_bad_epochs),
                            "history": history,
                        },
                    )
                if runtime.enabled:
                    import torch.distributed as dist

                    dist.barrier()

            if improved and args.best_checkpoint_out:
                if runtime.enabled:
                    import torch.distributed as dist

                    dist.barrier()
                if runtime.is_main:
                    save_training_checkpoint(
                        checkpoint_path=args.best_checkpoint_out,
                        agent_model=agent_model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        global_step=global_step,
                        args=args,
                        extra_state={
                            "ddp_active_batch_training": True,
                            "ddp_world_size": int(runtime.world_size),
                            "next_epoch": int(epoch + 1),
                            "next_wave_index": 1,
                            "next_wave_micro_index": 1,
                            "epoch_global_micro_index": 0,
                            "accum_micro_batches": 0,
                            "optimizer_steps": int(optimizer_steps),
                            "local_rows_total": int(local_rows_total),
                            "local_rows_total_per_rank": local_rows_total_per_rank,
                            "processed_rows_global": int(processed_rows_global_total),
                            "epoch_complete": True,
                            "best_checkpoint": True,
                            "best_monitor_value": float(best_monitor_value),
                            "best_epoch": int(best_epoch),
                            "early_stop_bad_epochs": int(early_stop_bad_epochs),
                            "history": history,
                        },
                    )
                if runtime.enabled:
                    import torch.distributed as dist

                    dist.barrier()

            if (
                int(args.early_stop_patience) > 0
                and int(epoch) >= int(args.early_stop_min_epochs)
                and int(early_stop_bad_epochs) >= int(args.early_stop_patience)
            ):
                stopped_early = True
                if runtime.is_main:
                    print(
                        json.dumps(
                            {
                                "stage": "early_stop",
                                "distributed": True,
                                "epoch": int(epoch),
                                "monitor": str(args.early_stop_monitor),
                                "best_epoch": int(best_epoch),
                                "best_monitor_value": float(best_monitor_value),
                                "bad_epochs": int(early_stop_bad_epochs),
                            },
                            ensure_ascii=False,
                        )
                    )
                break

        processed_rows_global_total = int(sum_across_ranks(local_rows_total, runtime))
        extra_metadata = {
            "latent_slot_count": int(args.latent_slot_count),
            "reasoning_alignment_mode": str(args.reasoning_alignment_mode),
            "reasoning_field_slot_counts": list(args.resolved_reasoning_field_slot_counts),
            "history_n": int(args.history_n),
            "active_batch_training": True,
            "ddp_active_batch_training": bool(runtime.enabled),
            "ddp_world_size": int(runtime.world_size),
            "batch_size_per_rank": int(args.batch_size),
            "effective_max_batch": int(args.batch_size) * int(runtime.world_size),
            "grad_accum_steps": int(args.grad_accum_steps),
            "pixel_pruned_visual": True,
            "pixel_prune_threshold": float(args.pixel_prune_threshold),
            "pixel_prune_predictor_order": str(args.pixel_prune_predictor_order),
            "pixel_temporal_reuse": bool(args.pixel_temporal_reuse),
            "pixel_temporal_threshold": float(args.pixel_temporal_threshold),
            "image_min_pixels": int(args.image_min_pixels),
            "image_max_pixels": int(args.image_max_pixels),
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
            "two_way_candidate_coord_loss_weight": float(
                args.two_way_candidate_coord_loss_weight
            ),
            "two_way_candidate_confidence_loss_weight": float(
                args.two_way_candidate_confidence_loss_weight
            ),
            "flow_continuous_source": str(args.flow_continuous_source),
            "action_head_loss_weight": float(args.action_head_loss_weight),
            "flow_action_loss_weight": float(args.flow_action_loss_weight),
            "flow_coord_loss_weight": float(args.flow_coord_loss_weight),
            "learnable_flow_coord_weight": bool(args.learnable_flow_coord_weight),
            "learned_flow_coord_weight": float(torch.exp(-agent_model.flow_coord_loss_log_var.detach()).cpu().item()),
            "flow_coord_loss_log_var": float(agent_model.flow_coord_loss_log_var.detach().cpu().item()),
            "flow_coord_loss_scale": float(args.flow_coord_loss_scale),
            "flow_coord_loss_space": str(args.flow_coord_loss_space),
            "flow_patch_loss_weight": float(args.flow_patch_loss_weight),
            "flow_patch_loss_mode": str(args.flow_patch_loss_mode),
            "flow_patch_gaussian_sigma": float(args.flow_patch_gaussian_sigma),
            "flow_pointer_coord_source": str(args.flow_pointer_coord_source),
            "flow_patch_logit_temperature": float(args.flow_patch_logit_temperature),
            "flow_patch_residual_scale": float(args.flow_patch_residual_scale),
            "action_hidden_source": str(args.action_hidden_source),
            "train_action_head_only": bool(args.train_action_head_only),
            "clean_observable_prompt": bool(args.clean_observable_prompt),
            "include_current_subtask_in_prompt": not bool(args.clean_observable_prompt),
            "include_expected_next_screen_in_prompt": not bool(args.clean_observable_prompt),
            "latent_scaffolds_in_prompt": not bool(args.clean_observable_prompt),
            "use_lora": bool(args.use_lora),
            "lora_r": int(args.lora_r),
            "lora_alpha": int(args.lora_alpha),
            "lora_dropout": float(args.lora_dropout),
            "checkpoint_out": str(args.checkpoint_out) if args.checkpoint_out else None,
            "best_checkpoint_out": str(args.best_checkpoint_out) if args.best_checkpoint_out else None,
            "best_monitor_value": float(best_monitor_value),
            "best_epoch": int(best_epoch),
            "early_stop_bad_epochs": int(early_stop_bad_epochs),
            "early_stop_monitor": str(args.early_stop_monitor),
            "early_stop_patience": int(args.early_stop_patience),
            "early_stop_min_delta": float(args.early_stop_min_delta),
            "early_stop_min_epochs": int(args.early_stop_min_epochs),
            "stopped_early": bool(stopped_early),
            "resumed_from_checkpoint": resumed_from_checkpoint,
        }
        if runtime.is_main:
            agent_model.save_adapter(args.adapter_out, extra_metadata=extra_metadata)
            report = {
                "adapter_path": str(args.adapter_out),
                "elapsed_seconds": time.time() - started,
                "sample_count": int(global_row_count),
                "trajectory_count": int(len(trajectories)),
                "epochs": int(args.epochs),
                "processed_rows_global": int(processed_rows_global_total),
                "optimizer_steps": int(optimizer_steps),
                "history": history,
                **extra_metadata,
            }
            if args.report_out:
                Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
                Path(args.report_out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps(report, ensure_ascii=False, indent=2))
        completed_without_error = True
    finally:
        cleanup_distributed(runtime, skip_destroy=not completed_without_error)


if __name__ == "__main__":
    main()
