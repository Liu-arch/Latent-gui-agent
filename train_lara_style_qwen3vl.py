from __future__ import annotations

import argparse
import json
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
from qwen3_gui_agent.rl.agentnet_adapter import _parse_action_code
from qwen3_gui_agent.training_checkpoint import (
    force_safe_adamw_runtime_flags,
    load_training_checkpoint,
    save_training_checkpoint,
)
from qwen3_gui_agent.rl.schema import read_jsonl
from qwen3_gui_agent.training_utils import (
    _build_loss_artifact_paths,
    _write_loss_artifacts,
    build_scheduler,
    current_lr,
    infer_trajectory_key,
    resolve_dataset_image,
    resolve_device_map,
    resolve_torch_dtype,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train LaRA-style latent reasoning Qwen3-VL with official generation and pixel prune/reuse."
    )
    parser.add_argument("--steps", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter-out", required=True)
    parser.add_argument("--report-out", default=None)
    parser.add_argument("--checkpoint-out", default=None)
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
    parser.add_argument("--history-n", type=int, default=5)
    parser.add_argument("--latent-slot-count", type=int, default=16)
    parser.add_argument(
        "--reasoning-alignment-mode",
        choices=["aggregate", "field_aligned"],
        default="aggregate",
    )
    parser.add_argument(
        "--reasoning-field-slot-counts",
        default="auto",
        help="Three comma-separated slot counts for actual_task,thought,reflection.",
    )
    parser.add_argument("--pixel-prune-threshold", type=float, default=0.0)
    parser.add_argument("--pixel-prune-predictor-order", default="pred2d,left,up")
    parser.add_argument("--pixel-temporal-reuse", action="store_true")
    parser.add_argument("--pixel-temporal-threshold", type=float, default=0.0)
    parser.add_argument(
        "--image-min-pixels",
        type=int,
        default=0,
        help="Optional Qwen3-VL processor min_pixels override. 0 keeps the model default.",
    )
    parser.add_argument(
        "--image-max-pixels",
        type=int,
        default=0,
        help="Optional Qwen3-VL processor max_pixels override. 0 keeps the model default.",
    )
    parser.add_argument("--training-stage", choices=["stage1", "stage2"], default="stage1")
    parser.add_argument(
        "--action-format",
        choices=["text", "action_tokens"],
        default="text",
        help="Supervise actions as legacy text fields or compact GUI action tokens.",
    )
    parser.add_argument(
        "--action-model",
        choices=["unified", "flow_matching"],
        default="unified",
        help="Action expert used when --action-head-loss-weight > 0. flow_matching is the LaRA-VLA-style GUI action model.",
    )
    parser.add_argument("--flow-action-sample-steps", type=int, default=8)
    parser.add_argument(
        "--lm-action-target",
        choices=["auto", "include", "omit"],
        default="auto",
        help="Whether LM targets include the action text. auto omits action text when an action expert loss is enabled.",
    )
    parser.add_argument(
        "--stage2-target-format",
        choices=["mixed_reasoning_action", "action_only"],
        default="mixed_reasoning_action",
    )
    parser.add_argument("--stage1-max-reasoning-chars", type=int, default=0)
    parser.add_argument("--stage2-explicit-keep-start", type=float, default=0.7)
    parser.add_argument("--stage2-explicit-keep-end", type=float, default=0.0)
    parser.add_argument("--stage2-min-explicit-tokens", type=int, default=0)
    parser.add_argument("--stage2-max-thinking-tokens", type=int, default=4)
    parser.add_argument("--lm-loss-weight", type=float, default=1.0)
    parser.add_argument("--reasoning-align-weight", type=float, default=1.0)
    parser.add_argument("--future-frame-loss-weight", type=float, default=1.0)
    parser.add_argument("--latent-diversity-weight", type=float, default=0.02)
    parser.add_argument(
        "--action-head-loss-weight",
        type=float,
        default=0.0,
        help="Optional hidden-state action head loss. Default 0 keeps the original text-generation path unchanged.",
    )
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
    parser.add_argument("--train-backbone", action="store_true")
    parser.add_argument("--train-embeddings", action="store_true")
    parser.add_argument("--shuffle-trajectories", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--prep-log-every", type=int, default=10)
    parser.add_argument(
        "--profile-stages",
        action="store_true",
        help="Log per-step timing for data prep, forward, auxiliary/action loss, backward, optimizer, and logging.",
    )
    parser.add_argument(
        "--log-full-debug",
        action="store_true",
        help="Keep long debug strings such as assistant_target in stdout/report/loss logs. Off by default.",
    )
    parser.add_argument(
        "--minimal-logging",
        action="store_true",
        help="Disable per-step JSON history/loss artifacts. Keep only concise stdout, checkpoints, and final report.",
    )
    parser.add_argument(
        "--no-progress-bar",
        action="store_true",
        help="Disable tqdm progress bar and print regular log lines only.",
    )
    parser.add_argument(
        "--log-gt",
        action="store_true",
        help="Print the supervised LM target and gold action for each logged training step.",
    )
    parser.add_argument(
        "--log-gt-max-chars",
        type=int,
        default=1200,
        help="Maximum characters to print for long GT text fields when --log-gt is enabled.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-every-steps", type=int, default=0)
    parser.add_argument("--loss-out-dir", default="E:/lara/results/loss")
    parser.add_argument("--loss-plot-every", type=int, default=10)
    args = parser.parse_args()
    args.max_samples = max(1, int(args.max_samples))
    args.history_n = max(1, int(args.history_n))
    args.log_every = max(1, int(args.log_every))
    args.prep_log_every = max(1, int(args.prep_log_every))
    args.loss_plot_every = max(1, int(args.loss_plot_every))
    args.checkpoint_every_steps = max(0, int(args.checkpoint_every_steps))
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
            "Field-aligned Stage 2 requires --stage2-max-thinking-tokens to equal --latent-slot-count."
        )
    args.flow_action_sample_steps = max(1, int(args.flow_action_sample_steps))
    args.image_min_pixels = max(0, int(args.image_min_pixels))
    args.image_max_pixels = max(0, int(args.image_max_pixels))
    args.flow_action_loss_weight = max(0.0, float(args.flow_action_loss_weight))
    args.flow_coord_loss_weight = max(0.0, float(args.flow_coord_loss_weight))
    args.flow_coord_loss_scale = max(1.0, float(args.flow_coord_loss_scale))
    args.flow_coord_loss_space = str(args.flow_coord_loss_space)
    args.flow_patch_loss_weight = max(0.0, float(args.flow_patch_loss_weight))
    if args.lm_action_target == "auto":
        args.resolved_lm_action_target = "omit" if float(args.action_head_loss_weight) > 0.0 else "include"
    else:
        args.resolved_lm_action_target = str(args.lm_action_target)
    args.flow_continuous_source = "direct" if args.flow_coord_loss_weight > 0.0 else "sample"
    return args


def compact_debug_info(debug_info: dict[str, Any] | None, *, keep_full_debug: bool = False) -> dict[str, Any] | None:
    if not debug_info:
        return debug_info
    if keep_full_debug:
        return debug_info
    compact = dict(debug_info)
    for key in (
        "assistant_target",
        "assistant_target_preview",
        "action_text",
        "raw_response_text",
        "messages",
        "prompt_text",
    ):
        compact.pop(key, None)
    if "assistant_target_char_count" in compact:
        compact["assistant_target_logged"] = False
    return compact


def truncate_for_log(value: Any, max_chars: int) -> Any:
    if max_chars <= 0:
        return value
    if isinstance(value, str) and len(value) > max_chars:
        return value[:max_chars] + f"...<truncated {len(value) - max_chars} chars>"
    if isinstance(value, list):
        return [truncate_for_log(item, max_chars) for item in value]
    if isinstance(value, dict):
        return {key: truncate_for_log(item, max_chars) for key, item in value.items()}
    return value


def build_trajectories(
    *,
    steps: list[dict[str, Any]],
    dataset_root: Path,
    max_samples: int,
    prep_log_every: int,
) -> list[dict[str, Any]]:
    total_rows = 0
    trajectories: list[dict[str, Any]] = []
    current_rows: list[dict[str, Any]] = []
    current_key: str | None = None
    for step in steps:
        image_name = str(step.get("before_screenshot", "") or "").strip()
        if not image_name:
            continue
        image_path = resolve_dataset_image(dataset_root, image_name)
        if image_path is None:
            continue
        after_name = str(step.get("after_screenshot", "") or "").strip()
        after_path = resolve_dataset_image(dataset_root, after_name) if after_name else image_path
        trajectory_key = infer_trajectory_key(step)
        if current_key is None:
            current_key = trajectory_key
        elif current_key != trajectory_key:
            if current_rows:
                trajectories.append({"trajectory_key": current_key, "task": current_rows[0]["task"], "rows": current_rows})
            current_rows = []
            current_key = trajectory_key

        instruction = str(step.get("instruction", step.get("task", "")) or "").strip()
        actual_task = str(step.get("actual_task", step.get("current_subtask", "")) or "").strip()
        thought = str(step.get("thought", "") or "").strip()
        reflection = str(
            step.get("reflection", step.get("expected_next_screen", step.get("predicted_next_screen_desc", ""))) or ""
        ).strip()
        code = str(step.get("code", "") or "").strip()
        img_next = step.get("img_next", [])
        explicit_supervision = str(step.get("explicit_supervision", "") or "").strip()
        gold_action = normalize_gold_action(extract_step_action(step))
        if not explicit_supervision:
            explicit_supervision = build_explicit_supervision_text(
                instruction=instruction,
                actual_task=actual_task,
                bbox=step.get("bbox"),
                thought=thought,
                reflection=reflection,
                img_next=img_next,
            )
        row = {
            "sample_id": str(step.get("sample_id", f"{trajectory_key}_step_{len(current_rows):04d}")),
            "trajectory_key": trajectory_key,
            "task": instruction,
            "instruction": instruction,
            "actual_task": actual_task,
            "thought": thought,
            "reflection": reflection,
            "code": code,
            "img_next": img_next,
            "bbox": step.get("bbox"),
            "explicit_supervision": explicit_supervision,
            "image_path": str(image_path),
            "after_image_path": str(after_path or image_path),
            "gold_action": gold_action,
            "raw_keys": sorted(str(key) for key in step.keys()),
        }
        current_rows.append(row)
        total_rows += 1
        if total_rows % prep_log_every == 0:
            print(
                json.dumps(
                    {
                        "stage": "prepare_lara_style_trajectories",
                        "prepared_samples": total_rows,
                        "current_trajectory_key": trajectory_key,
                    },
                    ensure_ascii=False,
                )
            )
        if total_rows >= max_samples:
            break
    if current_rows:
        trajectories.append({"trajectory_key": current_key, "task": current_rows[0]["task"], "rows": current_rows})
    return trajectories


def extract_step_action(step: dict[str, Any]) -> dict[str, Any]:
    parsed_action = step.get("parsed_action")
    if isinstance(parsed_action, dict) and parsed_action:
        return parsed_action
    gold_action = step.get("gold_action")
    if isinstance(gold_action, dict) and gold_action:
        return gold_action
    action = step.get("action")
    if isinstance(action, dict) and action:
        return action
    code = str(step.get("code", "") or "").strip()
    if code:
        parsed = _parse_action_code(code)
        if isinstance(parsed, dict) and parsed:
            return parsed
    bbox = step.get("bbox")
    if isinstance(bbox, list) and len(bbox) == 4:
        try:
            return {
                "type": "click",
                "x_norm": round(float(bbox[0]), 4),
                "y_norm": round(float(bbox[1]), 4),
            }
        except (TypeError, ValueError):
            pass
    return {"type": "wait", "status": "success"}


def build_explicit_supervision_text(
    *,
    instruction: str,
    actual_task: str,
    bbox: Any,
    thought: str,
    reflection: str,
    img_next: Any,
) -> str:
    lines = [
        f"instruction: {instruction.strip()}",
        f"actual_task: {actual_task.strip()}",
    ]
    if isinstance(bbox, list) and len(bbox) == 4:
        try:
            lines.append(
                f"bbox: [{float(bbox[0]):.4f} {float(bbox[1]):.4f} {float(bbox[2]):.4f} {float(bbox[3]):.4f}]"
            )
        except (TypeError, ValueError):
            lines.append("bbox: []")
    else:
        lines.append("bbox: []")
    lines.append(f"thought: {thought.strip()}")
    lines.append(f"reflection: {reflection.strip()}")
    if isinstance(img_next, list) and img_next:
        lines.append(" ".join(str(token) for token in img_next))
    elif isinstance(img_next, str) and img_next.strip():
        lines.append(img_next.strip())
    return "\n".join(lines).strip()


def normalize_gold_action(action: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {
        "type": str(action.get("type", "wait")),
        "x_norm": None,
        "y_norm": None,
        "text": None,
        "keys": None,
        "amount": None,
        "status": None,
    }
    if action.get("x_norm") is not None:
        normalized["x_norm"] = round(float(action["x_norm"]), 4)
    if action.get("y_norm") is not None:
        normalized["y_norm"] = round(float(action["y_norm"]), 4)
    if action.get("text") is not None:
        normalized["text"] = str(action["text"])
    if action.get("keys") is not None:
        normalized["keys"] = list(action["keys"])
    if action.get("amount") is not None:
        normalized["amount"] = int(action["amount"])
    if action.get("status") is not None:
        normalized["status"] = str(action["status"])
    return normalized


def configure_trainable_params(agent: LaRAStyleQwen3VLAgent, train_backbone: bool, train_embeddings: bool) -> None:
    if train_backbone:
        for parameter in agent.parameters():
            parameter.requires_grad = True
        return
    for name, parameter in agent.model.named_parameters():
        # Keep injected LoRA adapters trainable while freezing the base VLM.
        parameter.requires_grad = "lora_" in name
    # The agent overrides only the newly added latent/action token rows through
    # a small dedicated parameter. Never unfreeze Qwen's full vocabulary table.
    if hasattr(agent, "special_token_embeddings"):
        agent.special_token_embeddings.requires_grad = bool(train_embeddings)
    for module in [
        agent.reasoning_proj,
        agent.future_frame_head,
        agent.reasoning_norm,
        agent.action_state_norm,
        agent.action_head,
        agent.flow_action_head,
    ]:
        for parameter in module.parameters():
            parameter.requires_grad = True


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dtype = resolve_torch_dtype(args.torch_dtype)
    device_map = resolve_device_map(args.device_map)
    steps = read_jsonl(Path(args.steps))
    trajectories = build_trajectories(
        steps=steps,
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
        image_min_pixels=args.image_min_pixels or None,
        image_max_pixels=args.image_max_pixels or None,
        reasoning_alignment_mode=args.reasoning_alignment_mode,
        reasoning_field_slot_counts=args.resolved_reasoning_field_slot_counts,
    )
    if args.init_adapter:
        init_result = agent.load_adapter(args.init_adapter, strict=False)
        agent.set_reasoning_alignment_config(
            mode=args.reasoning_alignment_mode,
            field_slot_counts=args.resolved_reasoning_field_slot_counts,
        )
        agent.action_model = str(args.action_model)
        agent.flow_action_sample_steps = int(args.flow_action_sample_steps)
        agent.action_format = str(args.action_format)
        agent.lm_action_target = str(args.resolved_lm_action_target)
        agent.image_min_pixels = args.image_min_pixels or None
        agent.image_max_pixels = args.image_max_pixels or None
        agent._configure_processor_pixel_budget(
            agent.processor,
            image_min_pixels=agent.image_min_pixels,
            image_max_pixels=agent.image_max_pixels,
        )
        print(
            json.dumps(
                {
                    "stage": "init_adapter",
                    "init_adapter": args.init_adapter,
                    "missing_keys": list(init_result.get("missing_keys", [])),
                    "unexpected_keys": list(init_result.get("unexpected_keys", [])),
                },
                ensure_ascii=False,
            )
        )
    agent.flow_continuous_source = str(args.flow_continuous_source)
    agent.flow_action_loss_weight = float(args.flow_action_loss_weight)
    agent.flow_coord_loss_weight = float(args.flow_coord_loss_weight)
    agent.flow_coord_loss_scale = float(args.flow_coord_loss_scale)
    agent.flow_coord_loss_space = str(args.flow_coord_loss_space)
    agent.flow_patch_loss_weight = float(args.flow_patch_loss_weight)
    if args.train_backbone:
        raise NotImplementedError(
            "--train-backbone is not supported yet for adapter save/load on this LaRA-style line. "
            "Keep backbone frozen for now."
        )
    configure_trainable_params(agent, args.train_backbone, args.train_embeddings)
    agent.train()

    trainable_params = [parameter for parameter in agent.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, foreach=False)
    force_safe_adamw_runtime_flags(optimizer)
    total_rows = sum(len(trajectory["rows"]) for trajectory in trajectories)
    total_steps_estimate = max(1, total_rows * max(1, args.epochs))
    scheduler = build_scheduler(
        optimizer=optimizer,
        scheduler_name=args.lr_scheduler,
        total_steps=total_steps_estimate,
        warmup_ratio=args.warmup_ratio,
        min_lr_ratio=args.min_lr_ratio,
    )

    start_epoch = 1
    global_step = 0
    if args.resume_from:
        payload = load_training_checkpoint(
            checkpoint_path=args.resume_from,
            agent_model=agent,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        force_safe_adamw_runtime_flags(optimizer)
        start_epoch = int(payload.get("epoch", 0)) + 1
        global_step = int(payload.get("global_step", 0))
        print(
            json.dumps(
                {
                    "stage": "resume_checkpoint",
                    "resume_from": args.resume_from,
                    "resume_epoch": start_epoch,
                    "resume_global_step": global_step,
                },
                ensure_ascii=False,
            )
        )

    loss_artifact_paths = None
    if not args.minimal_logging:
        loss_artifact_paths = _build_loss_artifact_paths(args.loss_out_dir, args.adapter_out)
        loss_artifact_paths["root"].mkdir(parents=True, exist_ok=True)
    loss_rows: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    started = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_trajectories = list(trajectories)
        if args.shuffle_trajectories:
            random.Random(args.seed + epoch).shuffle(epoch_trajectories)
        epoch_loss_total = 0.0
        epoch_lm_total = 0.0
        epoch_reason_total = 0.0
        epoch_future_total = 0.0
        epoch_action_head_total = 0.0
        epoch_rows = 0
        progress = (
            tqdm(total=total_rows, desc=f"train_lara_style_epoch{epoch}", dynamic_ncols=True, mininterval=2.0)
            if tqdm is not None and not args.no_progress_bar
            else None
        )

        for trajectory_index, trajectory in enumerate(epoch_trajectories, start=1):
            history_image_paths: list[str] = []
            history_slots = max(0, int(args.history_n) - 1)
            trajectory_key = str(trajectory["trajectory_key"])
            for row in trajectory["rows"]:
                step_started = time.perf_counter()
                image_paths = list(history_image_paths[-history_slots:]) + [row["image_path"]]
                explicit_reasoning = str(row["explicit_supervision"])
                if args.training_stage == "stage1" and args.stage1_max_reasoning_chars > 0:
                    explicit_reasoning = explicit_reasoning[: int(args.stage1_max_reasoning_chars)].strip()
                if args.training_stage == "stage2":
                    progress_ratio = min(1.0, max(0.0, float(global_step) / float(max(1, total_steps_estimate - 1))))
                    current_stage2_keep_ratio = (
                        float(args.stage2_explicit_keep_start)
                        + (float(args.stage2_explicit_keep_end) - float(args.stage2_explicit_keep_start)) * progress_ratio
                    )
                else:
                    current_stage2_keep_ratio = 1.0
                prepared_at = time.perf_counter()
                output = agent.forward_train(
                    image_paths=image_paths,
                    task=row["instruction"],
                    history_frame_count=len(image_paths) - 1,
                    current_subtask=row["actual_task"],
                    expected_next_screen=None,
                    explicit_reasoning=explicit_reasoning,
                    gold_action=row["gold_action"],
                    next_image_path=row["after_image_path"],
                    training_stage=args.training_stage,
                    stage2_target_format=args.stage2_target_format,
                    stage2_explicit_keep_ratio=current_stage2_keep_ratio,
                    stage2_min_explicit_tokens=args.stage2_min_explicit_tokens,
                    stage2_max_thinking_tokens=args.stage2_max_thinking_tokens,
                    future_frame_enabled=args.future_frame_loss_weight > 0.0,
                    temporal_sample_key=trajectory_key,
                    action_format=args.action_format,
                    include_action_in_lm=(args.resolved_lm_action_target == "include"),
                )
                forward_done_at = time.perf_counter()
                aux_losses, aux_metrics = agent.compute_auxiliary_losses(
                    output,
                    training_stage=args.training_stage,
                )
                action_head_losses, action_head_metrics = agent.compute_action_head_losses(output)
                if "flow_action_loss" in action_head_losses:
                    action_head_losses["flow_action_loss"] = (
                        args.flow_action_loss_weight * action_head_losses["flow_action_loss"]
                    )
                if "flow_action_coord_loss" in action_head_losses:
                    action_head_losses["flow_action_coord_loss"] = (
                        args.flow_coord_loss_weight * action_head_losses["flow_action_coord_loss"]
                    )
                if "flow_action_patch_loss" in action_head_losses:
                    action_head_losses["flow_action_patch_loss"] = (
                        args.flow_patch_loss_weight * action_head_losses["flow_action_patch_loss"]
                    )
                loss_done_at = time.perf_counter()
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

                optimizer.zero_grad(set_to_none=True)
                backward_started_at = time.perf_counter()
                total_loss.backward()
                backward_done_at = time.perf_counter()
                optimizer.step()
                scheduler.step()
                optimizer_done_at = time.perf_counter()

                global_step += 1
                epoch_rows += 1
                epoch_loss_total += float(total_loss.detach().item())
                epoch_lm_total += float(lm_loss.detach().item())
                epoch_reason_total += float(reasoning_loss.detach().item())
                epoch_future_total += float(future_loss.detach().item())
                epoch_action_head_total += float(action_head_loss.detach().item())

                sample_log = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "loss": float(total_loss.detach().item()),
                    "action_loss": float(lm_loss.detach().item()),
                    "lm_loss": float(lm_loss.detach().item()),
                    "reasoning_alignment_loss": float(reasoning_loss.detach().item()),
                    "future_frame_loss": float(future_loss.detach().item()),
                    "latent_diversity_loss": float(diversity_loss.detach().item()),
                    "action_head_loss": float(action_head_loss.detach().item()),
                    "reasoning_cosine_similarity": float(aux_metrics.get("reasoning_cosine_similarity", 0.0)),
                    "future_frame_cosine_similarity": float(aux_metrics.get("future_frame_cosine_similarity", 0.0)),
                    "action_head_type_accuracy": float(action_head_metrics.get("action_head_type_accuracy", 0.0)),
                    "action_head_region_accuracy": float(action_head_metrics.get("action_head_region_accuracy", 0.0)),
                    "action_head_pointer_l1": float(action_head_metrics.get("action_head_pointer_l1", 0.0)),
                    "action_head_sampled_region_accuracy": float(
                        action_head_metrics.get("action_head_sampled_region_accuracy", 0.0)
                    ),
                    "action_head_coord_loss": float(action_head_metrics.get("action_head_coord_loss", 0.0)),
                    "action_head_teacher_pointer_l1": float(
                        action_head_metrics.get("action_head_teacher_pointer_l1", 0.0)
                    ),
                    "action_head_sampled_pointer_l1": float(
                        action_head_metrics.get("action_head_sampled_pointer_l1", 0.0)
                    ),
                    "action_head_coord_preview": action_head_metrics.get("action_head_coord_preview", []),
                    "lr": current_lr(optimizer),
                    "history_frame_count": len(image_paths) - 1,
                    "trajectory_index": trajectory_index,
                    "training_stage": args.training_stage,
                    "stage2_target_format": args.stage2_target_format,
                    "action_format": args.action_format,
                    "explicit_reasoning_chars": len(explicit_reasoning),
                    "lm_action_target": str(args.resolved_lm_action_target),
                    "stage2_explicit_keep_ratio": float(current_stage2_keep_ratio),
                    "debug_info": compact_debug_info(output.debug_info, keep_full_debug=args.log_full_debug),
                }
                sample_log.update(
                    {
                        key: float(value)
                        for key, value in aux_metrics.items()
                        if key.startswith("reasoning_")
                    }
                )
                if args.profile_stages:
                    visual_debug = {}
                    if output.debug_info and isinstance(output.debug_info.get("visual_debug"), dict):
                        visual_debug = output.debug_info["visual_debug"]
                    sample_log["timing"] = {
                        "data_prep_seconds": prepared_at - step_started,
                        "forward_seconds": forward_done_at - prepared_at,
                        "loss_seconds": loss_done_at - forward_done_at,
                        "backward_seconds": backward_done_at - backward_started_at,
                        "optimizer_seconds": optimizer_done_at - backward_done_at,
                        "step_seconds": optimizer_done_at - step_started,
                        "visual_pixel_prune_plan_seconds": float(visual_debug.get("pixel_prune_plan_seconds", 0.0)),
                        "visual_pixel_prune_reconstruct_seconds": float(
                            visual_debug.get("pixel_prune_reconstruct_seconds", 0.0)
                        ),
                    }

                if global_step % args.log_every == 0:
                    log_started_at = time.perf_counter()
                    if args.minimal_logging:
                        minimal_log = {
                            "epoch": epoch,
                            "global_step": global_step,
                            "loss": float(total_loss.detach().item()),
                            "lm_loss": float(lm_loss.detach().item()),
                            "action_head_loss": float(action_head_loss.detach().item()),
                            "lr": current_lr(optimizer),
                            "history_frame_count": len(image_paths) - 1,
                            "image_count": len(image_paths),
                            "trajectory_index": trajectory_index,
                            "current_image": Path(row["image_path"]).name,
                            "action_head_coord_loss": float(action_head_metrics.get("action_head_coord_loss", 0.0)),
                            "action_head_teacher_pointer_l1": float(
                                action_head_metrics.get("action_head_teacher_pointer_l1", 0.0)
                            ),
                            "action_head_sampled_pointer_l1": float(
                                action_head_metrics.get("action_head_sampled_pointer_l1", 0.0)
                            ),
                            "action_head_coord_preview": action_head_metrics.get("action_head_coord_preview", []),
                        }
                        if args.profile_stages:
                            timing = sample_log.get("timing", {})
                            minimal_log["timing"] = {
                                "fwd": float(timing.get("forward_seconds", 0.0)),
                                "step_t": float(timing.get("step_seconds", 0.0)),
                                "vrec": float(timing.get("visual_pixel_prune_reconstruct_seconds", 0.0)),
                            }
                        print(
                            json.dumps(
                                minimal_log,
                                ensure_ascii=False,
                            )
                        )
                    else:
                        history.append(sample_log)
                        loss_rows.append(sample_log)
                        print(json.dumps(sample_log, ensure_ascii=False))
                    if args.profile_stages:
                        sample_log["timing"]["log_seconds"] = time.perf_counter() - log_started_at
                    if args.log_gt:
                        raw_debug = output.debug_info or {}
                        gt_log = {
                            "stage": "train_gt",
                            "epoch": epoch,
                            "global_step": global_step,
                            "sample_id": row["sample_id"],
                            "trajectory_index": trajectory_index,
                            "trajectory_key": trajectory_key,
                            "history_frame_count": len(image_paths) - 1,
                            "image_paths": [Path(path).name for path in image_paths],
                            "current_image": Path(row["image_path"]).name,
                            "after_image": Path(row["after_image_path"]).name,
                            "instruction": row["instruction"],
                            "actual_task": row["actual_task"],
                            "bbox": row.get("bbox"),
                            "explicit_supervision": explicit_reasoning,
                            "gold_action": row["gold_action"],
                            "lm_action_target": str(args.resolved_lm_action_target),
                            "include_action_in_lm": bool(args.resolved_lm_action_target == "include"),
                            "assistant_lm_target": raw_debug.get("assistant_target"),
                            "assistant_target_char_count": raw_debug.get("assistant_target_char_count"),
                            "assistant_future_positions_count": raw_debug.get("assistant_future_positions_count"),
                            "assistant_latent_tokens": raw_debug.get("assistant_latent_tokens"),
                        }
                        print(json.dumps(truncate_for_log(gt_log, args.log_gt_max_chars), ensure_ascii=False))
                if (not args.minimal_logging) and loss_artifact_paths is not None and global_step % args.loss_plot_every == 0:
                    _write_loss_artifacts(artifact_paths=loss_artifact_paths, loss_rows=loss_rows)
                if args.checkpoint_out and args.checkpoint_every_steps > 0 and global_step % args.checkpoint_every_steps == 0:
                    save_training_checkpoint(
                        checkpoint_path=args.checkpoint_out,
                        agent_model=agent,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        global_step=global_step,
                        args=args,
                        extra_state={"training_stage": args.training_stage},
                    )

                history_image_paths.append(row["image_path"])
                if progress is not None:
                    timing = sample_log.get("timing", {}) if args.profile_stages else {}
                    postfix = {
                        "step": global_step,
                        "loss": f"{float(total_loss.detach().item()):.3f}",
                        "ah": f"{float(action_head_loss.detach().item()):.3f}",
                        "hist_imgs": len(image_paths) - 1,
                        "traj": trajectory_index,
                        "img": Path(row["image_path"]).name,
                        "ptr_l1": f"{float(action_head_metrics.get('action_head_teacher_pointer_l1', 0.0)):.3f}",
                        "samp_l1": f"{float(action_head_metrics.get('action_head_sampled_pointer_l1', 0.0)):.3f}",
                    }
                    if timing:
                        postfix.update(
                            {
                                "fwd": f"{float(timing.get('forward_seconds', 0.0)):.1f}s",
                                "step_t": f"{float(timing.get('step_seconds', 0.0)):.1f}s",
                                "vrec": f"{float(timing.get('visual_pixel_prune_reconstruct_seconds', 0.0)):.1f}s",
                            }
                        )
                    progress.update(1)
                    progress_text = str(progress).replace("\r", "")
                    postfix_text = ", ".join(f"{key}={value}" for key, value in postfix.items())
                    tqdm.write(f"{progress_text}, {postfix_text}")

        if progress is not None:
            progress.close()
        epoch_report = {
            "epoch": epoch,
            "avg_loss": epoch_loss_total / max(1, epoch_rows),
            "avg_lm_loss": epoch_lm_total / max(1, epoch_rows),
            "avg_reasoning_alignment_loss": epoch_reason_total / max(1, epoch_rows),
            "avg_future_frame_loss": epoch_future_total / max(1, epoch_rows),
            "avg_action_head_loss": epoch_action_head_total / max(1, epoch_rows),
            "lr": current_lr(optimizer),
            "epoch_elapsed_seconds": time.time() - started,
        }
        if not args.minimal_logging:
            history.append(epoch_report)
        print(json.dumps(epoch_report, ensure_ascii=False))

    if (not args.minimal_logging) and loss_artifact_paths is not None:
        _write_loss_artifacts(artifact_paths=loss_artifact_paths, loss_rows=loss_rows)
    extra_metadata = {
        "latent_slot_count": args.latent_slot_count,
        "reasoning_alignment_mode": str(args.reasoning_alignment_mode),
        "reasoning_field_slot_counts": list(args.resolved_reasoning_field_slot_counts),
        "init_adapter": args.init_adapter,
        "pixel_pruned_visual": True,
        "pixel_prune_threshold": args.pixel_prune_threshold,
        "pixel_prune_predictor_order": args.pixel_prune_predictor_order,
        "pixel_temporal_reuse": args.pixel_temporal_reuse,
        "pixel_temporal_threshold": args.pixel_temporal_threshold,
        "image_min_pixels": int(args.image_min_pixels),
        "image_max_pixels": int(args.image_max_pixels),
        "training_stage": args.training_stage,
        "stage2_target_format": args.stage2_target_format,
        "action_format": args.action_format,
        "action_model": args.action_model,
        "flow_action_sample_steps": args.flow_action_sample_steps,
        "flow_continuous_source": str(args.flow_continuous_source),
        "lm_action_target": str(args.resolved_lm_action_target),
        "action_coord_bins": int(agent.action_coord_bins),
        "stage1_max_reasoning_chars": args.stage1_max_reasoning_chars,
        "stage2_explicit_keep_start": args.stage2_explicit_keep_start,
        "stage2_explicit_keep_end": args.stage2_explicit_keep_end,
        "stage2_min_explicit_tokens": args.stage2_min_explicit_tokens,
        "action_head_loss_weight": args.action_head_loss_weight,
        "flow_action_loss_weight": float(args.flow_action_loss_weight),
        "flow_coord_loss_weight": float(args.flow_coord_loss_weight),
        "flow_coord_loss_scale": float(args.flow_coord_loss_scale),
        "flow_coord_loss_space": str(args.flow_coord_loss_space),
        "flow_patch_loss_weight": float(args.flow_patch_loss_weight),
        "use_action_head": bool(args.action_head_loss_weight > 0.0),
        "action_only_output": bool(args.training_stage == "stage2" and args.stage2_target_format == "action_only"),
        "official_generate_action": bool(args.resolved_lm_action_target == "include"),
        "lara_style_latent_reasoning": True,
        "history_n": args.history_n,
        "train_backbone": args.train_backbone,
        "train_embeddings": args.train_embeddings,
    }
    agent.save_adapter(args.adapter_out, extra_metadata=extra_metadata)
    elapsed = time.time() - started
    report = {
        "adapter_path": str(args.adapter_out),
        "elapsed_seconds": elapsed,
        "sample_count": int(total_rows),
        "trajectory_count": int(len(trajectories)),
        "epochs": int(args.epochs),
        "history_n": int(args.history_n),
        "latent_slot_count": int(args.latent_slot_count),
        "init_adapter": str(args.init_adapter) if args.init_adapter else None,
        "training_stage": str(args.training_stage),
        "stage2_target_format": str(args.stage2_target_format),
        "action_format": str(args.action_format),
        "action_model": str(args.action_model),
        "flow_action_sample_steps": int(args.flow_action_sample_steps),
        "flow_continuous_source": str(args.flow_continuous_source),
        "lm_action_target": str(args.resolved_lm_action_target),
        "action_coord_bins": int(agent.action_coord_bins),
        "stage1_max_reasoning_chars": int(args.stage1_max_reasoning_chars),
        "stage2_explicit_keep_start": float(args.stage2_explicit_keep_start),
        "stage2_explicit_keep_end": float(args.stage2_explicit_keep_end),
        "stage2_min_explicit_tokens": int(args.stage2_min_explicit_tokens),
        "action_head_loss_weight": float(args.action_head_loss_weight),
        "flow_action_loss_weight": float(args.flow_action_loss_weight),
        "flow_coord_loss_weight": float(args.flow_coord_loss_weight),
        "flow_coord_loss_scale": float(args.flow_coord_loss_scale),
        "flow_coord_loss_space": str(args.flow_coord_loss_space),
        "flow_patch_loss_weight": float(args.flow_patch_loss_weight),
        "pixel_pruned_visual": True,
        "pixel_temporal_reuse": bool(args.pixel_temporal_reuse),
        "image_min_pixels": int(args.image_min_pixels),
        "image_max_pixels": int(args.image_max_pixels),
        "train_backbone": bool(args.train_backbone),
        "train_embeddings": bool(args.train_embeddings),
        "history": history,
        "minimal_logging": bool(args.minimal_logging),
    }
    if loss_artifact_paths is not None:
        report["loss_artifacts"] = {
            "dir": str(loss_artifact_paths["root"]),
            "jsonl": str(loss_artifact_paths["jsonl"]),
            "csv": str(loss_artifact_paths["csv"]),
            "png": str(loss_artifact_paths["png"]),
        }
    if args.report_out:
        Path(args.report_out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
