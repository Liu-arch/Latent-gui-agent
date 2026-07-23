#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STORAGE_ROOT="${STORAGE_ROOT:-/workspace/storage}"
DATASET_ROOT="${DATASET_ROOT:-${STORAGE_ROOT}/datasets/AgentNet}"
SUBSET_ROOT="${SUBSET_ROOT:-${STORAGE_ROOT}/datasets/AgentNet_processed/clean_subset_s2000_t200_v1}"
PREFIX="${PREFIX:-agentnet_lara_clean_s2000_t200}"
PRIMARY_RUN_NAME="${PRIMARY_RUN_NAME:-lara_clean_s2000_t200_best_ppu89_seed42}"
PRIMARY_ROOT="${PRIMARY_ROOT:-${STORAGE_ROOT}/outputs/${PRIMARY_RUN_NAME}}"
OUT_ROOT="${OUT_ROOT:-${PRIMARY_ROOT}/post_ablation}"
MODEL="${MODEL:-${STORAGE_ROOT}/models/qwen3-vl-8b}"
TRAIN_STEPS="${SUBSET_ROOT}/fixed_for_training/${PREFIX}.train.jsonl"
TEST_STEPS="${SUBSET_ROOT}/fixed_for_training/${PREFIX}.test.jsonl"
STAGE1_BEST="${PRIMARY_ROOT}/stage1_explicit/best.ckpt.pt"
STAGE2_BEST="${PRIMARY_ROOT}/stage2_fully_latent/best.ckpt.pt"
WORLD_SIZE="${WORLD_SIZE:-2}"
BATCH_SIZE="${BATCH_SIZE:-4}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-2000}"
TEST_SAMPLES="${TEST_SAMPLES:-200}"

mkdir -p "${OUT_ROOT}"
exec > >(tee -a "${OUT_ROOT}/ablation_pipeline.log") 2>&1

require_file() {
  if [[ ! -s "$1" ]]; then
    echo "[scale_ablation] ERROR: missing required artifact: $1" >&2
    exit 2
  fi
}
for path in "${TRAIN_STEPS}" "${TEST_STEPS}" "${STAGE1_BEST}" "${STAGE2_BEST}"; do
  require_file "${path}"
done

train_and_eval() {
  local name="$1"
  local init_adapter="$2"
  local hidden_source="$3"
  local run_dir="${OUT_ROOT}/${name}"
  local final="${run_dir}/final.pt"
  local latest="${run_dir}/latest.ckpt.pt"
  local best="${run_dir}/best.ckpt.pt"
  mkdir -p "${run_dir}"

  if [[ ! -f "${run_dir}/train.done" ]]; then
    local load_args=(--init-adapter "${init_adapter}")
    if [[ -s "${latest}" ]]; then
      load_args=(--resume-from "${latest}")
    fi
    echo "[scale_ablation] training=${name} init=${init_adapter} hidden=${hidden_source}"
    torchrun --standalone --nnodes=1 --nproc_per_node="${WORLD_SIZE}" \
      "${ROOT_DIR}/train_lara_style_qwen3vl_active_batch_ddp.py" \
      --steps "${TRAIN_STEPS}" \
      --dataset-root "${DATASET_ROOT}" \
      --model "${MODEL}" \
      "${load_args[@]}" \
      --adapter-out "${final}" \
      --report-out "${run_dir}/train_report.json" \
      --checkpoint-out "${latest}" \
      --best-checkpoint-out "${best}" \
      --checkpoint-every-steps 5 \
      --epochs 20 \
      --max-samples "${TRAIN_SAMPLES}" \
      --batch-size "${BATCH_SIZE}" \
      --grad-accum-steps 1 \
      --history-n 5 \
      --image-max-pixels 589824 \
      --latent-slot-count 16 \
      --training-stage stage2 \
      --reasoning-alignment-mode field_aligned \
      --reasoning-field-slot-counts 6,5,5 \
      --stage2-target-format mixed_reasoning_action \
      --stage2-explicit-keep-start 0.0 \
      --stage2-explicit-keep-end 0.0 \
      --stage2-max-thinking-tokens 16 \
      --lm-action-target omit \
      --action-model flow_matching \
      --flow-head-hidden-dim 1024 \
      --flow-head-depth 4 \
      --action-hidden-source "${hidden_source}" \
      --action-head-loss-weight 1.0 \
      --flow-action-loss-weight 0.0 \
      --flow-coord-loss-weight 1.0 \
      --flow-coord-loss-space logit \
      --flow-patch-loss-weight 1.0 \
      --flow-patch-loss-mode ce \
      --flow-pointer-coord-source patch_residual \
      --lm-loss-weight 0.0 \
      --reasoning-align-weight 0.0 \
      --future-frame-loss-weight 0.0 \
      --latent-diversity-weight 0.0 \
      --pixel-prune-threshold 0.0 \
      --pixel-prune-predictor-order pred2d,left,up \
      --pixel-temporal-reuse \
      --pixel-temporal-threshold 0.0 \
      --clean-observable-prompt \
      --use-lora \
      --lora-r 16 \
      --lora-alpha 32 \
      --lora-dropout 0.05 \
      --gradient-checkpointing \
      --train-action-head-only \
      --torch-dtype bfloat16 \
      --device-map none \
      --shuffle-trajectories \
      --lr 1e-4 \
      --lr-scheduler constant \
      --early-stop-monitor action_head_loss \
      --early-stop-patience 4 \
      --early-stop-min-delta 0.001 \
      --early-stop-min-epochs 5 \
      --minimal-logging \
      --log-every 5 \
      --prep-log-every 20 \
      2>&1 | tee -a "${run_dir}/train.stdout.log"
    require_file "${best}"
    touch "${run_dir}/train.done"
  fi

  local eval_dir="${run_dir}/eval_test${TEST_SAMPLES}"
  mkdir -p "${eval_dir}"
  if [[ ! -f "${eval_dir}/done" ]]; then
    local resume_args=()
    if [[ -s "${eval_dir}/report.json.progress.json" ]]; then
      resume_args=(--resume-from "${eval_dir}/report.json.progress.json")
    fi
    echo "[scale_ablation] evaluating=${name} test_samples=${TEST_SAMPLES}"
    python "${ROOT_DIR}/evaluate_lara_style_qwen3vl_batched.py" \
      --steps "${TEST_STEPS}" \
      --dataset-root "${DATASET_ROOT}" \
      --model "${MODEL}" \
      --adapter "${best}" \
      --max-samples "${TEST_SAMPLES}" \
      --history-n 5 \
      --image-max-pixels 589824 \
      --batch-size 8 \
      --batching-mode active_pool \
      --eval-mode action_head \
      --flow-continuous-source direct \
      --flow-pointer-coord-source patch_residual \
      --action-hidden-source "${hidden_source}" \
      --max-new-tokens 64 \
      --torch-dtype bfloat16 \
      --device-map balanced \
      --log-every 10 \
      --save-every 10 \
      --step-out "${eval_dir}/steps.jsonl" \
      --step-out-every 1 \
      --report-out "${eval_dir}/report.json" \
      "${resume_args[@]}" \
      2>&1 | tee -a "${eval_dir}/stdout.log"
    require_file "${eval_dir}/steps.jsonl"
    require_file "${eval_dir}/report.json"
    touch "${eval_dir}/done"
  fi
}

# A: Does a fully-latent Stage 2 initialization improve the same action head?
train_and_eval stage1_init_prompt_slot_attn "${STAGE1_BEST}" prompt_slot_attn

# B: Does attention over prompt and role-specific slots beat the old three-summary input?
train_and_eval stage2_init_summary "${STAGE2_BEST}" summary

echo "[scale_ablation] complete=$(date '+%F %T')"
