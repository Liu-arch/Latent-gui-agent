#!/bin/bash
set -euo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export DDP_TIMEOUT_SECONDS="${DDP_TIMEOUT_SECONDS:-3600}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="${CODE_DIR:-${SCRIPT_DIR}}"
DATASET_ROOT="${DATASET_ROOT:-}"
MODEL="${MODEL:-}"
if [[ -z "${DATASET_ROOT}" || -z "${MODEL}" ]]; then
  echo "Set DATASET_ROOT and MODEL before starting the pipeline." >&2
  exit 2
fi
TRAIN_DEVICE_MAP="${TRAIN_DEVICE_MAP:-none}"
EVAL_DEVICE_MAP="${EVAL_DEVICE_MAP:-balanced}"
SOURCE_TRAIN="${SOURCE_TRAIN:-${DATASET_ROOT}/splits_lara_gui_shortreason/agentnet_ubuntu_lara_gui.stage1_full.train.jsonl}"
SOURCE_TEST="${SOURCE_TEST:-${DATASET_ROOT}/splits_lara_gui_shortreason/agentnet_ubuntu_lara_gui.stage1_full.test.jsonl}"
FIXED_DIR="${FIXED_DIR:-${DATASET_ROOT}/fixed_lara_clean_s100}"
FIXED_PREFIX="${FIXED_PREFIX:-lara_gui_clean_s100}"
TRAIN_STEPS="${FIXED_DIR}/${FIXED_PREFIX}.train.jsonl"
TEST_STEPS="${FIXED_DIR}/${FIXED_PREFIX}.test.jsonl"
RUN_NAME="${RUN_NAME:-lara_clean_s100_ddp_seed42}"
OUT_ROOT="${OUT_ROOT:-${CODE_DIR}/outputs/${RUN_NAME}}"

TRAIN_SAMPLES="${TRAIN_SAMPLES:-100}"
TEST_SAMPLES="${TEST_SAMPLES:-100}"
TRAIN_EVAL_SAMPLES="${TRAIN_EVAL_SAMPLES:-${TRAIN_SAMPLES}}"
TRAIN_WORLD_SIZE="${TRAIN_WORLD_SIZE:-2}"
HISTORY_N="${HISTORY_N:-5}"
IMAGE_MAX_PIXELS="${IMAGE_MAX_PIXELS:-589824}"
LATENT_SLOTS="${LATENT_SLOTS:-16}"
STAGE2_REASONING_ALIGNMENT_MODE="${STAGE2_REASONING_ALIGNMENT_MODE:-field_aligned}"
REASONING_FIELD_SLOT_COUNTS="${REASONING_FIELD_SLOT_COUNTS:-6,5,5}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
# Training batch sizes are per DDP rank. Gradient accumulation is configurable
# independently so memory-constrained accelerators can preserve the intended
# effective global batch without changing the optimization schedule.
STAGE1_BATCH_SIZE="${STAGE1_BATCH_SIZE:-4}"
STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-4}"
ACTION_BATCH_SIZE="${ACTION_BATCH_SIZE:-4}"
STAGE1_GRAD_ACCUM_STEPS="${STAGE1_GRAD_ACCUM_STEPS:-1}"
STAGE2_GRAD_ACCUM_STEPS="${STAGE2_GRAD_ACCUM_STEPS:-1}"
ACTION_GRAD_ACCUM_STEPS="${ACTION_GRAD_ACCUM_STEPS:-1}"
STAGE1_MAX_EPOCHS="${STAGE1_MAX_EPOCHS:-12}"
STAGE2_TRANSITION_EPOCHS="${STAGE2_TRANSITION_EPOCHS:-4}"
STAGE2_FULL_MAX_EPOCHS="${STAGE2_FULL_MAX_EPOCHS:-12}"
ACTION_MAX_EPOCHS="${ACTION_MAX_EPOCHS:-20}"
CHECKPOINT_EVERY_STEPS="${CHECKPOINT_EVERY_STEPS:-5}"
STAGE1_INIT_ADAPTER="${STAGE1_INIT_ADAPTER:-}"
STAGE1_REUSE_ADAPTER="${STAGE1_REUSE_ADAPTER:-}"
FUTURE_FRAME_LOSS_WEIGHT="${FUTURE_FRAME_LOSS_WEIGHT:-0.25}"
PREFLIGHT_MIN_FREE_GB="${PREFLIGHT_MIN_FREE_GB:-40}"
RUN_GENERATION_EVAL="${RUN_GENERATION_EVAL:-1}"
RUN_DDP_SMOKE="${RUN_DDP_SMOKE:-1}"
# First submission validates train-set overfitting only. Re-submit the same
# RUN_NAME with RUN_HELDOUT_EVAL=1 after those diagnostics look healthy.
RUN_HELDOUT_EVAL="${RUN_HELDOUT_EVAL:-0}"

mkdir -p "${OUT_ROOT}" "${FIXED_DIR}"
cd "${CODE_DIR}"

artifacts_ready() {
  local path
  for path in "$@"; do
    if [[ ! -s "${path}" ]]; then
      return 1
    fi
  done
}

require_artifacts() {
  local path
  for path in "$@"; do
    if [[ ! -s "${path}" ]]; then
      echo "[lara_clean_s100] ERROR: expected non-empty artifact was not produced: ${path}" >&2
      exit 3
    fi
  done
}

LIVE_LOG="${OUT_ROOT}/pipeline.live.log"
exec > >(tee -a "${LIVE_LOG}") 2>&1

echo "[lara_clean_s100] start=$(date '+%F %T') host=$(hostname)"
echo "[lara_clean_s100] run=${RUN_NAME} output=${OUT_ROOT}"
echo "[lara_clean_s100] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[lara_clean_s100] training=torchrun_ddp world_size=${TRAIN_WORLD_SIZE} train_device_map=${TRAIN_DEVICE_MAP}"
echo "[lara_clean_s100] evaluation=model_parallel eval_device_map=${EVAL_DEVICE_MAP}"
echo "[lara_clean_s100] per_rank_batches stage1=${STAGE1_BATCH_SIZE} stage2=${STAGE2_BATCH_SIZE} action=${ACTION_BATCH_SIZE}"
echo "[lara_clean_s100] global_batches stage1=$((TRAIN_WORLD_SIZE * STAGE1_BATCH_SIZE)) stage2=$((TRAIN_WORLD_SIZE * STAGE2_BATCH_SIZE)) action=$((TRAIN_WORLD_SIZE * ACTION_BATCH_SIZE))"
echo "[lara_clean_s100] grad_accum stage1=${STAGE1_GRAD_ACCUM_STEPS} stage2=${STAGE2_GRAD_ACCUM_STEPS} action=${ACTION_GRAD_ACCUM_STEPS}"
echo "[lara_clean_s100] effective_batches stage1=$((TRAIN_WORLD_SIZE * STAGE1_BATCH_SIZE * STAGE1_GRAD_ACCUM_STEPS)) stage2=$((TRAIN_WORLD_SIZE * STAGE2_BATCH_SIZE * STAGE2_GRAD_ACCUM_STEPS)) action=$((TRAIN_WORLD_SIZE * ACTION_BATCH_SIZE * ACTION_GRAD_ACCUM_STEPS))"
echo "[lara_clean_s100] train_samples=${TRAIN_SAMPLES} train_eval_samples=${TRAIN_EVAL_SAMPLES} test_samples=${TEST_SAMPLES}"
echo "[lara_clean_s100] stage2_reasoning_alignment=${STAGE2_REASONING_ALIGNMENT_MODE} field_slots=${REASONING_FIELD_SLOT_COUNTS}"
if [[ -n "${STAGE1_REUSE_ADAPTER}" ]]; then
  echo "[lara_clean_s100] stage1_direct_reuse=${STAGE1_REUSE_ADAPTER}"
elif [[ -n "${STAGE1_INIT_ADAPTER}" ]]; then
  echo "[lara_clean_s100] stage1_warm_start=${STAGE1_INIT_ADAPTER}"
fi
echo "[lara_clean_s100] run_generation_eval=${RUN_GENERATION_EVAL} run_heldout_eval=${RUN_HELDOUT_EVAL}"

VISIBLE_GPU_COUNT="$(python -c 'import torch; print(torch.cuda.device_count())')"
if [[ "${VISIBLE_GPU_COUNT}" -lt "${TRAIN_WORLD_SIZE}" ]]; then
  echo "[lara_clean_s100] ERROR: this pipeline requires ${TRAIN_WORLD_SIZE} visible GPUs; found ${VISIBLE_GPU_COUNT}." >&2
  exit 2
fi

TRAIN_LAUNCH=(
  torchrun
  --standalone
  --nnodes=1
  --nproc_per_node="${TRAIN_WORLD_SIZE}"
  "${CODE_DIR}/train_lara_style_qwen3vl_active_batch_ddp.py"
)

python "${CODE_DIR}/prepare_lara_fixed_splits.py" \
  --train-input "${SOURCE_TRAIN}" \
  --test-input "${SOURCE_TEST}" \
  --out-dir "${FIXED_DIR}" \
  --train-samples "${TRAIN_SAMPLES}" \
  --test-samples "${TEST_SAMPLES}" \
  --prefix "${FIXED_PREFIX}"

python "${CODE_DIR}/preflight_lara_clean_pipeline.py" \
  --train-steps "${TRAIN_STEPS}" \
  --test-steps "${TEST_STEPS}" \
  --dataset-root "${DATASET_ROOT}" \
  --model "${MODEL}" \
  --output-root "${OUT_ROOT}" \
  --train-samples "${TRAIN_SAMPLES}" \
  --test-samples "${TEST_SAMPLES}" \
  --world-size "${TRAIN_WORLD_SIZE}" \
  --batch-size-per-rank "${STAGE1_BATCH_SIZE}" \
  --min-free-gb "${PREFLIGHT_MIN_FREE_GB}"

COMMON_TRAIN_ARGS=(
  --steps "${TRAIN_STEPS}"
  --dataset-root "${DATASET_ROOT}"
  --model "${MODEL}"
  --max-samples "${TRAIN_SAMPLES}"
  --history-n "${HISTORY_N}"
  --image-max-pixels "${IMAGE_MAX_PIXELS}"
  --latent-slot-count "${LATENT_SLOTS}"
  --pixel-prune-threshold 0.0
  --pixel-prune-predictor-order pred2d,left,up
  --pixel-temporal-reuse
  --pixel-temporal-threshold 0.0
  --clean-observable-prompt
  --use-lora
  --lora-r "${LORA_R}"
  --lora-alpha "${LORA_ALPHA}"
  --lora-dropout "${LORA_DROPOUT}"
  --gradient-checkpointing
  --torch-dtype bfloat16
  --device-map "${TRAIN_DEVICE_MAP}"
  --shuffle-trajectories
  --checkpoint-every-steps "${CHECKPOINT_EVERY_STEPS}"
  --minimal-logging
  --log-every 5
  --prep-log-every 20
)

SMOKE_DIR="${OUT_ROOT}/ddp_runtime_smoke"
mkdir -p "${SMOKE_DIR}"
if [[ "${RUN_DDP_SMOKE}" == "1" ]]; then
  if [[ ! -f "${SMOKE_DIR}/done" ]] || ! artifacts_ready "${SMOKE_DIR}/final.pt" "${SMOKE_DIR}/report.json"; then
    echo "[lara_clean_s100] two-GPU real-model backward smoke begins"
    "${TRAIN_LAUNCH[@]}" \
      "${COMMON_TRAIN_ARGS[@]}" \
      --adapter-out "${SMOKE_DIR}/final.pt" \
      --report-out "${SMOKE_DIR}/report.json" \
      --epochs 1 \
      --max-samples 2 \
      --batch-size "${STAGE1_BATCH_SIZE}" \
      --grad-accum-steps "${STAGE1_GRAD_ACCUM_STEPS}" \
      --lr 5e-5 \
      --lr-scheduler constant \
      --training-stage stage1 \
      --stage1-max-reasoning-chars 0 \
      --lm-action-target include \
      --action-model flow_matching \
      --action-head-loss-weight 0.0 \
      --flow-action-loss-weight 0.0 \
      --flow-coord-loss-weight 0.0 \
      --flow-patch-loss-weight 0.0 \
      --lm-loss-weight 1.0 \
      --reasoning-align-weight 0.25 \
      --future-frame-loss-weight "${FUTURE_FRAME_LOSS_WEIGHT}" \
      --latent-diversity-weight 0.0 \
      --train-embeddings \
      2>&1 | tee -a "${SMOKE_DIR}/stdout.log"
    require_artifacts "${SMOKE_DIR}/final.pt" "${SMOKE_DIR}/report.json"
    touch "${SMOKE_DIR}/done"
    echo "[lara_clean_s100] two-GPU real-model backward smoke passed"
  else
    echo "[lara_clean_s100] two-GPU real-model backward smoke already passed"
  fi
fi

STAGE1_DIR="${OUT_ROOT}/stage1_explicit"
mkdir -p "${STAGE1_DIR}"
STAGE1_FINAL="${STAGE1_DIR}/final.pt"
STAGE1_CKPT="${STAGE1_DIR}/latest.ckpt.pt"
STAGE1_BEST="${STAGE1_DIR}/best.ckpt.pt"
if [[ -n "${STAGE1_REUSE_ADAPTER}" ]]; then
  require_artifacts "${STAGE1_REUSE_ADAPTER}"
  STAGE1_BEST="${STAGE1_REUSE_ADAPTER}"
  printf '%s\n' "${STAGE1_REUSE_ADAPTER}" > "${STAGE1_DIR}/reused_adapter.path"
  echo "[lara_clean_s100] Stage1 explicit training skipped; reusing ${STAGE1_BEST}"
else
  STAGE1_LOAD_ARGS=()
  if [[ -s "${STAGE1_CKPT}" ]]; then
    STAGE1_LOAD_ARGS=(--resume-from "${STAGE1_CKPT}")
  elif [[ -n "${STAGE1_INIT_ADAPTER}" ]]; then
    require_artifacts "${STAGE1_INIT_ADAPTER}"
    STAGE1_LOAD_ARGS=(--init-adapter "${STAGE1_INIT_ADAPTER}")
  fi
  if [[ ! -f "${STAGE1_DIR}/done" ]] || ! artifacts_ready "${STAGE1_FINAL}" "${STAGE1_CKPT}" "${STAGE1_BEST}" "${STAGE1_DIR}/report.json"; then
    echo "[lara_clean_s100] Stage1 explicit training begins"
    "${TRAIN_LAUNCH[@]}" \
      "${COMMON_TRAIN_ARGS[@]}" \
      "${STAGE1_LOAD_ARGS[@]}" \
      --adapter-out "${STAGE1_FINAL}" \
      --report-out "${STAGE1_DIR}/report.json" \
      --checkpoint-out "${STAGE1_CKPT}" \
      --best-checkpoint-out "${STAGE1_BEST}" \
      --epochs "${STAGE1_MAX_EPOCHS}" \
      --batch-size "${STAGE1_BATCH_SIZE}" \
      --grad-accum-steps "${STAGE1_GRAD_ACCUM_STEPS}" \
      --lr 5e-5 \
      --lr-scheduler constant \
      --training-stage stage1 \
      --stage1-max-reasoning-chars 0 \
      --lm-action-target include \
      --action-model flow_matching \
      --action-head-loss-weight 0.0 \
      --flow-action-loss-weight 0.0 \
      --flow-coord-loss-weight 0.0 \
      --flow-patch-loss-weight 0.0 \
      --lm-loss-weight 1.0 \
      --reasoning-align-weight 0.25 \
      --future-frame-loss-weight "${FUTURE_FRAME_LOSS_WEIGHT}" \
      --latent-diversity-weight 0.0 \
      --train-embeddings \
      --early-stop-monitor loss \
      --early-stop-patience 3 \
      --early-stop-min-delta 0.001 \
      --early-stop-min-epochs 4 \
      2>&1 | tee -a "${STAGE1_DIR}/train.stdout.log"
    require_artifacts "${STAGE1_FINAL}" "${STAGE1_CKPT}" "${STAGE1_BEST}" "${STAGE1_DIR}/report.json"
    touch "${STAGE1_DIR}/done"
  fi
fi
require_artifacts "${STAGE1_BEST}"

STAGE2_TRANSITION_DIR="${OUT_ROOT}/stage2_transition"
mkdir -p "${STAGE2_TRANSITION_DIR}"
STAGE2_TRANSITION_FINAL="${STAGE2_TRANSITION_DIR}/final.pt"
STAGE2_TRANSITION_CKPT="${STAGE2_TRANSITION_DIR}/latest.ckpt.pt"
STAGE2_TRANSITION_LOAD_ARGS=(--init-adapter "${STAGE1_BEST}")
if [[ -s "${STAGE2_TRANSITION_CKPT}" ]]; then
  STAGE2_TRANSITION_LOAD_ARGS=(--resume-from "${STAGE2_TRANSITION_CKPT}")
fi
if [[ ! -f "${STAGE2_TRANSITION_DIR}/done" ]] || ! artifacts_ready "${STAGE2_TRANSITION_FINAL}" "${STAGE2_TRANSITION_CKPT}" "${STAGE2_TRANSITION_DIR}/report.json"; then
  echo "[lara_clean_s100] Stage2 curriculum transition begins"
  "${TRAIN_LAUNCH[@]}" \
    "${COMMON_TRAIN_ARGS[@]}" \
    "${STAGE2_TRANSITION_LOAD_ARGS[@]}" \
    --adapter-out "${STAGE2_TRANSITION_FINAL}" \
    --report-out "${STAGE2_TRANSITION_DIR}/report.json" \
    --checkpoint-out "${STAGE2_TRANSITION_CKPT}" \
    --epochs "${STAGE2_TRANSITION_EPOCHS}" \
    --batch-size "${STAGE2_BATCH_SIZE}" \
    --grad-accum-steps "${STAGE2_GRAD_ACCUM_STEPS}" \
    --lr 3e-5 \
    --lr-scheduler constant \
    --training-stage stage2 \
    --reasoning-alignment-mode "${STAGE2_REASONING_ALIGNMENT_MODE}" \
    --reasoning-field-slot-counts "${REASONING_FIELD_SLOT_COUNTS}" \
    --stage2-target-format mixed_reasoning_action \
    --stage2-explicit-keep-start 0.75 \
    --stage2-explicit-keep-end 0.0 \
    --stage2-max-thinking-tokens "${LATENT_SLOTS}" \
    --lm-action-target include \
    --action-model flow_matching \
    --action-head-loss-weight 0.0 \
    --flow-action-loss-weight 0.0 \
    --flow-coord-loss-weight 0.0 \
    --flow-patch-loss-weight 0.0 \
    --lm-loss-weight 1.0 \
    --reasoning-align-weight 1.0 \
    --future-frame-loss-weight "${FUTURE_FRAME_LOSS_WEIGHT}" \
    --latent-diversity-weight 0.01 \
    --train-embeddings \
    2>&1 | tee -a "${STAGE2_TRANSITION_DIR}/train.stdout.log"
  require_artifacts "${STAGE2_TRANSITION_FINAL}" "${STAGE2_TRANSITION_CKPT}" "${STAGE2_TRANSITION_DIR}/report.json"
  touch "${STAGE2_TRANSITION_DIR}/done"
fi
require_artifacts "${STAGE2_TRANSITION_FINAL}"

STAGE2_FULL_DIR="${OUT_ROOT}/stage2_fully_latent"
mkdir -p "${STAGE2_FULL_DIR}"
STAGE2_FULL_FINAL="${STAGE2_FULL_DIR}/final.pt"
STAGE2_FULL_CKPT="${STAGE2_FULL_DIR}/latest.ckpt.pt"
STAGE2_FULL_BEST="${STAGE2_FULL_DIR}/best.ckpt.pt"
STAGE2_FULL_LOAD_ARGS=(--init-adapter "${STAGE2_TRANSITION_FINAL}")
if [[ -s "${STAGE2_FULL_CKPT}" ]]; then
  STAGE2_FULL_LOAD_ARGS=(--resume-from "${STAGE2_FULL_CKPT}")
fi
if [[ ! -f "${STAGE2_FULL_DIR}/done" ]] || ! artifacts_ready "${STAGE2_FULL_FINAL}" "${STAGE2_FULL_CKPT}" "${STAGE2_FULL_BEST}" "${STAGE2_FULL_DIR}/report.json"; then
  echo "[lara_clean_s100] Stage2 fully latent convergence begins"
  "${TRAIN_LAUNCH[@]}" \
    "${COMMON_TRAIN_ARGS[@]}" \
    "${STAGE2_FULL_LOAD_ARGS[@]}" \
    --adapter-out "${STAGE2_FULL_FINAL}" \
    --report-out "${STAGE2_FULL_DIR}/report.json" \
    --checkpoint-out "${STAGE2_FULL_CKPT}" \
    --best-checkpoint-out "${STAGE2_FULL_BEST}" \
    --epochs "${STAGE2_FULL_MAX_EPOCHS}" \
    --batch-size "${STAGE2_BATCH_SIZE}" \
    --grad-accum-steps "${STAGE2_GRAD_ACCUM_STEPS}" \
    --lr 3e-5 \
    --lr-scheduler constant \
    --training-stage stage2 \
    --reasoning-alignment-mode "${STAGE2_REASONING_ALIGNMENT_MODE}" \
    --reasoning-field-slot-counts "${REASONING_FIELD_SLOT_COUNTS}" \
    --stage2-target-format mixed_reasoning_action \
    --stage2-explicit-keep-start 0.0 \
    --stage2-explicit-keep-end 0.0 \
    --stage2-max-thinking-tokens "${LATENT_SLOTS}" \
    --lm-action-target include \
    --action-model flow_matching \
    --action-head-loss-weight 0.0 \
    --flow-action-loss-weight 0.0 \
    --flow-coord-loss-weight 0.0 \
    --flow-patch-loss-weight 0.0 \
    --lm-loss-weight 1.0 \
    --reasoning-align-weight 1.0 \
    --future-frame-loss-weight "${FUTURE_FRAME_LOSS_WEIGHT}" \
    --latent-diversity-weight 0.01 \
    --train-embeddings \
    --early-stop-monitor loss \
    --early-stop-patience 3 \
    --early-stop-min-delta 0.001 \
    --early-stop-min-epochs 4 \
    2>&1 | tee -a "${STAGE2_FULL_DIR}/train.stdout.log"
  require_artifacts "${STAGE2_FULL_FINAL}" "${STAGE2_FULL_CKPT}" "${STAGE2_FULL_BEST}" "${STAGE2_FULL_DIR}/report.json"
  touch "${STAGE2_FULL_DIR}/done"
fi
require_artifacts "${STAGE2_FULL_BEST}"

ACTION_DIR="${OUT_ROOT}/action_head"
mkdir -p "${ACTION_DIR}"
ACTION_FINAL="${ACTION_DIR}/final.pt"
ACTION_CKPT="${ACTION_DIR}/latest.ckpt.pt"
ACTION_BEST="${ACTION_DIR}/best.ckpt.pt"
ACTION_LOAD_ARGS=(--init-adapter "${STAGE2_FULL_BEST}")
if [[ -s "${ACTION_CKPT}" ]]; then
  ACTION_LOAD_ARGS=(--resume-from "${ACTION_CKPT}")
fi
if [[ ! -f "${ACTION_DIR}/done" ]] || ! artifacts_ready "${ACTION_FINAL}" "${ACTION_CKPT}" "${ACTION_BEST}" "${ACTION_DIR}/report.json"; then
  echo "[lara_clean_s100] Action-head convergence begins"
  "${TRAIN_LAUNCH[@]}" \
    "${COMMON_TRAIN_ARGS[@]}" \
    "${ACTION_LOAD_ARGS[@]}" \
    --adapter-out "${ACTION_FINAL}" \
    --report-out "${ACTION_DIR}/report.json" \
    --checkpoint-out "${ACTION_CKPT}" \
    --best-checkpoint-out "${ACTION_BEST}" \
    --epochs "${ACTION_MAX_EPOCHS}" \
    --batch-size "${ACTION_BATCH_SIZE}" \
    --grad-accum-steps "${ACTION_GRAD_ACCUM_STEPS}" \
    --lr 1e-4 \
    --lr-scheduler constant \
    --training-stage stage2 \
    --reasoning-alignment-mode "${STAGE2_REASONING_ALIGNMENT_MODE}" \
    --reasoning-field-slot-counts "${REASONING_FIELD_SLOT_COUNTS}" \
    --stage2-target-format mixed_reasoning_action \
    --stage2-explicit-keep-start 0.0 \
    --stage2-explicit-keep-end 0.0 \
    --stage2-max-thinking-tokens "${LATENT_SLOTS}" \
    --lm-action-target omit \
    --action-model flow_matching \
    --flow-head-hidden-dim 1024 \
    --flow-head-depth 4 \
    --action-hidden-source prompt_slot_attn \
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
    --train-action-head-only \
    --early-stop-monitor action_head_loss \
    --early-stop-patience 4 \
    --early-stop-min-delta 0.001 \
    --early-stop-min-epochs 5 \
    2>&1 | tee -a "${ACTION_DIR}/train.stdout.log"
  require_artifacts "${ACTION_FINAL}" "${ACTION_CKPT}" "${ACTION_BEST}" "${ACTION_DIR}/report.json"
  touch "${ACTION_DIR}/done"
fi
require_artifacts "${ACTION_BEST}"

run_eval() {
  local eval_name="$1"
  local adapter="$2"
  local steps="$3"
  local eval_mode="$4"
  local batch_size="$5"
  local max_new_tokens="$6"
  local max_samples="$7"
  local eval_dir="${OUT_ROOT}/eval/${eval_name}"
  mkdir -p "${eval_dir}"
  if [[ -f "${eval_dir}/done" ]] && artifacts_ready "${eval_dir}/report.json" "${eval_dir}/steps.jsonl"; then
    echo "[lara_clean_s100] evaluation ${eval_name} already complete"
    return
  fi
  local report="${eval_dir}/report.json"
  local progress="${report}.progress.json"
  local resume_args=()
  if [[ -s "${progress}" ]]; then
    resume_args=(--resume-from "${progress}")
  fi
  python "${CODE_DIR}/evaluate_lara_style_qwen3vl_batched.py" \
    --steps "${steps}" \
    --dataset-root "${DATASET_ROOT}" \
    --model "${MODEL}" \
    --adapter "${adapter}" \
    --max-samples "${max_samples}" \
    --history-n "${HISTORY_N}" \
    --image-max-pixels "${IMAGE_MAX_PIXELS}" \
    --batch-size "${batch_size}" \
    --batching-mode active_pool \
    --eval-mode "${eval_mode}" \
    --flow-continuous-source direct \
    --include-flow-alternatives \
    --max-new-tokens "${max_new_tokens}" \
    --torch-dtype bfloat16 \
    --device-map "${EVAL_DEVICE_MAP}" \
    --log-every 10 \
    --save-every 10 \
    --step-out "${eval_dir}/steps.jsonl" \
    --step-out-every 1 \
    --report-out "${report}" \
    "${resume_args[@]}" \
    2>&1 | tee -a "${eval_dir}/stdout.log"
  require_artifacts "${report}" "${eval_dir}/steps.jsonl"
  touch "${eval_dir}/done"
}

# Phase A: prove every learned representation can fit the exact training set
# before spending cluster time on held-out generalization.
if [[ "${RUN_GENERATION_EVAL}" == "1" ]]; then
  run_eval stage1_train_overfit_generate "${STAGE1_BEST}" "${TRAIN_STEPS}" generate 4 512 "${TRAIN_EVAL_SAMPLES}"
  run_eval stage2_transition_train_overfit_generate "${STAGE2_TRANSITION_FINAL}" "${TRAIN_STEPS}" generate 4 512 "${TRAIN_EVAL_SAMPLES}"
  run_eval stage2_fully_latent_train_overfit_generate "${STAGE2_FULL_BEST}" "${TRAIN_STEPS}" generate 4 256 "${TRAIN_EVAL_SAMPLES}"
fi
run_eval "action_train${TRAIN_EVAL_SAMPLES}" "${ACTION_BEST}" "${TRAIN_STEPS}" action_head 8 64 "${TRAIN_EVAL_SAMPLES}"
if [[ "${RUN_GENERATION_EVAL}" == "1" ]]; then
  run_eval final_train_overfit_hybrid "${ACTION_BEST}" "${TRAIN_STEPS}" hybrid 4 256 "${TRAIN_EVAL_SAMPLES}"
fi

# Phase B is deliberately opt-in. It reuses all done markers and checkpoints,
# so a second sbatch with the same RUN_NAME performs only the missing tests.
if [[ "${RUN_HELDOUT_EVAL}" == "1" ]]; then
  if [[ "${RUN_GENERATION_EVAL}" == "1" ]]; then
    run_eval stage1_test_generate "${STAGE1_BEST}" "${TEST_STEPS}" generate 4 512 "${TEST_SAMPLES}"
    run_eval stage2_test_generate "${STAGE2_FULL_BEST}" "${TEST_STEPS}" generate 4 256 "${TEST_SAMPLES}"
  fi
  run_eval "action_test${TEST_SAMPLES}" "${ACTION_BEST}" "${TEST_STEPS}" action_head 8 64 "${TEST_SAMPLES}"
  if [[ "${RUN_GENERATION_EVAL}" == "1" ]]; then
    run_eval "final_test${TEST_SAMPLES}_hybrid" "${ACTION_BEST}" "${TEST_STEPS}" hybrid 4 256 "${TEST_SAMPLES}"
  fi
else
  echo "[lara_clean_s100] held-out evaluation deferred; re-submit the same RUN_NAME with RUN_HELDOUT_EVAL=1"
fi

python "${CODE_DIR}/summarize_lara_clean_pipeline.py" --run-dir "${OUT_ROOT}"

echo "[lara_clean_s100] complete=$(date '+%F %T')"
echo "[lara_clean_s100] output=${OUT_ROOT}"
