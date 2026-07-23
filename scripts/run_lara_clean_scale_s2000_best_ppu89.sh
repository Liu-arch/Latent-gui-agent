#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STORAGE_ROOT="${STORAGE_ROOT:-/workspace/storage}"
DATASET_ROOT="${DATASET_ROOT:-${STORAGE_ROOT}/datasets/AgentNet}"
SUBSET_ROOT="${SUBSET_ROOT:-${STORAGE_ROOT}/datasets/AgentNet_processed/clean_subset_s2000_t200_v1}"
FINAL_DIR="${SUBSET_ROOT}/final"
PREFIX="${PREFIX:-agentnet_lara_clean_s2000_t200}"
READY_FILE="${FINAL_DIR}/READY"
MODEL="${MODEL:-${STORAGE_ROOT}/models/qwen3-vl-8b}"
RUN_NAME="${RUN_NAME:-lara_clean_s2000_t200_best_ppu89_seed42}"
OUT_ROOT="${OUT_ROOT:-${STORAGE_ROOT}/outputs/${RUN_NAME}}"
WAIT_INTERVAL_SECONDS="${WAIT_INTERVAL_SECONDS:-30}"
WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-0}"

mkdir -p "${OUT_ROOT}"
exec > >(tee -a "${OUT_ROOT}/scale_launcher.log") 2>&1
echo "[scale_s2000] start=$(date '+%F %T') host=$(hostname)"
echo "[scale_s2000] waiting_for=${READY_FILE}"
echo "[scale_s2000] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

start_time=$(date +%s)
while [[ ! -s "${READY_FILE}" ]]; do
  now=$(date +%s)
  if [[ "${WAIT_TIMEOUT_SECONDS}" -gt 0 ]] && (( now - start_time >= WAIT_TIMEOUT_SECONDS )); then
    echo "[scale_s2000] ERROR: timed out waiting for clean subset" >&2
    exit 1
  fi
  echo "[scale_s2000] data not ready; sleeping ${WAIT_INTERVAL_SECONDS}s"
  sleep "${WAIT_INTERVAL_SECONDS}"
done

SOURCE_TRAIN="${FINAL_DIR}/${PREFIX}.train.jsonl"
SOURCE_TEST="${FINAL_DIR}/${PREFIX}.test.jsonl"
for path in "${SOURCE_TRAIN}" "${SOURCE_TEST}" "${MODEL}/config.json"; do
  if [[ ! -s "${path}" ]]; then
    echo "[scale_s2000] ERROR: required artifact missing or empty: ${path}" >&2
    exit 2
  fi
done

python - "${READY_FILE}" "${SOURCE_TRAIN}" "${SOURCE_TEST}" <<'PY'
import json
import sys
from pathlib import Path

ready = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
counts = []
for path in map(Path, sys.argv[2:]):
    with path.open("rb") as handle:
        counts.append(sum(1 for line in handle if line.strip()))
expected = [int(ready["train"]["row_count"]), int(ready["test"]["row_count"])]
if counts != expected or counts != [2000, 200]:
    raise SystemExit(f"Fixed subset count mismatch: actual={counts}, expected={expected}")
print({"stage": "scale_data_preflight", "train_rows": counts[0], "test_rows": counts[1]})
PY

export CODE_DIR="${ROOT_DIR}"
export DATASET_ROOT
export MODEL
export SOURCE_TRAIN
export SOURCE_TEST
export FIXED_DIR="${SUBSET_ROOT}/fixed_for_training"
export FIXED_PREFIX="${PREFIX}"
export RUN_NAME
export OUT_ROOT
export TRAIN_SAMPLES=2000
export TEST_SAMPLES=200
export TRAIN_EVAL_SAMPLES="${TRAIN_EVAL_SAMPLES:-200}"
export TRAIN_WORLD_SIZE=2
export TRAIN_DEVICE_MAP=none
export EVAL_DEVICE_MAP="${EVAL_DEVICE_MAP:-balanced}"
export STAGE1_BATCH_SIZE="${STAGE1_BATCH_SIZE:-4}"
export STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-4}"
export ACTION_BATCH_SIZE="${ACTION_BATCH_SIZE:-4}"
export STAGE1_MAX_EPOCHS="${STAGE1_MAX_EPOCHS:-12}"
export STAGE2_TRANSITION_EPOCHS="${STAGE2_TRANSITION_EPOCHS:-4}"
export STAGE2_FULL_MAX_EPOCHS="${STAGE2_FULL_MAX_EPOCHS:-12}"
export ACTION_MAX_EPOCHS="${ACTION_MAX_EPOCHS:-20}"
export CHECKPOINT_EVERY_STEPS="${CHECKPOINT_EVERY_STEPS:-5}"
export STAGE2_REASONING_ALIGNMENT_MODE=field_aligned
export REASONING_FIELD_SLOT_COUNTS="${REASONING_FIELD_SLOT_COUNTS:-6,5,5}"
export FUTURE_FRAME_LOSS_WEIGHT="${FUTURE_FRAME_LOSS_WEIGHT:-0.25}"
export RUN_DDP_SMOKE="${RUN_DDP_SMOKE:-1}"
export RUN_GENERATION_EVAL="${RUN_GENERATION_EVAL:-1}"
export RUN_HELDOUT_EVAL="${RUN_HELDOUT_EVAL:-1}"

echo "[scale_s2000] data READY; starting field-aligned two-GPU pipeline"
echo "[scale_s2000] output=${OUT_ROOT}"
bash "${ROOT_DIR}/scripts/run_field_aligned_pipeline.sh"

if [[ "${RUN_POST_ABLATIONS:-1}" == "1" ]]; then
  echo "[scale_s2000] primary pipeline complete; starting two controlled action-head ablations"
  PRIMARY_RUN_NAME="${RUN_NAME}" PRIMARY_ROOT="${OUT_ROOT}" \
    bash "${ROOT_DIR}/scripts/run_lara_scale_action_ablation_ppu89.sh"
fi

echo "[scale_s2000] all requested experiments complete=$(date '+%F %T')"
