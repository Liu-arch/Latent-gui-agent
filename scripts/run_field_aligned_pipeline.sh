#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

: "${MODEL:?Set MODEL to the local Qwen3-VL checkpoint directory}"
: "${DATASET_ROOT:?Set DATASET_ROOT to the AgentNet dataset root}"
: "${SOURCE_TRAIN:?Set SOURCE_TRAIN to the normalized training JSONL}"
: "${SOURCE_TEST:?Set SOURCE_TEST to the normalized test JSONL}"

export CODE_DIR="${CODE_DIR:-${ROOT_DIR}}"
export OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/outputs/field_aligned_$(date +%Y%m%d_%H%M%S)}"
export STAGE2_REASONING_ALIGNMENT_MODE="field_aligned"
export REASONING_FIELD_SLOT_COUNTS="${REASONING_FIELD_SLOT_COUNTS:-6,5,5}"

exec bash "${ROOT_DIR}/run_lara_clean_s100_full_pipeline.sh"

