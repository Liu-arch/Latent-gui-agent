#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STORAGE_ROOT="${STORAGE_ROOT:-/workspace/storage}"
DATASET_ROOT="${DATASET_ROOT:-${STORAGE_ROOT}/datasets/AgentNet}"
STRUCTURAL_INPUT="${STRUCTURAL_INPUT:-${STORAGE_ROOT}/datasets/AgentNet_processed/clean_v1/structural/agentnet_ubuntu_lara_clean_structural.jsonl}"
OUT_ROOT="${OUT_ROOT:-${STORAGE_ROOT}/datasets/AgentNet_processed/clean_subset_s2000_t200_v1}"
PREFIX="${PREFIX:-agentnet_lara_clean_s2000_t200}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-2000}"
TEST_SAMPLES="${TEST_SAMPLES:-200}"
MODEL_NAME="${MODEL_NAME:-qwen3-vl-8b}"
BASE_URL_0="${BASE_URL_0:-http://127.0.0.1:18000/v1}"
BASE_URL_1="${BASE_URL_1:-http://127.0.0.1:18001/v1}"
CONCURRENCY="${CONCURRENCY:-4}"
OUTER_RETRIES="${OUTER_RETRIES:-20}"

STRUCTURAL_DIR="${OUT_ROOT}/structural"
ENRICHED_DIR="${OUT_ROOT}/enriched_shards"
FINAL_DIR="${OUT_ROOT}/final"
LOG_DIR="${OUT_ROOT}/logs"
mkdir -p "${STRUCTURAL_DIR}" "${ENRICHED_DIR}" "${FINAL_DIR}" "${LOG_DIR}"

exec > >(tee -a "${LOG_DIR}/pipeline.log") 2>&1
echo "[clean_subset] start=$(date '+%F %T') host=$(hostname)"
echo "[clean_subset] train=${TRAIN_SAMPLES} test=${TEST_SAMPLES} urls=${BASE_URL_0},${BASE_URL_1}"

python "${ROOT_DIR}/prepare_agentnet_clean_fixed_subset.py" \
  --input "${STRUCTURAL_INPUT}" \
  --out-dir "${STRUCTURAL_DIR}" \
  --prefix "${PREFIX}" \
  --train-samples "${TRAIN_SAMPLES}" \
  --test-samples "${TEST_SAMPLES}" \
  --seed 42

TRAIN_INPUT="${STRUCTURAL_DIR}/${PREFIX}.structural.train.jsonl"
TEST_INPUT="${STRUCTURAL_DIR}/${PREFIX}.structural.test.jsonl"
SELECTION_MANIFEST="${STRUCTURAL_DIR}/${PREFIX}.selection_manifest.json"

wait_for_server() {
  local url="$1"
  local attempt
  for attempt in $(seq 1 180); do
    if curl -fsS --max-time 5 "${url%/v1}/v1/models" >/dev/null 2>&1; then
      echo "[clean_subset] server ready: ${url}"
      return 0
    fi
    sleep 5
  done
  echo "[clean_subset] ERROR: server did not become ready: ${url}" >&2
  return 1
}

run_enrichment_shard() {
  local split="$1"
  local shard="$2"
  local url="$3"
  local input="$4"
  local expected="$5"
  local output="${ENRICHED_DIR}/${PREFIX}.${split}.shard${shard}.jsonl"
  local summary="${output%.jsonl}.summary.json"
  local progress="${output%.jsonl}.progress.json"
  local log="${LOG_DIR}/${split}.shard${shard}.log"
  local attempt

  for attempt in $(seq 1 "${OUTER_RETRIES}"); do
    echo "[clean_subset] split=${split} shard=${shard} attempt=${attempt} expected=${expected}"
    set +e
    python "${ROOT_DIR}/enrich_agentnet_lara_clean_with_vllm.py" \
      --steps "${input}" \
      --dataset-root "${DATASET_ROOT}" \
      --base-url "${url}" \
      --model "${MODEL_NAME}" \
      --out "${output}" \
      --summary-out "${summary}" \
      --progress-out "${progress}" \
      --shard-count 2 \
      --shard-index "${shard}" \
      --max-new-tokens 256 \
      --concurrency "${CONCURRENCY}" \
      --request-timeout 300 \
      --max-retries 5 \
      --save-every 10 \
      --log-every 20 \
      --resume 2>&1 | tee -a "${log}"
    local rc=${PIPESTATUS[0]}
    set -e

    if python - "${output}" "${expected}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected = int(sys.argv[2])
rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()] if path.exists() else []
ok = len(rows) == expected and all(row.get("enrich_status") == "ok" for row in rows)
raise SystemExit(0 if ok else 1)
PY
    then
      echo "[clean_subset] split=${split} shard=${shard} complete rows=${expected}"
      return 0
    fi
    echo "[clean_subset] split=${split} shard=${shard} incomplete rc=${rc}; retrying"
    sleep 10
  done
  echo "[clean_subset] ERROR: split=${split} shard=${shard} exhausted retries" >&2
  return 1
}

wait_for_server "${BASE_URL_0}"
wait_for_server "${BASE_URL_1}"

worker0() {
  run_enrichment_shard train 0 "${BASE_URL_0}" "${TRAIN_INPUT}" "$((TRAIN_SAMPLES / 2))"
  run_enrichment_shard test 0 "${BASE_URL_0}" "${TEST_INPUT}" "$((TEST_SAMPLES / 2))"
}

worker1() {
  run_enrichment_shard train 1 "${BASE_URL_1}" "${TRAIN_INPUT}" "$((TRAIN_SAMPLES - TRAIN_SAMPLES / 2))"
  run_enrichment_shard test 1 "${BASE_URL_1}" "${TEST_INPUT}" "$((TEST_SAMPLES - TEST_SAMPLES / 2))"
}

worker0 &
PID0=$!
worker1 &
PID1=$!
RC=0
wait "${PID0}" || RC=1
wait "${PID1}" || RC=1
if [[ "${RC}" -ne 0 ]]; then
  echo "[clean_subset] ERROR: at least one enrichment worker failed" >&2
  exit 1
fi

if [[ -f "${FINAL_DIR}/READY" ]]; then
  echo "[clean_subset] final subset already READY: ${FINAL_DIR}/READY"
  exit 0
fi
python "${ROOT_DIR}/merge_agentnet_clean_fixed_subset.py" \
  --train-shards \
    "${ENRICHED_DIR}/${PREFIX}.train.shard0.jsonl" \
    "${ENRICHED_DIR}/${PREFIX}.train.shard1.jsonl" \
  --test-shards \
    "${ENRICHED_DIR}/${PREFIX}.test.shard0.jsonl" \
    "${ENRICHED_DIR}/${PREFIX}.test.shard1.jsonl" \
  --selection-manifest "${SELECTION_MANIFEST}" \
  --dataset-root "${DATASET_ROOT}" \
  --out-dir "${FINAL_DIR}" \
  --prefix "${PREFIX}" \
  --overwrite

echo "[clean_subset] complete=$(date '+%F %T')"
echo "[clean_subset] train=${FINAL_DIR}/${PREFIX}.train.jsonl"
echo "[clean_subset] test=${FINAL_DIR}/${PREFIX}.test.jsonl"
