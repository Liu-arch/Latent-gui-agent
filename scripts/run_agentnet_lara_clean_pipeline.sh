#!/usr/bin/env bash
set -euo pipefail

CODE_DIR="${CODE_DIR:-/workspace/Latent-gui-agent}"
RAW_ROOT="${RAW_ROOT:-/workspace/storage/datasets/AgentNet}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/storage/datasets/AgentNet_processed/clean_v1}"
BASE_URL="${BASE_URL:-http://127.0.0.1:18000/v1}"
MODEL_NAME="${MODEL_NAME:-qwen3-vl-8b}"
CONCURRENCY="${CONCURRENCY:-4}"
SMOKE_SAMPLES="${SMOKE_SAMPLES:-100}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-384}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-300}"
REQUEST_RETRIES="${REQUEST_RETRIES:-5}"
PIPELINE_RETRIES="${PIPELINE_RETRIES:-12}"
PIPELINE_RETRY_SLEEP="${PIPELINE_RETRY_SLEEP:-10}"

RAW_JSONL="${RAW_ROOT}/agentnet_ubuntu_5k.jsonl"
STRUCTURAL_DIR="${OUTPUT_ROOT}/structural"
SMOKE_DIR="${OUTPUT_ROOT}/smoke${SMOKE_SAMPLES}"
FULL_DIR="${OUTPUT_ROOT}/full"
LOG_DIR="${OUTPUT_ROOT}/logs"
STRUCTURAL_JSONL="${STRUCTURAL_DIR}/agentnet_ubuntu_lara_clean_structural.jsonl"
UNSUPPORTED_JSONL="${STRUCTURAL_DIR}/agentnet_ubuntu_lara_clean_unsupported_actions.jsonl"
FULL_ENRICHED_JSONL="${FULL_DIR}/agentnet_ubuntu_lara_clean.enriched.jsonl"

mkdir -p "${STRUCTURAL_DIR}" "${SMOKE_DIR}" "${FULL_DIR}" "${LOG_DIR}"
export PYTHONUNBUFFERED=1

PIPELINE_LOG="${PIPELINE_LOG:-${OUTPUT_ROOT}/pipeline.live.log}"
touch "${PIPELINE_LOG}"
exec > >(tee -a "${PIPELINE_LOG}") 2>&1

echo "[agentnet-clean] start=$(date '+%F %T %z')"
echo "[agentnet-clean] live_log=${PIPELINE_LOG}"
echo "[agentnet-clean] raw=${RAW_JSONL}"
echo "[agentnet-clean] output_root=${OUTPUT_ROOT}"
echo "[agentnet-clean] base_url=${BASE_URL}"
echo "[agentnet-clean] concurrency=${CONCURRENCY}"
echo "[agentnet-clean] request_retries=${REQUEST_RETRIES} pipeline_retries=${PIPELINE_RETRIES}"

run_enrichment_with_resume() {
  local label="$1"
  local log_file="$2"
  shift 2
  local attempt=1

  while true; do
    echo "[agentnet-clean] ${label} enrichment attempt ${attempt}/${PIPELINE_RETRIES}"
    if python "${CODE_DIR}/enrich_agentnet_lara_clean_with_vllm.py" \
      "$@" \
      --request-timeout "${REQUEST_TIMEOUT}" \
      --max-retries "${REQUEST_RETRIES}" \
      --resume \
      2>&1 | tee -a "${log_file}"; then
      return 0
    fi
    if (( attempt >= PIPELINE_RETRIES )); then
      echo "[agentnet-clean] ${label} enrichment exhausted ${PIPELINE_RETRIES} attempts" >&2
      return 1
    fi
    attempt=$((attempt + 1))
    echo "[agentnet-clean] ${label} failed; retrying in ${PIPELINE_RETRY_SLEEP}s"
    sleep "${PIPELINE_RETRY_SLEEP}"
  done
}

if [[ ! -s "${RAW_JSONL}" ]]; then
  echo "Missing official AgentNet JSONL: ${RAW_JSONL}" >&2
  exit 2
fi

echo "[agentnet-clean] waiting for Qwen3-VL vLLM service"
ready=0
for _ in $(seq 1 180); do
  if curl -fsS "${BASE_URL}/models" >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 5
done
if [[ "${ready}" != "1" ]]; then
  echo "Qwen3-VL vLLM service did not become ready: ${BASE_URL}" >&2
  exit 3
fi
curl -fsS "${BASE_URL}/models"
echo

if [[ ! -s "${STRUCTURAL_JSONL}" || ! -f "${UNSUPPORTED_JSONL}" ]]; then
  echo "[agentnet-clean] building deterministic step-level data"
  python "${CODE_DIR}/prepare_agentnet_lara_clean.py" \
    --input "${RAW_JSONL}" \
    --dataset-root "${RAW_ROOT}" \
    --out "${STRUCTURAL_JSONL}" \
    --unsupported-out "${UNSUPPORTED_JSONL}" \
    --summary-out "${STRUCTURAL_DIR}/summary.json" \
    --img-next-count 16 \
    --overwrite \
    2>&1 | tee "${LOG_DIR}/prepare_structural.log"
else
  echo "[agentnet-clean] reuse structural data: ${STRUCTURAL_JSONL}"
fi

SMOKE_ENRICHED="${SMOKE_DIR}/enriched.jsonl"
echo "[agentnet-clean] running/resuming ${SMOKE_SAMPLES}-row smoke enrichment"
run_enrichment_with_resume "smoke" "${LOG_DIR}/enrich_smoke.log" \
  --steps "${STRUCTURAL_JSONL}" \
  --dataset-root "${RAW_ROOT}" \
  --base-url "${BASE_URL}" \
  --model "${MODEL_NAME}" \
  --out "${SMOKE_ENRICHED}" \
  --summary-out "${SMOKE_DIR}/enriched.summary.json" \
  --progress-out "${SMOKE_DIR}/enriched.progress.json" \
  --max-samples "${SMOKE_SAMPLES}" \
  --concurrency "${CONCURRENCY}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --save-every 20 \
  --log-every 20

python "${CODE_DIR}/finalize_agentnet_lara_clean.py" \
  --input "${SMOKE_ENRICHED}" \
  --dataset-root "${RAW_ROOT}" \
  --out-dir "${SMOKE_DIR}/final" \
  --prefix agentnet_ubuntu_lara_clean_smoke \
  --img-next-count 16 \
  --seed 42 \
  --overwrite \
  2>&1 | tee "${LOG_DIR}/finalize_smoke.log"

python - "${SMOKE_ENRICHED}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
print("[agentnet-clean] smoke examples")
with path.open("r", encoding="utf-8") as handle:
    for index, line in enumerate(handle):
        if index >= 3:
            break
        row = json.loads(line)
        print(json.dumps({
            "sample_id": row.get("sample_id"),
            "instruction": row.get("instruction"),
            "actual_task": row.get("actual_task"),
            "thought": row.get("thought"),
            "reflection": row.get("reflection"),
            "gold_action": row.get("gold_action"),
        }, ensure_ascii=False))
PY

echo "[agentnet-clean] smoke test passed; starting/resuming full enrichment"
run_enrichment_with_resume "full" "${LOG_DIR}/enrich_full.log" \
  --steps "${STRUCTURAL_JSONL}" \
  --dataset-root "${RAW_ROOT}" \
  --base-url "${BASE_URL}" \
  --model "${MODEL_NAME}" \
  --out "${FULL_ENRICHED_JSONL}" \
  --summary-out "${FULL_DIR}/enriched.summary.json" \
  --progress-out "${FULL_DIR}/enriched.progress.json" \
  --concurrency "${CONCURRENCY}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --save-every 20 \
  --log-every 100

echo "[agentnet-clean] validating and splitting full dataset"
python "${CODE_DIR}/finalize_agentnet_lara_clean.py" \
  --input "${FULL_ENRICHED_JSONL}" \
  --dataset-root "${RAW_ROOT}" \
  --out-dir "${FULL_DIR}/final" \
  --prefix agentnet_ubuntu_lara_clean \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --test-ratio 0.1 \
  --img-next-count 16 \
  --seed 42 \
  --overwrite \
  2>&1 | tee "${LOG_DIR}/finalize_full.log"

echo "[agentnet-clean] complete=$(date '+%F %T %z')"
echo "[agentnet-clean] final_dir=${FULL_DIR}/final"
