#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${1:-${ROOT_DIR}/environment_snapshots/cluster_${STAMP}}"

mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(cd "${OUTPUT_DIR}" && pwd)"

echo "[env-export] python=$(command -v python)"
echo "[env-export] output=${OUTPUT_DIR}"

python -m pip freeze --all > "${OUTPUT_DIR}/pip-freeze.full.txt"
python -m pip list --format=json > "${OUTPUT_DIR}/pip-list.json"
python "${ROOT_DIR}/scripts/make_portable_requirements.py" \
  "${OUTPUT_DIR}/pip-freeze.full.txt" \
  "${OUTPUT_DIR}/requirements.portable.txt"
python "${ROOT_DIR}/scripts/collect_runtime_info.py" \
  --output "${OUTPUT_DIR}/runtime-info.json" \
  > "${OUTPUT_DIR}/runtime-info.stdout.txt"

if python -m pip check > "${OUTPUT_DIR}/pip-check.txt" 2>&1; then
  echo "[env-export] pip check: PASS"
else
  echo "[env-export] pip check: WARN (see pip-check.txt)"
fi

{
  echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "hostname=$(hostname)"
  echo "uname=$(uname -a)"
  echo "shell=${SHELL:-unknown}"
  echo "python=$(command -v python)"
  echo "pip=$(python -m pip --version)"
} > "${OUTPUT_DIR}/system-info.txt"

if command -v conda >/dev/null 2>&1; then
  conda env export --no-builds > "${OUTPUT_DIR}/conda-env.full.yml"
  conda env export --from-history > "${OUTPUT_DIR}/conda-env.history.yml"
  conda list --explicit > "${OUTPUT_DIR}/conda-explicit.txt"
  conda info --json > "${OUTPUT_DIR}/conda-info.json"
else
  echo "[env-export] conda not found; skipped conda snapshots"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -q > "${OUTPUT_DIR}/nvidia-smi.txt" || true
fi

cp "${ROOT_DIR}/requirements.txt" "${OUTPUT_DIR}/requirements.project.txt"

ARCHIVE_PATH="${OUTPUT_DIR}.environment-snapshot.tar.gz"
if command -v tar >/dev/null 2>&1; then
  tar -czf "${ARCHIVE_PATH}" -C "$(dirname "${OUTPUT_DIR}")" "$(basename "${OUTPUT_DIR}")"
  echo "[env-export] archive=${ARCHIVE_PATH}"
fi

echo "[env-export] complete"
echo "[env-export] portable lock=${OUTPUT_DIR}/requirements.portable.txt"
