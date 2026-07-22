#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SNAPSHOT_DIR="${1:-}"
RESTORE_MODE="${RESTORE_MODE:-project}"

if [[ -z "${SNAPSHOT_DIR}" ]]; then
  echo "Usage: bash scripts/restore_ppu_environment.sh /path/to/environment_snapshot" >&2
  exit 2
fi

if [[ ! -f "${SNAPSHOT_DIR}/requirements.portable.txt" ]]; then
  echo "Missing ${SNAPSHOT_DIR}/requirements.portable.txt" >&2
  exit 2
fi

if ! python -c "import torch; print(torch.__version__)"; then
  cat >&2 <<'EOF'
PyTorch is not importable. Install the PPU vendor's matching PyTorch runtime first.
Do not install the source cluster's CUDA torch wheel on the PPU machine.
EOF
  exit 3
fi

echo "[ppu-restore] installing accelerator-neutral packages mode=${RESTORE_MODE}"
case "${RESTORE_MODE}" in
  project)
    python -m pip install -r "${ROOT_DIR}/requirements.txt"
    ;;
  locked)
    python -m pip install -r "${SNAPSHOT_DIR}/requirements.portable.txt"
    ;;
  *)
    echo "RESTORE_MODE must be 'project' or 'locked', got: ${RESTORE_MODE}" >&2
    exit 2
    ;;
esac
python -m pip install -e "${ROOT_DIR}"
python -m pip check
python "${ROOT_DIR}/scripts/collect_runtime_info.py" \
  --output "${SNAPSHOT_DIR}/runtime-info.ppu.json"

echo "[ppu-restore] environment restore complete"
echo "[ppu-restore] compare runtime-info.json with runtime-info.ppu.json"
