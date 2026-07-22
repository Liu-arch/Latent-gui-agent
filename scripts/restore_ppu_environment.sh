#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SNAPSHOT_DIR="${1:-${ROOT_DIR}/environment/cluster_qwen3_agentnet}"
RESTORE_MODE="${RESTORE_MODE:-project}"
PPU_RUNTIME_INFO="${PPU_RUNTIME_INFO:-${ROOT_DIR}/environment_snapshots/runtime-info.ppu.json}"

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
mkdir -p "$(dirname "${PPU_RUNTIME_INFO}")"
python "${ROOT_DIR}/scripts/collect_runtime_info.py" \
  --output "${PPU_RUNTIME_INFO}"

echo "[ppu-restore] environment restore complete"
echo "[ppu-restore] PPU runtime info=${PPU_RUNTIME_INFO}"
echo "[ppu-restore] compare it with ${SNAPSHOT_DIR}/runtime-info.public.json"
