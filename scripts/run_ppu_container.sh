#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_TAG="${IMAGE_TAG:-latent-gui-agent:ppu-sdk1.6.1}"
PPU_VISIBLE_DEVICES="${PPU_VISIBLE_DEVICES:-0}"
SHM_SIZE="${SHM_SIZE:-64g}"

if [[ $# -eq 0 ]]; then
  set -- bash
fi

TTY_ARGS=()
if [[ -t 0 && -t 1 ]]; then
  TTY_ARGS=(-it)
fi

docker run --rm "${TTY_ARGS[@]}" \
  --privileged \
  --ipc=host \
  --network=host \
  --shm-size "${SHM_SIZE}" \
  --env "CUDA_VISIBLE_DEVICES=${PPU_VISIBLE_DEVICES}" \
  --env TOKENIZERS_PARALLELISM=false \
  --env LARA_ATTN_IMPLEMENTATION=eager \
  --volume "${ROOT_DIR}:/workspace/Latent-gui-agent" \
  --workdir /workspace/Latent-gui-agent \
  "${IMAGE_TAG}" \
  "$@"
