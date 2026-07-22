#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_TAG="${IMAGE_TAG:-latent-gui-agent:ppu-sdk1.6.1}"
CONTAINER_NAME="${CONTAINER_NAME:-latent-gui-agent-dev}"
PPU_VISIBLE_DEVICES="${PPU_VISIBLE_DEVICES:-0,1}"
STORAGE_DIR="${STORAGE_DIR:-/data2/liuenqi/latent_gui_agent}"
SHM_SIZE="${SHM_SIZE:-64g}"

if [[ ! -d "${STORAGE_DIR}" ]]; then
  echo "Storage directory does not exist: ${STORAGE_DIR}" >&2
  exit 2
fi

if docker container inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
  if [[ "$(docker inspect -f '{{.State.Running}}' "${CONTAINER_NAME}")" != "true" ]]; then
    docker start "${CONTAINER_NAME}" >/dev/null
  fi
  echo "[ppu-dev] existing container is running: ${CONTAINER_NAME}"
else
  docker run --detach \
    --name "${CONTAINER_NAME}" \
    --privileged \
    --ipc=host \
    --network=host \
    --shm-size "${SHM_SIZE}" \
    --env "CUDA_VISIBLE_DEVICES=${PPU_VISIBLE_DEVICES}" \
    --env TOKENIZERS_PARALLELISM=false \
    --env HF_HOME=/workspace/storage/cache/huggingface \
    --volume "${ROOT_DIR}:/workspace/Latent-gui-agent" \
    --volume "${STORAGE_DIR}:/workspace/storage" \
    --workdir /workspace/Latent-gui-agent \
    "${IMAGE_TAG}" \
    sleep infinity >/dev/null
  echo "[ppu-dev] created container: ${CONTAINER_NAME}"
fi

echo "[ppu-dev] devices=${PPU_VISIBLE_DEVICES}"
echo "[ppu-dev] storage=${STORAGE_DIR}"
echo "[ppu-dev] enter with: docker exec -it ${CONTAINER_NAME} bash"
