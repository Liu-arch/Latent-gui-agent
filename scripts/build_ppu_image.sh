#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_IMAGE="${BASE_IMAGE:-reg.docker.alibaba-inc.com/aisw/llm:v1.6.1-pytorch2.6.0-ubuntu22.04-cuda12.6-vllm0.7.3-py310}"
IMAGE_TAG="${IMAGE_TAG:-latent-gui-agent:ppu-sdk1.6.1}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.org/simple}"

echo "[ppu-build] base=${BASE_IMAGE}"
echo "[ppu-build] target=${IMAGE_TAG}"

docker build \
  --network host \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  --build-arg "PIP_INDEX_URL=${PIP_INDEX_URL}" \
  --file "${ROOT_DIR}/docker/ppu.Dockerfile" \
  --tag "${IMAGE_TAG}" \
  "${ROOT_DIR}"

echo "[ppu-build] complete: ${IMAGE_TAG}"
