#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
import importlib.metadata as metadata

import torch
from transformers import Qwen3VLForConditionalGeneration

assert torch.cuda.is_available(), "The PPU CUDA-compatible runtime is unavailable"
assert torch.cuda.device_count() >= 1, "No PPU is visible"

values = torch.arange(8, dtype=torch.float32, device="cuda:0")
assert float(values.sum().cpu()) == 28.0

print(f"torch={torch.__version__}")
print(f"transformers={metadata.version('transformers')}")
print(f"device={torch.cuda.get_device_name(0)}")
print("Qwen3VLForConditionalGeneration=PASS")
print("PPU tensor operation=PASS")
PY

python -m pip check
python -m pytest -q
python -m py_compile \
  train_lara_style_qwen3vl.py \
  train_lara_style_qwen3vl_active_batch.py \
  train_lara_style_qwen3vl_active_batch_ddp.py \
  evaluate_lara_style_qwen3vl_batched.py

echo "[ppu-smoke] all checks passed"
