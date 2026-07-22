# Trajectory Latent GUI Agent

Research code for a Qwen3-VL GUI agent with:

- trajectory-conditioned multimodal training with up to five GUI frames;
- field-aligned latent reasoning (`actual_task`, `thought`, `reflection` = `6/5/5` slots);
- progressive explicit-to-latent Stage 2 curriculum;
- direct and flow-matching action heads;
- latent Two-Way visual grounding with patch-level coordinate supervision;
- pixel pruning and temporal patch reuse;
- active-trajectory batching, DDP, resumable checkpoints, and batched evaluation.

This repository intentionally excludes datasets, screenshots, model weights, checkpoints, logs, plots, PDFs, and obsolete ablation launchers.

## Layout

```text
qwen3_gui_agent/                         Core model, pruning, latent and action-head modules
train_lara_style_qwen3vl.py              Reference single-sample trainer
train_lara_style_qwen3vl_active_batch.py Active-trajectory batch trainer
train_lara_style_qwen3vl_active_batch_ddp.py  Multi-device DDP trainer
evaluate_lara_style_qwen3vl_batched.py   Batched generate/head/hybrid evaluation
run_lara_clean_s100_full_pipeline.sh     Resumable Stage 1 -> Stage 2 -> Action Head pipeline
scripts/run_field_aligned_pipeline.sh    Portable field-aligned launcher
tests/                                   Focused regression tests
```

## Environment

Use Python 3.10 or 3.11. Install the PyTorch build supplied for the target accelerator first, then install the remaining dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

The checked-in DDP runtime currently uses the standard PyTorch CUDA/NCCL interface. A PPU environment works without code changes only when its vendor runtime exposes a CUDA-compatible PyTorch device. For a native PPU device/backend, set up the vendor PyTorch extension first and adapt `qwen3_gui_agent/distributed_training.py` to its device name and collective backend.

### Export the source cluster environment

Do not migrate with `pip freeze` alone: it contains source-machine CUDA and compiled packages. Export both a forensic snapshot and an accelerator-neutral lock instead:

```bash
conda activate qwen3_agentnet
bash scripts/export_cluster_environment.sh /path/to/agentnet_environment_snapshot
```

The snapshot contains full pip/conda records, runtime and GPU metadata, and `requirements.portable.txt`. The portable file intentionally removes PyTorch, CUDA/NVIDIA packages, Triton, vLLM, FlashAttention, and local editable paths.

On the PPU machine, create a Python 3.10 environment and install the PPU vendor's PyTorch runtime first. Then restore the accelerator-neutral packages:

```bash
git clone https://github.com/Liu-arch/Latent-gui-agent.git
cd Latent-gui-agent
bash scripts/restore_ppu_environment.sh
```

For the Alibaba PPU-ZW810E host, the tested path is the cached SDK 1.6.1
container. It preserves the vendor's PyTorch 2.6.0 build and installs only
Python-level project dependencies:

```bash
sudo env IMAGE_TAG=latent-gui-agent:ppu-sdk1.6.1 \
  bash scripts/build_ppu_image.sh

sudo env PPU_VISIBLE_DEVICES=0 \
  bash scripts/run_ppu_container.sh bash scripts/smoke_test_ppu.sh

sudo env PPU_VISIBLE_DEVICES=0,1 \
  bash scripts/run_ppu_container.sh \
  env PPU_WORLD_SIZE=2 bash scripts/smoke_test_ppu_ddp.sh
```

Start a persistent development container after the checks pass. The container
itself sleeps without allocating PPU memory, while model/data/checkpoint files
live on the large `/data2` filesystem:

```bash
sudo env PPU_VISIBLE_DEVICES=0,1 \
  STORAGE_DIR=/data2/liuenqi/latent_gui_agent \
  bash scripts/start_ppu_dev_container.sh

sudo docker exec -it latent-gui-agent-dev bash
```

The verified package set is in `requirements-ppu.txt`. In particular, it pins
`transformers==4.57.1` for Qwen3-VL support and `numpy==1.25.2` to remain
compatible with the SDK image's vLLM 0.7.3 and SciPy 1.9.3. Do not install a
PyPI CUDA `torch` wheel into this image.

If a vendor Python environment already exists outside Docker, use the same
locked package set after confirming that its PyTorch can see the PPU:

```bash
RESTORE_MODE=ppu bash scripts/restore_ppu_environment.sh
```

The repository includes a sanitized source-environment manifest under `environment/cluster_qwen3_agentnet`. The default `RESTORE_MODE=project` uses the compatible ranges in `requirements.txt`. After that works, use `RESTORE_MODE=locked` to reproduce the source cluster's core package versions:

```bash
RESTORE_MODE=locked bash scripts/restore_ppu_environment.sh
```

Compare `environment/cluster_qwen3_agentnet/runtime-info.public.json` with `environment_snapshots/runtime-info.ppu.json` before training. The full tar snapshot remains private because it contains hostnames, internal filesystem paths, and the entire unrelated package inventory.

## Expected Data

Each line is one ordered trajectory step. The trainer consumes these principal fields:

```json
{
  "sample_id": "trajectory-id_step_0001",
  "instruction": "High-level GUI task",
  "before_screenshot": "current.png",
  "after_screenshot": "next.png",
  "actual_task": "Current fine-grained subtask",
  "thought": "Why this action should be taken",
  "reflection": "Observed execution feedback",
  "gold_action": {"type": "click", "x_norm": 0.42, "y_norm": 0.31},
  "img_next": ["<img next>"]
}
```

Coordinates are normalized to `[0, 1]`. Images are resolved below `DATASET_ROOT/ubuntu_images` or `DATASET_ROOT/win_mac_images`.

## Run the Full Pipeline

```bash
export MODEL=/models/Qwen3-VL-8B
export DATASET_ROOT=/datasets/agentnet
export SOURCE_TRAIN=/datasets/agentnet/train.jsonl
export SOURCE_TEST=/datasets/agentnet/test.jsonl
export CUDA_VISIBLE_DEVICES=0,1
export TRAIN_WORLD_SIZE=2
export TRAIN_SAMPLES=100
export TEST_SAMPLES=100
bash scripts/run_field_aligned_pipeline.sh
```

The pipeline runs:

1. Stage 1 explicit reasoning and action SFT.
2. Stage 2 transition with progressive field replacement.
3. Stage 2 fully latent convergence using `6/5/5` role-aligned slots.
4. Action-head training initialized from fully latent Stage 2.
5. Train-set overfit evaluation, with optional held-out evaluation.

Every stage writes `latest.ckpt.pt`, `best.ckpt.pt`, a report, and a `done` marker. Re-run with the same `RUN_NAME` and `OUT_ROOT` to resume. To reuse a completed Stage 1 checkpoint:

```bash
export STAGE1_REUSE_ADAPTER=/path/to/stage1_explicit/best.ckpt.pt
```

Run held-out evaluation after the overfit diagnostics pass:

```bash
export RUN_HELDOUT_EVAL=1
bash scripts/run_field_aligned_pipeline.sh
```

## Tests

```bash
python -m pytest -q
python -m py_compile train_lara_style_qwen3vl.py \
  train_lara_style_qwen3vl_active_batch.py \
  train_lara_style_qwen3vl_active_batch_ddp.py \
  evaluate_lara_style_qwen3vl_batched.py
```

## Research Status

This is research code. Validate train-set overfitting and per-action coordinate metrics before scaling. In particular, compare LM generation, direct action-head prediction, and selective hybrid routing on the same fixed trajectory split.
