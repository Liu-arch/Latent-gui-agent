from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable


ACCELERATOR_PACKAGES = {
    "apex",
    "bitsandbytes",
    "deepspeed",
    "flash-attn",
    "flashinfer-python",
    "intel-extension-for-pytorch",
    "pynvml",
    "pytorch-triton",
    "torch",
    "torch-mlu",
    "torch-musa",
    "torch-npu",
    "torchaudio",
    "torchvision",
    "triton",
    "vllm",
    "xformers",
}

ACCELERATOR_PREFIXES = (
    "cuda-",
    "cupy-",
    "cupy-cuda",
    "nvidia-",
)


def canonicalize_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def requirement_name(line: str) -> str | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if stripped.startswith(("-e ", "--editable ")):
        return None
    match = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)", stripped)
    return canonicalize_name(match.group(1)) if match else None


def is_portable_requirement(line: str) -> bool:
    stripped = line.strip()
    name = requirement_name(stripped)
    if name is None:
        return False
    if " @ file:" in stripped or " @ git+file:" in stripped:
        return False
    if name in ACCELERATOR_PACKAGES:
        return False
    return not any(name.startswith(prefix) for prefix in ACCELERATOR_PREFIXES)


def filter_requirements(lines: Iterable[str]) -> list[str]:
    portable = {
        line.strip()
        for line in lines
        if is_portable_requirement(line)
    }
    return sorted(portable, key=str.lower)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove accelerator-specific and local-path packages from pip freeze output."
    )
    parser.add_argument("input", type=Path, help="pip freeze input file")
    parser.add_argument("output", type=Path, help="portable requirements output file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lines = args.input.read_text(encoding="utf-8").splitlines()
    portable = filter_requirements(lines)
    header = [
        "# Generated from the source environment by make_portable_requirements.py.",
        "# Install the target accelerator vendor's PyTorch build before this file.",
        "# Known accelerator-bound packages such as CUDA, torch, vLLM, and FlashInfer are omitted.",
        "",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(header + portable) + "\n", encoding="utf-8")
    print(f"portable_requirements={args.output} packages={len(portable)}")


if __name__ == "__main__":
    main()
