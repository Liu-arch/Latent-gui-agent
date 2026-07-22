from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any


TRACKED_PACKAGES = (
    "torch",
    "torchvision",
    "torchaudio",
    "transformers",
    "accelerate",
    "peft",
    "qwen-vl-utils",
    "safetensors",
    "numpy",
    "Pillow",
    "einops",
    "sentencepiece",
    "tiktoken",
    "tqdm",
)

VENDOR_MODULES = (
    "torch_npu",
    "torch_mlu",
    "torch_musa",
    "torch_xla",
)


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in TRACKED_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def callable_bool(obj: Any, name: str) -> bool | None:
    value = getattr(obj, name, None)
    if not callable(value):
        return None
    try:
        return bool(value())
    except Exception:
        return None


def torch_runtime() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {"imported": False, "error": f"{type(exc).__name__}: {exc}"}

    runtime: dict[str, Any] = {
        "imported": True,
        "version": str(torch.__version__),
        "cuda_build_version": getattr(torch.version, "cuda", None),
        "hip_build_version": getattr(torch.version, "hip", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "distributed_available": bool(torch.distributed.is_available()),
        "vendor_modules": {
            module: importlib.util.find_spec(module) is not None for module in VENDOR_MODULES
        },
    }
    if torch.cuda.is_available():
        runtime["cuda_devices"] = [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "capability": list(torch.cuda.get_device_capability(index)),
                "total_memory_bytes": int(torch.cuda.get_device_properties(index).total_memory),
            }
            for index in range(torch.cuda.device_count())
        ]
        try:
            runtime["cudnn_version"] = torch.backends.cudnn.version()
        except Exception:
            runtime["cudnn_version"] = None

    dist = torch.distributed
    runtime["distributed_backends"] = {
        backend: callable_bool(dist, checker)
        for backend, checker in (
            ("nccl", "is_nccl_available"),
            ("gloo", "is_gloo_available"),
            ("mpi", "is_mpi_available"),
            ("ucc", "is_ucc_available"),
        )
    }
    return runtime


def build_report() -> dict[str, Any]:
    return {
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "conda": {
            "prefix": os.environ.get("CONDA_PREFIX"),
            "default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        },
        "packages": package_versions(),
        "torch": torch_runtime(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture accelerator-safe runtime metadata.")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
