from __future__ import annotations

from scripts.make_portable_requirements import filter_requirements


def test_filter_requirements_removes_accelerator_and_local_packages() -> None:
    frozen = [
        "torch==2.7.0",
        "torchvision==0.22.0",
        "nvidia-cublas-cu12==12.6.4.1",
        "triton==3.3.0",
        "flash-attn==2.7.4",
        "transformers==4.57.1",
        "numpy==2.0.2",
        "local-project @ file:///workspace/local-project",
        "-e file:///workspace/editable-project",
    ]

    assert filter_requirements(frozen) == [
        "numpy==2.0.2",
        "transformers==4.57.1",
    ]


def test_filter_requirements_is_sorted_and_deduplicated() -> None:
    frozen = ["tqdm==4.67.0", "Pillow==11.0.0", "tqdm==4.67.0"]

    assert filter_requirements(frozen) == ["Pillow==11.0.0", "tqdm==4.67.0"]
