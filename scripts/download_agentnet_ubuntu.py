#!/usr/bin/env python3
"""Download, verify, and optionally extract the official AgentNet Ubuntu data."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from huggingface_hub import snapshot_download


REPO_ID = "xlangai/AgentNet"
DEFAULT_REVISION = "d76ee50a63fad81cfdbe576416757d7c2091ed50"
EXPECTED_FILES = {
    "agentnet_ubuntu_5k.jsonl": 282_313_437,
    **{
        f"ubuntu_images/images.z{index:02d}": 5_368_709_120
        for index in range(1, 14)
    },
    "ubuntu_images/images.zip": 3_727_419_649,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"),
    )
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--keep-merged-archive", action="store_true")
    return parser.parse_args()


def write_marker(target: Path, name: str, payload: dict[str, object]) -> None:
    marker = target / name
    temporary = marker.with_suffix(marker.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(marker)


def verify_download(target: Path) -> int:
    total = 0
    failures: list[str] = []
    for relative_path, expected_size in EXPECTED_FILES.items():
        path = target / relative_path
        if not path.is_file():
            failures.append(f"missing: {relative_path}")
            continue
        actual_size = path.stat().st_size
        total += actual_size
        if actual_size != expected_size:
            failures.append(
                f"size mismatch: {relative_path}: expected={expected_size}, actual={actual_size}"
            )
    if failures:
        raise RuntimeError("AgentNet verification failed:\n" + "\n".join(failures))
    return total


def extract_archives(target: Path, *, keep_merged_archive: bool) -> dict[str, object]:
    zip_binary = shutil.which("zip")
    unzip_binary = shutil.which("unzip")
    if not zip_binary or not unzip_binary:
        raise RuntimeError("Both zip and unzip are required to extract AgentNet")

    archive_dir = target / "ubuntu_images"
    split_archive = archive_dir / "images.zip"
    full_archive = archive_dir / "images-full.zip"
    partial_archive = archive_dir / "images-full.partial.zip"
    extract_marker = target / ".agentnet_ubuntu_extract.done.json"
    if extract_marker.is_file():
        return json.loads(extract_marker.read_text(encoding="utf-8"))

    if full_archive.is_file():
        archive_check = subprocess.run(
            [unzip_binary, "-tq", str(full_archive)],
            check=False,
            capture_output=True,
        )
        if archive_check.returncode != 0:
            full_archive.unlink()

    if not full_archive.is_file():
        partial_archive.unlink(missing_ok=True)
        subprocess.run(
            [
                zip_binary,
                "-s",
                "0",
                str(split_archive),
                "--out",
                str(partial_archive),
            ],
            check=True,
        )
        partial_archive.replace(full_archive)
    subprocess.run([unzip_binary, "-tq", str(full_archive)], check=True)

    listing = subprocess.run(
        [unzip_binary, "-Z1", str(full_archive)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    first_member = next((name for name in listing if name.strip()), "")
    destination = target if first_member.startswith("ubuntu_images/") else archive_dir
    subprocess.run(
        [unzip_binary, "-q", "-n", str(full_archive), "-d", str(destination)],
        check=True,
    )
    image_count = sum(
        1
        for suffix in ("*.png", "*.jpg", "*.jpeg", "*.webp")
        for _ in archive_dir.rglob(suffix)
    )
    payload: dict[str, object] = {
        "status": "complete",
        "archive": str(full_archive),
        "destination": str(destination),
        "first_member": first_member,
        "archive_member_count": len(listing),
        "extracted_image_count": image_count,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    if not keep_merged_archive:
        full_archive.unlink()
        payload["merged_archive_removed"] = True
    write_marker(target, extract_marker.name, payload)
    return payload


def main() -> None:
    args = parse_args()
    target = args.target.resolve()
    target.mkdir(parents=True, exist_ok=True)
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    running_marker = target / ".agentnet_ubuntu_download.running.json"
    write_marker(
        target,
        running_marker.name,
        {
            "status": "running",
            "repo_id": REPO_ID,
            "revision": args.revision,
            "endpoint": args.endpoint,
            "started_at": started_at,
        },
    )
    try:
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            revision=args.revision,
            local_dir=target,
            allow_patterns=[
                "README.md",
                "LICENSE.txt",
                "agentnet_ubuntu_5k.jsonl",
                "ubuntu_images/*",
            ],
            endpoint=args.endpoint,
        )
        total_bytes = verify_download(target)
        download_payload: dict[str, object] = {
            "status": "complete",
            "repo_id": REPO_ID,
            "revision": args.revision,
            "verified_file_count": len(EXPECTED_FILES),
            "verified_total_bytes": total_bytes,
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        write_marker(target, ".agentnet_ubuntu_download.done.json", download_payload)
        if args.extract:
            download_payload["extraction"] = extract_archives(
                target,
                keep_merged_archive=bool(args.keep_merged_archive),
            )
        running_marker.unlink(missing_ok=True)
        print(json.dumps(download_payload, ensure_ascii=True))
    except BaseException as exc:
        write_marker(
            target,
            ".agentnet_ubuntu_download.failed.json",
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "failed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            },
        )
        running_marker.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    main()
