from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


REASONING_FIELDS = ("actual_task", "thought", "reflection")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize the clean Stage1/Stage2/action-head pipeline.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-csv", default=None)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def train_summary(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    epochs = [row for row in payload.get("history", []) if row.get("stage") == "train_epoch"]
    if not epochs:
        return {"available": False}
    return {
        "available": True,
        "epochs_ran": len(epochs),
        "first_avg_loss": float(epochs[0].get("avg_loss", 0.0)),
        "last_avg_loss": float(epochs[-1].get("avg_loss", 0.0)),
        "best_avg_loss": min(float(row.get("avg_loss", float("inf"))) for row in epochs),
        "best_epoch": int(payload.get("best_epoch", 0) or 0),
        "stopped_early": bool(payload.get("stopped_early", False)),
        "optimizer_steps": int(payload.get("optimizer_steps", 0) or 0),
    }


def train_epoch_rows(path: Path, stage_name: str) -> list[dict[str, Any]]:
    payload = load_json(path)
    rows: list[dict[str, Any]] = []
    for item in payload.get("history", []):
        if item.get("stage") != "train_epoch":
            continue
        rows.append(
            {
                "stage": stage_name,
                "epoch": int(item.get("epoch", 0) or 0),
                "avg_loss": item.get("avg_loss"),
                "avg_lm_loss": item.get("avg_lm_loss"),
                "avg_action_head_loss": item.get("avg_action_head_loss"),
                "monitor_value": item.get("monitor_value"),
                "best_monitor_value": item.get("best_monitor_value"),
                "improved": item.get("improved"),
            }
        )
    return rows


def parse_reasoning_fields(text: Any) -> dict[str, str]:
    fields = {key: "" for key in REASONING_FIELDS}
    current_key = ""
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        match = re.match(r"^(actual_task|thought|reflection):\s*(.*)$", line, flags=re.I)
        if match:
            current_key = match.group(1).lower()
            fields[current_key] = match.group(2).strip()
        elif current_key and line and not line.startswith("<") and not line.startswith("Action:"):
            fields[current_key] = (fields[current_key] + " " + line).strip()
    return fields


def token_f1(prediction: str, target: str) -> float | None:
    pred_tokens = re.findall(r"[a-z0-9_]+", prediction.lower())
    target_tokens = re.findall(r"[a-z0-9_]+", target.lower())
    if not target_tokens:
        return None
    if not pred_tokens:
        return 0.0
    overlap = sum((Counter(pred_tokens) & Counter(target_tokens)).values())
    precision = overlap / len(pred_tokens)
    recall = overlap / len(target_tokens)
    return 2.0 * precision * recall / max(1e-12, precision + recall)


def reasoning_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"available": False}
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        return {"available": False}

    presence = {key: 0 for key in REASONING_FIELDS}
    f1_values: dict[str, list[float]] = {key: [] for key in ("actual_task", "thought", "reflection")}
    latent_counts: list[int] = []
    for row in rows:
        pred_text = str(row.get("pred_reasoning_text", row.get("raw_response_text", "")) or "")
        gt_text = str(row.get("gt_reasoning", "") or "")
        pred_fields = parse_reasoning_fields(pred_text)
        gt_fields = parse_reasoning_fields(gt_text)
        for key in REASONING_FIELDS:
            presence[key] += int(bool(pred_fields[key]))
        for key in f1_values:
            value = token_f1(pred_fields[key], gt_fields[key])
            if value is not None:
                f1_values[key].append(value)
        raw = str(row.get("raw_response_text", "") or "")
        latent_counts.append(len(re.findall(r"<LATENT_\d+>|<\|thinking\|>", raw)))

    return {
        "available": True,
        "sample_count": len(rows),
        "field_presence_accuracy": {key: presence[key] / len(rows) for key in REASONING_FIELDS},
        "field_token_f1": {
            key: (sum(values) / len(values) if values else None) for key, values in f1_values.items()
        },
        "avg_generated_latent_token_count": sum(latent_counts) / len(latent_counts),
        "full_16_latent_scaffold_rate": sum(count >= 16 for count in latent_counts) / len(latent_counts),
    }


def eval_summary(report_path: Path, steps_path: Path) -> dict[str, Any]:
    report = load_json(report_path)
    if not report:
        return {"available": False}
    metrics = report.get("metrics", {})
    keys = (
        "action_type_accuracy",
        "region_accuracy",
        "pointer_exact_match_accuracy",
        "coord_hit_accuracy@0p01",
        "coord_hit_accuracy@0p03",
        "coord_hit_accuracy@0p05",
        "action_exact_match_with_coord_accuracy@0p01",
        "action_exact_match_with_coord_accuracy@0p03",
        "action_exact_match_with_coord_accuracy@0p05",
    )
    return {
        "available": True,
        "sample_count": int(report.get("sample_count", 0) or 0),
        "metrics": {key: metrics.get(key) for key in keys},
        "reasoning": reasoning_summary(steps_path),
    }


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    summary = {
        "run_dir": str(run_dir),
        "training": {
            "stage1_explicit": train_summary(run_dir / "stage1_explicit" / "report.json"),
            "stage2_transition": train_summary(run_dir / "stage2_transition" / "report.json"),
            "stage2_fully_latent": train_summary(run_dir / "stage2_fully_latent" / "report.json"),
            "action_head": train_summary(run_dir / "action_head" / "report.json"),
        },
        "evaluation": {},
    }
    eval_root = run_dir / "eval"
    if eval_root.exists():
        for eval_dir in sorted(path for path in eval_root.iterdir() if path.is_dir()):
            summary["evaluation"][eval_dir.name] = eval_summary(
                eval_dir / "report.json",
                eval_dir / "steps.jsonl",
            )

    out_json = Path(args.out_json) if args.out_json else run_dir / "pipeline_summary.json"
    out_csv = Path(args.out_csv) if args.out_csv else run_dir / "pipeline_summary.csv"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    flat_rows: list[dict[str, Any]] = []
    for name, payload in summary["training"].items():
        flat_rows.append({"section": "training", "name": name, **payload})
    for name, payload in summary["evaluation"].items():
        row: dict[str, Any] = {"section": "evaluation", "name": name, "sample_count": payload.get("sample_count")}
        row.update(payload.get("metrics", {}))
        flat_rows.append(row)
    fieldnames = sorted({key for row in flat_rows for key in row})
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)

    loss_rows: list[dict[str, Any]] = []
    for stage_name, relative_path in (
        ("stage1_explicit", Path("stage1_explicit/report.json")),
        ("stage2_transition", Path("stage2_transition/report.json")),
        ("stage2_fully_latent", Path("stage2_fully_latent/report.json")),
        ("action_head", Path("action_head/report.json")),
    ):
        loss_rows.extend(train_epoch_rows(run_dir / relative_path, stage_name))
    loss_csv = run_dir / "pipeline_loss_curves.csv"
    loss_fields = [
        "stage",
        "epoch",
        "avg_loss",
        "avg_lm_loss",
        "avg_action_head_loss",
        "monitor_value",
        "best_monitor_value",
        "improved",
    ]
    with loss_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=loss_fields)
        writer.writeheader()
        writer.writerows(loss_rows)

    loss_png: Path | None = None
    try:
        import matplotlib.pyplot as plt

        if loss_rows:
            figure, axis = plt.subplots(figsize=(10, 6))
            for stage_name in sorted({str(row["stage"]) for row in loss_rows}):
                stage_rows = [row for row in loss_rows if row["stage"] == stage_name]
                axis.plot(
                    [int(row["epoch"]) for row in stage_rows],
                    [float(row["avg_loss"]) for row in stage_rows],
                    marker="o",
                    label=stage_name,
                )
            axis.set_xlabel("Epoch")
            axis.set_ylabel("Average training loss")
            axis.set_title("LaRA GUI clean s100 convergence")
            axis.grid(alpha=0.25)
            axis.legend()
            figure.tight_layout()
            loss_png = run_dir / "pipeline_loss_curves.png"
            figure.savefig(loss_png, dpi=160)
            plt.close(figure)
    except Exception as exc:
        print(json.dumps({"stage": "loss_plot_skipped", "reason": str(exc)}, ensure_ascii=False))

    print(
        json.dumps(
            {
                "summary_json": str(out_json),
                "summary_csv": str(out_csv),
                "loss_csv": str(loss_csv),
                "loss_png": str(loss_png) if loss_png else None,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
