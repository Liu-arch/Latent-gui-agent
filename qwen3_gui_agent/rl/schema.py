from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import json


@dataclass
class ReasoningCandidate:
    candidate_id: str
    compression_level: int
    source: str
    tokens: list[str]
    decoded_action: dict[str, Any]
    reward: float
    reward_breakdown: dict[str, float]


@dataclass
class RLSample:
    sample_id: str
    task: str
    before_screenshot: str
    after_screenshot: str
    temporal_anchor: str
    current_subtask: str
    explicit_reasoning: str
    predicted_next_screen_desc: str
    semantic_anchors: list[dict[str, Any]] = field(default_factory=list)
    gold_action: dict[str, Any] = field(default_factory=dict)
    candidates: list[ReasoningCandidate] = field(default_factory=list)
    best_candidate_id: str = ""


@dataclass
class ScreenInfo:
    width: int
    height: int


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_coerce(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_coerce(payload), ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[Any]:
    rows: list[Any] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _coerce(payload: Any) -> Any:
    if hasattr(payload, "__dataclass_fields__"):
        return asdict(payload)
    return payload
