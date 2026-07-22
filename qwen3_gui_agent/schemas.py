from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _require(payload: dict[str, Any], key: str) -> Any:
    if key not in payload:
        raise ValueError(f"Missing required field: {key}")
    return payload[key]


@dataclass
class GUIAction:
    type: str
    reason: str
    expected_outcome: str
    x: int | None = None
    y: int | None = None
    text: str | None = None
    keys: list[str] | None = None
    amount: int | None = None
    seconds: float | None = None

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "GUIAction":
        return GUIAction(
            type=str(_require(payload, "type")),
            reason=str(_require(payload, "reason")),
            expected_outcome=str(_require(payload, "expected_outcome")),
            x=int(payload["x"]) if payload.get("x") is not None else None,
            y=int(payload["y"]) if payload.get("y") is not None else None,
            text=str(payload["text"]) if payload.get("text") is not None else None,
            keys=[str(k) for k in payload.get("keys", [])] or None,
            amount=int(payload["amount"]) if payload.get("amount") is not None else None,
            seconds=float(payload["seconds"]) if payload.get("seconds") is not None else None,
        )


@dataclass
class AgentDecision:
    current_subtask: str
    target_element: str
    temporal_anchor: str
    predicted_next_screen: str
    confidence: float
    action: GUIAction

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "AgentDecision":
        return AgentDecision(
            current_subtask=str(_require(payload, "current_subtask")),
            target_element=str(_require(payload, "target_element")),
            temporal_anchor=str(_require(payload, "temporal_anchor")),
            predicted_next_screen=str(_require(payload, "predicted_next_screen")),
            confidence=float(_require(payload, "confidence")),
            action=GUIAction.from_dict(_require(payload, "action")),
        )


@dataclass
class BootstrapPlan:
    semantic_anchors: list[str] = field(default_factory=list)
    stage_plan: list[str] = field(default_factory=list)
    success_signal: str = ""

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "BootstrapPlan":
        return BootstrapPlan(
            semantic_anchors=[str(x) for x in payload.get("semantic_anchors", [])],
            stage_plan=[str(x) for x in payload.get("stage_plan", [])],
            success_signal=str(payload.get("success_signal", "")),
        )
