from __future__ import annotations

from typing import Any


ACTION_TYPES = frozenset(
    {
        "click",
        "double_click",
        "right_click",
        "type",
        "hotkey",
        "scroll",
        "wait",
        "terminate",
    }
)
POINTER_ACTION_TYPES = frozenset({"click", "double_click", "right_click"})
TERMINATE_STATUSES = frozenset({"success", "failure"})
HARD_LM_ACTION_TYPES = frozenset({"type", "hotkey"})


def _as_action_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        payload = value.to_dict()
        return dict(payload) if isinstance(payload, dict) else {}
    return {}


def _action_type(action: dict[str, Any]) -> str | None:
    value = str(action.get("type", "") or "").strip().lower()
    return value or None


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result != result or result in {float("inf"), float("-inf")}:
        return None
    return result


def route_typed_hybrid_action(
    *,
    head_action: Any,
    lm_action: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fuse an action-head decision with LM-only parameters using hard type gates.

    The action head owns the action type. The LM may only fill parameters that the
    head cannot predict (currently text and hotkey keys), and only when both models
    agree on the action type. Parameters from all inactive branches are discarded.
    """

    head = _as_action_dict(head_action)
    lm = _as_action_dict(lm_action)
    raw_head_type = _action_type(head)
    lm_type = _action_type(lm)
    head_type = raw_head_type if raw_head_type in ACTION_TYPES else "wait"

    action: dict[str, Any] = {"type": head_type}
    parameter_source = "action_head"
    parameter_valid = True
    parameter_error: str | None = None

    if raw_head_type not in ACTION_TYPES:
        parameter_valid = False
        parameter_error = f"unsupported_head_action_type:{raw_head_type or 'missing'}"
    elif head_type in POINTER_ACTION_TYPES:
        x_norm = _finite_float(head.get("x_norm"))
        y_norm = _finite_float(head.get("y_norm"))
        if x_norm is None or y_norm is None:
            parameter_valid = False
            parameter_error = "missing_pointer_coordinates"
        else:
            action["x_norm"] = round(max(0.0, min(1.0, x_norm)), 4)
            action["y_norm"] = round(max(0.0, min(1.0, y_norm)), 4)
            region = str(head.get("region", "") or "").strip()
            if region:
                action["region"] = region
    elif head_type == "scroll":
        amount = _finite_float(head.get("amount"))
        if amount is None:
            parameter_valid = False
            parameter_error = "missing_scroll_amount"
        else:
            action["amount"] = int(max(-1000.0, min(1000.0, amount)))
    elif head_type == "terminate":
        status = str(head.get("status", "") or "").strip().lower()
        if status not in TERMINATE_STATUSES:
            parameter_valid = False
            parameter_error = "missing_or_invalid_terminate_status"
        else:
            action["status"] = status
    elif head_type == "wait":
        status = str(head.get("status", "") or "").strip().lower()
        if status in TERMINATE_STATUSES:
            action["status"] = status
    elif head_type == "type":
        parameter_source = "lm_same_type"
        text = str(lm.get("text", "") or "")
        if lm_type != "type":
            parameter_valid = False
            parameter_error = f"lm_action_type_mismatch:{lm_type or 'missing'}"
        elif not text:
            parameter_valid = False
            parameter_error = "missing_type_text"
        else:
            action["text"] = text
    elif head_type == "hotkey":
        parameter_source = "lm_same_type"
        raw_keys = lm.get("keys")
        keys = (
            [str(key).strip() for key in raw_keys if str(key).strip()]
            if isinstance(raw_keys, (list, tuple))
            else []
        )
        if lm_type != "hotkey":
            parameter_valid = False
            parameter_error = f"lm_action_type_mismatch:{lm_type or 'missing'}"
        elif not keys:
            parameter_valid = False
            parameter_error = "missing_hotkey_keys"
        else:
            action["keys"] = keys

    diagnostics = {
        "hybrid_head_action_type": head_type,
        "hybrid_lm_action_type": lm_type,
        "hybrid_action_type_agreement": None if lm_type is None else lm_type == head_type,
        "hybrid_parameter_source": parameter_source,
        "hybrid_parameter_valid": parameter_valid,
        "hybrid_parameter_error": parameter_error,
        "execution_allowed": parameter_valid,
    }
    return action, diagnostics


def route_selective_hybrid_action(
    *,
    head_action: Any,
    lm_action: Any,
    hard_action_policy: str = "parameters_only",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Route easy actions to the head and optionally let the LM own hard actions.

    ``parameters_only`` preserves the strict typed router: the head owns the
    action type and the LM may only provide text or hotkey keys. ``full_action``
    uses the head as a cheap gate, but lets the LM own the complete action when
    the head predicts a variable-length action type.
    """

    policy = str(hard_action_policy or "parameters_only").strip().lower()
    if policy not in {"parameters_only", "full_action"}:
        raise ValueError(f"Unsupported hard action policy: {hard_action_policy!r}")

    head = _as_action_dict(head_action)
    head_type = _action_type(head)
    lm = _as_action_dict(lm_action)
    lm_type = _action_type(lm)
    lm_invoked = head_type in HARD_LM_ACTION_TYPES

    if policy == "full_action" and lm_invoked:
        if not lm:
            return (
                {"type": head_type},
                {
                    "hybrid_head_action_type": head_type,
                    "hybrid_lm_action_type": None,
                    "hybrid_action_type_agreement": None,
                    "hybrid_parameter_source": "lm_full_action",
                    "hybrid_parameter_valid": False,
                    "hybrid_parameter_error": "missing_lm_full_action",
                    "execution_allowed": False,
                    "selective_hard_action_policy": policy,
                    "selective_lm_owns_action": True,
                },
            )

        # Passing the LM action as both arguments reuses the typed sanitizer for
        # every possible full action while allowing the LM to change the type.
        action, diagnostics = route_typed_hybrid_action(head_action=lm, lm_action=lm)
        diagnostics.update(
            {
                "hybrid_head_action_type": head_type,
                "hybrid_lm_action_type": lm_type,
                "hybrid_action_type_agreement": lm_type == head_type,
                "hybrid_parameter_source": "lm_full_action",
                "selective_hard_action_policy": policy,
                "selective_lm_owns_action": True,
            }
        )
        return action, diagnostics

    action, diagnostics = route_typed_hybrid_action(head_action=head, lm_action=lm)
    diagnostics.update(
        {
            "selective_hard_action_policy": policy,
            "selective_lm_owns_action": False,
        }
    )
    return action, diagnostics
