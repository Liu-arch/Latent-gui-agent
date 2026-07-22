from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


ACTION_TYPE_TO_TOKEN = {
    "click": "<ACT_CLICK>",
    "double_click": "<ACT_DOUBLE_CLICK>",
    "right_click": "<ACT_RIGHT_CLICK>",
    "type": "<ACT_TYPE>",
    "hotkey": "<ACT_HOTKEY>",
    "scroll": "<ACT_SCROLL>",
    "wait": "<ACT_WAIT>",
    "terminate": "<ACT_TERMINATE>",
}
TOKEN_TO_ACTION_TYPE = {token: action for action, token in ACTION_TYPE_TO_TOKEN.items()}

STATUS_TO_TOKEN = {
    "success": "<STATUS_SUCCESS>",
    "failure": "<STATUS_FAILURE>",
}
TOKEN_TO_STATUS = {token: status for status, token in STATUS_TO_TOKEN.items()}

SCROLL_TO_TOKEN = {
    "up": "<SCROLL_UP>",
    "down": "<SCROLL_DOWN>",
    "zero": "<SCROLL_ZERO>",
}
TOKEN_TO_SCROLL = {token: direction for direction, token in SCROLL_TO_TOKEN.items()}

TEXT_START_TOKEN = "<TEXT_START>"
TEXT_END_TOKEN = "<TEXT_END>"
KEY_UNKNOWN_TOKEN = "<KEY_UNKNOWN>"

COMMON_KEYS = [
    "ctrl",
    "control",
    "shift",
    "alt",
    "meta",
    "cmd",
    "command",
    "super",
    "enter",
    "return",
    "esc",
    "escape",
    "tab",
    "space",
    "backspace",
    "delete",
    "insert",
    "home",
    "end",
    "pageup",
    "pagedown",
    "up",
    "down",
    "left",
    "right",
]
COMMON_KEYS.extend([chr(code) for code in range(ord("a"), ord("z") + 1)])
COMMON_KEYS.extend([str(index) for index in range(10)])
COMMON_KEYS.extend([f"f{index}" for index in range(1, 13)])


def _key_to_token_name(key: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", str(key).strip()).strip("_")
    if not cleaned:
        return "UNKNOWN"
    return cleaned.upper()


KEY_TO_TOKEN = {key: f"<KEY_{_key_to_token_name(key)}>" for key in COMMON_KEYS}
TOKEN_TO_KEY = {token: key for key, token in KEY_TO_TOKEN.items()}


@dataclass(frozen=True)
class DecodedActionTokens:
    action: dict[str, Any]
    raw_text: str


class GUIActionTokenizer:
    """
    Discrete GUI action tokenizer for native LM generation.

    Pointer actions become compact action tokens, for example:
    <ACT_CLICK> <X_512> <Y_288>

    This keeps the action in the language-model output stream, but removes
    fragile free-form numeric formatting from the supervised target.
    """

    def __init__(self, coord_bins: int = 1000) -> None:
        if coord_bins < 2:
            raise ValueError("coord_bins must be >= 2")
        self.coord_bins = int(coord_bins)
        self.coord_width = len(str(self.coord_bins - 1))

    @property
    def special_tokens(self) -> list[str]:
        x_tokens = [self.x_token(index) for index in range(self.coord_bins)]
        y_tokens = [self.y_token(index) for index in range(self.coord_bins)]
        return (
            list(ACTION_TYPE_TO_TOKEN.values())
            + list(STATUS_TO_TOKEN.values())
            + list(SCROLL_TO_TOKEN.values())
            + [TEXT_START_TOKEN, TEXT_END_TOKEN, KEY_UNKNOWN_TOKEN]
            + list(dict.fromkeys(KEY_TO_TOKEN.values()))
            + x_tokens
            + y_tokens
        )

    def x_token(self, index: int) -> str:
        return f"<X_{int(index):0{self.coord_width}d}>"

    def y_token(self, index: int) -> str:
        return f"<Y_{int(index):0{self.coord_width}d}>"

    def _coord_to_bin(self, value: Any) -> int:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            parsed = 0.5
        if parsed != parsed:
            parsed = 0.5
        parsed = min(1.0, max(0.0, parsed))
        return int(round(parsed * (self.coord_bins - 1)))

    def _bin_to_coord(self, index: int) -> float:
        index = min(self.coord_bins - 1, max(0, int(index)))
        return round(index / float(self.coord_bins - 1), 4)

    def encode(self, action: dict[str, Any]) -> str:
        action_type = str(action.get("type", "wait") or "wait").strip().lower()
        if action_type not in ACTION_TYPE_TO_TOKEN:
            action_type = "wait"
        tokens = [ACTION_TYPE_TO_TOKEN[action_type]]

        if action_type in {"click", "double_click", "right_click"}:
            x_index = self._coord_to_bin(action.get("x_norm", action.get("x", 0.5)))
            y_index = self._coord_to_bin(action.get("y_norm", action.get("y", 0.5)))
            tokens.extend([self.x_token(x_index), self.y_token(y_index)])
        elif action_type == "type":
            text = str(action.get("text", "") or "")
            tokens.extend([TEXT_START_TOKEN, text, TEXT_END_TOKEN])
        elif action_type == "hotkey":
            keys = list(action.get("keys") or [])
            if not keys:
                tokens.append(KEY_UNKNOWN_TOKEN)
            for key in keys:
                normalized = str(key).strip().lower()
                tokens.append(KEY_TO_TOKEN.get(normalized, KEY_UNKNOWN_TOKEN))
        elif action_type == "scroll":
            amount = action.get("amount", 0) or 0
            try:
                amount_value = float(amount)
            except (TypeError, ValueError):
                amount_value = 0.0
            if amount_value < 0:
                tokens.append(SCROLL_TO_TOKEN["down"])
            elif amount_value > 0:
                tokens.append(SCROLL_TO_TOKEN["up"])
            else:
                tokens.append(SCROLL_TO_TOKEN["zero"])
        elif action_type in {"wait", "terminate"}:
            status = str(action.get("status", "success") or "success").strip().lower()
            tokens.append(STATUS_TO_TOKEN.get(status, STATUS_TO_TOKEN["success"]))
        return " ".join(tokens)

    def decode(self, text: str) -> DecodedActionTokens | None:
        raw_text = str(text or "")
        action_match = re.search(r"<ACT_[A-Z_]+>", raw_text)
        if not action_match:
            return None
        action_token = action_match.group(0)
        action_type = TOKEN_TO_ACTION_TYPE.get(action_token)
        if not action_type:
            return None
        action: dict[str, Any] = {"type": action_type}

        if action_type in {"click", "double_click", "right_click"}:
            x_match = re.search(r"<X_(\d+)>", raw_text)
            y_match = re.search(r"<Y_(\d+)>", raw_text)
            if x_match and y_match:
                action["x_norm"] = self._bin_to_coord(int(x_match.group(1)))
                action["y_norm"] = self._bin_to_coord(int(y_match.group(1)))
        elif action_type == "type":
            text_match = re.search(
                re.escape(TEXT_START_TOKEN) + r"(.*?)" + re.escape(TEXT_END_TOKEN),
                raw_text,
                flags=re.S,
            )
            action["text"] = text_match.group(1).strip() if text_match else ""
        elif action_type == "hotkey":
            keys = []
            for token in re.findall(r"<KEY_[A-Z0-9_]+>", raw_text):
                key = TOKEN_TO_KEY.get(token)
                if key and key not in {"control", "command", "return", "escape"}:
                    keys.append(key)
                elif key == "control":
                    keys.append("ctrl")
                elif key == "command":
                    keys.append("cmd")
                elif key == "return":
                    keys.append("enter")
                elif key == "escape":
                    keys.append("esc")
            action["keys"] = keys
        elif action_type == "scroll":
            direction = "zero"
            for token, candidate in TOKEN_TO_SCROLL.items():
                if token in raw_text:
                    direction = candidate
                    break
            action["amount"] = -512 if direction == "down" else 512 if direction == "up" else 0
        elif action_type in {"wait", "terminate"}:
            status = "success"
            for token, candidate in TOKEN_TO_STATUS.items():
                if token in raw_text:
                    status = candidate
                    break
            action["status"] = status
        return DecodedActionTokens(action=action, raw_text=raw_text)
