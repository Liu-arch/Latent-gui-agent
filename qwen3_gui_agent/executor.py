from __future__ import annotations

import time

from qwen3_gui_agent.schemas import GUIAction


class GUIExecutor:
    def __init__(self, dry_run: bool = True) -> None:
        self.dry_run = dry_run
        self.pyautogui = None
        if not self.dry_run:
            import pyautogui

            pyautogui.FAILSAFE = True
            pyautogui.PAUSE = 0.2
            self.pyautogui = pyautogui

    def execute(self, action: GUIAction) -> None:
        self._validate_action(action)
        self._print_action(action)
        if self.dry_run:
            return

        if self.pyautogui is None:
            raise RuntimeError("pyautogui is not available")

        if action.type == "click":
            self.pyautogui.click(x=action.x, y=action.y)
        elif action.type == "double_click":
            self.pyautogui.doubleClick(x=action.x, y=action.y)
        elif action.type == "right_click":
            self.pyautogui.rightClick(x=action.x, y=action.y)
        elif action.type == "type":
            self.pyautogui.write(action.text, interval=0.02)
        elif action.type == "hotkey":
            self.pyautogui.hotkey(*(action.keys or []))
        elif action.type == "scroll":
            self.pyautogui.scroll(action.amount or 0, x=action.x, y=action.y)
        elif action.type == "wait":
            time.sleep(action.seconds or 1.0)
        elif action.type in {"done", "fail"}:
            return
        else:
            raise ValueError(f"Unsupported action type: {action.type}")

    @staticmethod
    def _validate_action(action: GUIAction) -> None:
        if action.type in {"click", "double_click", "right_click"}:
            if action.x is None or action.y is None:
                raise ValueError(f"{action.type} requires x and y coordinates")
            if action.text is not None or action.keys is not None:
                raise ValueError(f"{action.type} cannot carry text or hotkey parameters")
        elif action.type == "type":
            if not action.text:
                raise ValueError("type requires non-empty text")
            if action.x is not None or action.y is not None:
                raise ValueError("type cannot carry coordinates; emit a separate click action first")
            if action.keys is not None or action.amount is not None:
                raise ValueError("type cannot carry hotkey or scroll parameters")
        elif action.type == "hotkey":
            if not action.keys:
                raise ValueError("hotkey requires at least one key")
            if action.x is not None or action.y is not None or action.text is not None:
                raise ValueError("hotkey cannot carry coordinates or text")
        elif action.type == "scroll":
            if action.amount is None:
                raise ValueError("scroll requires amount")

    @staticmethod
    def _print_action(action: GUIAction) -> None:
        print(
            {
                "type": action.type,
                "x": action.x,
                "y": action.y,
                "text": action.text,
                "keys": action.keys,
                "amount": action.amount,
                "seconds": action.seconds,
                "reason": action.reason,
                "expected_outcome": action.expected_outcome,
            }
        )
