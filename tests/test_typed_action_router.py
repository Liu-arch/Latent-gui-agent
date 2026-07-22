from __future__ import annotations

import unittest

from qwen3_gui_agent.executor import GUIExecutor
from qwen3_gui_agent.schemas import GUIAction
from qwen3_gui_agent.typed_action_router import (
    route_selective_hybrid_action,
    route_typed_hybrid_action,
)


class TypedActionRouterTest(unittest.TestCase):
    def test_type_never_consumes_mismatched_pointer_coordinates(self) -> None:
        action, diagnostics = route_typed_hybrid_action(
            head_action={"type": "type", "x_norm": 0.2, "y_norm": 0.3},
            lm_action={"type": "click", "x_norm": 0.8, "y_norm": 0.9},
        )

        self.assertEqual(action, {"type": "type"})
        self.assertFalse(diagnostics["execution_allowed"])
        self.assertEqual(diagnostics["hybrid_parameter_error"], "lm_action_type_mismatch:click")

    def test_click_uses_only_head_coordinates(self) -> None:
        action, diagnostics = route_typed_hybrid_action(
            head_action={"type": "click", "x_norm": 0.2, "y_norm": 0.3, "region": "top_left"},
            lm_action={"type": "type", "text": "ignored", "x_norm": 0.8, "y_norm": 0.9},
        )

        self.assertEqual(
            action,
            {"type": "click", "x_norm": 0.2, "y_norm": 0.3, "region": "top_left"},
        )
        self.assertTrue(diagnostics["execution_allowed"])

    def test_matching_type_action_receives_lm_text(self) -> None:
        action, diagnostics = route_typed_hybrid_action(
            head_action={"type": "type"},
            lm_action={"type": "type", "text": "hello", "x_norm": 0.8, "y_norm": 0.9},
        )

        self.assertEqual(action, {"type": "type", "text": "hello"})
        self.assertEqual(diagnostics["hybrid_parameter_source"], "lm_same_type")
        self.assertTrue(diagnostics["execution_allowed"])

    def test_matching_hotkey_action_receives_only_lm_keys(self) -> None:
        action, diagnostics = route_typed_hybrid_action(
            head_action={"type": "hotkey"},
            lm_action={"type": "hotkey", "keys": ["ctrl", "s"], "text": "ignored"},
        )

        self.assertEqual(action, {"type": "hotkey", "keys": ["ctrl", "s"]})
        self.assertTrue(diagnostics["execution_allowed"])

    def test_full_action_policy_lets_lm_own_hard_action(self) -> None:
        action, diagnostics = route_selective_hybrid_action(
            head_action={"type": "type"},
            lm_action={"type": "click", "x_norm": 0.8, "y_norm": 0.9},
            hard_action_policy="full_action",
        )

        self.assertEqual(action, {"type": "click", "x_norm": 0.8, "y_norm": 0.9})
        self.assertTrue(diagnostics["selective_lm_owns_action"])
        self.assertEqual(diagnostics["hybrid_parameter_source"], "lm_full_action")

    def test_full_action_policy_keeps_easy_action_on_head(self) -> None:
        action, diagnostics = route_selective_hybrid_action(
            head_action={"type": "click", "x_norm": 0.2, "y_norm": 0.3},
            lm_action={"type": "terminate", "status": "success"},
            hard_action_policy="full_action",
        )

        self.assertEqual(action, {"type": "click", "x_norm": 0.2, "y_norm": 0.3})
        self.assertFalse(diagnostics["selective_lm_owns_action"])

    def test_full_action_policy_rejects_missing_lm_action(self) -> None:
        action, diagnostics = route_selective_hybrid_action(
            head_action={"type": "hotkey"},
            lm_action=None,
            hard_action_policy="full_action",
        )

        self.assertEqual(action, {"type": "hotkey"})
        self.assertFalse(diagnostics["execution_allowed"])
        self.assertEqual(diagnostics["hybrid_parameter_error"], "missing_lm_full_action")

    def test_executor_rejects_coordinates_on_type_action(self) -> None:
        executor = GUIExecutor(dry_run=True)
        action = GUIAction(
            type="type",
            reason="test",
            expected_outcome="test",
            x=10,
            y=20,
            text="hello",
        )

        with self.assertRaisesRegex(ValueError, "type cannot carry coordinates"):
            executor.execute(action)


if __name__ == "__main__":
    unittest.main()
