from __future__ import annotations

import pytest

from qwen3_gui_agent.lara_style_qwen3vl_agent import (
    LaRAStyleQwen3VLAgent,
    resolve_reasoning_field_slot_counts,
)


def _schedule_agent(*, mode: str, counts: tuple[int, int, int]) -> LaRAStyleQwen3VLAgent:
    agent = object.__new__(LaRAStyleQwen3VLAgent)
    agent.latent_slot_count = sum(counts)
    agent.reasoning_alignment_mode = mode
    agent.reasoning_field_slot_counts = counts
    return agent


def test_reasoning_field_slot_counts_resolve_to_six_five_five() -> None:
    assert resolve_reasoning_field_slot_counts("auto", latent_slot_count=16) == (6, 5, 5)
    assert resolve_reasoning_field_slot_counts("6,5,5", latent_slot_count=16) == (6, 5, 5)


@pytest.mark.parametrize("value", ["6,5", "6,5,4", "6,0,10"])
def test_reasoning_field_slot_counts_reject_invalid_partitions(value: str) -> None:
    with pytest.raises(ValueError):
        resolve_reasoning_field_slot_counts(value, latent_slot_count=16)


def test_field_aligned_transition_replaces_whole_fields() -> None:
    agent = _schedule_agent(mode="field_aligned", counts=(6, 5, 5))

    assert agent._build_stage2_field_schedule(
        explicit_keep_ratio=1.0,
        max_thinking_tokens=16,
    ) == (["actual_task", "thought", "reflection"], 0)
    assert agent._build_stage2_field_schedule(
        explicit_keep_ratio=0.75,
        max_thinking_tokens=16,
    ) == (["thought", "reflection"], 6)
    assert agent._build_stage2_field_schedule(
        explicit_keep_ratio=0.34,
        max_thinking_tokens=16,
    ) == (["reflection"], 11)
    assert agent._build_stage2_field_schedule(
        explicit_keep_ratio=0.0,
        max_thinking_tokens=16,
    ) == ([], 16)


def test_aggregate_schedule_remains_legacy_compatible() -> None:
    agent = _schedule_agent(mode="aggregate", counts=(6, 5, 5))

    assert agent._build_stage2_field_schedule(
        explicit_keep_ratio=0.75,
        max_thinking_tokens=8,
    ) == (["thought", "reflection"], 3)
    assert agent._build_stage2_field_schedule(
        explicit_keep_ratio=0.0,
        max_thinking_tokens=8,
    ) == ([], 8)


def test_aggregate_config_keeps_legacy_small_slot_runs_valid() -> None:
    agent = object.__new__(LaRAStyleQwen3VLAgent)
    agent.latent_slot_count = 1
    agent.set_reasoning_alignment_config(mode="aggregate")

    assert agent.reasoning_alignment_mode == "aggregate"
    assert agent.reasoning_field_slot_counts == (1, 0, 0)


if __name__ == "__main__":
    test_reasoning_field_slot_counts_resolve_to_six_five_five()
    test_field_aligned_transition_replaces_whole_fields()
    test_aggregate_schedule_remains_legacy_compatible()
    test_aggregate_config_keeps_legacy_small_slot_runs_valid()
    print("FIELD_ALIGNED_REASONING_SMOKE: PASS")
