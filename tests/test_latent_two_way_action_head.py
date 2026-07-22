from __future__ import annotations

import torch

from qwen3_gui_agent.latent_two_way_action_head import LatentTwoWayActionHead


def test_latent_two_way_action_head_forward_backward() -> None:
    torch.manual_seed(7)
    head = LatentTwoWayActionHead(
        input_dim=32,
        action_type_count=8,
        terminate_count=2,
        hidden_dim=16,
        depth=2,
        num_heads=4,
        location_query_count=3,
        max_latent_tokens=16,
    )
    latent = torch.randn(2, 5, 32, requires_grad=True)
    latent_mask = torch.tensor(
        [[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]],
        dtype=torch.bool,
    )
    visual = torch.randn(2, 12, 32, requires_grad=True)
    visual_mask = torch.tensor(
        [[1] * 12, [1] * 10 + [0] * 2],
        dtype=torch.bool,
    )
    sequence = torch.randn(2, 32, requires_grad=True)
    img_next = torch.randn(2, 32, requires_grad=True)

    output = head(
        latent_states=latent,
        latent_valid_mask=latent_mask,
        current_visual_tokens=visual,
        current_visual_token_mask=visual_mask,
        target_patch_grid_sizes=[(3, 4), (2, 5)],
        sequence_summary=sequence,
        img_next_state=img_next,
    )

    assert output.action_type_logits.shape == (2, 8)
    assert output.terminate_logits.shape == (2, 2)
    assert output.direct_continuous_action.shape == (2, 3)
    assert output.target_patch_logits.shape == (2, 12)
    assert output.candidate_continuous_action.shape == (2, 3, 2)
    assert output.candidate_confidence_logits.shape == (2, 3)
    assert output.two_way_query_mode == "semantic_pool"
    assert output.pos_query_state is None
    assert output.pos_latent_attention is None
    assert torch.isfinite(output.direct_continuous_action).all()
    pointer = output.direct_continuous_action[:, :2]
    assert ((pointer > 0.0) & (pointer < 1.0)).all()

    loss = (
        output.action_type_logits.square().mean()
        + output.terminate_logits.square().mean()
        + output.direct_continuous_action.square().mean()
        + output.target_patch_logits[:, :10].square().mean()
        + output.candidate_continuous_action.square().mean()
        + output.candidate_confidence_logits.square().mean()
    )
    loss.backward()
    assert latent.grad is not None and torch.isfinite(latent.grad).all()
    assert visual.grad is not None and torch.isfinite(visual.grad).all()
    assert head.location_queries.grad is not None


def test_latent_pos_query_forward_backward() -> None:
    torch.manual_seed(11)
    head = LatentTwoWayActionHead(
        input_dim=32,
        action_type_count=8,
        terminate_count=2,
        hidden_dim=16,
        depth=2,
        num_heads=4,
        location_query_count=3,
        max_latent_tokens=16,
        query_mode="latent_pos",
    )
    latent = torch.randn(2, 5, 32, requires_grad=True)
    latent_mask = torch.tensor(
        [[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]],
        dtype=torch.bool,
    )
    visual = torch.randn(2, 12, 32, requires_grad=True)
    visual_mask = torch.tensor(
        [[1] * 12, [1] * 10 + [0] * 2],
        dtype=torch.bool,
    )
    sequence = torch.randn(2, 32)
    img_next = torch.randn(2, 32)

    output = head(
        latent_states=latent,
        latent_valid_mask=latent_mask,
        current_visual_tokens=visual,
        current_visual_token_mask=visual_mask,
        target_patch_grid_sizes=[(3, 4), (2, 5)],
        sequence_summary=sequence,
        img_next_state=img_next,
    )

    assert output.two_way_query_mode == "latent_pos"
    assert output.pos_query_state.shape == (2, 16)
    assert output.pos_latent_attention.shape == (2, 5)
    assert output.pos_latent_attention_entropy.shape == (2,)
    assert output.pos_latent_attention_max.shape == (2,)
    assert torch.isfinite(output.pos_query_state).all()
    assert torch.isfinite(output.pos_latent_attention).all()
    assert torch.allclose(
        output.pos_latent_attention[1, 3:],
        torch.zeros_like(output.pos_latent_attention[1, 3:]),
        atol=1e-7,
    )
    assert torch.allclose(
        output.pos_latent_attention.sum(dim=-1),
        torch.ones(2),
        atol=1e-6,
    )

    loss = (
        output.action_type_logits.square().mean()
        + output.terminate_logits.square().mean()
        + output.direct_continuous_action.square().mean()
        + output.target_patch_logits[:, :10].square().mean()
        + output.pos_query_state.square().mean()
    )
    loss.backward()
    assert latent.grad is not None and torch.isfinite(latent.grad).all()
    assert visual.grad is not None and torch.isfinite(visual.grad).all()
    assert head.pos_query.grad is not None and torch.isfinite(head.pos_query.grad).all()
    assert head.pos_to_latent.in_proj_weight.grad is not None

    # <|POS|> is an action router, not the spatial prompt. Perturbing only its
    # learned seed may change type logits, but must not change pointer x/y.
    pointer_before = output.direct_continuous_action[:, :2].detach().clone()
    with torch.no_grad():
        head.pos_query.add_(3.0)
        rerouted = head(
            latent_states=latent.detach(),
            latent_valid_mask=latent_mask,
            current_visual_tokens=visual.detach(),
            current_visual_token_mask=visual_mask,
            target_patch_grid_sizes=[(3, 4), (2, 5)],
            sequence_summary=sequence,
            img_next_state=img_next,
        )
    assert torch.allclose(
        pointer_before,
        rerouted.direct_continuous_action[:, :2],
        atol=1e-6,
    )


if __name__ == "__main__":
    test_latent_two_way_action_head_forward_backward()
    test_latent_pos_query_forward_backward()
    print("LATENT_TWO_WAY_SMOKE: PASS")
