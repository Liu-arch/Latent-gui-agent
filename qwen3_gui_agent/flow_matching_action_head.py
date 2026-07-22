from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

    class _NNNamespace:
        Module = object
        Linear = object
        Sequential = object
        GELU = object
        LayerNorm = object

    nn = _NNNamespace()  # type: ignore[assignment]


@dataclass
class FlowMatchingActionHeadOutput:
    action_type_logits: Any
    terminate_logits: Any
    flow_velocity: Any
    flow_target_velocity: Any
    flow_noisy_action: Any
    flow_t: Any
    direct_continuous_action: Any | None = None
    direct_continuous_raw: Any | None = None
    fused: Any | None = None
    sampled_continuous_action: Any | None = None
    target_patch_logits: Any | None = None
    target_patch_probs: Any | None = None
    patch_continuous_action: Any | None = None
    patch_argmax_action: Any | None = None
    patch_residual: Any | None = None
    target_patch_grid_sizes: Any | None = None
    pointer_coord_source: str = "patch_residual"
    candidate_confidence_logits: Any | None = None
    candidate_continuous_action: Any | None = None
    candidate_patch_logits: Any | None = None
    location_confidence: Any | None = None
    pos_query_state: Any | None = None
    pos_latent_attention: Any | None = None
    pos_latent_attention_entropy: Any | None = None
    pos_latent_attention_max: Any | None = None
    two_way_query_mode: str = "semantic_pool"


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)

    def forward(self, t: Any) -> Any:
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        half = max(1, self.dim // 2)
        freqs = torch.exp(
            torch.arange(half, device=t.device, dtype=t.dtype)
            * (-torch.log(t.new_tensor(10000.0)) / max(1, half - 1))
        )
        args = t * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb[:, : self.dim]


def _build_mlp(
    *,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    depth: int,
    input_norm: bool = False,
) -> Any:
    layers: list[Any] = []
    if input_norm:
        layers.append(nn.LayerNorm(input_dim))
    hidden_layers = max(1, int(depth))
    layers.append(nn.Linear(input_dim, hidden_dim))
    layers.append(nn.GELU())
    for _ in range(hidden_layers - 1):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.GELU())
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


class GUIFlowMatchingActionHead(nn.Module):
    """
    LaRA-VLA-style action expert adapted to GUI actions.

    GUI actions have mixed structure: action type is categorical, while pointer
    coordinates and scroll amount are continuous. This module therefore keeps a
    small CE classifier for discrete fields and uses rectified-flow matching for
    continuous action parameters [x_norm, y_norm, scroll_norm].
    """

    def __init__(
        self,
        *,
        fused_dim: int,
        action_type_count: int,
        terminate_count: int,
        continuous_dim: int = 3,
        hidden_dim: int | None = None,
        head_depth: int = 2,
        time_embed_dim: int = 128,
        visual_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.continuous_dim = int(continuous_dim)
        hidden = int(hidden_dim or max(512, fused_dim // 2))
        depth = max(1, int(head_depth))
        self.visual_dim = int(visual_dim or hidden)
        self.condition_proj = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
        self.flow_net = _build_mlp(
            input_dim=hidden + self.continuous_dim + time_embed_dim,
            hidden_dim=hidden,
            output_dim=self.continuous_dim,
            depth=depth,
        )
        direct_depth = 1 if hidden_dim is None and depth <= 2 else max(1, depth // 2)
        self.direct_action_head = _build_mlp(
            input_dim=fused_dim,
            hidden_dim=hidden,
            output_dim=self.continuous_dim,
            depth=direct_depth,
            input_norm=True,
        )
        self.ground_query_proj = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, self.visual_dim),
        )
        self.ground_visual_proj = nn.Sequential(
            nn.LayerNorm(self.visual_dim),
            nn.Linear(self.visual_dim, self.visual_dim),
            nn.GELU(),
            nn.Linear(self.visual_dim, self.visual_dim),
        )
        self.ground_key_proj = nn.Linear(self.visual_dim, self.visual_dim)
        self.patch_residual_head = _build_mlp(
            input_dim=fused_dim + self.visual_dim,
            hidden_dim=hidden,
            output_dim=2,
            depth=direct_depth,
            input_norm=True,
        )
        # Coordinate source used by the direct pointer branch.
        # - mlp:          old pure MLP direct head
        # - patch:        soft visual patch expectation, no learned residual
        # - argmax_patch: hard visual patch center, no learned residual
        # - patch_residual: soft patch expectation plus learned local offset
        self.pointer_coord_source = "patch_residual"
        self.patch_logit_temperature = 1.0
        self.patch_residual_scale = 1.0
        if hidden_dim is None and depth <= 2:
            self.action_type_head = nn.Linear(fused_dim, action_type_count)
            self.terminate_head = nn.Linear(fused_dim, terminate_count)
        else:
            self.action_type_head = _build_mlp(
                input_dim=fused_dim,
                hidden_dim=hidden,
                output_dim=action_type_count,
                depth=direct_depth,
                input_norm=True,
            )
            self.terminate_head = _build_mlp(
                input_dim=fused_dim,
                hidden_dim=hidden,
                output_dim=terminate_count,
                depth=direct_depth,
                input_norm=True,
            )

    def _continuous_min(self, reference: Any) -> Any:
        mins = reference.new_full((self.continuous_dim,), -1.0)
        if self.continuous_dim >= 1:
            mins[0] = 0.0
        if self.continuous_dim >= 2:
            mins[1] = 0.0
        return mins

    def _continuous_max(self, reference: Any) -> Any:
        maxs = reference.new_full((self.continuous_dim,), 1.0)
        return maxs

    def _clamp_continuous(self, value: Any) -> Any:
        mins = self._continuous_min(value).to(device=value.device, dtype=value.dtype)
        maxs = self._continuous_max(value).to(device=value.device, dtype=value.dtype)
        return value.clamp(min=mins, max=maxs)

    def _sample_prior(self, reference: Any) -> Any:
        # GUI coordinates live in [0, 1], scroll lives in [-1, 1]. Sampling
        # inside the valid action domain keeps flow targets numerically stable.
        noise = torch.rand_like(reference)
        if self.continuous_dim >= 3:
            noise[:, 2:] = noise[:, 2:] * 2.0 - 1.0
        return self._clamp_continuous(noise)

    @staticmethod
    def _fallback_grid_size(token_count: int) -> tuple[int, int]:
        if token_count <= 0:
            return 1, 1
        width = max(1, int(round(token_count ** 0.5)))
        while width > 1 and token_count % width != 0:
            width -= 1
        height = max(1, token_count // width)
        if height * width != token_count:
            height = token_count
            width = 1
        return height, width

    def _patch_centers(
        self,
        *,
        target_patch_probs: Any,
        target_patch_grid_sizes: list[tuple[int, int]] | None,
    ) -> tuple[Any, Any]:
        centers: list[Any] = []
        residual_scales: list[Any] = []
        token_count = int(target_patch_probs.shape[1])
        for batch_index in range(int(target_patch_probs.shape[0])):
            if target_patch_grid_sizes and batch_index < len(target_patch_grid_sizes):
                grid_height, grid_width = target_patch_grid_sizes[batch_index]
            else:
                grid_height, grid_width = self._fallback_grid_size(token_count)
            grid_height = max(1, int(grid_height))
            grid_width = max(1, int(grid_width))
            valid_token_count = grid_height * grid_width
            if valid_token_count > token_count:
                grid_height, grid_width = self._fallback_grid_size(token_count)
                valid_token_count = grid_height * grid_width
            x_coords = []
            y_coords = []
            for patch_index in range(token_count):
                if patch_index < valid_token_count:
                    row = patch_index // grid_width
                    col = patch_index % grid_width
                    x_coords.append((col + 0.5) / grid_width)
                    y_coords.append((row + 0.5) / grid_height)
                else:
                    # Padded visual tokens are masked before softmax; their centers are placeholders.
                    x_coords.append(0.5)
                    y_coords.append(0.5)
            centers.append(
                torch.stack(
                    [
                        target_patch_probs.new_tensor(x_coords),
                        target_patch_probs.new_tensor(y_coords),
                    ],
                    dim=-1,
                )
            )
            residual_scales.append(
                target_patch_probs.new_tensor([1.0 / float(grid_width), 1.0 / float(grid_height)])
            )
        return torch.stack(centers, dim=0), torch.stack(residual_scales, dim=0)

    def _ground_pointer_from_visual_tokens(
        self,
        *,
        fused: Any,
        current_visual_tokens: Any | None,
        current_visual_token_mask: Any | None,
        target_patch_grid_sizes: list[tuple[int, int]] | None,
        fallback_raw: Any,
    ) -> tuple[Any, Any, Any | None, Any | None, Any | None, Any | None, Any | None, str]:
        fallback_parts = []
        if self.continuous_dim >= 1:
            fallback_parts.append(torch.sigmoid(fallback_raw[:, 0:1]))
        if self.continuous_dim >= 2:
            fallback_parts.append(torch.sigmoid(fallback_raw[:, 1:2]))
        if self.continuous_dim >= 3:
            fallback_parts.append(torch.tanh(fallback_raw[:, 2:]))
        fallback_action = torch.cat(fallback_parts, dim=-1) if fallback_parts else fallback_raw
        fallback_action = self._clamp_continuous(fallback_action)

        source = str(getattr(self, "pointer_coord_source", "patch_residual") or "patch_residual").lower()
        if current_visual_tokens is None or int(current_visual_tokens.shape[1]) <= 0:
            return fallback_action, fallback_raw, None, None, None, None, None, "mlp_no_visual_tokens"

        visual_ctx = current_visual_tokens.to(device=fused.device, dtype=fused.dtype)
        visual_ctx = visual_ctx + self.ground_visual_proj(visual_ctx)
        query = self.ground_query_proj(fused)
        keys = self.ground_key_proj(visual_ctx)
        scale = float(keys.shape[-1]) ** -0.5
        target_patch_logits = torch.matmul(keys, query.unsqueeze(-1)).squeeze(-1) * scale
        if current_visual_token_mask is not None:
            valid_mask = current_visual_token_mask.to(device=target_patch_logits.device, dtype=torch.bool)
            target_patch_logits = target_patch_logits.masked_fill(~valid_mask, -1e9)
        temperature = max(1e-4, float(getattr(self, "patch_logit_temperature", 1.0) or 1.0))
        target_patch_probs = torch.softmax(target_patch_logits.float() / temperature, dim=-1).to(dtype=fused.dtype)
        centers, residual_scales = self._patch_centers(
            target_patch_probs=target_patch_probs,
            target_patch_grid_sizes=target_patch_grid_sizes,
        )
        patch_xy = (target_patch_probs.unsqueeze(-1) * centers).sum(dim=1)
        argmax_indices = target_patch_probs.detach().float().argmax(dim=-1)
        argmax_xy = centers.gather(
            dim=1,
            index=argmax_indices.view(-1, 1, 1).expand(-1, 1, 2),
        ).squeeze(1)
        visual_summary = (target_patch_probs.unsqueeze(-1) * visual_ctx).sum(dim=1)
        residual_raw = self.patch_residual_head(torch.cat([fused, visual_summary], dim=-1))
        residual_scale = float(getattr(self, "patch_residual_scale", 1.0) or 1.0)
        residual = torch.tanh(residual_raw) * residual_scales * residual_scale
        if source == "mlp":
            direct_action = fallback_action
            direct_raw = fallback_raw
            return (
                direct_action,
                direct_raw,
                target_patch_logits,
                target_patch_probs,
                patch_xy,
                argmax_xy,
                residual,
                source,
            )
        if source == "patch":
            xy = patch_xy.clamp(min=1e-4, max=1.0 - 1e-4)
        elif source == "argmax_patch":
            xy = argmax_xy.clamp(min=1e-4, max=1.0 - 1e-4)
        else:
            source = "patch_residual"
            xy = (patch_xy + residual).clamp(min=1e-4, max=1.0 - 1e-4)

        parts = [xy]
        raw_parts = [torch.logit(xy.float()).to(dtype=fused.dtype)]
        if self.continuous_dim >= 3:
            scroll_raw = fallback_raw[:, 2:]
            parts.append(torch.tanh(scroll_raw))
            raw_parts.append(scroll_raw)
        direct_action = torch.cat(parts, dim=-1)
        direct_raw = torch.cat(raw_parts, dim=-1)
        return (
            self._clamp_continuous(direct_action),
            direct_raw,
            target_patch_logits,
            target_patch_probs,
            patch_xy,
            argmax_xy,
            residual,
            source,
        )

    def forward(
        self,
        *,
        fused: Any,
        target_continuous_action: Any | None = None,
        noise: Any | None = None,
        t: Any | None = None,
        current_visual_tokens: Any | None = None,
        current_visual_token_mask: Any | None = None,
        target_patch_grid_sizes: list[tuple[int, int]] | None = None,
    ) -> FlowMatchingActionHeadOutput:
        batch_size = int(fused.shape[0])
        device = fused.device
        dtype = fused.dtype
        if target_continuous_action is None:
            target_continuous_action = fused.new_zeros((batch_size, self.continuous_dim))
        else:
            target_continuous_action = target_continuous_action.to(device=device, dtype=dtype)
        target_continuous_action = torch.nan_to_num(target_continuous_action, nan=0.0, posinf=1.0, neginf=0.0)
        target_continuous_action = self._clamp_continuous(target_continuous_action)
        if noise is None:
            noise = self._sample_prior(target_continuous_action)
        else:
            noise = noise.to(device=device, dtype=dtype)
            noise = torch.nan_to_num(noise, nan=0.0, posinf=1.0, neginf=0.0)
            noise = self._clamp_continuous(noise)
        if t is None:
            t = torch.rand((batch_size, 1), device=device, dtype=dtype)
        elif t.ndim == 1:
            t = t.unsqueeze(-1)
        t = t.to(device=device, dtype=dtype)
        t = t.clamp(min=1e-4, max=1.0 - 1e-4)

        noisy_action = (1.0 - t) * noise + t * target_continuous_action
        noisy_action = self._clamp_continuous(noisy_action)
        target_velocity = target_continuous_action - noise
        velocity = self.predict_velocity(fused=fused, continuous_action=noisy_action, t=t)
        fallback_raw = torch.nan_to_num(self.direct_action_head(fused), nan=0.0, posinf=20.0, neginf=-20.0)
        (
            direct_action,
            direct_raw,
            target_patch_logits,
            target_patch_probs,
            patch_action,
            patch_argmax_action,
            patch_residual,
            pointer_coord_source,
        ) = self._ground_pointer_from_visual_tokens(
            fused=fused,
            current_visual_tokens=current_visual_tokens,
            current_visual_token_mask=current_visual_token_mask,
            target_patch_grid_sizes=target_patch_grid_sizes,
            fallback_raw=fallback_raw,
        )
        return FlowMatchingActionHeadOutput(
            action_type_logits=self.action_type_head(fused),
            terminate_logits=self.terminate_head(fused),
            flow_velocity=velocity,
            flow_target_velocity=target_velocity,
            flow_noisy_action=noisy_action,
            flow_t=t,
            direct_continuous_action=direct_action,
            direct_continuous_raw=direct_raw,
            fused=fused,
            target_patch_logits=target_patch_logits,
            target_patch_probs=target_patch_probs,
            patch_continuous_action=patch_action,
            patch_argmax_action=patch_argmax_action,
            patch_residual=patch_residual,
            target_patch_grid_sizes=target_patch_grid_sizes,
            pointer_coord_source=pointer_coord_source,
        )

    def predict_velocity(self, *, fused: Any, continuous_action: Any, t: Any) -> Any:
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        condition = self.condition_proj(fused)
        time_features = self.time_embed(t.to(dtype=fused.dtype))
        flow_input = torch.cat(
            [
                condition,
                continuous_action.to(device=fused.device, dtype=fused.dtype),
                time_features.to(device=fused.device, dtype=fused.dtype),
            ],
            dim=-1,
        )
        velocity = self.flow_net(flow_input)
        velocity = torch.nan_to_num(velocity, nan=0.0, posinf=4.0, neginf=-4.0)
        return velocity.clamp(min=-4.0, max=4.0)

    def sample(
        self,
        *,
        fused: Any,
        steps: int = 8,
        initial_action: Any | None = None,
    ) -> Any:
        with torch.no_grad():
            step_count = max(1, int(steps))
            batch_size = int(fused.shape[0])
            if initial_action is None:
                action = fused.new_zeros((batch_size, self.continuous_dim))
                if self.continuous_dim >= 2:
                    action[:, 0:2] = 0.5
            else:
                action = initial_action.to(device=fused.device, dtype=fused.dtype)
            dt = 1.0 / float(step_count)
            for step in range(step_count):
                t_value = (step + 0.5) / float(step_count)
                t = fused.new_full((batch_size, 1), float(t_value))
                velocity = self.predict_velocity(fused=fused, continuous_action=action, t=t)
                action = action + dt * velocity
                action = self._clamp_continuous(action)
            return self._clamp_continuous(action)
