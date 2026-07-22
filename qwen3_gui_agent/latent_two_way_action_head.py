from __future__ import annotations

from typing import Any

import torch
from torch import nn

from qwen3_gui_agent.flow_matching_action_head import FlowMatchingActionHeadOutput


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, output_dim),
    )


class TwoWayDecoderLayer(nn.Module):
    """A compact SAM-style two-way query/image transformer layer."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        ffn_dim = max(hidden_dim, int(round(hidden_dim * float(mlp_ratio))))
        self.query_self_norm = nn.LayerNorm(hidden_dim)
        self.query_self_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.query_cross_norm = nn.LayerNorm(hidden_dim)
        self.image_key_norm = nn.LayerNorm(hidden_dim)
        self.query_to_image = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.query_ffn_norm = nn.LayerNorm(hidden_dim)
        self.query_ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.image_cross_norm = nn.LayerNorm(hidden_dim)
        self.query_key_norm = nn.LayerNorm(hidden_dim)
        self.image_to_query = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.image_ffn_norm = nn.LayerNorm(hidden_dim)
        self.image_ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
        )

    def forward(
        self,
        queries: torch.Tensor,
        image_tokens: torch.Tensor,
        *,
        query_valid_mask: torch.Tensor,
        image_valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query_padding_mask = ~query_valid_mask.to(dtype=torch.bool)
        image_padding_mask = ~image_valid_mask.to(dtype=torch.bool)

        normalized_queries = self.query_self_norm(queries)
        attended_queries, _ = self.query_self_attn(
            normalized_queries,
            normalized_queries,
            normalized_queries,
            key_padding_mask=query_padding_mask,
            need_weights=False,
        )
        queries = queries + attended_queries

        attended_image, _ = self.query_to_image(
            self.query_cross_norm(queries),
            self.image_key_norm(image_tokens),
            self.image_key_norm(image_tokens),
            key_padding_mask=image_padding_mask,
            need_weights=False,
        )
        queries = queries + attended_image
        queries = queries + self.query_ffn(self.query_ffn_norm(queries))

        attended_queries_from_image, _ = self.image_to_query(
            self.image_cross_norm(image_tokens),
            self.query_key_norm(queries),
            self.query_key_norm(queries),
            key_padding_mask=query_padding_mask,
            need_weights=False,
        )
        image_tokens = image_tokens + attended_queries_from_image
        image_tokens = image_tokens + self.image_ffn(self.image_ffn_norm(image_tokens))
        return queries, image_tokens


class LatentTwoWayActionHead(nn.Module):
    """Ground GUI actions by decoding Stage-2 latent states against visual patches.

    In latent_pos mode, a learned internal <|POS|> router reads the
    input-dependent Stage-2 latent reasoning states and predicts whether the
    action is pointer-like. The router is not used as the spatial prompt: the
    Stage-2 latent states themselves interact with current visual patches and
    drive the location queries.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        action_type_count: int,
        terminate_count: int,
        hidden_dim: int = 512,
        depth: int = 2,
        num_heads: int = 8,
        location_query_count: int = 3,
        max_latent_tokens: int = 16,
        dropout: float = 0.0,
        query_mode: str = "semantic_pool",
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"two-way hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})."
            )
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.location_query_count = max(1, int(location_query_count))
        self.max_latent_tokens = max(1, int(max_latent_tokens))
        self.query_mode = self._normalize_query_mode(query_mode)
        self.patch_logit_temperature = 1.0
        self.patch_residual_scale = 1.0
        self.pointer_coord_source = "two_way_patch_residual"

        self.latent_projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
        )
        self.context_projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
        )
        self.visual_projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
        )
        self.latent_position_embeddings = nn.Parameter(
            torch.empty(self.max_latent_tokens, hidden_dim)
        )
        self.context_type_embeddings = nn.Parameter(torch.empty(2, hidden_dim))
        self.location_queries = nn.Parameter(
            torch.empty(self.location_query_count, hidden_dim)
        )
        self.pos_query = nn.Parameter(torch.empty(hidden_dim))
        nn.init.normal_(self.latent_position_embeddings, mean=0.0, std=0.02)
        nn.init.normal_(self.context_type_embeddings, mean=0.0, std=0.02)
        nn.init.normal_(self.location_queries, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_query, mean=0.0, std=0.02)

        self.pos_query_norm = nn.LayerNorm(hidden_dim)
        self.pos_latent_norm = nn.LayerNorm(hidden_dim)
        self.pos_to_latent = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.pos_ffn_norm = nn.LayerNorm(hidden_dim)
        self.pos_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        self.patch_position_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.layers = nn.ModuleList(
            [
                TwoWayDecoderLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(max(1, int(depth)))
            ]
        )
        self.query_output_norm = nn.LayerNorm(hidden_dim)
        self.image_output_norm = nn.LayerNorm(hidden_dim)
        self.semantic_pool_query = nn.Parameter(torch.empty(hidden_dim))
        nn.init.normal_(self.semantic_pool_query, mean=0.0, std=0.02)

        self.action_type_head = _mlp(hidden_dim, hidden_dim, action_type_count)
        self.terminate_head = _mlp(hidden_dim, hidden_dim, terminate_count)
        self.scroll_head = _mlp(hidden_dim, hidden_dim, 1)
        self.location_confidence_head = _mlp(hidden_dim, hidden_dim, 1)
        self.location_residual_head = _mlp(hidden_dim, hidden_dim, 2)
        self.patch_key_projection = nn.Linear(hidden_dim, hidden_dim)

    @staticmethod
    def _normalize_query_mode(query_mode: str) -> str:
        normalized = str(query_mode or "semantic_pool").strip().lower()
        aliases = {
            "pool": "semantic_pool",
            "legacy": "semantic_pool",
            "pos": "latent_pos",
            "<|pos|>": "latent_pos",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized not in {"semantic_pool", "latent_pos"}:
            raise ValueError(
                "two-way query mode must be 'semantic_pool' or 'latent_pos', "
                f"got {query_mode!r}."
            )
        return normalized

    def _latent_pos_readout(
        self,
        latent_states: torch.Tensor,
        latent_valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build an input-dependent <|POS|> state from Stage-2 latent tokens."""
        batch_size = int(latent_states.shape[0])
        pos_query = self.pos_query.to(
            device=latent_states.device,
            dtype=latent_states.dtype,
        ).view(1, 1, -1).expand(batch_size, -1, -1)
        normalized_latents = self.pos_latent_norm(latent_states)
        attended, attention = self.pos_to_latent(
            self.pos_query_norm(pos_query),
            normalized_latents,
            normalized_latents,
            key_padding_mask=~latent_valid_mask.to(dtype=torch.bool),
            need_weights=True,
            average_attn_weights=True,
        )
        pos_state = pos_query + attended
        pos_state = pos_state + self.pos_ffn(self.pos_ffn_norm(pos_state))
        return pos_state, attention.squeeze(1)

    @staticmethod
    def _fallback_grid_size(token_count: int) -> tuple[int, int]:
        if token_count <= 0:
            return 1, 1
        width = max(1, int(round(token_count**0.5)))
        while width > 1 and token_count % width != 0:
            width -= 1
        height = max(1, token_count // width)
        return (height, width) if height * width == token_count else (token_count, 1)

    def _patch_geometry(
        self,
        *,
        image_tokens: torch.Tensor,
        image_valid_mask: torch.Tensor,
        grid_sizes: list[tuple[int, int]] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, token_count = image_valid_mask.shape
        centers = image_tokens.new_full((batch_size, token_count, 2), 0.5)
        residual_scales = image_tokens.new_ones((batch_size, 2))
        for row_index in range(batch_size):
            valid_count = int(image_valid_mask[row_index].sum().item())
            if grid_sizes and row_index < len(grid_sizes):
                grid_height, grid_width = grid_sizes[row_index]
            else:
                grid_height, grid_width = self._fallback_grid_size(valid_count)
            grid_height = max(1, int(grid_height))
            grid_width = max(1, int(grid_width))
            if grid_height * grid_width != valid_count:
                grid_height, grid_width = self._fallback_grid_size(valid_count)
            if valid_count > 0:
                indices = torch.arange(valid_count, device=image_tokens.device)
                centers[row_index, :valid_count, 0] = (
                    (indices % grid_width).to(dtype=image_tokens.dtype) + 0.5
                ) / float(grid_width)
                centers[row_index, :valid_count, 1] = (
                    (indices // grid_width).to(dtype=image_tokens.dtype) + 0.5
                ) / float(grid_height)
            residual_scales[row_index, 0] = 1.0 / float(grid_width)
            residual_scales[row_index, 1] = 1.0 / float(grid_height)
        return centers, residual_scales

    def _semantic_pool(
        self,
        semantic_states: torch.Tensor,
        semantic_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        query = self.semantic_pool_query.to(
            device=semantic_states.device,
            dtype=semantic_states.dtype,
        )
        logits = torch.matmul(semantic_states.float(), query.float())
        logits = logits.masked_fill(~semantic_valid_mask, -1e9)
        weights = torch.softmax(logits, dim=-1).to(dtype=semantic_states.dtype)
        return (weights.unsqueeze(-1) * semantic_states).sum(dim=1)

    def forward(
        self,
        *,
        latent_states: torch.Tensor,
        latent_valid_mask: torch.Tensor,
        current_visual_tokens: torch.Tensor,
        current_visual_token_mask: torch.Tensor,
        target_patch_grid_sizes: list[tuple[int, int]] | None,
        sequence_summary: torch.Tensor | None = None,
        img_next_state: torch.Tensor | None = None,
    ) -> FlowMatchingActionHeadOutput:
        batch_size = int(latent_states.shape[0])
        latent_count = min(int(latent_states.shape[1]), self.max_latent_tokens)
        latent_states = latent_states[:, :latent_count, :]
        latent_valid_mask = latent_valid_mask[:, :latent_count].to(dtype=torch.bool)

        semantic_states = self.latent_projection(latent_states)
        semantic_states = semantic_states + self.latent_position_embeddings[:latent_count].to(
            device=semantic_states.device,
            dtype=semantic_states.dtype,
        ).unsqueeze(0)
        semantic_masks = [latent_valid_mask]
        pos_state = None
        pos_latent_attention = None
        query_mode = self._normalize_query_mode(self.query_mode)
        if query_mode == "latent_pos":
            pos_state, pos_latent_attention = self._latent_pos_readout(
                semantic_states,
                latent_valid_mask,
            )
        context_states: list[torch.Tensor] = []
        if sequence_summary is not None:
            projected = self.context_projection(sequence_summary).unsqueeze(1)
            context_type = self.context_type_embeddings[0].to(
                device=projected.device,
                dtype=projected.dtype,
            )
            context_states.append(projected + context_type.view(1, 1, -1))
        if img_next_state is not None:
            projected = self.context_projection(img_next_state).unsqueeze(1)
            context_type = self.context_type_embeddings[1].to(
                device=projected.device,
                dtype=projected.dtype,
            )
            context_states.append(projected + context_type.view(1, 1, -1))
        if context_states:
            context = torch.cat(context_states, dim=1)
            semantic_states = torch.cat([semantic_states, context], dim=1)
            semantic_masks.append(
                torch.ones(
                    (batch_size, int(context.shape[1])),
                    device=latent_valid_mask.device,
                    dtype=torch.bool,
                )
            )
        semantic_valid_mask = torch.cat(semantic_masks, dim=1)

        image_valid_mask = current_visual_token_mask.to(
            device=semantic_states.device,
            dtype=torch.bool,
        )
        image_tokens = current_visual_tokens.to(
            device=semantic_states.device,
            dtype=semantic_states.dtype,
        )
        centers, residual_scales = self._patch_geometry(
            image_tokens=image_tokens,
            image_valid_mask=image_valid_mask,
            grid_sizes=target_patch_grid_sizes,
        )
        image_tokens = self.visual_projection(image_tokens)
        image_tokens = image_tokens + self.patch_position_encoder(centers.to(dtype=image_tokens.dtype))

        location_states = self.location_queries.to(
            device=semantic_states.device,
            dtype=semantic_states.dtype,
        ).unsqueeze(0).expand(batch_size, -1, -1)
        queries = torch.cat([semantic_states, location_states], dim=1)
        query_valid_mask = torch.cat(
            [
                semantic_valid_mask,
                torch.ones(
                    (batch_size, self.location_query_count),
                    device=semantic_valid_mask.device,
                    dtype=torch.bool,
                ),
            ],
            dim=1,
        )
        for layer in self.layers:
            queries, image_tokens = layer(
                queries,
                image_tokens,
                query_valid_mask=query_valid_mask,
                image_valid_mask=image_valid_mask,
            )
        queries = self.query_output_norm(queries)
        image_tokens = self.image_output_norm(image_tokens)

        semantic_count = int(semantic_states.shape[1])
        decoded_semantics = queries[:, :semantic_count, :]
        decoded_locations = queries[:, semantic_count:, :]
        if query_mode == "latent_pos":
            # <|POS|> is only the action router. Grounding above is driven by
            # the full latent reasoning sequence, not by this marker state.
            pooled_semantics = pos_state.squeeze(1)
        else:
            pooled_semantics = self._semantic_pool(decoded_semantics, semantic_valid_mask)

        patch_keys = self.patch_key_projection(image_tokens)
        candidate_patch_logits = torch.einsum(
            "bkd,bnd->bkn",
            decoded_locations.float(),
            patch_keys.float(),
        ) * (float(self.hidden_dim) ** -0.5)
        candidate_patch_logits = candidate_patch_logits.masked_fill(
            ~image_valid_mask.unsqueeze(1),
            -1e9,
        )
        temperature = max(1e-4, float(self.patch_logit_temperature))
        candidate_patch_probs = torch.softmax(candidate_patch_logits / temperature, dim=-1).to(
            dtype=decoded_locations.dtype
        )
        candidate_patch_xy = torch.einsum(
            "bkn,bnd->bkd",
            candidate_patch_probs,
            centers.to(dtype=candidate_patch_probs.dtype),
        )
        candidate_residual = torch.tanh(self.location_residual_head(decoded_locations))
        candidate_residual = (
            candidate_residual
            * residual_scales.unsqueeze(1)
            * float(self.patch_residual_scale)
        )
        candidate_xy = (candidate_patch_xy + candidate_residual).clamp(1e-4, 1.0 - 1e-4)
        candidate_confidence_logits = self.location_confidence_head(decoded_locations).squeeze(-1)
        candidate_weights = torch.softmax(candidate_confidence_logits.float(), dim=-1).to(
            dtype=candidate_xy.dtype
        )

        pointer_xy = (candidate_weights.unsqueeze(-1) * candidate_xy).sum(dim=1)
        aggregate_patch_probs = (
            candidate_weights.unsqueeze(-1) * candidate_patch_probs
        ).sum(dim=1).clamp_min(1e-12)
        target_patch_logits = aggregate_patch_probs.float().log()
        patch_xy = (
            aggregate_patch_probs.unsqueeze(-1)
            * centers.to(dtype=aggregate_patch_probs.dtype)
        ).sum(dim=1)
        argmax_indices = aggregate_patch_probs.argmax(dim=-1)
        patch_argmax_xy = centers.gather(
            1,
            argmax_indices.view(-1, 1, 1).expand(-1, 1, 2),
        ).squeeze(1)

        scroll_raw = self.scroll_head(pooled_semantics)
        continuous_action = torch.cat([pointer_xy, torch.tanh(scroll_raw)], dim=-1)
        continuous_raw = torch.cat(
            [torch.logit(pointer_xy.float()).to(dtype=scroll_raw.dtype), scroll_raw],
            dim=-1,
        )
        patch_action = torch.cat([patch_xy, torch.tanh(scroll_raw)], dim=-1)
        patch_argmax_action = torch.cat([patch_argmax_xy, torch.tanh(scroll_raw)], dim=-1)
        zeros = continuous_action.new_zeros(continuous_action.shape)
        zero_t = continuous_action.new_zeros((batch_size, 1))

        output = FlowMatchingActionHeadOutput(
            action_type_logits=self.action_type_head(pooled_semantics),
            terminate_logits=self.terminate_head(pooled_semantics),
            flow_velocity=zeros,
            flow_target_velocity=zeros,
            flow_noisy_action=zeros,
            flow_t=zero_t,
            direct_continuous_action=continuous_action,
            direct_continuous_raw=continuous_raw,
            fused=pooled_semantics,
            target_patch_logits=target_patch_logits,
            target_patch_probs=aggregate_patch_probs,
            patch_continuous_action=patch_action,
            patch_argmax_action=patch_argmax_action,
            patch_residual=pointer_xy - patch_xy,
            target_patch_grid_sizes=target_patch_grid_sizes,
            pointer_coord_source=self.pointer_coord_source,
            candidate_confidence_logits=candidate_confidence_logits,
            candidate_continuous_action=candidate_xy,
            candidate_patch_logits=candidate_patch_logits,
            location_confidence=candidate_weights.max(dim=-1).values,
            pos_query_state=pos_state.squeeze(1) if pos_state is not None else None,
            pos_latent_attention=pos_latent_attention,
            pos_latent_attention_entropy=(
                -(
                    pos_latent_attention.float()
                    * pos_latent_attention.float().clamp_min(1e-12).log()
                ).sum(dim=-1)
                if pos_latent_attention is not None
                else None
            ),
            pos_latent_attention_max=(
                pos_latent_attention.float().max(dim=-1).values
                if pos_latent_attention is not None
                else None
            ),
            two_way_query_mode=query_mode,
        )
        return output
