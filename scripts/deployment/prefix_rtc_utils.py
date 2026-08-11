"""Prefix-RTC helpers shared by ONNX export, verification, and TRT runtime."""

from __future__ import annotations

from typing import Any, Optional
import types

import torch
import torch.nn.functional as F
from transformers.feature_extraction_utils import BatchFeature


_PREFIX_TIMESTEP_MODES = {"legacy_zero", "groot_clean"}


def normalize_prefix_timestep_mode(mode: str) -> str:
    aliases = {
        "legacy": "legacy_zero",
        "zero": "legacy_zero",
        "groot": "groot_clean",
        "groot_style": "groot_clean",
        "clean": "groot_clean",
    }
    normalized = aliases.get(str(mode).strip().lower(), str(mode).strip().lower())
    if normalized not in _PREFIX_TIMESTEP_MODES:
        raise ValueError(
            "prefix_rtc_timestep_mode must be one of "
            f"{sorted(_PREFIX_TIMESTEP_MODES)}, got {mode!r}"
        )
    return normalized


def resolve_prefix_timestep_mode(action_head, explicit_mode: str | None) -> str:
    if explicit_mode is None:
        explicit_mode = getattr(
            action_head.config,
            "prefix_rtc_timestep_mode",
            "legacy_zero",
        )
    return normalize_prefix_timestep_mode(explicit_mode)


def configure_prefix_rtc(action_head, prefix_timestep_mode: str | None = None) -> str:
    resolved_mode = resolve_prefix_timestep_mode(action_head, prefix_timestep_mode)
    action_head._prefix_rtc_timestep_mode = resolved_mode
    action_head._prefix_rtc_enabled = True
    return resolved_mode


def prefix_timestep_bucket(action_head) -> int:
    mode = getattr(action_head, "_prefix_rtc_timestep_mode", "legacy_zero")
    if mode == "legacy_zero":
        return 0
    if mode == "groot_clean":
        return max(int(action_head.num_timestep_buckets) - 1, 0)
    raise RuntimeError(f"Unsupported resolved prefix timestep mode: {mode!r}")


def ensure_prefix_rtc_policy_options(policy) -> None:
    """Fill harmless soft-RTC defaults required by Gr00tPolicy._get_action."""
    if getattr(policy, "_prefix_rtc_options_wrapped", False):
        return

    original_get_action = policy.get_action

    def _get_action_prefix_rtc(observation, options=None):
        if options is not None and options.get("rtc_prev_action") is not None:
            options = dict(options)
            options.setdefault("rtc_frozen_steps", 0)
            options.setdefault("rtc_ramp_rate", 1.0)
        return original_get_action(observation, options)

    policy.get_action = _get_action_prefix_rtc
    policy._prefix_rtc_options_wrapped = True


def _patched_action_encoder_forward(self, actions, timesteps, cat_ids):
    """MultiEmbodimentActionEncoder.forward with optional per-frame timesteps."""
    from gr00t.model.modules.embodiment_conditioned_mlp import swish

    batch_size, horizon, _ = actions.shape
    if timesteps.dim() == 1 and timesteps.shape[0] == batch_size:
        timesteps = timesteps.unsqueeze(1).expand(-1, horizon)
    elif timesteps.dim() == 2 and timesteps.shape == (batch_size, horizon):
        pass
    else:
        raise ValueError(
            "Expected timesteps shape "
            f"(B,) or (B, T)=({batch_size}, {horizon}), got {tuple(timesteps.shape)}"
        )

    action_emb = self.W1(actions, cat_ids)
    time_emb = self.pos_encoding(timesteps).to(dtype=action_emb.dtype)
    x = torch.cat([action_emb, time_emb], dim=-1)
    x = swish(self.W2(x, cat_ids))
    return self.W3(x, cat_ids)


def _patched_timestep_encoder_forward(self, timesteps):
    """Encode (B,) or (B, T) timesteps into matching time embeddings."""
    dtype = next(self.parameters()).dtype
    if timesteps.dim() == 1:
        timesteps_proj = self.time_proj(timesteps).to(dtype)
        return self.timestep_embedder(timesteps_proj)
    if timesteps.dim() == 2:
        batch_size, seq_len = timesteps.shape
        flat = timesteps.reshape(batch_size * seq_len)
        timesteps_proj = self.time_proj(flat).to(dtype)
        emb = self.timestep_embedder(timesteps_proj)
        return emb.reshape(batch_size, seq_len, -1)
    raise ValueError(f"Expected timesteps dim 1 or 2, got shape {tuple(timesteps.shape)}")


def _patched_ada_layer_norm_forward(
    self,
    x: torch.Tensor,
    temb: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """AdaLayerNorm with batch-global or per-token time embeddings."""
    if temb is None:
        raise ValueError("AdaLayerNorm requires temb")
    if temb.dim() == 2:
        temb = self.linear(self.silu(temb))
        scale, shift = temb.chunk(2, dim=1)
        return self.norm(x) * (1 + scale[:, None]) + shift[:, None]
    if temb.dim() == 3:
        if temb.shape[:2] != x.shape[:2]:
            raise ValueError(
                f"per-token temb shape {tuple(temb.shape)} incompatible with x {tuple(x.shape)}"
            )
        temb = self.linear(self.silu(temb))
        scale, shift = temb.chunk(2, dim=-1)
        return self.norm(x) * (1 + scale) + shift
    raise ValueError(f"Expected temb dim 2 or 3, got shape {tuple(temb.shape)}")


def _apply_output_adaln(module, hidden_states: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
    if temb.dim() == 2:
        shift, scale = module.proj_out_1(F.silu(temb)).chunk(2, dim=1)
        return module.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
    if temb.dim() == 3:
        if temb.shape[:2] != hidden_states.shape[:2]:
            raise ValueError(
                f"per-token temb {tuple(temb.shape)} vs hidden {tuple(hidden_states.shape)}"
            )
        shift, scale = module.proj_out_1(F.silu(temb)).chunk(2, dim=-1)
        return module.norm_out(hidden_states) * (1 + scale) + shift
    raise ValueError(f"Expected temb dim 2 or 3, got {tuple(temb.shape)}")


def _patched_dit_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    timestep: Optional[torch.LongTensor] = None,
    encoder_attention_mask: Optional[torch.Tensor] = None,
    return_all_hidden_states: bool = False,
    encoder_kv_cache: Optional[dict[int, tuple[torch.Tensor, torch.Tensor]]] = None,
):
    temb = self.timestep_encoder(timestep)
    hidden_states = hidden_states.contiguous()
    encoder_hidden_states = encoder_hidden_states.contiguous()
    all_hidden_states = [hidden_states]

    for idx, block in enumerate(self.transformer_blocks):
        if idx % 2 == 1 and self.config.interleave_self_attention:
            hidden_states = block(
                hidden_states,
                attention_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                temb=temb,
            )
        else:
            hidden_states = block(
                hidden_states,
                attention_mask=None,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=None,
                temb=temb,
                encoder_kv_cache=encoder_kv_cache,
                encoder_kv_cache_key=idx,
            )
        all_hidden_states.append(hidden_states)

    hidden_states = _apply_output_adaln(self, hidden_states, temb)
    if return_all_hidden_states:
        return self.proj_out_2(hidden_states), all_hidden_states
    return self.proj_out_2(hidden_states)


def _patched_alternate_vl_dit_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    timestep: Optional[torch.LongTensor] = None,
    encoder_attention_mask: Optional[torch.Tensor] = None,
    return_all_hidden_states: bool = False,
    image_mask: Optional[torch.Tensor] = None,
    backbone_attention_mask: Optional[torch.Tensor] = None,
    encoder_kv_cache: Optional[dict[int, tuple[torch.Tensor, torch.Tensor]]] = None,
):
    assert image_mask is not None, "Image mask is required"
    temb = self.timestep_encoder(timestep)
    hidden_states = hidden_states.contiguous()
    encoder_hidden_states = encoder_hidden_states.contiguous()

    image_attention_mask = image_mask & backbone_attention_mask
    non_image_attention_mask = (~image_mask) & backbone_attention_mask
    all_hidden_states = [hidden_states]
    assert self.config.interleave_self_attention, "Interleave self attention must be enabled"

    for idx, block in enumerate(self.transformer_blocks):
        if idx % 2 == 1:
            hidden_states = block(
                hidden_states,
                attention_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                temb=temb,
            )
        else:
            if idx % (2 * self.attend_text_every_n_blocks) == 0:
                curr_encoder_attention_mask = non_image_attention_mask
            else:
                curr_encoder_attention_mask = image_attention_mask
            hidden_states = block(
                hidden_states,
                attention_mask=None,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=curr_encoder_attention_mask,
                temb=temb,
                encoder_kv_cache=encoder_kv_cache,
                encoder_kv_cache_key=idx,
            )
        all_hidden_states.append(hidden_states)

    hidden_states = _apply_output_adaln(self, hidden_states, temb)
    if return_all_hidden_states:
        return self.proj_out_2(hidden_states), all_hidden_states
    return self.proj_out_2(hidden_states)


def patch_dit_stack(dit_module) -> None:
    """Patch TimestepEncoder, AdaLayerNorm, and DiT forwards on a live DiT."""
    from gr00t.model.modules import dit as dit_mod
    from gr00t.model.modules.dit import AlternateVLDiT

    dit_mod.AdaLayerNorm.forward = _patched_ada_layer_norm_forward
    dit_module.timestep_encoder.forward = types.MethodType(
        _patched_timestep_encoder_forward, dit_module.timestep_encoder
    )

    if isinstance(dit_module, AlternateVLDiT):
        dit_module.forward = types.MethodType(_patched_alternate_vl_dit_forward, dit_module)
    else:
        dit_module.forward = types.MethodType(_patched_dit_forward, dit_module)


@torch.no_grad()
def _prefix_rtc_get_action_with_features(
    self,
    backbone_features: torch.Tensor,
    state_features: torch.Tensor,
    embodiment_id: torch.Tensor,
    backbone_output: BatchFeature,
    action_input: BatchFeature,
    options: dict[str, Any] | None = None,
    timing: dict[str, float] | None = None,
    timing_sync_cuda: bool = False,
) -> BatchFeature:
    vl_embeds = backbone_features
    batch_size = vl_embeds.shape[0]
    device = vl_embeds.device
    action_horizon = self.config.action_horizon

    actions = torch.randn(
        size=(batch_size, action_horizon, self.action_dim),
        dtype=vl_embeds.dtype,
        device=device,
    )
    dt = 1.0 / self.num_inference_timesteps

    rtc_prefix = None
    rtc_overlap_steps = 0
    if "action" in action_input:
        assert options is not None, "options is required for RTC"
        assert "action_horizon" in options, "action_horizon is not in options"
        assert "rtc_overlap_steps" in options, "rtc_overlap_steps is not in options"
        action_horizon_before_padding = int(options["action_horizon"])
        rtc_overlap_steps = int(options["rtc_overlap_steps"])
        if not (0 < rtc_overlap_steps < action_horizon):
            raise ValueError(
                f"rtc_overlap_steps must be in (0, {action_horizon}), got {rtc_overlap_steps}"
            )
        rtc_prefix = action_input["action"][
            :,
            action_horizon_before_padding - rtc_overlap_steps : action_horizon_before_padding,
            :,
        ].to(dtype=actions.dtype, device=device)
        actions[:, :rtc_overlap_steps, :] = rtc_prefix

    for step in range(self.num_inference_timesteps):
        if rtc_prefix is not None:
            actions[:, :rtc_overlap_steps, :] = rtc_prefix

        t_cont = step / float(self.num_inference_timesteps)
        t_discretized = int(t_cont * self.num_timestep_buckets)
        t_global = torch.full(
            size=(batch_size,), fill_value=t_discretized, device=device, dtype=torch.long
        )

        t_action = t_global[:, None].expand(-1, action_horizon).clone()
        if rtc_prefix is not None:
            t_action[:, :rtc_overlap_steps] = prefix_timestep_bucket(self)
        action_features = self.action_encoder(actions, t_action, embodiment_id)
        t_sa = torch.cat([t_global[:, None], t_action], dim=1)

        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        sa_embs = torch.cat((state_features, action_features), dim=1)

        if self.config.use_alternate_vl_dit:
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                timestep=t_sa,
                image_mask=backbone_output.image_mask,
                backbone_attention_mask=backbone_output.backbone_attention_mask,
            )
        else:
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                timestep=t_sa,
            )

        pred = self.action_decoder(model_output, embodiment_id)
        pred_velocity = pred[:, -action_horizon:]
        actions = actions + dt * pred_velocity

        if rtc_prefix is not None:
            actions[:, :rtc_overlap_steps, :] = rtc_prefix

    return BatchFeature(
        data={
            "action_pred": actions,
            "backbone_features": vl_embeds,
            "state_features": state_features,
        }
    )


def apply_prefix_rtc(policy, prefix_timestep_mode: str | None = None) -> None:
    """Enable Prefix-RTC PyTorch inference on a loaded Gr00tPolicy."""
    action_head = policy.model.action_head
    resolved_mode = configure_prefix_rtc(action_head, prefix_timestep_mode)
    action_head.action_encoder.forward = types.MethodType(
        _patched_action_encoder_forward, action_head.action_encoder
    )
    patch_dit_stack(action_head.model)
    action_head.get_action_with_features = types.MethodType(
        _prefix_rtc_get_action_with_features, action_head
    )
    ensure_prefix_rtc_policy_options(policy)
    policy._prefix_rtc_enabled = True
    print(
        "[prefix-rtc] enabled on policy: hard-rewrite overlap + "
        f"prefix_timestep_mode={resolved_mode} "
        f"(bucket={prefix_timestep_bucket(action_head)}) + DiT per-token AdaLN"
    )
