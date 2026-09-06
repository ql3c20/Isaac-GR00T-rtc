#!/usr/bin/env python3
"""Replay an online prefix-RTC TensorRT layer trace through PyTorch modules."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy
import torch
import torch.nn.functional as F


def load_prefix_adapter(path: Path, policy: Gr00tPolicy) -> None:
    spec = importlib.util.spec_from_file_location("prefix_rtc_capture_adapter", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load prefix-RTC adapter: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.apply_prefix_rtc(policy)


def metrics(name: str, expected: torch.Tensor, actual: torch.Tensor) -> dict[str, float]:
    expected = expected.float().reshape(-1)
    actual = actual.float().reshape(-1)
    if expected.shape != actual.shape:
        raise ValueError(f"{name}: shape mismatch {expected.shape} vs {actual.shape}")
    delta = expected - actual
    result = {
        "mae": delta.abs().mean().item(),
        "max": delta.abs().max().item(),
        "cos": F.cosine_similarity(expected[None], actual[None]).item(),
    }
    print(f"{name:34s} cos={result['cos']:.9f} mae={result['mae']:.7f} max={result['max']:.7f}")
    return result


def cuda_tensor(value: torch.Tensor, dtype=None) -> torch.Tensor:
    tensor = value.cuda()
    return tensor.to(dtype=dtype) if dtype is not None else tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--backbone-path", type=Path, required=True)
    parser.add_argument("--prefix-rtc-module-path", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    capture = torch.load(args.capture, map_location="cpu", weights_only=False)
    if int(capture.get("format_version", -1)) != 1:
        raise ValueError(f"Unsupported capture format: {capture.get('format_version')}")

    policy = Gr00tPolicy(
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        model_path=str(args.model_path),
        backbone_path=str(args.backbone_path),
        device="cuda",
        strict=True,
    )
    load_prefix_adapter(args.prefix_rtc_module_path, policy)
    head = policy.model.action_head
    embodiment_id = cuda_tensor(capture["embodiment_id"], torch.int64)

    print(
        f"capture={args.capture} call={capture['call_index']} "
        f"rtc={capture['has_rtc_prefix']} steps={len(capture['steps'])}"
    )
    worst: tuple[str, float] = ("", -1.0)
    with torch.inference_mode():
        vl_input = cuda_tensor(capture["backbone_after_vlln"], torch.bfloat16)
        vl_expected = capture["vl_self_attention_output"]
        vl_actual = head.vl_self_attention(vl_input).cpu()
        result = metrics("vl_self_attention", vl_expected, vl_actual)
        worst = ("vl_self_attention", result["max"])

        state = cuda_tensor(capture["state_encoder_input"], torch.bfloat16)
        state_actual = head.state_encoder(state, embodiment_id).cpu()
        result = metrics("state_encoder", capture["state_encoder_output"], state_actual)
        if result["max"] > worst[1]:
            worst = ("state_encoder", result["max"])

        for step in capture["steps"]:
            index = int(step["step"])
            actions = cuda_tensor(step["actions_input"], torch.bfloat16)
            t_action = cuda_tensor(step["t_action"], torch.int64)
            action_actual = head.action_encoder(actions, t_action, embodiment_id).cpu()
            result = metrics(
                f"step{index}.action_encoder",
                step["action_encoder_output"],
                action_actual,
            )
            if result["max"] > worst[1]:
                worst = (f"step{index}.action_encoder", result["max"])

            sa_embs = cuda_tensor(step["sa_embs"], torch.bfloat16)
            vl_embs = cuda_tensor(step["vl_embs"], torch.bfloat16)
            t_sa = cuda_tensor(step["t_sa"], torch.int64)
            dit_actual = head.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                timestep=t_sa,
                image_mask=cuda_tensor(step["image_mask"], torch.bool),
                backbone_attention_mask=cuda_tensor(step["backbone_attention_mask"], torch.bool),
            ).cpu()
            result = metrics(f"step{index}.dit", step["dit_output"], dit_actual)
            if result["max"] > worst[1]:
                worst = (f"step{index}.dit", result["max"])

            decoder_input = cuda_tensor(step["dit_output"], torch.bfloat16)
            decoder_actual = head.action_decoder(decoder_input, embodiment_id).cpu()
            result = metrics(
                f"step{index}.action_decoder",
                step["action_decoder_output"],
                decoder_actual,
            )
            if result["max"] > worst[1]:
                worst = (f"step{index}.action_decoder", result["max"])

        # End-to-end replay is deliberately separate from the layer-local
        # comparisons above. It starts from the exact online TRT noise/prefix
        # and lets small per-layer differences accumulate through all Euler
        # denoising steps, while following the original PyTorch prefix-RTC
        # control flow (global timestep for an initial chunk, per-token
        # timesteps only for an RTC continuation).
        actions = cuda_tensor(capture["initial_actions"], torch.bfloat16).clone()
        rtc_prefix = capture.get("rtc_prefix")
        rtc_overlap_steps = int(capture.get("rtc_overlap_steps", 0))
        if rtc_prefix is not None:
            rtc_prefix = cuda_tensor(rtc_prefix, torch.bfloat16)
            actions[:, :rtc_overlap_steps] = rtc_prefix
        dt = 1.0 / len(capture["steps"])
        for step in capture["steps"]:
            index = int(step["step"])
            if rtc_prefix is not None:
                actions[:, :rtc_overlap_steps] = rtc_prefix
                action_timestep = cuda_tensor(step["t_action"], torch.int64)
                dit_timestep = cuda_tensor(step["t_sa"], torch.int64)
            else:
                action_timestep = cuda_tensor(step["t_action"][:, 0], torch.int64)
                dit_timestep = action_timestep

            action_features = head.action_encoder(actions, action_timestep, embodiment_id)
            if head.config.add_pos_embed:
                pos_ids = torch.arange(
                    action_features.shape[1],
                    dtype=torch.long,
                    device=action_features.device,
                )
                action_features = action_features + head.position_embedding(pos_ids).unsqueeze(0)
            sa_embs = torch.cat((state_actual.cuda(), action_features), dim=1)
            model_output = head.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_actual.cuda(),
                timestep=dit_timestep,
                image_mask=cuda_tensor(step["image_mask"], torch.bool),
                backbone_attention_mask=cuda_tensor(step["backbone_attention_mask"], torch.bool),
            )
            pred = head.action_decoder(model_output, embodiment_id)
            actions = actions + dt * pred[:, -actions.shape[1] :]
            if rtc_prefix is not None:
                actions[:, :rtc_overlap_steps] = rtc_prefix
            metrics(
                f"step{index}.accumulated_actions",
                step["actions_output"],
                actions.cpu(),
            )

        metrics(
            "FINAL.normalized_action_132",
            capture["final_actions"],
            actions.cpu(),
        )
        metrics(
            "FINAL.normalized_action_first59",
            capture["final_actions"][..., :59],
            actions.cpu()[..., :59],
        )

    print(f"WORST_LAYER={worst[0]} max_abs_error={worst[1]:.7f}")


if __name__ == "__main__":
    main()
