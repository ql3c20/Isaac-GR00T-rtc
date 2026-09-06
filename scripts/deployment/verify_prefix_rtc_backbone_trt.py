#!/usr/bin/env python3
"""Compare consecutive Prefix-RTC calls with PyTorch vs a selected TRT mode."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F


sys.path.insert(0, str(Path(__file__).resolve().parent))

from export_onnx_n1d7 import prepare_observation
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy
from trt_model_forward import setup_tensorrt_engines


def load_prefix_adapter(path: Path, policy: Gr00tPolicy) -> None:
    spec = importlib.util.spec_from_file_location("prefix_rtc_verify_adapter", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Prefix-RTC adapter: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.apply_prefix_rtc(policy)


def normalized(result) -> torch.Tensor:
    _, info = result
    value = torch.as_tensor(info["normalized_action_pred"]).float().cpu()
    if value.ndim == 2:
        value = value.unsqueeze(0)
    return value


def report(name: str, expected: torch.Tensor, actual: torch.Tensor) -> dict[str, float]:
    expected = expected.reshape(-1)
    actual = actual.reshape(-1)
    delta = (expected - actual).abs()
    cosine = F.cosine_similarity(expected[None], actual[None]).item()
    result = {
        "cosine": cosine,
        "mean_abs_error": delta.mean().item(),
        "max_abs_error": delta.max().item(),
    }
    print(
        f"{name:28s} cos={cosine:.9f} "
        f"mae={result['mean_abs_error']:.7f} max={result['max_abs_error']:.7f}"
    )
    return result


def run_sequence(policy: Gr00tPolicy, observation, seed: int, overlap: int, calls: int):
    policy.reset()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    outputs = [normalized(policy.get_action(observation))]
    for _ in range(1, calls):
        previous = outputs[-1]
        options = {
            "rtc_prev_action": previous.numpy(),
            "action_horizon": int(previous.shape[1]),
            "rtc_overlap_steps": overlap,
        }
        outputs.append(normalized(policy.get_action(observation, options)))
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--backbone-path", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--request-npz", type=Path)
    parser.add_argument("--server-module-path", type=Path)
    parser.add_argument("--engine-dir", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=(
            "vit_llm_only",
            "prefix_rtc_llm_only",
            "prefix_rtc_action_head",
            "prefix_rtc_vit_action_head",
            "prefix_rtc_full_pipeline",
        ),
        default="vit_llm_only",
    )
    parser.add_argument("--prefix-rtc-module-path", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overlap", type=int, default=6)
    parser.add_argument("--calls", type=int, default=2)
    parser.add_argument("--min-cosine", type=float, default=0.99)
    parser.add_argument("--max-mean-abs-error", type=float, default=0.02)
    parser.add_argument("--max-abs-error", type=float, default=0.2)
    return parser.parse_args()


def observation_from_request_npz(path: Path, server_module_path: Path):
    spec = importlib.util.spec_from_file_location("gr00t_live_server_schema", server_module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load GR00T server schema: {server_module_path}")
    server = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(server)
    with np.load(path, allow_pickle=False) as capture:
        state49 = server.as_btd(capture["state49"])
        state = {name: state49[..., start:end] for name, start, end in server.STATE_PARTS}
        root6 = state49[..., 43:49]
        state["rot59_root"] = np.concatenate(
            (root6[..., :3], server.rpy_to_rot6d(root6[..., 3:6])), axis=-1
        ).astype(np.float32)
        observation = {
            "video": {"ego_view": server.as_bthwc(capture["image"])},
            "state": state,
            "language": {
                "annotation.human.task_description": [[str(capture["instruction"].item())]]
            },
        }
        archived = torch.from_numpy(capture["normalized_action"].copy()).float()
    return observation, archived


def main() -> None:
    args = parse_args()
    policy = Gr00tPolicy(
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        model_path=str(args.model_path),
        backbone_path=str(args.backbone_path),
        device="cuda",
        strict=True,
    )
    load_prefix_adapter(args.prefix_rtc_module_path, policy)
    archived = None
    if args.request_npz is not None:
        if args.server_module_path is None:
            raise ValueError("--server-module-path is required with --request-npz")
        observation, archived = observation_from_request_npz(
            args.request_npz, args.server_module_path
        )
    else:
        if args.dataset_path is None:
            raise ValueError("Pass either --request-npz or --dataset-path")
        dataset = LeRobotEpisodeLoader(
            dataset_path=str(args.dataset_path),
            modality_configs=policy.get_modality_config(),
        )
        observation = prepare_observation(policy, dataset, traj_idx=0)

    if args.calls < 2:
        raise ValueError("--calls must be at least 2")
    print(f"[1/2] PyTorch initial + {args.calls - 1} Prefix-RTC continuations")
    pt_outputs = run_sequence(policy, observation, args.seed, args.overlap, args.calls)
    if archived is not None:
        report("archived_vs_pytorch", archived, pt_outputs[0])

    print(f"[2/2] {args.mode} initial + {args.calls - 1} Prefix-RTC continuations")
    setup_tensorrt_engines(policy, str(args.engine_dir), mode=args.mode)
    trt_outputs = run_sequence(policy, observation, args.seed, args.overlap, args.calls)

    failures = []
    for index, (pt_output, trt_output) in enumerate(zip(pt_outputs, trt_outputs)):
        result = report(f"call{index:02d}.normalized_action", pt_output, trt_output)
        if result["cosine"] < args.min_cosine:
            failures.append(f"call{index:02d} cosine {result['cosine']} < {args.min_cosine}")
        if result["mean_abs_error"] > args.max_mean_abs_error:
            failures.append(
                f"call{index:02d} MAE {result['mean_abs_error']} > {args.max_mean_abs_error}"
            )
        if result["max_abs_error"] > args.max_abs_error:
            failures.append(f"call{index:02d} max {result['max_abs_error']} > {args.max_abs_error}")
        if index:
            report(
                f"call{index:02d}.generated_suffix",
                pt_output[:, args.overlap :],
                trt_output[:, args.overlap :],
            )
    if failures:
        raise RuntimeError("Prefix-RTC parity failed: " + "; ".join(failures))
    print("PASS: consecutive Prefix-RTC TensorRT parity")


if __name__ == "__main__":
    main()
