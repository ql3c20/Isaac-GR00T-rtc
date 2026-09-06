#!/usr/bin/env python3
"""Offline parity check for PyTorch vs Prefix-RTC TensorRT action head.

This tool is intentionally standalone. It is not imported by GR00T serving or
the TensorRT runtime, so adding it cannot change the default PyTorch chain.
"""

from __future__ import annotations

import argparse
from contextlib import AbstractContextManager
import gc
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


ENGINE_CONTRACTS = {
    "state_encoder.engine": {
        "state": (1, 1, 132),
        "embodiment_id": (1,),
        "output": (1, 1, 1536),
    },
    "action_encoder_prefix_rtc.engine": {
        "actions": (1, 40, 132),
        "timesteps": (1, 40),
        "embodiment_id": (1,),
        "output": (1, 40, 1536),
    },
    "dit_prefix_rtc_bf16.engine": {
        "sa_embs": (1, 41, 1536),
        "vl_embs": (1, -1, 2048),
        "timestep": (1, 41),
        "image_mask": (1, -1),
        "backbone_attention_mask": (1, -1),
        "output": (1, 41, 1024),
    },
    "action_decoder.engine": {
        "model_output": (1, 41, 1024),
        "embodiment_id": (1,),
        "output": (1, 41, 132),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Prefix-RTC PyTorch/TRT action-head parity.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--backbone-path", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--engine-dir", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--embodiment-tag", default="new_embodiment")
    parser.add_argument("--overlap", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--prefix-timestep-mode",
        choices=("groot_clean", "legacy_zero"),
        default="legacy_zero",
        help="Must match serving semantics; this checkpoint's baseline uses legacy_zero.",
    )
    parser.add_argument("--min-cosine", type=float, default=0.99)
    parser.add_argument("--max-abs-error", type=float, default=0.1)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def require_dir(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def validate_paths(args: argparse.Namespace) -> dict[str, Path]:
    model_path = require_dir(args.model_path, "model directory")
    backbone_path = require_dir(args.backbone_path, "backbone directory")
    dataset_path = require_dir(args.dataset_path, "dataset directory")
    engine_dir = require_dir(args.engine_dir, "engine directory")
    adapter_path = require_file(args.adapter_path, "Prefix-RTC adapter")

    for relative in (
        "config.json",
        "model.safetensors.index.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ):
        require_file(model_path / relative, f"model asset {relative}")
    for filename in (
        "processor_config.json",
        "statistics.json",
        "embodiment_id.json",
    ):
        root_layout = model_path / filename
        split_layout = model_path / "processor" / filename
        if root_layout.is_file():
            require_file(root_layout, f"model asset {filename}")
        else:
            require_file(split_layout, f"model asset processor/{filename}")
    for relative in ("config.json", "tokenizer.json", "model.safetensors"):
        require_file(backbone_path / relative, f"backbone asset {relative}")
    require_file(dataset_path / "meta/info.json", "LeRobot dataset metadata")
    for filename in ENGINE_CONTRACTS:
        require_file(engine_dir / filename, f"TensorRT engine {filename}")

    if not 0 < args.overlap < 40:
        raise ValueError(f"--overlap must be in [1,39], got {args.overlap}")
    if not 0.0 < args.min_cosine <= 1.0:
        raise ValueError("--min-cosine must be in (0,1]")
    if args.max_abs_error <= 0:
        raise ValueError("--max-abs-error must be positive")
    return {
        "model": model_path,
        "backbone": backbone_path,
        "dataset": dataset_path,
        "engines": engine_dir,
        "adapter": adapter_path,
    }


def validate_engine_contracts(engine_dir: Path) -> dict[str, Any]:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    details: dict[str, Any] = {}
    for filename, expected in ENGINE_CONTRACTS.items():
        path = engine_dir / filename
        engine = runtime.deserialize_cuda_engine(path.read_bytes())
        if engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {path}")
        actual = {
            engine.get_tensor_name(index): tuple(
                int(value) for value in engine.get_tensor_shape(engine.get_tensor_name(index))
            )
            for index in range(engine.num_io_tensors)
        }
        missing = sorted(set(expected) - set(actual))
        if missing:
            raise ValueError(f"{filename} is missing tensors: {missing}")
        mismatched = {
            name: {"expected": shape, "actual": actual[name]}
            for name, shape in expected.items()
            if actual[name] != shape
        }
        if mismatched:
            raise ValueError(f"{filename} shape mismatch: {mismatched}")
        details[filename] = actual
        del engine
        gc.collect()
    return details


class FixedNoise(AbstractContextManager[None]):
    """Make both policies start their action denoising from identical noise."""

    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor
        self.original = torch.randn

    def __enter__(self) -> None:
        fixed = self.tensor

        def fake_randn(*args, **kwargs):
            size = kwargs.get("size", args[0] if args else None)
            if tuple(size) != tuple(fixed.shape):
                raise AssertionError(
                    f"Unexpected torch.randn shape {tuple(size)}; "
                    f"fixed action noise is {tuple(fixed.shape)}"
                )
            return fixed.to(
                dtype=kwargs.get("dtype", fixed.dtype),
                device=kwargs.get("device", fixed.device),
            )

        torch.randn = fake_randn
        return None

    def __exit__(self, *_exc) -> None:
        torch.randn = self.original


def load_adapter(adapter_path: Path):
    adapter_dir = str(adapter_path.parent)
    if adapter_dir not in sys.path:
        sys.path.insert(0, adapter_dir)
    module_name = adapter_path.stem
    module = __import__(module_name)
    apply_prefix_rtc = getattr(module, "apply_prefix_rtc", None)
    if not callable(apply_prefix_rtc):
        raise RuntimeError(f"Adapter has no callable apply_prefix_rtc: {adapter_path}")
    return apply_prefix_rtc


def make_policy(paths: dict[str, Path], args: argparse.Namespace):
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    tag = EmbodimentTag.resolve(args.embodiment_tag)
    policy = Gr00tPolicy(
        model_path=str(paths["model"]),
        backbone_path=str(paths["backbone"]),
        embodiment_tag=tag,
        device="cuda",
        strict=True,
    )
    apply_prefix_rtc = load_adapter(paths["adapter"])
    apply_prefix_rtc(policy, prefix_timestep_mode=args.prefix_timestep_mode)
    return policy, tag


def load_observation(policy, dataset_path: Path, embodiment_tag):
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data

    modality_config = policy.get_modality_config()
    dataset = LeRobotEpisodeLoader(dataset_path=str(dataset_path), modality_configs=modality_config)
    episode_data = dataset[0]
    step_data = extract_step_data(
        episode_data,
        step_index=0,
        modality_configs=modality_config,
        embodiment_tag=embodiment_tag,
        allow_padding=False,
    )
    observation = {
        "video": {key: np.stack(step_data.images[key])[None] for key in step_data.images},
        "state": {key: step_data.states[key][None] for key in step_data.states},
        "language": {modality_config["language"].modality_keys[0]: [[step_data.text]]},
    }
    return observation


def normalized_action(info: dict[str, Any]) -> np.ndarray:
    if "normalized_action_pred" not in info:
        raise KeyError(f"Policy info has no normalized_action_pred; keys={sorted(info.keys())}")
    return np.asarray(info["normalized_action_pred"], dtype=np.float32)


def bf16_roundtrip(value: np.ndarray) -> np.ndarray:
    return torch.from_numpy(value).to(torch.bfloat16).float().numpy()


def run_parity(paths: dict[str, Path], args: argparse.Namespace) -> dict[str, float]:
    print("[A] loading PyTorch policy + Prefix-RTC adapter", flush=True)
    policy_torch, tag = make_policy(paths, args)
    observation = load_observation(policy_torch, paths["dataset"], tag)
    action_head = policy_torch.model.action_head
    horizon = int(action_head.config.action_horizon)
    action_dim = int(action_head.action_dim)
    if horizon != 40 or action_dim != 132:
        raise ValueError(
            f"Engine contract requires horizon=40/action_dim=132, got {horizon}/{action_dim}"
        )

    rng = np.random.default_rng(0)
    previous = rng.normal(size=(horizon, action_dim)).astype(np.float32)
    options = {
        "rtc_prev_action": previous,
        "rtc_overlap_steps": args.overlap,
        "action_horizon": horizon,
        "rtc_frozen_steps": 0,
        "rtc_ramp_rate": 1.0,
    }
    torch.manual_seed(args.seed)
    noise = torch.randn(1, horizon, action_dim, dtype=torch.bfloat16, device="cuda")
    with FixedNoise(noise), torch.inference_mode():
        _, info_torch = policy_torch.get_action(observation, options)
    action_torch = normalized_action(info_torch)
    del policy_torch
    gc.collect()
    torch.cuda.empty_cache()

    print("[B] loading PyTorch backbone + TensorRT Prefix-RTC action head", flush=True)
    policy_trt, _ = make_policy(paths, args)
    deployment_dir = str(Path(__file__).resolve().parent)
    if deployment_dir not in sys.path:
        sys.path.insert(0, deployment_dir)
    from trt_model_forward import setup_tensorrt_engines

    setup_tensorrt_engines(policy_trt, str(paths["engines"]), mode="prefix_rtc_action_head")
    with FixedNoise(noise), torch.inference_mode():
        _, info_trt = policy_trt.get_action(observation, options)
    action_trt = normalized_action(info_trt)

    flat_torch = torch.from_numpy(action_torch).float().flatten()
    flat_trt = torch.from_numpy(action_trt).float().flatten()
    cosine = float(
        torch.nn.functional.cosine_similarity(flat_torch.unsqueeze(0), flat_trt.unsqueeze(0)).item()
    )
    absolute = np.abs(action_torch - action_trt)
    max_abs = float(absolute.max())
    mean_abs = float(absolute.mean())
    overlap = args.overlap
    expected_prefix = bf16_roundtrip(previous[None, horizon - overlap : horizon])
    torch_prefix_error = float(np.abs(action_torch[:, :overlap] - expected_prefix).max())
    trt_prefix_error = float(np.abs(action_trt[:, :overlap] - expected_prefix).max())
    prefix_cross_error = float(np.abs(action_torch[:, :overlap] - action_trt[:, :overlap]).max())
    suffix_max_abs = float(absolute[:, overlap:].max())
    result = {
        "cosine": cosine,
        "mean_abs_error": mean_abs,
        "max_abs_error": max_abs,
        "suffix_max_abs_error": suffix_max_abs,
        "torch_prefix_error": torch_prefix_error,
        "trt_prefix_error": trt_prefix_error,
        "prefix_cross_error": prefix_cross_error,
    }
    print("parity:", json.dumps(result, indent=2, sort_keys=True), flush=True)
    if torch_prefix_error > 1e-6 or trt_prefix_error > 1e-6:
        raise RuntimeError("Prefix hard-rewrite parity failed")
    if cosine < args.min_cosine:
        raise RuntimeError(f"Cosine parity failed: {cosine} < {args.min_cosine}")
    if max_abs > args.max_abs_error:
        raise RuntimeError(f"Absolute parity failed: {max_abs} > {args.max_abs_error}")
    print("PASS: Prefix-RTC action-head TensorRT parity", flush=True)
    return result


def main() -> None:
    args = parse_args()
    paths = validate_paths(args)
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    import tensorrt as trt

    environment = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "tensorrt": trt.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_capability": torch.cuda.get_device_capability(0),
    }
    print("environment:", json.dumps(environment, indent=2, default=list))
    contracts = validate_engine_contracts(paths["engines"])
    print("engine contracts:", json.dumps(contracts, indent=2))
    if args.check_only:
        print("check-only passed; no model loaded and no files written")
        return
    run_parity(paths, args)


if __name__ == "__main__":
    main()
