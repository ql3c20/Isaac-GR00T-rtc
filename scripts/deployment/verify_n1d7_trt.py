#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Quick verification: compare PyTorch vs TRT action head outputs for N1.7."""

from dataclasses import dataclass
import os
import sys

import numpy as np
import torch
from torch.nn.functional import cosine_similarity
import tyro


# Make sibling imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from export_onnx_n1d7 import prepare_observation
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.deployment.modes import VerifyMode
from gr00t.policy.gr00t_policy import Gr00tPolicy
from modality_config_utils import import_modality_config
from prefix_rtc_utils import apply_prefix_rtc


@dataclass
class VerifyConfig:
    """Configuration for TRT verification."""

    model_path: str
    """Path to model checkpoint (required)."""

    dataset_path: str = "demo_data/libero_demo"
    """Path to dataset."""

    engine_dir: str = "./gr00t_n1d7_engines"
    """Directory with TRT engines."""

    mode: VerifyMode = VerifyMode.action_head
    """TRT setup mode: 'vit_llm_only' uses ViT+LLM TRT with PyTorch action head."""

    embodiment_tag: EmbodimentTag = EmbodimentTag.LIBERO_PANDA
    """Embodiment tag to use."""

    modality_config_path: str | None = None
    """Optional Python file that registers a custom embodiment modality config."""

    batch_size: int = 1
    """Batch size for TRT inference. If > 1, tiles the observation and takes slice [0] for comparison."""

    traj_idx: int = 0
    """Trajectory index to sample from the dataset for PyTorch/TRT comparison."""

    step_idx: int = 0
    """Step index within the selected trajectory for PyTorch/TRT comparison."""

    prefix_rtc_timestep_mode: str | None = None
    """Prefix-RTC timestep convention for Prefix-RTC TRT verification."""

    rtc_overlap_steps: int = 6
    """Number of RTC overlap frames for Prefix-RTC TRT verification."""


def _tile_observation(obs, n):
    """Tile a single observation dict to batch size n."""
    tiled = {}
    for modality, entries in obs.items():
        tiled[modality] = {}
        for key, val in entries.items():
            if isinstance(val, list):
                # language: [["text"]] -> [["text"]] * n
                tiled[modality][key] = val * n
            else:
                # numpy/tensor: repeat along batch dim 0
                import numpy as np

                if isinstance(val, np.ndarray):
                    tiled[modality][key] = np.repeat(val, n, axis=0)
                else:
                    tiled[modality][key] = val.repeat(n, *([1] * (val.ndim - 1)))
    return tiled


def main(args: VerifyConfig | None = None):
    if args is None:
        args = tyro.cli(VerifyConfig)
    verify_mode = str(args.mode)
    prefix_rtc_mode = verify_mode in ("prefix_rtc_action_head", "prefix_rtc_full_pipeline")

    print("=" * 60)
    print("N1.7 TRT Action Head Verification")
    print("=" * 60)

    # Step 1: Load policy and get PyTorch reference output
    print("\n[1] Loading policy...")
    import_modality_config(args.modality_config_path)
    policy = Gr00tPolicy(
        embodiment_tag=args.embodiment_tag,
        model_path=args.model_path,
        device="cuda",
    )
    if prefix_rtc_mode:
        apply_prefix_rtc(policy, prefix_timestep_mode=args.prefix_rtc_timestep_mode)

    print("[2] Loading dataset...")
    dataset = LeRobotEpisodeLoader(
        dataset_path=args.dataset_path,
        modality_configs=policy.get_modality_config(),
    )

    # --- Capture ViT input/output and backbone output from PyTorch ---
    pt_backbone_features = None
    pt_vit_output = None
    pt_vit_input = None

    def _capture_backbone_hook(module, args, output):
        nonlocal pt_backbone_features
        pt_backbone_features = output["backbone_features"].detach().clone()
        return output

    def _capture_vit_hook(module, args, kwargs, output):
        nonlocal pt_vit_output, pt_vit_input
        # Capture input pixel_values
        if args:
            pt_vit_input = args[0].detach().clone()
        elif "pixel_values" in kwargs:
            pt_vit_input = kwargs["pixel_values"].detach().clone()
        # Capture output (image_embeds after merger)
        if isinstance(output, tuple):
            pt_vit_output = output[0].detach().clone()
        else:
            pt_vit_output = output.detach().clone()
        return output

    backbone_hook = policy.model.backbone.register_forward_hook(_capture_backbone_hook)
    vit_hook = policy.model.backbone.model.model.visual.register_forward_hook(
        _capture_vit_hook, with_kwargs=True
    )

    obs = prepare_observation(policy, dataset, traj_idx=args.traj_idx, step_idx=args.step_idx)
    rtc_options = None
    if prefix_rtc_mode:
        action_horizon = int(policy.model.action_head.config.action_horizon)
        rtc_overlap_steps = int(args.rtc_overlap_steps)
        if not (0 < rtc_overlap_steps < action_horizon):
            raise ValueError(
                f"rtc_overlap_steps must be in (0, {action_horizon}), got {rtc_overlap_steps}"
            )
        print("[3a] Preparing Prefix-RTC previous action from PyTorch unconditioned output...")
        torch.manual_seed(123)
        with torch.inference_mode():
            prev_result = policy.get_action(obs)
        prev_info = prev_result[1] if isinstance(prev_result, tuple) else {}
        if "normalized_action_pred" not in prev_info:
            raise RuntimeError("PyTorch policy did not return normalized_action_pred for RTC")
        rtc_prev_action = np.asarray(prev_info["normalized_action_pred"], dtype=np.float32)
        rtc_options = {
            "rtc_prev_action": rtc_prev_action,
            "action_horizon": int(rtc_prev_action.shape[1]),
            "rtc_overlap_steps": rtc_overlap_steps,
        }
        print(
            "  rtc_prev_action: "
            f"shape={rtc_prev_action.shape}, overlap={rtc_overlap_steps}"
        )

    print("[3] Running PyTorch inference...")
    torch.manual_seed(42)
    with torch.inference_mode():
        result = policy.get_action(obs, rtc_options)

    backbone_hook.remove()
    vit_hook.remove()

    # get_action returns (action_dict, info_dict)
    action_dict = result[0] if isinstance(result, tuple) else result
    pt_info = result[1] if isinstance(result, tuple) else {}
    print(f"  Action keys: {list(action_dict.keys())}")

    # Concatenate all action arrays into a single tensor for comparison
    pt_arrays = []
    for k in sorted(action_dict.keys()):
        v = action_dict[k]
        t = torch.tensor(v) if not isinstance(v, torch.Tensor) else v
        pt_arrays.append(t.float().flatten())
        print(f"  {k}: shape={v.shape if hasattr(v, 'shape') else len(v)}")
    pt_action = torch.cat(pt_arrays)

    # Step 2: Setup TRT engines and run
    print("\n[4] Loading TRT engines...")
    from trt_model_forward import setup_tensorrt_engines

    setup_tensorrt_engines(policy, args.engine_dir, mode=args.mode)

    # --- Capture backbone output from TRT ---
    trt_backbone_features = None

    def _capture_trt_backbone_hook(module, args, output):
        nonlocal trt_backbone_features
        trt_backbone_features = output["backbone_features"].detach().clone()
        return output

    backbone_hook2 = policy.model.backbone.register_forward_hook(_capture_trt_backbone_hook)

    # Run ViT TRT with the same pixel_values captured during PyTorch pass
    trt_vit_output = None
    if pt_vit_input is not None and getattr(policy.model.backbone, "vit_engine", None) is not None:
        vit_dtype = policy.model.backbone.vit_engine.dtype_of("pixel_values")
        pv = pt_vit_input.to(vit_dtype).cuda().contiguous()
        # For batch_size > 1, tile pixel_values to match engine's expected num_patches
        if args.batch_size > 1:
            pv = pv.repeat(args.batch_size, 1)
        policy.model.backbone.vit_engine.set_runtime_tensor_shape("pixel_values", pv.shape)
        vit_result = policy.model.backbone.vit_engine(pv)
        # Take first batch's merged patches for comparison
        num_merged = (
            pt_vit_output.shape[0]
            if pt_vit_output is not None
            else vit_result["image_embeds"].shape[0] // args.batch_size
        )
        trt_vit_output = vit_result["image_embeds"][:num_merged].detach().clone()

    print("[5] Running TRT inference...")
    obs2 = prepare_observation(policy, dataset, traj_idx=args.traj_idx, step_idx=args.step_idx)
    if args.batch_size > 1:
        print(f"  Tiling observation to batch_size={args.batch_size}")
        obs2 = _tile_observation(obs2, args.batch_size)
    torch.manual_seed(42)
    with torch.inference_mode():
        result2 = policy.get_action(obs2, rtc_options)

    backbone_hook2.remove()

    action_dict2 = result2[0] if isinstance(result2, tuple) else result2
    trt_info = result2[1] if isinstance(result2, tuple) else {}
    trt_arrays = []
    for k in sorted(action_dict2.keys()):
        v = action_dict2[k]
        # For batch_size > 1, take slice [0] to compare against single PyTorch output
        if args.batch_size > 1 and hasattr(v, "shape") and v.shape[0] == args.batch_size:
            v = v[0:1]
        t = torch.tensor(v) if not isinstance(v, torch.Tensor) else v
        trt_arrays.append(t.float().flatten())
    trt_act = torch.cat(trt_arrays)

    # Step 3a: Compare ViT outputs
    if pt_vit_output is not None and trt_vit_output is not None:
        vit_pt = pt_vit_output.float().flatten()
        vit_trt = trt_vit_output.float().flatten()
        vit_cosine = cosine_similarity(vit_pt.unsqueeze(0), vit_trt.unsqueeze(0)).item()
        vit_l1 = (vit_pt - vit_trt).abs().mean().item()
        vit_linf = (vit_pt - vit_trt).abs().max().item()
        print("\n[6a] ViT output comparison (image_embeds):")
        print(f"  Cosine Similarity: {vit_cosine:.6f}")
        print(f"  L1 Mean Error:     {vit_l1:.6f}")
        print(f"  L∞ Max Error:      {vit_linf:.6f}")
    else:
        print("\n[6a] ViT comparison skipped (PyTorch ViT was deleted before capture)")

    # Step 3b: Compare backbone outputs (before vl_self_attention)
    if pt_backbone_features is not None and trt_backbone_features is not None:
        bb_pt = pt_backbone_features.float().flatten()
        # For batch_size > 1, take slice [0] to match PyTorch single-batch output
        trt_bb = trt_backbone_features[:1] if args.batch_size > 1 else trt_backbone_features
        bb_trt = trt_bb.float().flatten()
        bb_cosine = cosine_similarity(bb_pt.unsqueeze(0), bb_trt.unsqueeze(0)).item()
        bb_l1 = (bb_pt - bb_trt).abs().mean().item()
        bb_linf = (bb_pt - bb_trt).abs().max().item()
        print("\n[6b] Backbone output comparison (LLM output, before vl_self_attention):")
        print(f"  Cosine Similarity: {bb_cosine:.6f}")
        print(f"  L1 Mean Error:     {bb_l1:.6f}")
        print(f"  L∞ Max Error:      {bb_linf:.6f}")

    # Step 4: Compare final action outputs
    print("\n[6b] Final action output comparison:")
    pt_flat = pt_action.float().flatten()
    trt_flat = trt_act.float().flatten()

    cosine = cosine_similarity(pt_flat.unsqueeze(0), trt_flat.unsqueeze(0)).item()
    l1 = (pt_flat - trt_flat).abs().mean().item()
    linf = (pt_flat - trt_flat).abs().max().item()

    print(f"\n  Cosine Similarity: {cosine:.6f}")
    print(f"  L1 Mean Error:     {l1:.6f}")
    print(f"  L∞ Max Error:      {linf:.6f}")

    normalized_suffix_cosine = None
    prefix_linf = None
    if prefix_rtc_mode:
        if "normalized_action_pred" not in pt_info or "normalized_action_pred" not in trt_info:
            raise RuntimeError("Prefix-RTC verification requires normalized_action_pred in info")
        pt_norm = torch.as_tensor(pt_info["normalized_action_pred"]).float()
        trt_norm = torch.as_tensor(trt_info["normalized_action_pred"]).float()
        if args.batch_size > 1 and trt_norm.shape[0] == args.batch_size:
            trt_norm = trt_norm[0:1]
        if pt_norm.shape != trt_norm.shape:
            raise RuntimeError(
                f"normalized_action_pred shape mismatch: PyTorch {pt_norm.shape}, TRT {trt_norm.shape}"
            )
        overlap = int(args.rtc_overlap_steps)
        pt_prefix = pt_norm[:, :overlap]
        trt_prefix = trt_norm[:, :overlap]
        pt_suffix = pt_norm[:, overlap:]
        trt_suffix = trt_norm[:, overlap:]

        prefix_flat = pt_prefix.flatten()
        trt_prefix_flat = trt_prefix.flatten()
        suffix_flat = pt_suffix.flatten()
        trt_suffix_flat = trt_suffix.flatten()

        prefix_cosine = cosine_similarity(
            prefix_flat.unsqueeze(0), trt_prefix_flat.unsqueeze(0)
        ).item()
        prefix_l1 = (prefix_flat - trt_prefix_flat).abs().mean().item()
        prefix_linf = (prefix_flat - trt_prefix_flat).abs().max().item()
        normalized_suffix_cosine = cosine_similarity(
            suffix_flat.unsqueeze(0), trt_suffix_flat.unsqueeze(0)
        ).item()
        suffix_l1 = (suffix_flat - trt_suffix_flat).abs().mean().item()
        suffix_linf = (suffix_flat - trt_suffix_flat).abs().max().item()

        print("\n[6c] Prefix-RTC normalized action comparison:")
        print(f"  Prefix Cosine:     {prefix_cosine:.6f}")
        print(f"  Prefix L1 Mean:    {prefix_l1:.6f}")
        print(f"  Prefix L∞ Max:     {prefix_linf:.6f}")
        print(f"  Suffix Cosine:     {normalized_suffix_cosine:.6f}")
        print(f"  Suffix L1 Mean:    {suffix_l1:.6f}")
        print(f"  Suffix L∞ Max:     {suffix_linf:.6f}")

    if prefix_rtc_mode and (
        cosine > 0.999
        and normalized_suffix_cosine is not None
        and normalized_suffix_cosine > 0.999
        and prefix_linf is not None
        and prefix_linf < 1e-3
    ):
        print("\n  PASS — Prefix-RTC TRT matches PyTorch")
    elif not prefix_rtc_mode and cosine > 0.999:
        print("\n  PASS — TRT matches PyTorch")
    elif cosine > 0.99:
        print("\n  WARN — Minor drift detected")
    else:
        print("\n  FAIL — Significant divergence")

    return cosine


if __name__ == "__main__":
    config = tyro.cli(VerifyConfig)
    main(config)
