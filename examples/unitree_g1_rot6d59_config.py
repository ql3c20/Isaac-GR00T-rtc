# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""GR00T N1.7 custom modality for the Psi0 rot6d59 -> Kimodo -> TextOp path.

This uses the same observation/action semantics as the Psi0 rot6d59 policy:

  observation.full_state_rot6d: 52D
    hand14 + body29 + root9(xyz, rot6d)

  action.policy_action_rot6d59: 59D
    hand14 + root9(xyz, rot6d) + 4 EE poses * 9(xyz, rot6d)

The dataset meta/modality.json exposes these columns as rot59_* groups, so GR00T
can train a NEW_EMBODIMENT policy without changing the LeRobot parquet columns.
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


unitree_g1_rot6d59_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["ego_view"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "rot59_left_hand",
            "rot59_right_hand",
            "rot59_left_leg",
            "rot59_right_leg",
            "rot59_waist",
            "rot59_left_arm",
            "rot59_right_arm",
            "rot59_root",
        ],
    ),
    "action": ModalityConfig(
        # Match the GR00T N1.7 G1/SONIC prediction horizon.
        # Execution can still use a shorter receding horizon downstream.
        delta_indices=list(range(40)),
        modality_keys=[
            "rot59_hand",
            "rot59_root",
            "rot59_left_hand_pose",
            "rot59_right_hand_pose",
            "rot59_left_foot_pose",
            "rot59_right_foot_pose",
        ],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}


register_modality_config(unitree_g1_rot6d59_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
