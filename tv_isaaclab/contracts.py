from __future__ import annotations

from pathlib import Path
from typing import Optional

import h5py
import numpy as np


TELEOP_TASK_ID = "television_lab"
H1_TASK_ID = "television_h1"

TELEOP_CMD_SCHEMA = "teleop_command_v1"
TELEOP_STATE_SCHEMA = "teleop_state_v1"
H1_ACTION_SCHEMA = "h1_joint_action_v1"
H1_STATE_SCHEMA = "h1_joint_state_v1"
H1_LEGACY_ACTION_SCHEMA = "h1_joint_action_legacy28_v1"

TELEOP_ACTION_DIM = 38
H1_ACTION_DIM = 26
H1_LEGACY_ACTION_DIM = 28


def expand_inspire_driver_qpos(qpos: np.ndarray) -> np.ndarray:
    """Expand 6-dim Inspire driver joints into the legacy 12-dim command layout."""
    q = np.asarray(qpos, dtype=np.float32).reshape(-1)
    if q.shape[0] == 12:
        return q
    if q.shape[0] != 6:
        raise ValueError(f"Unexpected retarget qpos dim: {q.shape[0]} (expected 6 or 12)")
    index_prox, middle_prox, pinky_prox, ring_prox, thumb_yaw, thumb_pitch = q.tolist()
    return np.array(
        [
            index_prox,
            index_prox,
            middle_prox,
            middle_prox,
            pinky_prox,
            pinky_prox,
            ring_prox,
            ring_prox,
            thumb_yaw,
            thumb_pitch,
            1.6 * thumb_pitch,
            2.4 * thumb_pitch,
        ],
        dtype=np.float32,
    )


def assemble_teleop_action(
    left_pose: np.ndarray,
    right_pose: np.ndarray,
    left_qpos: np.ndarray,
    right_qpos: np.ndarray,
) -> np.ndarray:
    left_pose = np.asarray(left_pose, dtype=np.float32).reshape(-1)
    right_pose = np.asarray(right_pose, dtype=np.float32).reshape(-1)
    left_qpos = expand_inspire_driver_qpos(left_qpos)
    right_qpos = expand_inspire_driver_qpos(right_qpos)
    if left_pose.shape[0] != 7 or right_pose.shape[0] != 7:
        raise ValueError("Teleop wrist poses must be 7D [xyz, xyzw].")
    return np.concatenate([left_pose, right_pose, left_qpos, right_qpos], axis=0)


def adapt_h1_action(action: np.ndarray) -> np.ndarray:
    """Normalize H1 replay actions to the canonical 26D control vector."""
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.shape[0] == H1_ACTION_DIM:
        return action
    if action.shape[0] == H1_LEGACY_ACTION_DIM:
        return action[:H1_ACTION_DIM].copy()
    raise ValueError(
        f"Expected H1 action dim {H1_ACTION_DIM} or legacy dim {H1_LEGACY_ACTION_DIM}, got {action.shape[0]}."
    )


def h1_action_to_qpos(action: np.ndarray) -> np.ndarray:
    """Convert compact H1 arm+hand actions into the 51-DoF URDF joint vector."""
    compact = adapt_h1_action(action)
    qpos = np.zeros(51, dtype=np.float32)
    qpos[13:20] = compact[0:7]
    qpos[20:22] = compact[7]
    qpos[22:24] = compact[8]
    qpos[24:26] = compact[9]
    qpos[26:28] = compact[10]
    qpos[28] = compact[11]
    qpos[29:32] = compact[12] * np.array([1.0, 1.6, 2.4], dtype=np.float32)
    qpos[32:39] = compact[13:20]
    qpos[39:41] = compact[20]
    qpos[41:43] = compact[21]
    qpos[43:45] = compact[22]
    qpos[45:47] = compact[23]
    qpos[47] = compact[24]
    qpos[48:51] = compact[25] * np.array([1.0, 1.6, 2.4], dtype=np.float32)
    return qpos


def infer_action_schema(
    action_dim: int,
    *,
    cmd_dim: Optional[int] = None,
) -> str:
    if action_dim == TELEOP_ACTION_DIM:
        return TELEOP_CMD_SCHEMA
    if action_dim == H1_ACTION_DIM:
        return H1_ACTION_SCHEMA
    if action_dim == H1_LEGACY_ACTION_DIM:
        return H1_LEGACY_ACTION_SCHEMA
    if cmd_dim == TELEOP_ACTION_DIM and action_dim in (H1_ACTION_DIM, H1_LEGACY_ACTION_DIM):
        return H1_ACTION_SCHEMA if action_dim == H1_ACTION_DIM else H1_LEGACY_ACTION_SCHEMA
    return f"unknown_action_dim_{action_dim}"


def infer_task_from_schemas(
    action_schema: Optional[str],
    action_dim: Optional[int],
    *,
    fallback: str = TELEOP_TASK_ID,
) -> str:
    if action_schema in (H1_ACTION_SCHEMA, H1_LEGACY_ACTION_SCHEMA):
        return H1_TASK_ID
    if action_schema == TELEOP_CMD_SCHEMA:
        return TELEOP_TASK_ID
    if action_dim in (H1_ACTION_DIM, H1_LEGACY_ACTION_DIM):
        return H1_TASK_ID
    if action_dim == TELEOP_ACTION_DIM:
        return TELEOP_TASK_ID
    return fallback


def infer_task_from_episode(
    episode_path: Path | str,
    *,
    fallback: str = TELEOP_TASK_ID,
) -> str:
    episode_path = Path(episode_path)
    if not episode_path.exists():
        return fallback
    with h5py.File(str(episode_path), "r") as handle:
        action_schema = handle.attrs.get("action_schema")
        if isinstance(action_schema, bytes):
            action_schema = action_schema.decode("utf-8")
        action_dim = None
        if "qpos_action" in handle:
            action_dim = int(handle["qpos_action"].shape[-1])
    return infer_task_from_schemas(action_schema, action_dim, fallback=fallback)
