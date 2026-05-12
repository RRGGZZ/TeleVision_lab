from pathlib import Path
from typing import List, Optional

import h5py
import numpy as np

from .contracts import TELEOP_CMD_SCHEMA, TELEOP_STATE_SCHEMA


def _to_chw_uint8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[2] in (3, 4):
        arr = arr[..., :3]
        arr = np.transpose(arr, (2, 0, 1))
    if arr.ndim != 3:
        raise ValueError(f"Expected image with 3 dims, got shape={arr.shape}")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


class EpisodeRecorder:
    def __init__(
        self,
        *,
        action_schema: str = TELEOP_CMD_SCHEMA,
        cmd_schema: str = TELEOP_CMD_SCHEMA,
        state_schema: str = TELEOP_STATE_SCHEMA,
    ):
        self.left_images: List[np.ndarray] = []
        self.right_images: List[np.ndarray] = []
        self.states: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.cmds: List[np.ndarray] = []
        self.action_schema = action_schema
        self.cmd_schema = cmd_schema
        self.state_schema = state_schema

    def append(
        self,
        left_rgb: np.ndarray,
        right_rgb: np.ndarray,
        state: np.ndarray,
        action: np.ndarray,
        cmd: Optional[np.ndarray] = None,
    ):
        self.left_images.append(_to_chw_uint8(left_rgb))
        self.right_images.append(_to_chw_uint8(right_rgb))
        self.states.append(np.asarray(state, dtype=np.float32))
        self.actions.append(np.asarray(action, dtype=np.float32))
        if cmd is None:
            cmd = np.asarray(action, dtype=np.float32)
        self.cmds.append(np.asarray(cmd, dtype=np.float32))

    def save(self, output_file: Path):
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(str(output_file), "w") as hf:
            hf.create_dataset("observation.image.left", data=np.stack(self.left_images, axis=0))
            hf.create_dataset("observation.image.right", data=np.stack(self.right_images, axis=0))
            hf.create_dataset("observation.state", data=np.stack(self.states, axis=0).astype(np.float32))
            hf.create_dataset("qpos_action", data=np.stack(self.actions, axis=0).astype(np.float32))
            hf.create_dataset("cmds", data=np.stack(self.cmds, axis=0).astype(np.float32))
            hf.attrs["sim"] = True
            hf.attrs["init_action"] = np.asarray(self.cmds[0], dtype=np.float32)
            hf.attrs["action_schema"] = self.action_schema
            hf.attrs["cmd_schema"] = self.cmd_schema
            hf.attrs["state_schema"] = self.state_schema
