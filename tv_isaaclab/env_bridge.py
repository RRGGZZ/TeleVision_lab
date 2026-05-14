import importlib
import os
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import numpy as np

from .contracts import infer_action_schema, infer_task_from_schemas


def _try_import_isaaclab_tasks():
    """Try importing Isaac Lab tasks, including custom television_lab."""
    # Try custom television_lab first
    try:
        import tv_isaaclab.tasks  # noqa: F401

        return
    except Exception:
        pass

    # Try standard Isaac Lab tasks
    for module_name in ("isaaclab_tasks", "omni.isaac.lab_tasks"):
        try:
            importlib.import_module(module_name)
            return
        except ModuleNotFoundError:
            continue


def _as_numpy(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if isinstance(value, np.ndarray):
        return value
    try:
        return np.asarray(value)
    except Exception:
        return None


def _resolve_key_path(data: Any, key_path: str) -> Any:
    current = data
    for key in key_path.split("."):
        if isinstance(current, dict):
            if key not in current:
                return None
            current = current[key]
        else:
            return None
    return current


def _to_hwc_uint8(img: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if img is None:
        return None
    array = _as_numpy(img)
    if array is None:
        return None
    if array.ndim == 4:
        array = array[0]
    if (
        array.ndim == 3
        and array.shape[0] in (1, 3, 4)
        and array.shape[2] not in (1, 3, 4)
    ):
        array = np.transpose(array, (1, 2, 0))
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    if array.shape[-1] == 4:
        array = array[..., :3]
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


@dataclass
class ObsPack:
    left_rgb: np.ndarray
    right_rgb: np.ndarray
    state: np.ndarray
    raw_obs: Any


class IsaacLabEnvBridge:
    def __init__(
        self,
        task: str,
        render_mode: str = "rgb_array",
        left_image_keys: Optional[Iterable[str]] = None,
        right_image_keys: Optional[Iterable[str]] = None,
        state_keys: Optional[Iterable[str]] = None,
    ):
        import gymnasium as gym

        _try_import_isaaclab_tasks()
        self.task = task
        self.env = self._create_env(task=task, render_mode=render_mode)

        self.left_image_keys = list(
            left_image_keys
            or [
                "observation.image.left",  # television_lab
                "policy.left_rgb",
                "left_rgb",
                "images.left",
            ]
        )
        self.right_image_keys = list(
            right_image_keys
            or [
                "observation.image.right",  # television_lab
                "policy.right_rgb",
                "right_rgb",
                "images.right",
            ]
        )
        self.state_keys = list(
            state_keys
            or [
                "observation.state",  # television_lab
                "policy.state",
                "state",
                "policy",
            ]
        )
        self.supports_teleop_to_action = bool(
            getattr(self._env_target, "supports_teleop_to_action", False)
        )
        self.is_real_env = bool(getattr(self._env_target, "is_real_env", False))
        self.action_schema = getattr(
            self._env_target,
            "action_schema",
            infer_action_schema(self.action_dim),
        )
        self.state_schema = getattr(
            self._env_target,
            "state_schema",
            f"unknown_state_dim_{self.action_dim}",
        )
        self.task_contract = infer_task_from_schemas(self.action_schema, self.action_dim, fallback=task)

    def _create_env(self, task: str, render_mode: str):
        import gymnasium as gym

        real_env_registered = False
        try:
            env_spec = gym.spec(task)
            kwargs = getattr(env_spec, "kwargs", {}) or {}
            real_env_registered = "cfg" in kwargs or "env_cfg_entry_point" in kwargs
        except Exception:
            real_env_registered = False

        if real_env_registered:
            try:
                return gym.make(task, render_mode=render_mode)
            except Exception as exc:
                print(f"[Warning] Real registered env creation failed for {task}: {exc}")

        try:
            from isaaclab_tasks.utils import parse_env_cfg

            env_cfg = parse_env_cfg(task, device="cuda:0", num_envs=1)
            os.environ.setdefault("ENABLE_CAMERAS", "1")
            return gym.make(task, cfg=env_cfg, render_mode=render_mode)
        except Exception as exc:
            print(f"[Warning] Isaac Lab env creation failed for {task}, falling back to adapter env: {exc}")

        try:
            return gym.make(task, render_mode=render_mode)
        except Exception as exc:
            print(f"[Warning] Gym registration path failed for {task}, using direct fallback adapter: {exc}")
            fallback_cfg = self._fallback_config_for_task(task)
            from .tasks.television_lab import TelevisionLabEnv

            return TelevisionLabEnv(cfg=fallback_cfg, render_mode=render_mode)

    @staticmethod
    def _fallback_config_for_task(task: str):
        from .tasks.television_lab import TelevisionLabConfig
        from .contracts import (
            H1_ACTION_DIM,
            H1_STATE_SCHEMA,
            H1_TASK_ID,
            TELEOP_ACTION_DIM,
            TELEOP_STATE_SCHEMA,
            TELEOP_TASK_ID,
        )

        if task == H1_TASK_ID:
            return TelevisionLabConfig(
                action_dim=H1_ACTION_DIM,
                state_dim=H1_ACTION_DIM,
                scene_mode="h1",
                task_id=H1_TASK_ID,
                state_schema=H1_STATE_SCHEMA,
            )
        return TelevisionLabConfig(
            action_dim=TELEOP_ACTION_DIM,
            state_dim=TELEOP_ACTION_DIM,
            scene_mode="teleop",
            task_id=TELEOP_TASK_ID,
            state_schema=TELEOP_STATE_SCHEMA,
        )

    @property
    def action_dim(self) -> int:
        shape = getattr(self.env.action_space, "shape", None)
        if shape is None or len(shape) == 0:
            raise RuntimeError("Action space has no shape, cannot infer action_dim")
        return int(shape[-1])

    def reset(self) -> ObsPack:
        obs, _ = self.env.reset()
        return self._build_obs_pack(obs)

    @property
    def _env_target(self):
        return getattr(self.env, "unwrapped", self.env)

    def teleop_to_action(
        self,
        left_pose: np.ndarray,
        right_pose: np.ndarray,
        left_qpos: np.ndarray,
        right_qpos: np.ndarray,
    ) -> np.ndarray:
        if not self.supports_teleop_to_action:
            raise NotImplementedError(
                "This environment expects already-assembled actions."
            )
        return self._env_target.teleop_to_action(
            left_pose,
            right_pose,
            left_qpos,
            right_qpos,
        )

    def step(self, action: np.ndarray, **_: Any) -> ObsPack:
        self._apply_runtime_metadata(**_)
        action = self._adapt_action(action)
        if action.ndim == 1:
            action = action[None, :]
        obs, _, terminated, truncated, _ = self.env.step(action)
        if np.any(terminated) or np.any(truncated):
            obs, _ = self.env.reset()
        return self._build_obs_pack(obs)

    def close(self):
        self.env.close()

    def _apply_runtime_metadata(self, **kwargs: Any) -> None:
        head_rmat = kwargs.get("head_rmat")
        if head_rmat is not None and hasattr(self._env_target, "set_head_rotation"):
            self._env_target.set_head_rotation(head_rmat)

    def _adapt_action(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        if hasattr(self._env_target, "adapt_action"):
            return self._env_target.adapt_action(action)
        return action

    def _find_by_keys(
        self, obs: Any, key_candidates: Iterable[str]
    ) -> Optional[np.ndarray]:
        for key in key_candidates:
            value = _resolve_key_path(obs, key)
            array = _as_numpy(value)
            if array is not None:
                return array
        return None

    def _build_obs_pack(self, obs: Any) -> ObsPack:
        left_rgb = _to_hwc_uint8(self._find_by_keys(obs, self.left_image_keys))
        right_rgb = _to_hwc_uint8(self._find_by_keys(obs, self.right_image_keys))
        state = _as_numpy(self._find_by_keys(obs, self.state_keys))

        if left_rgb is None or right_rgb is None:
            rendered = self.env.render()
            rendered = _to_hwc_uint8(_as_numpy(rendered))
            if rendered is None:
                raise RuntimeError(
                    "Cannot extract RGB images from observations or env.render()."
                )
            half_w = rendered.shape[1] // 2
            left_rgb = rendered[:, :half_w]
            right_rgb = rendered[:, half_w:]
            if right_rgb.size == 0:
                right_rgb = left_rgb.copy()

        if state is None:
            state = np.zeros((1, 1), dtype=np.float32)
        state = np.asarray(state)
        if state.ndim > 1:
            state = state[0]

        return ObsPack(
            left_rgb=left_rgb,
            right_rgb=right_rgb,
            state=state.astype(np.float32),
            raw_obs=obs,
        )
