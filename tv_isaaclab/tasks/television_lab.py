"""television_lab adapter env with real Isaac Lab scene fallback."""

import os
from dataclasses import dataclass
from typing import Any, Optional

import cv2
import gymnasium as gym
from gymnasium import spaces
import numpy as np


@dataclass
class TelevisionLabConfig:
    """Configuration for television_lab adapter environment."""

    action_dim: int = 38
    image_width: int = 512
    image_height: int = 512
    state_dim: int = 38
    # Use one explicit real-scene Isaac task to avoid costly multi-task probing.
    base_task_id: str = "Isaac-Cartpole-Direct-v0"


class TelevisionLabEnv(gym.Env):
    """Adapter that exposes TeleVision schema over a real Isaac Lab task."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        cfg: Optional[TelevisionLabConfig] = None,
        render_mode: str = "rgb_array",
    ):
        self.cfg = cfg or TelevisionLabConfig()
        self.render_mode = render_mode
        self.step_count = 0
        self.max_steps = 1000

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.cfg.action_dim,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Dict(
                    {
                        "image": spaces.Dict(
                            {
                                "left": spaces.Box(
                                    low=0,
                                    high=255,
                                    shape=(
                                        self.cfg.image_height,
                                        self.cfg.image_width,
                                        3,
                                    ),
                                    dtype=np.uint8,
                                ),
                                "right": spaces.Box(
                                    low=0,
                                    high=255,
                                    shape=(
                                        self.cfg.image_height,
                                        self.cfg.image_width,
                                        3,
                                    ),
                                    dtype=np.uint8,
                                ),
                            }
                        ),
                        "state": spaces.Box(
                            low=-np.inf,
                            high=np.inf,
                            shape=(self.cfg.state_dim,),
                            dtype=np.float32,
                        ),
                    }
                ),
            }
        )

        self.base_env = None
        self.base_task = None
        self._last_raw_obs = None
        self._state = np.zeros(self.cfg.state_dim, dtype=np.float32)

        self._try_create_base_env()

    def _try_create_base_env(self):
        """Try to attach a real Isaac Lab task for scene camera rendering."""
        try:
            import isaaclab_tasks  # noqa: F401
        except Exception:
            return

        try:
            from isaaclab_tasks.utils import parse_env_cfg
        except ImportError:
            print("[!] Could not import parse_env_cfg from isaaclab_tasks")
            return

        task = os.getenv("TELEVISION_LAB_BASE_TASK", self.cfg.base_task_id)
        try:
            env_cfg = parse_env_cfg(task, device="cuda:0", num_envs=1)
            os.environ.setdefault("ENABLE_CAMERAS", "1")
            env = gym.make(task, cfg=env_cfg, render_mode="rgb_array")
            raw_obs, _ = env.reset()
            self.base_env = env
            self.base_task = task
            self._last_raw_obs = raw_obs
            print(f"[*] television_lab using real Isaac task: {task}")
        except Exception as e:
            self.base_env = None
            self.base_task = None
            self._last_raw_obs = None
            print(f"[!] television_lab failed to bind base task {task}: {e}")
            print("[*] Falling back to synthetic visible frame")

    @staticmethod
    def _to_numpy(value: Any) -> Optional[np.ndarray]:
        if value is None:
            return None
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        try:
            return np.asarray(value)
        except Exception:
            return None

    def _flatten_state(self, raw_obs: Any) -> np.ndarray:
        """Flatten observation payload into fixed-dim state vector."""
        chunks = []

        def walk(node: Any):
            if isinstance(node, dict):
                for key in sorted(node.keys()):
                    walk(node[key])
                return
            arr = self._to_numpy(node)
            if arr is None:
                return
            arr = arr.astype(np.float32, copy=False).reshape(-1)
            if arr.size > 0:
                chunks.append(arr)

        walk(raw_obs)
        if not chunks:
            return np.zeros(self.cfg.state_dim, dtype=np.float32)

        vec = np.concatenate(chunks, axis=0)
        out = np.zeros(self.cfg.state_dim, dtype=np.float32)
        n = min(out.size, vec.size)
        out[:n] = vec[:n]
        return out

    @staticmethod
    def _to_hwc_uint8(image: np.ndarray) -> np.ndarray:
        arr = np.asarray(image)
        if arr.ndim == 4:
            arr = arr[0]
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=2)
        if (
            arr.ndim == 3
            and arr.shape[0] in (1, 3, 4)
            and arr.shape[-1] not in (1, 3, 4)
        ):
            arr = np.transpose(arr, (1, 2, 0))
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr

    def _generate_synthetic_frame(
        self, h: int, w: int, action: np.ndarray = None
    ) -> np.ndarray:
        """Generate a synthetic frame that shows some visual structure.

        This is used when Isaac Sim is not available or has memory issues.
        """
        # Create a gradient background
        y_grad = np.linspace(30, 80, h, dtype=np.uint8)
        x_grad = np.linspace(40, 100, w, dtype=np.uint8)
        xx, yy = np.meshgrid(x_grad, y_grad)
        bg = np.stack([xx, yy, np.full_like(xx, 50)], axis=-1)

        # Add a "robot" representation - simple geometric shapes
        frame = bg.copy()

        # Add grid lines to simulate a lab environment
        grid_color = np.array([60, 60, 60], dtype=np.uint8)
        for i in range(0, h, 64):
            frame[i : i + 2, :] = grid_color
        for j in range(0, w, 64):
            frame[:, j : j + 2] = grid_color

        # Add a "table" surface at bottom
        table_y = int(h * 0.7)
        frame[table_y:, :] = np.array([80, 70, 60], dtype=np.uint8)

        # If we have action data, visualize the hand positions
        if action is not None and len(action) >= 38:
            # Left hand position (indices 0-6: xyz + quat)
            lx = int(w * 0.3 + action[0] * 50)
            ly = int(h * 0.5 + action[1] * 50)
            # Draw left hand marker
            if 0 <= lx < w and 0 <= ly < h:
                cv = min(30, w // 8)
                frame[
                    max(0, ly - cv) : min(h, ly + cv), max(0, lx - cv) : min(w, lx + cv)
                ] = np.array([100, 150, 200], dtype=np.uint8)

            # Right hand position
            rx = int(w * 0.7 + action[7] * 50)
            ry = int(h * 0.5 + action[8] * 50)
            if 0 <= rx < w and 0 <= ry < h:
                cv = min(30, w // 8)
                frame[
                    max(0, ry - cv) : min(h, ry + cv), max(0, rx - cv) : min(w, rx + cv)
                ] = np.array([200, 150, 100], dtype=np.uint8)

        return frame

    def _extract_images(self, raw_obs: Any) -> tuple[np.ndarray, np.ndarray]:
        """Prefer rendered scene from base env; fallback to synthetic frame."""
        frame = None
        if self.base_env is not None:
            try:
                frame = self.base_env.render()
            except Exception:
                frame = None

        if frame is None:
            h, w = self.cfg.image_height, self.cfg.image_width
            left = self._generate_synthetic_frame(h, w, self._state)
            right = left.copy()
            return left, right

        frame = self._to_hwc_uint8(frame)
        if frame.shape[1] < 2:
            return frame, frame.copy()

        half = frame.shape[1] // 2
        left = frame[:, :half]
        right = frame[:, half:]
        if right.size == 0:
            right = left.copy()
        return left, right

    def _resize_rgb(self, image: np.ndarray) -> np.ndarray:
        """Resize to configured output shape without cv2 dependency."""
        h, w = image.shape[:2]
        if h == self.cfg.image_height and w == self.cfg.image_width:
            return image
        y_idx = np.linspace(0, h - 1, self.cfg.image_height).astype(np.int32)
        x_idx = np.linspace(0, w - 1, self.cfg.image_width).astype(np.int32)
        return image[y_idx][:, x_idx]

    def _build_obs(self, raw_obs: Any) -> dict:
        left, right = self._extract_images(raw_obs)
        left = self._resize_rgb(left)
        right = self._resize_rgb(right)
        self._state = self._flatten_state(raw_obs)
        return {
            "observation": {
                "image": {
                    "left": left,
                    "right": right,
                },
                "state": self._state.copy(),
            }
        }

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple:
        super().reset(seed=seed)
        self.step_count = 0
        if self.base_env is not None:
            raw_obs, _ = self.base_env.reset(seed=seed, options=options)
            self._last_raw_obs = raw_obs
            return self._build_obs(raw_obs), {}
        self._last_raw_obs = None
        return self._build_obs(None), {}

    def _map_action(self, action: np.ndarray) -> np.ndarray:
        if self.base_env is None:
            return action
        target_shape = getattr(self.base_env.action_space, "shape", None)
        if not target_shape:
            return action
        target_dim = int(target_shape[-1])
        mapped = np.zeros(target_dim, dtype=np.float32)
        n = min(target_dim, action.shape[0])
        mapped[:n] = action[:n]
        # Respect base env action limits when available.
        low = getattr(self.base_env.action_space, "low", None)
        high = getattr(self.base_env.action_space, "high", None)
        if low is not None and high is not None:
            mapped = np.clip(mapped, low, high)
        return mapped

    def step(self, action: np.ndarray) -> tuple:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        self.step_count += 1

        if self.base_env is None:
            obs = self._build_obs(None)
            terminated = False
            truncated = self.step_count >= self.max_steps
            return obs, 0.0, terminated, truncated, {}

        mapped_action = self._map_action(action)
        raw_obs, reward, terminated, truncated, info = self.base_env.step(mapped_action)

        term = (
            bool(np.any(terminated))
            if isinstance(terminated, np.ndarray)
            else bool(terminated)
        )
        trunc = (
            bool(np.any(truncated))
            if isinstance(truncated, np.ndarray)
            else bool(truncated)
        )
        if term or trunc:
            raw_obs, _ = self.base_env.reset()

        self._last_raw_obs = raw_obs
        obs = self._build_obs(raw_obs)
        return obs, float(reward), term, trunc, info

    def render(self) -> Optional[np.ndarray]:
        if self.base_env is not None:
            try:
                frame = self.base_env.render()
                if frame is not None:
                    return self._to_hwc_uint8(frame)
            except Exception:
                pass

        left, right = self._extract_images(None)
        return np.hstack([left, right])

    def close(self):
        if self.base_env is not None:
            self.base_env.close()
            self.base_env = None


def register_television_lab():
    """Register television_lab environment in Gymnasium."""
    gym.register(
        id="television_lab",
        entry_point="tv_isaaclab.tasks.television_lab:TelevisionLabEnv",
        max_episode_steps=1000,
        kwargs={"cfg": TelevisionLabConfig()},
    )
