import numpy as np
import cv2
import sys
from pathlib import Path

# Add teleop directory to path for local imports
TELEOP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TELEOP_DIR))

from TeleVision import OpenTeleVision
from Preprocessor import VuerPreprocessor
from constants_vuer import tip_indices
from dex_retargeting.retargeting_config import RetargetingConfig
from pytransform3d import rotations

import argparse
import time
import yaml
from multiprocessing import shared_memory, Queue, Event

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from tv_isaaclab import (  # noqa: E402
    add_app_launcher_args,
    launch_simulation_app,
    IsaacLabEnvBridge,
    EpisodeRecorder,
)
from tv_isaaclab.contracts import TELEOP_TASK_ID, expand_inspire_driver_qpos


def _resolve_local_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute() and path.exists():
        return path
    for candidate in (Path.cwd() / path, TELEOP_DIR / path, ROOT_DIR / path):
        if candidate.exists():
            return candidate.resolve()
    return (TELEOP_DIR / path).resolve()

class VuerTeleop:
    def __init__(self, config_file_path, ngrok=False, vuer_port=8012):
        self.resolution = (720, 1280)
        self.crop_size_w = 0
        self.crop_size_h = 0
        self.resolution_cropped = (
            self.resolution[0] - self.crop_size_h,
            self.resolution[1] - 2 * self.crop_size_w,
        )

        self.img_shape = (self.resolution_cropped[0], 2 * self.resolution_cropped[1], 3)
        self.img_height, self.img_width = self.resolution_cropped[:2]

        self.shm = shared_memory.SharedMemory(create=True, size=np.prod(self.img_shape) * np.uint8().itemsize)
        self.img_array = np.ndarray(
            (self.img_shape[0], self.img_shape[1], 3),
            dtype=np.uint8,
            buffer=self.shm.buf,
        )
        # Set a visible startup frame before the first simulation image arrives.
        self.img_array[:] = 32
        image_queue = Queue()
        toggle_streaming = Event()
        self.tv = OpenTeleVision(
            self.resolution_cropped,
            self.shm.name,
            image_queue,
            toggle_streaming,
            ngrok=ngrok,
            vuer_port=vuer_port,
        )
        self.processor = VuerPreprocessor()

        RetargetingConfig.set_default_urdf_dir((ROOT_DIR / "assets").as_posix())
        config_file_path = _resolve_local_path(config_file_path)
        with config_file_path.open('r') as f:
            cfg = yaml.safe_load(f)
        left_retargeting_config = RetargetingConfig.from_dict(cfg['left'])
        right_retargeting_config = RetargetingConfig.from_dict(cfg['right'])
        self.left_retargeting = left_retargeting_config.build()
        self.right_retargeting = right_retargeting_config.build()

    @staticmethod
    def _expand_inspire_qpos(qpos):
        return expand_inspire_driver_qpos(qpos)

    def step(self):
        head_mat, left_wrist_mat, right_wrist_mat, left_hand_mat, right_hand_mat = self.processor.process(self.tv)

        head_rmat = head_mat[:3, :3]

        left_pose = np.concatenate([left_wrist_mat[:3, 3] + np.array([-0.6, 0, 1.6]),
                                    rotations.quaternion_from_matrix(left_wrist_mat[:3, :3])[[1, 2, 3, 0]]])
        right_pose = np.concatenate([right_wrist_mat[:3, 3] + np.array([-0.6, 0, 1.6]),
                                     rotations.quaternion_from_matrix(right_wrist_mat[:3, :3])[[1, 2, 3, 0]]])
        left_qpos = self._expand_inspire_qpos(
            self.left_retargeting.retarget(left_hand_mat[tip_indices])
        )
        right_qpos = self._expand_inspire_qpos(
            self.right_retargeting.retarget(right_hand_mat[tip_indices])
        )

        return head_rmat, left_pose, right_pose, left_qpos, right_qpos

class ActionMapper:
    def __init__(self, config_path):
        config_path = _resolve_local_path(config_path)
        with config_path.open("r") as f:
            cfg = yaml.safe_load(f)
        self.action_dim = int(cfg["action_dim"])
        self.mapping = cfg["mapping"]

    def assemble(self, left_pose, right_pose, left_qpos, right_qpos):
        sources = {
            "left_pose": np.asarray(left_pose, dtype=np.float32),
            "right_pose": np.asarray(right_pose, dtype=np.float32),
            "left_qpos": np.asarray(left_qpos, dtype=np.float32),
            "right_qpos": np.asarray(right_qpos, dtype=np.float32),
        }
        action = np.zeros(self.action_dim, dtype=np.float32)
        for item in self.mapping:
            src_name = item["source"]
            src_start, src_end = item["src"]
            dst_start, dst_end = item["dst"]
            action[dst_start:dst_end] = sources[src_name][src_start:src_end]
        return action


def _parse_keys(key_str):
    if not key_str:
        return None
    return [x.strip() for x in key_str.split(",") if x.strip()]


def _resize_pair(left_img, right_img, height, width):
    left_resized = cv2.resize(
        left_img,
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )
    right_resized = cv2.resize(
        right_img,
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )
    return left_resized, right_resized


def _fit_action_dim(action, expected_dim):
    if action.shape[0] == expected_dim:
        return action
    resized = np.zeros(expected_dim, dtype=np.float32)
    copy_n = min(expected_dim, action.shape[0])
    resized[:copy_n] = action[:copy_n]
    return resized


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VisionPro teleoperation on Isaac Lab")
    parser.add_argument("--task", type=str, default=TELEOP_TASK_ID, help="Isaac Lab task name")
    parser.add_argument("--retarget_config", type=str, default="inspire_hand.yml", help="Retargeting config")
    parser.add_argument(
        "--action_mapping",
        type=str,
        default="teleop_action_mapping_isaaclab.yml",
        help="YAML file defining teleop command to env action mapping",
    )
    parser.add_argument("--record", action="store_true", help="Record episode to HDF5")
    parser.add_argument("--output", type=str, default="../data/recordings/isaaclab/processed_episode_0.hdf5")
    parser.add_argument("--max_steps", type=int, default=0, help="0 means run forever")
    parser.add_argument("--left_image_keys", type=str, default="")
    parser.add_argument("--right_image_keys", type=str, default="")
    parser.add_argument("--state_keys", type=str, default="")
    parser.add_argument("--ngrok", action="store_true", help="Enable ngrok mode for Vuer server")
    parser.add_argument("--vuer_port", type=int, default=8012, help="Port for Vuer websocket server")
    parser.add_argument("--loop_hz", type=float, default=30.0, help="Main control/render loop frequency")
    add_app_launcher_args(parser)
    args = parser.parse_args()

    simulation_app = launch_simulation_app(args)

    teleoperator = VuerTeleop(
        args.retarget_config,
        ngrok=args.ngrok,
        vuer_port=args.vuer_port,
    )
    mapper = ActionMapper(args.action_mapping)
    try:
        env = IsaacLabEnvBridge(
            task=args.task,
            left_image_keys=_parse_keys(args.left_image_keys),
            right_image_keys=_parse_keys(args.right_image_keys),
            state_keys=_parse_keys(args.state_keys),
        )
    except Exception as e:
        simulation_app.close()
        raise RuntimeError(f"Failed to create IsaacLabEnvBridge for task {args.task}: {e}") from e
    recorder = EpisodeRecorder() if args.record else None
    env.reset()

    step_count = 0
    target_dt = 1.0 / max(args.loop_hz, 1.0)
    try:
        while simulation_app.is_running():
            loop_start = time.perf_counter()
            head_rmat, left_pose, right_pose, left_qpos, right_qpos = teleoperator.step()
            raw_action = mapper.assemble(left_pose, right_pose, left_qpos, right_qpos)
            if getattr(env, "supports_teleop_to_action", False):
                action = env.teleop_to_action(left_pose, right_pose, left_qpos, right_qpos)
            else:
                action = _fit_action_dim(raw_action, env.action_dim)

            obs = env.step(action, head_rmat=head_rmat)

            left_img, right_img = _resize_pair(
                obs.left_rgb,
                obs.right_rgb,
                teleoperator.img_height,
                teleoperator.img_width,
            )
            np.copyto(teleoperator.img_array, np.hstack((left_img, right_img)))

            if recorder is not None:
                recorder.append(
                    left_rgb=obs.left_rgb,
                    right_rgb=obs.right_rgb,
                    state=obs.state,
                    action=action,
                    cmd=raw_action,
                )

            step_count += 1
            if args.max_steps > 0 and step_count >= args.max_steps:
                break

            elapsed = time.perf_counter() - loop_start
            if elapsed < target_dt:
                time.sleep(target_dt - elapsed)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"[!] Runtime loop error: {e}")
        # Exit loop to allow graceful cleanup in finally block.
        pass
    finally:
        if recorder is not None and step_count > 0:
            recorder.save(Path(args.output))
            print(f"Saved episode to: {args.output}")
        env.close()
        simulation_app.close()
