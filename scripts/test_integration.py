"""Smoke-test TeleVision Isaac Lab integration paths."""

import argparse
import sys
import tempfile
import traceback
from pathlib import Path

import h5py
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    import gymnasium as gym
except ModuleNotFoundError as exc:
    raise SystemExit(
        "gymnasium is required to run scripts/test_integration.py. "
        "Install repo dependencies first."
    ) from exc

from scripts.replay_demo import Player
from tv_isaaclab import EpisodeRecorder, IsaacLabEnvBridge
from tv_isaaclab.contracts import H1_TASK_ID, TELEOP_TASK_ID


def _print_header(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def test_environment_creation(task: str) -> bool:
    _print_header(f"[TEST 1] Environment Creation ({task})")
    try:
        import tv_isaaclab  # noqa: F401

        env = gym.make(task, render_mode="rgb_array")
        print("Environment created")
        obs, _ = env.reset()
        print(f"Observation keys: {list(obs.keys()) if isinstance(obs, dict) else type(obs).__name__}")
        print(f"Action space: {env.action_space}")
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        print(f"Step ok, reward={reward}, terminated={terminated}, truncated={truncated}")
        env.close()
        return True
    except Exception as exc:
        print(f"Environment test failed: {exc}")
        traceback.print_exc()
        return False


def test_env_bridge(task: str) -> bool:
    _print_header(f"[TEST 2] IsaacLabEnvBridge ({task})")
    try:
        bridge = IsaacLabEnvBridge(task=task)
        print(f"Action dim: {bridge.action_dim}")
        print(f"Action schema: {bridge.action_schema}")
        print(f"State schema: {bridge.state_schema}")
        print(f"Supports teleop mapping: {bridge.supports_teleop_to_action}")
        print(f"Real env backend: {bridge.is_real_env}")
        obs = bridge.reset()
        print(f"Left RGB: {obs.left_rgb.shape}")
        print(f"Right RGB: {obs.right_rgb.shape}")
        print(f"State: {obs.state.shape}")
        action = np.random.randn(bridge.action_dim).astype(np.float32)
        bridge.step(action)
        bridge.close()
        return True
    except Exception as exc:
        print(f"Bridge test failed: {exc}")
        traceback.print_exc()
        return False


def test_recording(task: str) -> bool:
    _print_header(f"[TEST 3] Episode Recording ({task})")
    try:
        bridge = IsaacLabEnvBridge(task=task)
        recorder = EpisodeRecorder(
            action_schema=bridge.action_schema,
            cmd_schema=bridge.action_schema,
            state_schema=bridge.state_schema,
        )
        bridge.reset()

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_file = Path(tmp_dir) / "test_episode.hdf5"
            for _ in range(5):
                action = np.random.randn(bridge.action_dim).astype(np.float32)
                obs = bridge.step(action)
                recorder.append(
                    left_rgb=obs.left_rgb,
                    right_rgb=obs.right_rgb,
                    state=obs.state,
                    action=action,
                    cmd=action,
                )

            recorder.save(output_file)
            with h5py.File(output_file, "r") as handle:
                print(f"HDF5 keys: {sorted(handle.keys())}")
                print(f"action_schema={handle.attrs.get('action_schema')}")
                print(f"state_schema={handle.attrs.get('state_schema')}")
                print(f"episode_len={len(handle['observation.image.left'])}")

        bridge.close()
        return True
    except Exception as exc:
        print(f"Recording test failed: {exc}")
        traceback.print_exc()
        return False


def test_replay(task: str) -> bool:
    _print_header(f"[TEST 4] Episode Replay ({task})")
    try:
        bridge = IsaacLabEnvBridge(task=task)
        recorder = EpisodeRecorder(
            action_schema=bridge.action_schema,
            cmd_schema=bridge.action_schema,
            state_schema=bridge.state_schema,
        )
        bridge.reset()

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_file = Path(tmp_dir) / "test_replay_episode.hdf5"
            for _ in range(6):
                action = np.random.randn(bridge.action_dim).astype(np.float32)
                obs = bridge.step(action)
                recorder.append(
                    left_rgb=obs.left_rgb,
                    right_rgb=obs.right_rgb,
                    state=obs.state,
                    action=action,
                    cmd=action,
                )
            recorder.save(output_file)

            player = Player(task=task, show_plot=False)
            with h5py.File(output_file, "r") as handle:
                actions = np.array(handle["qpos_action"])
                left_imgs = np.array(handle["observation.image.left"])
                right_imgs = np.array(handle["observation.image.right"])

            for index in range(min(3, len(actions))):
                player.step(actions[index], left_imgs[index], right_imgs[index])
                print(f"Replay step {index} ok")
            player.end()

        bridge.close()
        return True
    except Exception as exc:
        print(f"Replay test failed: {exc}")
        traceback.print_exc()
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test TeleVision Isaac Lab integration")
    parser.add_argument("--task", default=TELEOP_TASK_ID, choices=[TELEOP_TASK_ID, H1_TASK_ID])
    args = parser.parse_args()

    results = {
        "Environment Creation": test_environment_creation(args.task),
        "IsaacLabEnvBridge": test_env_bridge(args.task),
        "Episode Recording": test_recording(args.task),
        "Episode Replay": test_replay(args.task),
    }

    _print_header("TEST SUMMARY")
    passed = sum(1 for result in results.values() if result)
    for test_name, result in results.items():
        print(f"{'PASS' if result else 'FAIL'}: {test_name}")
    print(f"Total: {passed}/{len(results)} tests passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
