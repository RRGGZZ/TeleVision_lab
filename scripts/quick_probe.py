"""Quick schema probe without full AppLauncher."""

import argparse
import os
import sys
from pathlib import Path

os.environ["LOGLEVEL"] = "WARN"

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tv_isaaclab.contracts import H1_TASK_ID, TELEOP_TASK_ID


def probe(task: str):
    try:
        import gymnasium as gym
        import numpy as np

        import tv_isaaclab  # noqa: F401

        print(f"\n[+] Creating {task} environment...")
        env = gym.make(task, render_mode="rgb_array")

        print("[+] Resetting...")
        obs, _ = env.reset()

        print(f"\n========== {task.upper()} SCHEMA ==========\n")
        print("[ACTION]")
        print(f"  Action space: {env.action_space}")
        print(f"  Action shape: {env.action_space.shape}")
        print(f"  Action dtype: {env.action_space.dtype}")
        action_dim = env.action_space.shape[0] if len(env.action_space.shape) > 0 else 1
        print(f"  Action dim: {action_dim}\n")

        print("[OBSERVATION]")
        if hasattr(env, "observation_space"):
            print(f"  Observation space: {env.observation_space}")

        print(f"  Observation type: {type(obs).__name__}")
        if isinstance(obs, dict):
            print(f"  Observation keys: {list(obs.keys())}")
            for key in sorted(obs.keys()):
                value = obs[key]
                if isinstance(value, dict):
                    print(f"    {key}:")
                    for subkey in sorted(value.keys()):
                        subval = value[subkey]
                        if isinstance(subval, dict):
                            print(f"      {subkey}:")
                            for nested_key in sorted(subval.keys()):
                                nested_val = subval[nested_key]
                                if hasattr(nested_val, "shape"):
                                    print(
                                        f"        - {nested_key}: shape={nested_val.shape}, "
                                        f"dtype={getattr(nested_val, 'dtype', 'N/A')}"
                                    )
                                else:
                                    print(f"        - {nested_key}: {type(nested_val).__name__}")
                        elif hasattr(subval, "shape"):
                            print(f"      - {subkey}: shape={subval.shape}, dtype={subval.dtype}")
                        else:
                            print(f"      - {subkey}: {type(subval).__name__}")
                elif hasattr(value, "shape"):
                    print(f"    - {key}: shape={value.shape}, dtype={value.dtype}")
                else:
                    print(f"    - {key}: {type(value).__name__}")
        elif hasattr(obs, "shape"):
            print(f"  Observation shape: {obs.shape}")

        print("\n[STEP TEST]")
        test_action = np.zeros(action_dim, dtype=np.float32)
        obs, reward, terminated, truncated, _ = env.step(test_action)
        print(f"  Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
        if isinstance(obs, dict):
            print(f"  Observation keys after step: {list(obs.keys())}")

        env.close()
        print("\n[+] Schema probe complete!\n")
        return action_dim, obs
    except Exception as exc:
        print(f"\n[!] Error: {exc}")
        import traceback

        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quick task probe without AppLauncher")
    parser.add_argument("--task", default=TELEOP_TASK_ID, choices=[TELEOP_TASK_ID, H1_TASK_ID])
    args = parser.parse_args()
    probe(args.task)
