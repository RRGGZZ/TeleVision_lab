"""Probe television_lab environment schema."""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from tv_isaaclab import launch_simulation_app, IsaacLabEnvBridge
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="television_lab")
    from tv_isaaclab import add_app_launcher_args
    add_app_launcher_args(parser)
    args = parser.parse_args()

    print(f"[*] Launching {args.task}...")
    simulation_app = launch_simulation_app(args)

    print("[*] Creating environment bridge...")
    env = IsaacLabEnvBridge(task=args.task)

    print("[*] Resetting environment...")
    obs = env.reset()

    print("\n=== ENVIRONMENT SCHEMA ===\n")
    print(f"Action dimension: {env.action_dim}")
    print(f"Left RGB shape: {obs.left_rgb.shape}")
    print(f"Right RGB shape: {obs.right_rgb.shape}")
    print(f"State shape: {obs.state.shape}")
    print(f"State sample: {obs.state[:10]}")

    print("\n=== RAW OBSERVATION (first level keys) ===")
    if isinstance(obs.raw_obs, dict):
        for key in sorted(obs.raw_obs.keys()):
            val = obs.raw_obs[key]
            if hasattr(val, "shape"):
                print(f"  {key}: shape={val.shape}, dtype={getattr(val, 'dtype', 'N/A')}")
            elif isinstance(val, dict):
                print(f"  {key}: dict with keys={list(val.keys())[:5]}")
            else:
                print(f"  {key}: {type(val).__name__}")

    print("\n[*] Taking 3 steps...")
    for i in range(3):
        import numpy as np
        action = np.zeros(env.action_dim, dtype=np.float32)
        obs = env.step(action)
        print(f"  Step {i+1}: state shape={obs.state.shape}")

    print("\n[*] Done. Closing environment...")
    env.close()
    simulation_app.close()
    print("[+] Success!")


if __name__ == "__main__":
    main()
