"""Quick schema probe without full AppLauncher."""

import sys
import os
from pathlib import Path

# Suppress Isaac Sim startup logs
os.environ["LOGLEVEL"] = "WARN"

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

def probe():
    try:
        import gymnasium as gym
        
        # Import TV Isaac Lab tasks (includes television_lab registration)
        import tv_isaaclab  # noqa: F401
        
        print("\n[+] Creating television_lab environment...")
        env = gym.make("television_lab", render_mode="rgb_array")
        
        print("[+] Resetting...")
        obs, info = env.reset()
        
        print("\n========== TELEVISION_LAB SCHEMA ==========\n")
        
        # Action space
        print(f"[ACTION]")
        print(f"  Action space: {env.action_space}")
        print(f"  Action shape: {env.action_space.shape}")
        print(f"  Action dtype: {env.action_space.dtype}")
        action_dim = env.action_space.shape[0] if len(env.action_space.shape) > 0 else 1
        print(f"  Action dim: {action_dim}\n")
        
        # Observation space
        print(f"[OBSERVATION]")
        if hasattr(env, "observation_space"):
            print(f"  Observation space: {env.observation_space}")
        
        print(f"  Observation type: {type(obs).__name__}")
        if isinstance(obs, dict):
            print(f"  Observation keys: {list(obs.keys())}")
            for key in sorted(obs.keys()):
                val = obs[key]
                if isinstance(val, dict):
                    print(f"    {key}:")
                    for subkey in sorted(val.keys()):
                        subval = val[subkey]
                        if isinstance(subval, dict):
                            print(f"      {subkey}:")
                            for k2 in sorted(subval.keys()):
                                v2 = subval[k2]
                                if hasattr(v2, "shape"):
                                    print(f"        - {k2}: shape={v2.shape}, dtype={v2.dtype}")
                                else:
                                    print(f"        - {k2}: {type(v2).__name__}")
                        else:
                            if hasattr(subval, "shape"):
                                print(f"      - {subkey}: shape={subval.shape}, dtype={subval.dtype}")
                            else:
                                print(f"      - {subkey}: {type(subval).__name__}")
                else:
                    if hasattr(val, "shape"):
                        print(f"    - {key}: shape={val.shape}, dtype={val.dtype}")
                    else:
                        print(f"    - {key}: {type(val).__name__}")
        else:
            if hasattr(obs, "shape"):
                print(f"  Observation shape: {obs.shape}")
        
        print(f"\n[STEP TEST]")
        import numpy as np
        test_action = np.zeros(action_dim, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(test_action)
        print(f"  Step executed successfully")
        print(f"  Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
        if isinstance(obs, dict):
            print(f"  Observation keys after step: {list(obs.keys())}")
        
        env.close()
        print(f"\n[✓] Schema probe complete!\n")
        
        return action_dim, obs
        
    except Exception as e:
        print(f"\n[!] Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    probe()
