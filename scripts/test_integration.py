"""Integration test for television_lab Isaac Lab migration."""

import h5py
import numpy as np
from pathlib import Path
import sys
import gymnasium as gym

# Add root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from tv_isaaclab import IsaacLabEnvBridge, EpisodeRecorder
from scripts.replay_demo import Player


def test_environment_creation():
    """Test 1: Can we create and interact with the television_lab environment?"""
    print("\n" + "="*60)
    print("[TEST 1] Environment Creation")
    print("="*60)
    
    try:
        import tv_isaaclab  # noqa: F401
        env = gym.make("television_lab", render_mode="rgb_array")
        print("✓ Environment created successfully")
        
        obs, _ = env.reset()
        print(f"✓ Environment reset successful")
        print(f"  - Observation keys: {obs.keys()}")
        print(f"  - Action space: {env.action_space}")
        
        # Step once
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"✓ Environment step successful")
        print(f"  - Reward: {reward}")
        print(f"  - Terminated: {terminated}, Truncated: {truncated}")
        
        env.close()
        print("✓ Environment closed successfully")
        return True
        
    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_env_bridge():
    """Test 2: Can IsaacLabEnvBridge correctly bridge observations?"""
    print("\n" + "="*60)
    print("[TEST 2] IsaacLabEnvBridge")
    print("="*60)
    
    try:
        import tv_isaaclab  # noqa: F401
        bridge = IsaacLabEnvBridge(task="television_lab")
        print(f"✓ IsaacLabEnvBridge created")
        print(f"  - Action dim: {bridge.action_dim}")
        
        obs = bridge.reset()
        print(f"✓ Bridge reset successful")
        print(f"  - Left RGB shape: {obs.left_rgb.shape}")
        print(f"  - Right RGB shape: {obs.right_rgb.shape}")
        print(f"  - State shape: {obs.state.shape}")
        
        # Step with random action
        action = np.random.randn(bridge.action_dim).astype(np.float32)
        obs = bridge.step(action)
        print(f"✓ Bridge step successful")
        
        bridge.close()
        print("✓ Bridge closed successfully")
        return True
        
    except Exception as e:
        print(f"✗ Bridge test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_recording():
    """Test 3: Can we record and save episodes to HDF5?"""
    print("\n" + "="*60)
    print("[TEST 3] Episode Recording")
    print("="*60)
    
    try:
        import tv_isaaclab  # noqa: F401
        
        # Record a short episode
        output_file = Path("/tmp/test_episode.hdf5")
        recorder = EpisodeRecorder()
        bridge = IsaacLabEnvBridge(task="television_lab")
        obs = bridge.reset()
        
        print(f"✓ Recorder created")
        
        # Record 5 steps
        for i in range(5):
            action = np.random.randn(bridge.action_dim).astype(np.float32)
            obs = bridge.step(action)
            recorder.append(
                left_rgb=obs.left_rgb,
                right_rgb=obs.right_rgb,
                state=obs.state,
                action=action,
                cmd=action,  # For test, cmd == action
            )
        
        recorder.save(output_file)
        print(f"✓ Episode saved to {output_file}")
        
        # Verify HDF5 structure
        with h5py.File(output_file, 'r') as f:
            keys = list(f.keys())
            print(f"  - HDF5 keys: {keys}")
            print(f"  - Episode length: {len(f['observation.image.left'])}")
            for key in sorted(keys):
                dataset = f[key]
                print(f"    • {key}: shape={dataset.shape}, dtype={dataset.dtype}")
        
        bridge.close()
        output_file.unlink()
        print("✓ Recording test successful")
        return True
        
    except Exception as e:
        print(f"✗ Recording test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_replay():
    """Test 4: Can we create a Player and replay episodes?"""
    print("\n" + "="*60)
    print("[TEST 4] Episode Replay")
    print("="*60)
    
    try:
        import tv_isaaclab  # noqa: F401
        
        # First, create a test episode
        output_file = Path("/tmp/test_replay_episode.hdf5")
        recorder = EpisodeRecorder()
        bridge = IsaacLabEnvBridge(task="television_lab")
        obs = bridge.reset()
        
        for i in range(10):
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
        bridge.close()
        print(f"✓ Test episode created at {output_file}")
        
        # Now test replay
        player = Player(task="television_lab", show_plot=False)
        print(f"✓ Player created")
        
        # Load episode data
        with h5py.File(str(output_file), 'r') as f:
            actions = np.array(f['qpos_action'])
            left_imgs = np.array(f['observation.image.left'])
            right_imgs = np.array(f['observation.image.right'])
        
        print(f"✓ Episode loaded (length={len(actions)})")
        
        # Replay a few steps
        for i in range(min(3, len(actions))):
            obs = player.step(actions[i], left_imgs[i], right_imgs[i])
            print(f"  - Step {i}: observation captured")
        
        player.end()
        output_file.unlink()
        print("✓ Replay test successful")
        return True
        
    except Exception as e:
        print(f"✗ Replay test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all integration tests."""
    print("\n" + "="*60)
    print("TELEVISION_LAB ISAAC LAB INTEGRATION TESTS")
    print("="*60)
    
    results = {
        "Environment Creation": test_environment_creation(),
        "IsaacLabEnvBridge": test_env_bridge(),
        "Episode Recording": test_recording(),
        "Episode Replay": test_replay(),
    }
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Isaac Lab migration is successful.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. See details above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
