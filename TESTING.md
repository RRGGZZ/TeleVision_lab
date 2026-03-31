# Isaac Lab Migration Testing Guide

This document describes how to test and validate the Isaac Lab migration of TeleVision.

## Overview

The migration includes:
- ✅ Custom `television_lab` Gymnasium environment (minimal implementation)
- ✅ `IsaacLabEnvBridge` for observation/action bridging
- ✅ Episode recording in HDF5 format
- ✅ Episode replay functionality  
- ✅ Action mapping configuration
- ✅ Integration test suite

## Quick Start

### 1. Environment Setup

```bash
# The television_lab conda environment should already be configured with:
# - Python 3.11
# - NumPy, PyTorch, OpenCV, h5py
# - Gymnasium

conda activate television_lab
```

### 2. Run Integration Tests

```bash
cd /path/to/TeleVision_lab

# Run complete test suite (all 4 tests)
python scripts/test_integration.py
```

Expected output:
```
🎉 All tests passed! Isaac Lab migration is successful.

✓ PASS: Environment Creation
✓ PASS: IsaacLabEnvBridge
✓ PASS: Episode Recording
✓ PASS: Episode Replay

Total: 4/4 tests passed
```

## Test Descriptions

### Test 1: Environment Creation
- Creates the `television_lab` Gymnasium environment
- Verifies action/observation spaces
- Tests reset and step operations
- **Expected Schema:**
  - Action: Box(-1.0, 1.0, (38,), float32)
  - Observation: Dict with image.left, image.right (512x512 RGB), state (38D)

### Test 2: IsaacLabEnvBridge
- Verifies bridge layer correctly maps observations
- Tests action dimension resolution (38)
- Validates RGB image extraction and formatting (HWC uint8)
- Validates state extraction and normalization (float32)

### Test 3: Episode Recording
- Records 5 steps of simulated interaction
- Saves to HDF5 format
- Verifies all expected keys present:
  - `observation.image.left`: (N, 3, 512, 512) uint8
  - `observation.image.right`: (N, 3, 512, 512) uint8
  - `observation.state`: (N, 38) float32
  - `qpos_action`: (N, 38) float32
  - `cmds`: (N, 38) float32

### Test 4: Episode Replay
- Creates Player object
- Loads episode from HDF5
- Steps through episode with action replay
- Validates observation generation

## Manual Testing

### Quick Schema Probe

Get environment schema without Isaac Sim:
```bash
cd teleop
python ../scripts/quick_probe.py
```

Output shows:
- Action space dimensions and bounds
- Observation keys and shapes
- Dtype information

### Record Test Episode

Record a single episode (requires VisionPro setup):
```bash
cd teleop

# Record 100 steps to file
python teleop_hand.py \
  --task television_lab \
  --record \
  --output ../data/recordings/isaaclab/test_episode_000.hdf5 \
  --max_steps 100
```

### Replay Recorded Episode

```bash
cd scripts

# Replay with stride=1 (every frame)
python replay_demo.py \
  --task television_lab \
  --episode_path ../data/recordings/isaaclab/test_episode_000.hdf5 \
  --stride 1
```

### Batch Collect Episodes

Collect multiple episodes for training:
```bash
cd scripts

# Collect 5 episodes
python collect_episodes.py \
  --num_episodes 5 \
  --task television_lab \
  --output_dir ../data/recordings/isaaclab \
  --max_steps 500
```

## Environment Schema Details

### `television_lab`

Minimal gymnasium environment for teleoperation testing.

**Configuration:**
- Located: `tv_isaaclab/tasks/television_lab.py`
- Action Dim: 38 (adjustable via `TelevisionLabConfig.action_dim`)
- Image Size: 512x512 (adjustable via config)
- State Dim: 38 (matches action_dim by default)

**Action Breakdown (38D):**
```
[0:7]    - left_wrist_pose (xyz + quat)
[7:14]   - right_wrist_pose (xyz + quat)
[14:26]  - left_hand_joints (12 angles)
[26:38]  - right_hand_joints (12 angles)
```

**Observation Structure:**
```python
{
  "observation": {
    "image": {
      "left": (512, 512, 3) uint8,
      "right": (512, 512, 3) uint8,
    },
    "state": (38,) float32,
  }
}
```

## Troubleshooting

### Test Fails: "No module named 'tv_isaaclab'"

Solution: Ensure project root is in Python path.
```bash
export PYTHONPATH="/path/to/TeleVision_lab:$PYTHONPATH"
```

### Test Fails: HDF5 Keys Missing

Solution: Verify `EpisodeRecorder` is saving all expected keys.
```bash
python -c "
import h5py
with h5py.File('episode.hdf5', 'r') as f:
    print('Keys:', list(f.keys()))
"
```

### Environment Reset Hangs

Solution: Current implementation uses synthetic data. If using real Isaac Sim:
```bash
# Launch Isaac Sim first, then run tests
python -m isaacsim.launch.python
```

## Integration Test Coverage

| Component | Test | Status |
|-----------|------|--------|
| Gymnasium Creation | Test 1 | ✅ |
| Action Space | Test 1 | ✅ |
| Observation Space | Test 1 | ✅ |
| Image Bridge | Test 2 | ✅ |  
| State Bridge | Test 2 | ✅ |
| HDF5 Recording | Test 3 | ✅ |
| Episode Replay | Test 4 | ✅ |
| Action Mapping | Manual | ✅ |
| Batch Collection | Manual | ✅ |

## Next Steps

After passing all tests:

1. **Real Environment Integration** (if full Isaac Lab available):
   - Replace `TelevisionLabEnv` with full Isaac Lab scene
   - Integrate H1 robot model with dual inspire hands
   - Add physics-based interactions

2. **Training Pipeline**:
   ```bash
   python act/imitate_episodes.py \
     --dataset_path data/recordings/isaaclab \
     --ckpt_dir experiments/models
   ```

3. **Policy Deployment**:
   ```bash
   scripts/deploy_sim.py \
     --task television_lab \
     --resume_ckpt experiments/models/model.pt
   ```

## Files Changed/Created

### New Files
- `tv_isaaclab/tasks/television_lab.py` - Gymnasium task definition
- `tv_isaaclab/tasks/__init__.py` - Task export
- `scripts/test_integration.py` - Integration test suite
- `scripts/collect_episodes.py` - Batch episode collection
- `scripts/quick_probe.py` - Schema introspection

### Modified Files
- `tv_isaaclab/__init__.py` - Task registration
- `tv_isaaclab/env_bridge.py` - Updated observation keys
- `tv_isaaclab/bootstrap.py` - App launcher support
- `teleop/teleop_hand.py` - Migrated to IsaacLabEnvBridge
- `scripts/replay_demo.py` - Updated for headless operation
- `scripts/deploy_sim.py` - Isaac Lab integration

## References

- [Isaac Lab Documentation](https://isaac-sim.github.io/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [HDF5 Format Reference](https://www.h5py.org/)
