# TeleVision_lab

Isaac Lab / Isaac Sim migration workspace for [Open-TeleVision](https://github.com/OpenTeleVision/TeleVision).

This repository keeps the original teleoperation and imitation-learning workflow, while progressively replacing the old IsaacGym simulation path with Isaac Lab tasks, tooling, and dataset contracts.

## Current Migration Status

The migration is now organized around two explicit task contracts:

- `television_lab`: teleoperation collection scene with dual floating Inspire hands and legacy 38D commands
- `television_h1`: H1 replay / policy-consumption scene with canonical 26D actions and compatibility for legacy 28D recordings

The codebase also records schema metadata into processed HDF5 episodes:

- `action_schema`
- `cmd_schema`
- `state_schema`

Replay and deploy scripts use this metadata to route episodes to the correct task automatically.

## Runtime Model

Task registration now prefers real Isaac Lab tasks first:

- real path: DirectRLEnv-style task definitions in [tv_isaaclab/tasks/television_lab_real.py](tv_isaaclab/tasks/television_lab_real.py)
- fallback path: adapter environments in [tv_isaaclab/tasks/television_lab.py](tv_isaaclab/tasks/television_lab.py)

If Isaac Lab is unavailable, the repository still exposes a fallback environment so dataset tooling, replay tooling, and regression tests continue to work.

## Installation

### Base Python Environment

```bash
conda create -n tv python=3.8
conda activate tv
pip install -r requirements.txt
cd act/detr && pip install -e .
```

### Python 3.11 / 3.12 Environment

For Python 3.11 or 3.12 setups such as `television_lab`, see [SETUP_PYTHON311.md](SETUP_PYTHON311.md).

Important differences:

- `dex-retargeting` is pinned to `<0.5.0` so it remains compatible with NumPy 1.x
- `numpy` is constrained to `<2.0` for Isaac Lab compatibility
- `packaging` is pinned to `23.0` because Isaac Sim 5.1 requires that exact version
- `wheel` is pinned below `0.47` because newer wheel releases require `packaging>=24`
- `gymnasium>=0.29.1` is required
- `dex-retargeting` currently does not support Python 3.13, so teleoperation should be run under Python 3.11 or 3.12
- quick setup script: `bash install_deps.sh`

### Isaac Lab / Isaac Sim

To run the real simulation path:

1. Install Isaac Lab / Isaac Sim.
2. Install this repository's Python dependencies in the same environment.
3. Launch scripts through the wrappers in [tv_isaaclab/bootstrap.py](tv_isaaclab/bootstrap.py).

Notes:

- camera pipelines are enabled before `AppLauncher` starts
- scripts assume the Isaac Lab app can create stereo camera outputs
- if `gymnasium` is missing, runtime task registration will fail even if the static tests pass

### ZED SDK

Install the ZED SDK from [StereoLabs](https://www.stereolabs.com/developers/release/).

Install the Python API:

```bash
cd /usr/local/zed/
python get_python_api.py
```

## Teleoperation

### Local Streaming

For Quest local streaming, see the upstream Open-TeleVision discussion:

- [Issue #12](https://github.com/OpenTeleVision/TeleVision/issues/12#issue-2401541144)

For Vision Pro over local HTTPS, you still need to provision certificates in `teleop/` and open the streaming port. The original Open-TeleVision workflow is preserved here.

From the `teleop/` directory, generate local HTTPS certificates with:

```bash
mkcert -install
mkcert -cert-file cert.pem -key-file key.pem localhost 127.0.0.1 <your-server-ip>
```

If you are using ngrok, run teleoperation with `--ngrok` instead of local certificate files.

### Network Streaming

For Meta Quest 3 or remote secure access, `ngrok` is still supported:

```bash
ngrok http 8012
```

When using network streaming, initialize `OpenTeleVision` with `ngrok=True`.

### Isaac Lab Teleoperation Workflow

#### 1. Smoke-Test the Task

Teleop task:

```bash
cd scripts
python test_integration.py --task television_lab
```

H1 replay task:

```bash
cd scripts
python test_integration.py --task television_h1
```

Server-side headless full-run smoke test:

```bash
cd scripts
python headless_full_run.py --tasks television_lab,television_h1 --memory_mode low --headless
```

When launching through Isaac Lab, keep `--device` for Isaac Lab's simulator
device and use `--policy_device` only for the PyTorch dataset/deploy smoke path:

```bash
python headless_full_run.py --tasks television_lab,television_h1 --memory_mode low --headless --device cuda:0 --policy_device cuda
```

This script runs the main non-interactive paths end-to-end for each task:

- environment creation and reset/step
- episode recording to HDF5
- replay through the bridge
- dataset loading smoke test
- dummy JIT policy deployment smoke test

The outputs are written under `data/headless_runs/` by default.

#### 2. Inspect the Schema

Quick probe without full Isaac Lab app launch:

```bash
python quick_probe.py --task television_lab
```

Real Isaac Lab schema probe:

```bash
python probe_schema.py --task television_lab
```

Typical contracts:

- `television_lab`: 38D action/state, stereo RGB observations, native teleop-to-action mapping
- `television_h1`: 26D canonical replay action/state, stereo RGB observations, legacy 28D replay adaptation

#### 3. Record Teleoperation Episodes

Single episode:

```bash
cd teleop
python teleop_hand.py --task television_lab --record --output ../data/recordings/isaaclab/processed_episode_0.hdf5
```

Batch collection:

```bash
cd scripts
python collect_episodes.py --num_episodes 5 --task television_lab --output_dir ../data/recordings/isaaclab
```

#### 4. Replay Episodes

Task is inferred from episode metadata by default:

```bash
cd scripts
python replay_demo.py --episode_path ../data/recordings/isaaclab/processed_episode_0.hdf5
```

Manual override is still available:

```bash
cd scripts
python replay_demo.py --task television_h1 --episode_path ../data/recordings/isaaclab/processed_episode_0.hdf5
```

If your environment uses different observation key names:

```bash
python replay_demo.py --episode_path ../data/recordings/isaaclab/processed_episode_0.hdf5 --left_image_keys observation.image.left --right_image_keys observation.image.right --state_keys observation.state
```

## Dataset and Training

1. Collect or obtain episodes.
2. Place them under `data/recordings/...`.
3. If starting from raw teleop / real-robot captures, run [scripts/post_process.py](scripts/post_process.py).
4. Use [scripts/replay_demo.py](scripts/replay_demo.py) to verify image/action alignment before training.

Processed datasets now carry schema metadata, and the training loader:

- discovers `processed_episode_*.hdf5` by glob instead of assuming contiguous numbering
- rejects mixed task contracts within one processed dataset directory
- derives `state_dim` and `action_dim` from the actual dataset stats

### Train ACT

```bash
python imitate_episodes.py --policy_class ACT --kl_weight 10 --chunk_size 60 --hidden_dim 512 --batch_size 45 --dim_feedforward 3200 --num_epochs 50000 --lr 5e-5 --seed 0 --taskid 00 --exptid 01-sample-expt
```

### Export JIT

```bash
python imitate_episodes.py --policy_class ACT --kl_weight 10 --chunk_size 60 --hidden_dim 512 --batch_size 45 --dim_feedforward 3200 --num_epochs 50000 --lr 5e-5 --seed 0 --taskid 00 --exptid 01-sample-expt --save_jit --resume_ckpt 25000
```

### Deploy in Isaac Lab

By default the deploy script infers the task from episode metadata:

```bash
cd scripts
python deploy_sim.py --taskid 00 --exptid 01 --resume_ckpt 25000
```

Explicit H1 deployment:

```bash
cd scripts
python deploy_sim.py --task television_h1 --taskid 00 --exptid 01 --resume_ckpt 25000
```

## Key Files

- [tv_isaaclab/contracts.py](tv_isaaclab/contracts.py): shared action/state/task contracts
- [tv_isaaclab/env_bridge.py](tv_isaaclab/env_bridge.py): schema-aware environment bridge
- [tv_isaaclab/tasks/television_lab_real.py](tv_isaaclab/tasks/television_lab_real.py): real Isaac Lab task skeletons
- [tv_isaaclab/tasks/television_lab.py](tv_isaaclab/tasks/television_lab.py): fallback adapter tasks
- [tests/test_action_contracts.py](tests/test_action_contracts.py): migration contract regressions
- [tests/test_migration_contracts.py](tests/test_migration_contracts.py): task registration and bridge regressions

## Validation

Static verification already used in this migration:

```bash
py -3 -m py_compile tv_isaaclab/contracts.py tv_isaaclab/env_bridge.py tv_isaaclab/recording.py tv_isaaclab/tasks/television_lab.py tv_isaaclab/tasks/television_lab_real.py teleop/teleop_hand.py scripts/collect_episodes.py scripts/replay_demo.py scripts/deploy_sim.py scripts/post_process.py act/utils.py act/imitate_episodes.py
py -3 -m unittest discover -s tests -v
```

Known limitation:

- without a working Isaac Lab + `gymnasium` runtime environment, these checks validate migration structure and contracts but not full Isaac Sim runtime execution

## Citation

```bibtex
@article{cheng2024tv,
  title={Open-TeleVision: Teleoperation with Immersive Active Visual Feedback},
  author={Cheng, Xuxin and Li, Jialong and Yang, Shiqi and Yang, Ge and Wang, Xiaolong},
  journal={arXiv preprint arXiv:2407.01512},
  year={2024}
}
```
