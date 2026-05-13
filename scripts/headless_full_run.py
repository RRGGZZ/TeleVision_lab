"""Headless end-to-end smoke runner for TeleVision Isaac Lab tasks."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from act.utils import EpisodicDataset, get_norm_stats, list_processed_episode_paths
from scripts.deploy_sim import merge_act, normalize_input
from scripts.replay_demo import Player
from tv_isaaclab import EpisodeRecorder, IsaacLabEnvBridge, add_app_launcher_args, launch_simulation_app
from tv_isaaclab.contracts import H1_TASK_ID, TELEOP_TASK_ID, infer_task_from_episode


class DummyChunkPolicy(nn.Module):
    """Simple policy used to smoke-test the deployment path."""

    def __init__(self, state_dim: int, action_dim: int, num_queries: int):
        super().__init__()
        self.num_queries = num_queries
        self.action_dim = action_dim
        self.state_proj = nn.Linear(state_dim, action_dim)

    def forward(self, qpos: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        del image
        base = torch.tanh(self.state_proj(qpos))
        return base.unsqueeze(1).repeat(1, self.num_queries, 1)


def _parse_tasks(raw: str) -> list[str]:
    tasks = [item.strip() for item in raw.split(",") if item.strip()]
    if not tasks:
        raise ValueError("At least one task must be provided.")
    invalid = [task for task in tasks if task not in (TELEOP_TASK_ID, H1_TASK_ID)]
    if invalid:
        raise ValueError(f"Unsupported task ids: {invalid}")
    return tasks


def _choose_policy_device(raw: str, sim_device: str | None = None) -> str:
    if raw != "auto":
        return raw
    if sim_device:
        normalized = str(sim_device).lower()
        if normalized.startswith("cpu"):
            return "cpu"
        if normalized.startswith("cuda") and torch.cuda.is_available():
            return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _sample_action(bridge: IsaacLabEnvBridge, step_idx: int) -> np.ndarray:
    action = np.asarray(bridge.env.action_space.sample(), dtype=np.float32).reshape(-1)
    if action.size != bridge.action_dim:
        resized = np.zeros(bridge.action_dim, dtype=np.float32)
        copy_n = min(action.size, bridge.action_dim)
        resized[:copy_n] = action[:copy_n]
        action = resized
    phase = np.float32(0.15 * step_idx)
    return np.tanh(action * 0.5 + phase).astype(np.float32)


def _record_episode(task: str, steps: int, output_dir: Path) -> tuple[Path, dict]:
    bridge = IsaacLabEnvBridge(task=task)
    recorder = EpisodeRecorder(
        action_schema=bridge.action_schema,
        cmd_schema=bridge.action_schema,
        state_schema=bridge.state_schema,
    )

    first_obs = bridge.reset()
    recorded_state_shapes = [list(first_obs.state.shape)]
    for step_idx in range(steps):
        action = _sample_action(bridge, step_idx)
        obs = bridge.step(action)
        recorder.append(
            left_rgb=obs.left_rgb,
            right_rgb=obs.right_rgb,
            state=obs.state,
            action=action,
            cmd=action,
        )
        recorded_state_shapes.append(list(obs.state.shape))

    processed_dir = output_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    episode_path = processed_dir / "processed_episode_0.hdf5"
    recorder.save(episode_path)

    bridge.close()
    summary = {
        "task": task,
        "action_dim": int(bridge.action_dim),
        "action_schema": bridge.action_schema,
        "state_schema": bridge.state_schema,
        "supports_teleop_to_action": bool(bridge.supports_teleop_to_action),
        "is_real_env": bool(bridge.is_real_env),
        "recorded_state_shapes": recorded_state_shapes,
    }
    return episode_path, summary


def _replay_episode(task: str, episode_path: Path, steps: int) -> dict:
    player = Player(task=task, show_plot=False)
    with h5py.File(str(episode_path), "r") as handle:
        actions = np.asarray(handle["qpos_action"], dtype=np.float32)
        left_imgs = np.asarray(handle["observation.image.left"])
        right_imgs = np.asarray(handle["observation.image.right"])

    replay_steps = min(steps, actions.shape[0])
    state_shapes: list[list[int]] = []
    for step_idx in range(replay_steps):
        obs = player.step(actions[step_idx], left_imgs[step_idx], right_imgs[step_idx])
        state_shapes.append(list(obs.state.shape))
    player.end()
    return {
        "steps": replay_steps,
        "state_shapes": state_shapes,
    }


def _build_dataset_smoke(dataset_dir: Path, batch_size: int) -> tuple[dict, tuple[int, ...]]:
    norm_stats, episode_len = get_norm_stats(dataset_dir, num_episodes=0)
    dataset = EpisodicDataset(
        episode_paths=list_processed_episode_paths(dataset_dir),
        camera_names=["left", "right"],
        norm_stats=norm_stats,
        episode_len=episode_len,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    sample_batch = next(iter(loader))
    image_data, qpos_data, action_data, is_pad = sample_batch
    sample_shape = tuple(int(dim) for dim in image_data.shape[1:])
    return (
        {
            "task": dataset.task,
            "action_schema": dataset.action_schema,
            "state_schema": dataset.state_schema,
            "action_dim": dataset.action_dim,
            "state_dim": dataset.state_dim,
            "image_batch_shape": list(image_data.shape),
            "qpos_batch_shape": list(qpos_data.shape),
            "action_batch_shape": list(action_data.shape),
            "is_pad_batch_shape": list(is_pad.shape),
        },
        sample_shape,
    )


def _create_dummy_policy(
    output_dir: Path,
    state_dim: int,
    image_shape: tuple[int, ...],
    action_dim: int,
    device: str,
    num_queries: int,
) -> Path:
    policy = DummyChunkPolicy(state_dim=state_dim, action_dim=action_dim, num_queries=num_queries).to(device)
    policy.eval()
    example_qpos = torch.zeros((1, state_dim), dtype=torch.float32, device=device)
    example_image = torch.zeros((1, *image_shape), dtype=torch.float32, device=device)
    traced = torch.jit.trace(policy, (example_qpos, example_image))
    policy_dir = output_dir / "policy"
    policy_dir.mkdir(parents=True, exist_ok=True)
    policy_path = policy_dir / "traced_jit_smoke.pt"
    traced.save(str(policy_path))
    return policy_path


def _save_dataset_stats(output_dir: Path, norm_stats: dict) -> Path:
    stats_path = output_dir / "policy" / "dataset_stats.pkl"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "wb") as handle:
        pickle.dump(norm_stats, handle)
    return stats_path


def _deploy_policy_smoke(
    task: str,
    episode_path: Path,
    norm_stats: dict,
    policy_path: Path,
    device: str,
    rollout_steps: int,
    num_queries: int,
) -> dict:
    policy = torch.jit.load(str(policy_path), map_location=device)
    policy.eval()
    player = Player(task=task, show_plot=False)

    with h5py.File(str(episode_path), "r") as handle:
        states = np.asarray(handle["observation.state"], dtype=np.float32)
        left_imgs = np.asarray(handle["observation.image.left"])
        right_imgs = np.asarray(handle["observation.image.right"])
        reference_actions = np.asarray(handle["qpos_action"], dtype=np.float32)

    steps = min(rollout_steps, states.shape[0])
    all_time_actions = np.zeros((steps, steps + num_queries, reference_actions.shape[-1]), dtype=np.float32)
    action_norm_mean = np.asarray(norm_stats["action_mean"], dtype=np.float32).reshape(-1)
    action_norm_std = np.asarray(norm_stats["action_std"], dtype=np.float32).reshape(-1)

    executed_shapes: list[list[int]] = []
    for step_idx in range(steps):
        inputs = normalize_input(
            states[step_idx],
            left_imgs[step_idx],
            right_imgs[step_idx],
            norm_stats,
            device=device,
        )
        output = policy(*inputs)[0].detach().cpu().numpy()
        all_time_actions[[step_idx], step_idx : step_idx + num_queries] = output
        normalized_action = merge_act(all_time_actions[:, step_idx])
        action = normalized_action * action_norm_std + action_norm_mean
        obs = player.step(action, left_imgs[step_idx], right_imgs[step_idx])
        executed_shapes.append(list(obs.state.shape))

    player.end()
    return {
        "steps": steps,
        "executed_state_shapes": executed_shapes,
    }


def run_task(task: str, args, output_root: Path, policy_device: str) -> dict:
    task_dir = output_root / task
    task_dir.mkdir(parents=True, exist_ok=True)

    episode_path, record_summary = _record_episode(task, args.record_steps, task_dir)
    inferred_task = infer_task_from_episode(episode_path, fallback=task)
    replay_summary = _replay_episode(task, episode_path, args.replay_steps)

    dataset_summary, image_shape = _build_dataset_smoke(episode_path.parent, args.batch_size)
    norm_stats, _ = get_norm_stats(episode_path.parent, num_episodes=0)
    _save_dataset_stats(task_dir, norm_stats)
    policy_path = _create_dummy_policy(
        output_dir=task_dir,
        state_dim=dataset_summary["state_dim"],
        image_shape=image_shape,
        action_dim=dataset_summary["action_dim"],
        device=policy_device,
        num_queries=args.policy_queries,
    )
    deploy_summary = _deploy_policy_smoke(
        task=task,
        episode_path=episode_path,
        norm_stats=norm_stats,
        policy_path=policy_path,
        device=policy_device,
        rollout_steps=args.deploy_steps,
        num_queries=args.policy_queries,
    )

    task_summary = {
        "task": task,
        "episode_path": str(episode_path),
        "inferred_task": inferred_task,
        "record": record_summary,
        "replay": replay_summary,
        "dataset": dataset_summary,
        "policy_path": str(policy_path),
        "deploy": deploy_summary,
    }
    with open(task_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(task_summary, handle, indent=2, ensure_ascii=False)
    return task_summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Headless end-to-end TeleVision smoke runner")
    parser.add_argument("--tasks", default=f"{TELEOP_TASK_ID},{H1_TASK_ID}")
    parser.add_argument("--output_root", default=str(ROOT_DIR / "data" / "headless_runs"))
    parser.add_argument("--record_steps", type=int, default=8)
    parser.add_argument("--replay_steps", type=int, default=4)
    parser.add_argument("--deploy_steps", type=int, default=4)
    parser.add_argument("--policy_queries", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--policy_device",
        "--policy-device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help=(
            "Device for PyTorch dataset/deploy smoke steps. Isaac Lab's AppLauncher owns "
            "--device for the simulator."
        ),
    )
    add_app_launcher_args(parser)
    parser.set_defaults(headless=True)
    args = parser.parse_args()

    tasks = _parse_tasks(args.tasks)
    policy_device = _choose_policy_device(args.policy_device, sim_device=getattr(args, "device", None))
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    simulation_app = launch_simulation_app(args)
    summary = {
        "tasks": tasks,
        "device": policy_device,
        "policy_device": policy_device,
        "sim_device": getattr(args, "device", None),
        "output_root": str(output_root),
        "headless": bool(getattr(args, "headless", True)),
        "results": [],
    }

    try:
        for task in tasks:
            if hasattr(simulation_app, "is_running") and not simulation_app.is_running():
                raise RuntimeError("Simulation app exited before the smoke run completed.")
            print(f"\n[HeadlessSmoke] Running task: {task}")
            task_summary = run_task(task, args, output_root, policy_device)
            summary["results"].append(task_summary)
            print(
                f"[HeadlessSmoke] Completed {task}: "
                f"recorded={task_summary['record']['action_dim']}D, "
                f"inferred={task_summary['inferred_task']}"
            )
    finally:
        simulation_app.close()

    summary_path = output_root / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(f"\n[HeadlessSmoke] Summary written to: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
