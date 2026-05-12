import numpy as np
from replay_demo import Player

from pathlib import Path
import h5py
from tqdm import tqdm
import pickle
import torch
from collections import deque
import argparse
import sys
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from act.utils import parse_id
from tv_isaaclab import add_app_launcher_args, launch_simulation_app
from tv_isaaclab.contracts import H1_TASK_ID, infer_task_from_episode
# from act.imitate_episodes import RECORD_DIR, DATA_DIR, LOG_DIR

current_dir = Path(__file__).parent.resolve()
DATA_DIR = (current_dir.parent / 'data/').resolve()
RECORD_DIR = (DATA_DIR / 'recordings/').resolve()
LOG_DIR = (DATA_DIR / 'logs/').resolve()
# print(f"\nDATA dir: {DATA_DIR}")

def get_norm_stats(data_path):
    # norm_stats = {
    #     "action_mean": np.array([]), "action_std": np.array([]),
    #     "qpos_mean": np.array([]), "qpos_std": np.array([]),
    # }
    with open(data_path, "rb") as f:
        norm_stats = pickle.load(f)
    return norm_stats


def load_policy(policy_path, device):
    policy = torch.jit.load(policy_path, map_location=device)
    return policy


def _to_chw_image(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError(f"Expected image with 3 dims, got shape={array.shape}")
    if array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        return array[:3]
    if array.shape[-1] in (3, 4):
        return np.transpose(array[..., :3], (2, 0, 1))
    raise ValueError(f"Unsupported image layout for shape={array.shape}")


def normalize_input(
    state,
    left_img,
    right_img,
    norm_stats,
    last_action_data=None,
    device="cuda",
):
    left_chw = _to_chw_image(left_img)
    right_chw = _to_chw_image(right_img)
    image_np = np.stack([left_chw, right_chw], axis=0).astype(np.float32) / 255.0
    image_data = torch.from_numpy(image_np).unsqueeze(0).to(device=device)

    qpos_mean = np.asarray(norm_stats["qpos_mean"], dtype=np.float32).reshape(-1)
    qpos_std = np.asarray(norm_stats["qpos_std"], dtype=np.float32).reshape(-1)
    state_np = np.asarray(state, dtype=np.float32).reshape(-1)
    qpos_data = (torch.from_numpy(state_np) - torch.from_numpy(qpos_mean)) / torch.from_numpy(qpos_std)
    qpos_data = qpos_data.view((1, -1)).to(device=device)

    if last_action_data is not None:
        last_action_data = (
            torch.from_numpy(np.asarray(last_action_data, dtype=np.float32))
            .to(device=device)
            .view((1, -1))
            .to(torch.float)
        )
        qpos_data = torch.cat((qpos_data, last_action_data), dim=1)
    return (qpos_data, image_data)


def merge_act(actions_for_curr_step, k = 0.01):
    actions_populated = np.all(actions_for_curr_step != 0, axis=1)
    actions_for_curr_step = actions_for_curr_step[actions_populated]

    exp_weights = np.exp(-k * np.arange(actions_for_curr_step.shape[0]))
    exp_weights = (exp_weights / exp_weights.sum()).reshape((-1, 1))
    raw_action = (actions_for_curr_step * exp_weights).sum(axis=0)

    return raw_action


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Deploy ACT policy in Isaac Lab', add_help=False)
    parser.add_argument('--taskid', action='store', type=str, help='task id', required=True)
    parser.add_argument('--exptid', action='store', type=str, help='experiment id', required=True)
    parser.add_argument('--resume_ckpt', action='store', type=str, help='resume checkpoint', required=True)
    parser.add_argument('--task', type=str, default='', help='Isaac Lab task name')
    parser.add_argument('--left_image_keys', type=str, default='')
    parser.add_argument('--right_image_keys', type=str, default='')
    parser.add_argument('--state_keys', type=str, default='')
    add_app_launcher_args(parser)
    args = vars(parser.parse_args())

    simulation_app = launch_simulation_app(argparse.Namespace(**args))

    episode_name = "processed_episode_0.hdf5"
    task_dir, task_name = parse_id(RECORD_DIR, args['taskid'])
    episode_path = (Path(task_dir) / 'processed' / episode_name).resolve()
    exp_path, _ = parse_id((Path(LOG_DIR) / task_name).resolve(), args['exptid'])
    resolved_task = args['task'] or infer_task_from_episode(episode_path, fallback=H1_TASK_ID)
    
    norm_stat_path = Path(exp_path) / "dataset_stats.pkl"
    policy_path = Path(exp_path) / f"traced_jit_{args['resume_ckpt']}.pt"
    
    temporal_agg = True
    chunk_size = 60
    device = "cuda"

    data = h5py.File(str(episode_path), 'r')
    actions = np.array(data['qpos_action'])
    left_imgs = np.array(data['observation.image.left'])
    right_imgs = np.array(data['observation.image.right'])
    states = np.array(data['observation.state'])
    init_action = np.array(data.attrs['init_action'])
    data.close()
    timestamps = states.shape[0]
    action_dim = actions.shape[-1]

    norm_stats = get_norm_stats(norm_stat_path)
    policy = load_policy(policy_path, device)
    policy.cuda()
    policy.eval()

    history_stack = 0
    if history_stack > 0:
        last_action_queue = deque(maxlen=history_stack)
        for i in range(history_stack):
            last_action_queue.append(actions[0])
    else:
        last_action_queue = None
        last_action_data = None

    def parse_keys(raw):
        return [x.strip() for x in raw.split(",") if x.strip()] if raw else None

    player = Player(
        task=resolved_task,
        left_image_keys=parse_keys(args['left_image_keys']),
        right_image_keys=parse_keys(args['right_image_keys']),
        state_keys=parse_keys(args['state_keys']),
    )

    if temporal_agg:
        all_time_actions = np.zeros([timestamps, timestamps+chunk_size, action_dim])
    else:
        num_actions_exe = chunk_size
    
    try:
        output = None
        act_index = 0
        for t in tqdm(range(timestamps)):
            if not simulation_app.is_running():
                break
            if history_stack > 0:
                last_action_data = np.array(last_action_queue)

            data = normalize_input(
                states[t],
                left_imgs[t],
                right_imgs[t],
                norm_stats,
                last_action_data,
                device=device,
            )

            if temporal_agg:
                output = policy(*data)[0].detach().cpu().numpy() # (1,chuck_size,action_dim)
                all_time_actions[[t], t:t+chunk_size] = output
                act = merge_act(all_time_actions[:, t])
            else:
                if output is None or act_index == num_actions_exe-1:
                    print("Inference...")
                    output = policy(*data)[0].detach().cpu().numpy()
                    act_index = 0
                act = output[act_index]
                act_index += 1
            # import ipdb; ipdb.set_trace()
            if history_stack > 0:
                last_action_queue.append(act)
            act = act * norm_stats["action_std"] + norm_stats["action_mean"]
            player.step(act, left_imgs[t], right_imgs[t])
    except KeyboardInterrupt:
        pass
    finally:
        player.end()
        simulation_app.close()
