import os
import sys
import time
from pathlib import Path

import h5py
import IPython
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tv_isaaclab.contracts import infer_task_from_schemas

e = IPython.embed


def _processed_episode_sort_key(path_like):
    stem = Path(path_like).stem
    suffix = stem.split("processed_episode_")[-1]
    try:
        return (0, int(suffix))
    except ValueError:
        return (1, stem)


def list_processed_episode_paths(dataset_dir):
    dataset_dir = Path(dataset_dir)
    episode_paths = sorted(dataset_dir.glob("processed_episode_*.hdf5"), key=_processed_episode_sort_key)
    if not episode_paths:
        raise FileNotFoundError(f"No processed episodes found under {dataset_dir}")
    return episode_paths


def _read_episode_metadata(root):
    action_schema = root.attrs.get("action_schema")
    state_schema = root.attrs.get("state_schema")
    if isinstance(action_schema, bytes):
        action_schema = action_schema.decode("utf-8")
    if isinstance(state_schema, bytes):
        state_schema = state_schema.decode("utf-8")
    action_dim = int(root["qpos_action"].shape[-1])
    state_dim = int(root["observation.state"].shape[-1])
    return {
        "action_schema": action_schema,
        "state_schema": state_schema,
        "action_dim": action_dim,
        "state_dim": state_dim,
        "task": infer_task_from_schemas(action_schema, action_dim),
    }

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_paths, camera_names, norm_stats, episode_len, history_stack=0):
        super(EpisodicDataset).__init__()
        self.episode_paths = [Path(path) for path in episode_paths]
        self.dataset_dir = str(self.episode_paths[0].parent) if self.episode_paths else ""
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = None
        self.max_pad_len = 200
        action_str = 'qpos_action'

        self.history_stack = history_stack

        self.dataset_paths = []
        self.roots = []
        self.is_sims = []
        self.original_action_shapes = []

        self.states = []
        self.image_dict = dict()
        for cam_name in self.camera_names:
            self.image_dict[cam_name] = []
        self.actions = []
        self.metadata = []

        for episode_path in self.episode_paths:
            self.dataset_paths.append(str(episode_path))
            root = h5py.File(str(episode_path), 'r')
            self.roots.append(root)
            self.is_sims.append(root.attrs['sim'])
            self.original_action_shapes.append(root[action_str].shape)
            self.metadata.append(_read_episode_metadata(root))

            self.states.append(np.array(root['observation.state']))
            for cam_name in self.camera_names:
                self.image_dict[cam_name].append(root[f'observation.image.{cam_name}'])
            self.actions.append(np.array(root[action_str]))

        self.is_sim = self.is_sims[0]
        self.task = self.metadata[0]["task"]
        self.action_schema = self.metadata[0]["action_schema"]
        self.state_schema = self.metadata[0]["state_schema"]
        self.action_dim = self.metadata[0]["action_dim"]
        self.state_dim = self.metadata[0]["state_dim"]

        for meta in self.metadata[1:]:
            if meta["action_dim"] != self.action_dim or meta["state_dim"] != self.state_dim:
                raise ValueError(
                    "Processed episodes mix different action/state dimensions; "
                    f"got {(self.action_dim, self.state_dim)} and {(meta['action_dim'], meta['state_dim'])}."
                )
            if meta["task"] != self.task:
                raise ValueError(
                    "Processed episodes mix different task contracts; "
                    f"got {self.task!r} and {meta['task']!r}."
                )

        self.episode_len = episode_len
        self.cumulative_len = np.cumsum(self.episode_len)

        # self.__getitem__(0) # initialize self.is_sim

    def __len__(self):
        if len(self.cumulative_len) == 0:
            return 0
        return int(self.cumulative_len[-1])

    def _locate_transition(self, index):
        assert index < self.cumulative_len[-1]
        episode_index = np.argmax(self.cumulative_len > index) # argmax returns first True index
        start_ts = index - (self.cumulative_len[episode_index] - self.episode_len[episode_index])
        return episode_index, start_ts
    
    def __getitem__(self, ts_index):
        sample_full_episode = False # hardcode

        index, start_ts = self._locate_transition(ts_index)

        original_action_shape = self.original_action_shapes[index]
        episode_len = original_action_shape[0]

        if sample_full_episode:
            start_ts = 0
        else:
            start_ts = np.random.choice(episode_len)

        # get observation at start_ts only
        qpos = self.states[index][start_ts]
        # qvel = root['/observations/qvel'][start_ts]

        if self.history_stack > 0:
            last_indices = np.maximum(0, np.arange(start_ts-self.history_stack, start_ts)).astype(int)
            last_action = self.actions[index][last_indices, :]

        image_dict = dict()
        for cam_name in self.camera_names:
            image_dict[cam_name] = self.image_dict[cam_name][index][start_ts]
        # get all actions after and including start_ts
        all_time_action = self.actions[index][:]

        all_time_action_padded = np.zeros((self.max_pad_len+original_action_shape[0], original_action_shape[1]), dtype=np.float32)
        all_time_action_padded[:episode_len] = all_time_action
        all_time_action_padded[episode_len:] = all_time_action[-1]
        
        padded_action = all_time_action_padded[start_ts:start_ts+self.max_pad_len] 
        real_len = episode_len - start_ts

        is_pad = np.zeros(self.max_pad_len)
        is_pad[real_len:] = 1

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()
        if self.history_stack > 0:
            last_action_data = torch.from_numpy(last_action).float()

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]
        if self.history_stack > 0:
            last_action_data = (last_action_data - self.norm_stats['action_mean']) / self.norm_stats['action_std']
            qpos_data = torch.cat((qpos_data, last_action_data.flatten()))
        # print(f"qpos_data: {qpos_data.shape}, action_data: {action_data.shape}, image_data: {image_data.shape}, is_pad: {is_pad.shape}")
        return image_data, qpos_data, action_data, is_pad


def get_norm_stats(dataset_dir, num_episodes):
    del num_episodes
    action_str = 'qpos_action'
    all_qpos_data = []
    all_action_data = []
    all_episode_len = []
    metadata = None
    dataset_paths = list_processed_episode_paths(dataset_dir)
    for dataset_path in dataset_paths:
        with h5py.File(str(dataset_path), 'r') as root:
            qpos = root['observation.state'][()]
            action = root[action_str][()]
            episode_meta = _read_episode_metadata(root)
            if metadata is None:
                metadata = episode_meta
            elif (
                episode_meta["action_dim"] != metadata["action_dim"]
                or episode_meta["state_dim"] != metadata["state_dim"]
                or episode_meta["task"] != metadata["task"]
            ):
                raise ValueError(
                    "Processed dataset mixes incompatible schemas or dimensions; "
                    f"got {metadata} and {episode_meta}."
                )
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))
        all_episode_len.append(len(qpos))
    all_qpos_data = torch.cat(all_qpos_data)
    all_action_data = torch.cat(all_action_data)
    all_action_data = all_action_data

    # normalize action data
    action_mean = all_action_data.mean(dim=0, keepdim=True)  # (episode, timstep, action_dim)
    action_std = all_action_data.std(dim=0, keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf) # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=0, keepdim=True)
    qpos_std = all_qpos_data.std(dim=0, keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping

    stats = {
        "action_mean": action_mean.numpy().squeeze(),
        "action_std": action_std.numpy().squeeze(),
        "qpos_mean": qpos_mean.numpy().squeeze(),
        "qpos_std": qpos_std.numpy().squeeze(),
        "example_qpos": qpos,
    }
    if metadata is not None:
        stats.update(metadata)

    return stats, all_episode_len

def find_all_processed_episodes(path):
    return [path.name for path in list_processed_episode_paths(path)]

def BatchSampler(batch_size, episode_len_l, sample_weights=None):
    sample_probs = np.array(sample_weights) / np.sum(sample_weights) if sample_weights is not None else None
    sum_dataset_len_l = np.cumsum([0] + [np.sum(episode_len) for episode_len in episode_len_l])
    while True:
        batch = []
        for _ in range(batch_size):
            episode_idx = np.random.choice(len(episode_len_l), p=sample_probs)
            step_idx = np.random.randint(sum_dataset_len_l[episode_idx], sum_dataset_len_l[episode_idx + 1])
            batch.append(step_idx)
        yield batch

def load_data(dataset_dir, camera_names, batch_size_train, batch_size_val):
    print(f'\nData from: {dataset_dir}\n')

    episode_paths = list_processed_episode_paths(dataset_dir)
    num_episodes = len(episode_paths)
    
    # obtain train test split
    train_ratio = 0.99
    shuffled_indices = np.random.permutation(num_episodes)
    train_cutoff = int(train_ratio * num_episodes)
    train_cutoff = min(max(train_cutoff, 1), num_episodes)
    train_indices = shuffled_indices[:train_cutoff]
    val_indices = shuffled_indices[train_cutoff:]
    if len(val_indices) == 0:
        val_indices = train_indices[:1]
    print(f"Train episodes: {len(train_indices)}, Val episodes: {len(val_indices)}")
    # obtain normalization stats for qpos and action
    norm_stats, all_episode_len = get_norm_stats(dataset_dir, num_episodes)

    train_episode_len_l = [all_episode_len[i] for i in train_indices]
    val_episode_len_l = [all_episode_len[i] for i in val_indices]
    batch_sampler_train = BatchSampler(batch_size_train, train_episode_len_l)
    batch_sampler_val = BatchSampler(batch_size_val, val_episode_len_l, None)

    # construct dataset and dataloader
    train_paths = [episode_paths[i] for i in train_indices]
    val_paths = [episode_paths[i] for i in val_indices]
    train_dataset = EpisodicDataset(train_paths, camera_names, norm_stats, train_episode_len_l)
    val_dataset = EpisodicDataset(val_paths, camera_names, norm_stats, val_episode_len_l)
    train_dataloader = DataLoader(train_dataset, batch_sampler=batch_sampler_train, pin_memory=True, num_workers=24, prefetch_factor=2)
    val_dataloader = DataLoader(val_dataset, batch_sampler=batch_sampler_val, pin_memory=True, num_workers=16, prefetch_factor=2)

    dataset_meta = {
        "task": train_dataset.task,
        "action_schema": train_dataset.action_schema,
        "state_schema": train_dataset.state_schema,
        "action_dim": train_dataset.action_dim,
        "state_dim": train_dataset.state_dim,
    }

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim, dataset_meta

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def parse_id(base_dir, prefix):
    base_path = Path(base_dir)
    # Ensure the base path exists and is a directory
    if not base_path.exists() or not base_path.is_dir():
        raise ValueError(f"The provided base directory does not exist or is not a directory: \n{base_path}")

    # Loop through all subdirectories of the base path
    for subfolder in base_path.iterdir():
        if subfolder.is_dir() and subfolder.name.startswith(prefix):
            return str(subfolder), subfolder.name
    
    # If no matching subfolder is found
    return None, None

def find_all_ckpt(base_dir, prefix="policy_epoch_"):
    base_path = Path(base_dir)
    # Ensure the base path exists and is a directory
    if not base_path.exists() or not base_path.is_dir():
        raise ValueError("The provided base directory does not exist or is not a directory.")

    ckpt_files = []
    for file in base_path.iterdir():
        if file.is_file() and file.name.startswith(prefix):
            ckpt_files.append(file.name)
    # find latest ckpt
    ckpt_files = sorted(ckpt_files, key=lambda x: int(x.split(prefix)[-1].split('_')[0]), reverse=True)
    epoch = int(ckpt_files[0].split(prefix)[-1].split('_')[0])
    return ckpt_files[0], epoch
