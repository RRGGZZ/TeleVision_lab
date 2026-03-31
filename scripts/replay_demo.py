import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import h5py
from tqdm import tqdm
import argparse
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from tv_isaaclab import add_app_launcher_args, launch_simulation_app, IsaacLabEnvBridge  # noqa: E402

class Player:
    def __init__(self, task="television_lab", left_image_keys=None, right_image_keys=None, state_keys=None, show_plot=True):
        self.bridge = IsaacLabEnvBridge(
            task=task,
            left_image_keys=left_image_keys,
            right_image_keys=right_image_keys,
            state_keys=state_keys,
        )
        self.bridge.reset()
        self.show_plot = show_plot

        if show_plot:
            plt.figure(figsize=(12, 6))
            plt.ion()

    def step(self, action, left_img=None, right_img=None):
        action = np.asarray(action, dtype=np.float32)
        if action.shape[0] != self.bridge.action_dim:
            resized = np.zeros(self.bridge.action_dim, dtype=np.float32)
            n = min(self.bridge.action_dim, action.shape[0])
            resized[:n] = action[:n]
            action = resized

        obs = self.bridge.step(action)
        if left_img is None or right_img is None:
            left_img = np.transpose(obs.left_rgb, (2, 0, 1))
            right_img = np.transpose(obs.right_rgb, (2, 0, 1))

        if self.show_plot:
            img = np.concatenate((left_img.transpose((1, 2, 0)), right_img.transpose((1, 2, 0))), axis=1)
            plt.cla()
            plt.title('VisionPro View')
            plt.imshow(img, aspect='equal')
            plt.pause(0.001)

        return obs

    def end(self):
        self.bridge.close()
        if self.show_plot:
            plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Replay processed episode in Isaac Lab")
    parser.add_argument("--task", type=str, default="television_lab")
    parser.add_argument("--episode_path", type=str, required=True)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--left_image_keys", type=str, default="")
    parser.add_argument("--right_image_keys", type=str, default="")
    parser.add_argument("--state_keys", type=str, default="")
    add_app_launcher_args(parser)
    args = parser.parse_args()

    simulation_app = launch_simulation_app(args)
    data = h5py.File(str(Path(args.episode_path).resolve()), 'r')
    actions = np.array(data['qpos_action'])[::args.stride]
    left_imgs = np.array(data['observation.image.left'])[::args.stride]
    right_imgs = np.array(data['observation.image.right'])[::args.stride]
    data.close()

    def parse_keys(raw):
        return [x.strip() for x in raw.split(",") if x.strip()] if raw else None

    player = Player(
        task=args.task,
        left_image_keys=parse_keys(args.left_image_keys),
        right_image_keys=parse_keys(args.right_image_keys),
        state_keys=parse_keys(args.state_keys),
    )

    try:
        for t in tqdm(range(actions.shape[0])):
            if not simulation_app.is_running():
                break
            player.step(actions[t], left_imgs[t, :], right_imgs[t, :])
    except KeyboardInterrupt:
        pass
    finally:
        player.end()
        simulation_app.close()
