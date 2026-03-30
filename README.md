<h1 align="center"><img src="img/logo.png" width="40"> Open-TeleVision: Teleoperation with

Immersive Active Visual Feedback</h1>

<p align="center">
    <a href="https://chengxuxin.github.io/"><strong>Xuxin Cheng*</strong></a>
    ·
    <a href=""><strong>Jialong Li*</strong></a>
    ·
    <a href="https://aaronyang1223.github.io/"><strong>Shiqi Yang</strong></a>
    <br>
    <a href="https://www.episodeyang.com/"><strong>Ge Yang</strong></a>
    ·
    <a href="https://xiaolonw.github.io/"><strong>Xiaolong Wang</strong></a>
</p>

<p align="center">
    <img src="img/UCSanDiegoLogo-BlueGold.png" height=50"> &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;
    <img src="img/mit-logo.png" height="50">
</p>

<h3 align="center"> CoRL 2024 </h3>

<p align="center">
<h3 align="center"><a href="https://robot-tv.github.io/">Website</a> | <a href="https://arxiv.org/abs/2407.01512/">arXiv</a> | <a href="">Video</a> | <a href="">Summary</a> </h3>
  <div align="center"></div>
</p>

<p align="center">
<img src="./img/main.webp" width="80%"/>
</p>

## Introduction
This code contains implementation for teleoperation and imitation learning of Open-TeleVision.

## Installation

```bash
    conda create -n tv python=3.8
    conda activate tv
    pip install -r requirements.txt
    cd act/detr && pip install -e .
```

**For Python 3.11+ environments** (like `television_lab`):

See [SETUP_PYTHON311.md](SETUP_PYTHON311.md) for complete setup instructions. Key differences:
- `dex-retargeting` installed from GitHub (0.5.0 supports Python 3.11+)
- numpy constrained to `<2.0` for Isaac Lab compatibility
- Quick setup: `bash install_deps.sh`

Install ZED sdk: https://www.stereolabs.com/developers/release/

Install ZED Python API:
```
    cd /usr/local/zed/ && python get_python_api.py
```

If you want to run simulation teleoperation, replay, and policy deployment in Isaac Lab:

Install Isaac Lab first, and make sure your task environment is registered (for this repo: `television_lab`).

## Teleoperation Guide

### Local streaming
For **Quest** local streaming, follow [this](https://github.com/OpenTeleVision/TeleVision/issues/12#issue-2401541144) issue.

**Apple** does not allow WebXR on non-https connections. To test the application locally, we need to create a self-signed certificate and install it on the client. You need a ubuntu machine and a router. Connect the VisionPro and the ubuntu machine to the same router. 
1. install mkcert: https://github.com/FiloSottile/mkcert
2. check local ip address: 

```
    ifconfig | grep inet
```
Suppose the local ip address of the ubuntu machine is `192.168.8.102`.

3. create certificate: 

```
    mkcert -install && mkcert -cert-file cert.pem -key-file key.pem 192.168.8.102 localhost 127.0.0.1
```
ps. place the generated `cert.pem` and `key.pem` files in `teleop`.

4. open firewall on server
```
    sudo iptables -A INPUT -p tcp --dport 8012 -j ACCEPT
    sudo iptables-save
    sudo iptables -L
```
or can be done with `ufw`:
```
    sudo ufw allow 8012
```
5.
```
    tv = OpenTeleVision(self.resolution_cropped, shm.name, image_queue, toggle_streaming, ngrok=False)
```

6. install ca-certificates on VisionPro
```
    mkcert -CAROOT
```
Copy the rootCA.pem via AirDrop to VisionPro and install it.

Settings > General > About > Certificate Trust Settings. Under "Enable full trust for root certificates", turn on trust for the certificate.

settings > Apps > Safari > Advanced > Feature Flags > Enable WebXR Related Features

7. open the browser on Safari on VisionPro and go to `https://192.168.8.102:8012?ws=wss://192.168.8.102:8012`

8. Click `Enter VR` and ``Allow`` to start the VR session.

### Network Streaming
For Meta Quest3, installation of the certificate is not trivial. We need to use a network streaming solution. We use `ngrok` to create a secure tunnel to the server. This method will work for both VisionPro and Meta Quest3.

1. Install ngrok: https://ngrok.com/download
2. Run ngrok
```
    ngrok http 8012
```
3. Copy the https address and open the browser on Meta Quest3 and go to the address.

ps. When using ngrok for network streaming, remember to call `OpenTeleVision` with:
```
    self.tv = OpenTeleVision(self.resolution_cropped, self.shm.name, image_queue, toggle_streaming, ngrok=True)
```

### Isaac Lab Teleoperation Example

**1. Verify Isaac Lab Installation**

Run integration tests to verify migration:
```
    cd scripts && python test_integration.py
```

Expected output: `🎉 All tests passed! Isaac Lab migration is successful.`

**2. Get Environment Schema**

```
    python ../scripts/quick_probe.py
```

This shows:
- Action space: Box(-1.0, 1.0, (38,), float32)
- Observation keys: observation.image.{left,right}, observation.state
- Image dimensions: 512x512 RGB
- State dimension: 38D

**3. Record Episodes with VisionPro**

After setting up streaming with either local or network streaming, record teleoperation episodes:

```
    cd teleop && python teleop_hand.py --task television_lab --record --output ../data/recordings/isaaclab/processed_episode_0.hdf5
```

Alternatively, batch collect multiple episodes:

```
    cd scripts && python collect_episodes.py --num_episodes 5 --task television_lab --output_dir ../data/recordings/isaaclab
```

**4. Replay Episodes**

Verify collected episodes by replaying them in the environment:

```
    cd scripts && python replay_demo.py --task television_lab --episode_path ../data/recordings/isaaclab/processed_episode_0.hdf5
```

If your environment uses different observation key names, pass key paths explicitly:

```
    cd teleop && python teleop_hand.py --task television_lab \
      --left_image_keys observation.image.left --right_image_keys observation.image.right --state_keys observation.state
```

For more detailed testing and troubleshooting, see [TESTING.md](TESTING.md).

## Training Guide
1. Collect or download dataset. See "Record Episodes with VisionPro" above or download from https://drive.google.com/drive/folders/11WO96mUMjmxRo9Hpvm4ADz7THuuGNEMY?usp=sharing.

2. Place the collected dataset in ``data/recordings/isaaclab/``.

3. Process the specified dataset for training using ``scripts/post_process.py`` (if needed).

4. You can verify the image and action sequences of a specific episode in Isaac Lab using ``scripts/replay_demo.py`` (see above).

5. To train ACT, run:
```
    python imitate_episodes.py --policy_class ACT --kl_weight 10 --chunk_size 60 --hidden_dim 512 --batch_size 45 --dim_feedforward 3200 --num_epochs 50000 --lr 5e-5 --seed 0 --taskid 00 --exptid 01-sample-expt
```

6. After training, save jit for the desired checkpoint:
```
    python imitate_episodes.py --policy_class ACT --kl_weight 10 --chunk_size 60 --hidden_dim 512 --batch_size 45 --dim_feedforward 3200 --num_epochs 50000 --lr 5e-5 --seed 0 --taskid 00 --exptid 01-sample-expt\
                               --save_jit --resume_ckpt 25000
```

7. You can visualize the trained policy in Isaac Lab with inputs from dataset using ``scripts/deploy_sim.py``, example usage:
```
    cd scripts && python deploy_sim.py --task television_lab --taskid 00 --exptid 01 --resume_ckpt 25000
```

## Citation
```
@article{cheng2024tv,
title={Open-TeleVision: Teleoperation with Immersive Active Visual Feedback},
author={Cheng, Xuxin and Li, Jialong and Yang, Shiqi and Yang, Ge and Wang, Xiaolong},
journal={arXiv preprint arXiv:2407.01512},
year={2024}
}
```
# TeleVision_lab
