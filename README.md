<div align="center">

### DePi: $\pi$-series model training with large Decentralized-collected so100 dataset

</div>


[![deploy](https://img.shields.io/badge/Hugging%20Face-DePi0-FFEB3B)](https://huggingface.co/sengi/DePi0)


---
This repository (DePi) focuses on the PyTorch training of $\pi$-series policies using large, decentrally collected SO100/SO101 datasets. It is built upon 🤗 [LeRobot](https://github.com/huggingface/lerobot). To maintain a stable environment for reproducible evaluation, this repository pins specific versions of the hardware configuration and inference code. For the latest upstream features, please refer to the main [LeRobot](https://github.com/huggingface/lerobot) repository.

The objective is to enable all contributors to the LeRobot dataset pool to leverage the full potential of this data collection effort. By providing the complete data collection and annotation pipeline as well as efficient and bug-free training script, this repository aims to create a virtuous cycle. 

Mirroring the philosophy of DeFi (Decentralized Finance), **DePi** aims to incentivize decentralized data contributions to build a more powerful, community-shared resource: a diverse robotic training dataset that can be used to train state-of-the-art VLA models.

## News
- [2025.10.22] DePi code and model is released.

## Features
 - Fast Multi-GPU training of Pi0 policy with torch compile and accelerate
 - Decentralized data collection and annotation pipeline for publicly available SO100 datasets
 - MultiLeRobotDataset class that supports training on multiple LeRobot datasets at once
 - More to come...

## Installation

Download the source code:
```bash
git clone https://github.com/chinsengi/depi.git
cd depi
```

Create a virtual environment with Python 3.10 and activate it, e.g. with [`miniconda`](https://docs.anaconda.com/free/miniconda/index.html):
```bash
conda create -y -n depi python=3.10
conda activate depi
```

When using `miniconda`, install `ffmpeg` in your environment:
Note, need to specify the version of ffmpeg, otherwise torchcodec won't work.
```bash
conda install -c conda-forge ffmpeg=6.1.1 -y
```

> **NOTE:** This usually installs `ffmpeg 7.X` for your platform compiled with the `libsvtav1` encoder. If `libsvtav1` is not supported (check supported encoders with `ffmpeg -encoders`), you can:
>  - _[On any platform]_ Explicitly install `ffmpeg 7.X` using:
>  ```bash
>  conda install ffmpeg=7.1.1 -c conda-forge
>  ```
>  - _[On Linux only]_ Install [ffmpeg build dependencies](https://trac.ffmpeg.org/wiki/CompilationGuide/Ubuntu#GettheDependencies) and [compile ffmpeg from source with libsvtav1](https://trac.ffmpeg.org/wiki/CompilationGuide/Ubuntu#libsvtav1), and make sure you use the corresponding ffmpeg binary to your install with `which ffmpeg`.

Install the dependencies for Pi0 training:
```bash
uv sync
uv pip install -e .
source .venv/bin/activate
```

> **NOTE:** If you encounter build errors, you may need to install additional dependencies (`cmake`, `build-essential`, and `ffmpeg libs`). On Linux, run:
`sudo apt-get install cmake build-essential python3-dev pkg-config libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libswscale-dev libswresample-dev libavfilter-dev pkg-config`. For other systems, see: [Compiling PyAV](https://pyav.org/docs/develop/overview/installation.html#bring-your-own-ffmpeg)

To use [Weights and Biases](https://docs.wandb.ai/quickstart) for experiment tracking, log in with
```bash
wandb login
```

(note: you will also need to enable WandB in the configuration. See below.)

## Walkthrough

```
.
├── examples             # contains demonstration examples, start here to learn about LeRobot
|   └── advanced         # contains even more examples for those who have mastered the basics
├── lerobot
|   ├── configs          # contains config classes with all options that you can override in the command line
|   ├── common           # contains classes and utilities
|   |   ├── datasets       # various datasets of human demonstrations: aloha, pusht, xarm
|   |   ├── envs           # various sim environments: aloha, pusht, xarm
|   |   ├── policies       # various policies: act, diffusion, tdmpc
|   |   ├── robot_devices  # various real devices: dynamixel motors, opencv cameras, koch robots
|   |   └── utils          # various utilities
|   └── scripts          # contains functions to execute via command line
|       ├── eval.py                 # load policy and evaluate it on an environment
|       ├── train.py                # train a policy via imitation learning and/or reinforcement learning
|       ├── control_robot.py        # teleoperate a real robot, record data, run a policy
|       ├── push_dataset_to_hub.py  # convert your dataset into LeRobot dataset format and upload it to the Hugging Face hub
|       └── visualize_dataset.py    # load a dataset and render its demonstrations
├── outputs               # contains results of scripts execution: logs, videos, model checkpoints
├── tests                 # contains pytest utilities for continuous integration
├── so100_data            # contains the annotation tools to improve the quality of decentralized collected so100 datasets 
└── data_ids              # contains the decentralized collected so100 dataset ids and the code to collect them.
```

### Visualize datasets

Check out [example 1](./examples/1_load_lerobot_dataset.py) that illustrates how to use our dataset class which automatically downloads data from the Hugging Face hub.

You can also locally visualize episodes from a dataset on the hub by executing our script from the command line:
```bash
python lerobot/scripts/visualize_dataset.py \
    --repo-id lerobot/pusht \
    --episode-index 0
```

or from a dataset in a local folder with the `root` option and the `--local-files-only` (in the following case the dataset will be searched for in `./my_local_data_dir/lerobot/pusht`)
```bash
python lerobot/scripts/visualize_dataset.py \
    --repo-id lerobot/pusht \
    --root ./my_local_data_dir \
    --local-files-only 1 \
    --episode-index 0
```


It will open `rerun.io` and display the camera streams, robot states and actions, like this:

https://github-production-user-asset-6210df.s3.amazonaws.com/4681518/328035972-fd46b787-b532-47e2-bb6f-fd536a55a7ed.mov?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20240505%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240505T172924Z&X-Amz-Expires=300&X-Amz-Signature=d680b26c532eeaf80740f08af3320d22ad0b8a4e4da1bcc4f33142c15b509eda&X-Amz-SignedHeaders=host&actor_id=24889239&key_id=0&repo_id=748713144


Our script can also visualize datasets stored on a distant server. See `python lerobot/scripts/visualize_dataset.py --help` for more instructions.

### The `LeRobotDataset` format (V2/2.1)

A dataset in `LeRobotDataset` format is very simple to use. It can be loaded from a repository on the Hugging Face hub or a local folder simply with e.g. `dataset = LeRobotDataset("lerobot/aloha_static_coffee")` and can be indexed into like any Hugging Face and PyTorch dataset. For instance `dataset[0]` will retrieve a single temporal frame from the dataset containing observation(s) and an action as PyTorch tensors ready to be fed to a model.

A specificity of `LeRobotDataset` is that, rather than retrieving a single frame by its index, we can retrieve several frames based on their temporal relationship with the indexed frame, by setting `delta_timestamps` to a list of relative times with respect to the indexed frame. For example, with `delta_timestamps = {"observation.image": [-1, -0.5, -0.2, 0]}`  one can retrieve, for a given index, 4 frames: 3 "previous" frames 1 second, 0.5 seconds, and 0.2 seconds before the indexed frame, and the indexed frame itself (corresponding to the 0 entry). See example [1_load_lerobot_dataset.py](examples/1_load_lerobot_dataset.py) for more details on `delta_timestamps`.

Under the hood, the `LeRobotDataset` format makes use of several ways to serialize data which can be useful to understand if you plan to work more closely with this format. We tried to make a flexible yet simple dataset format that would cover most type of features and specificities present in reinforcement learning and robotics, in simulation and in real-world, with a focus on cameras and robot states but easily extended to other types of sensory inputs as long as they can be represented by a tensor.

Here are the important details and internal structure organization of a typical `LeRobotDataset` instantiated with `dataset = LeRobotDataset("lerobot/aloha_static_coffee")`. The exact features will change from dataset to dataset but not the main aspects:

```
dataset attributes:
  ├ hf_dataset: a Hugging Face dataset (backed by Arrow/parquet). Typical features example:
  │  ├ observation.images.cam_high (VideoFrame):
  │  │   VideoFrame = {'path': path to a mp4 video, 'timestamp' (float32): timestamp in the video}
  │  ├ observation.state (list of float32): position of an arm joints (for instance)
  │  ... (more observations)
  │  ├ action (list of float32): goal position of an arm joints (for instance)
  │  ├ episode_index (int64): index of the episode for this sample
  │  ├ frame_index (int64): index of the frame for this sample in the episode ; starts at 0 for each episode
  │  ├ timestamp (float32): timestamp in the episode
  │  ├ next.done (bool): indicates the end of an episode ; True for the last frame in each episode
  │  └ index (int64): general index in the whole dataset
  ├ episode_data_index: contains 2 tensors with the start and end indices of each episode
  │  ├ from (1D int64 tensor): first frame index for each episode — shape (num episodes,) starts with 0
  │  └ to: (1D int64 tensor): last frame index for each episode — shape (num episodes,)
  ├ stats: a dictionary of statistics (max, mean, min, std) for each feature in the dataset, for instance
  │  ├ observation.images.cam_high: {'max': tensor with same number of dimensions (e.g. `(c, 1, 1)` for images, `(c,)` for states), etc.}
  │  ...
  ├ info: a dictionary of metadata on the dataset
  │  ├ codebase_version (str): this is to keep track of the codebase version the dataset was created with
  │  ├ fps (float): frame per second the dataset is recorded/synchronized to
  │  ├ video (bool): indicates if frames are encoded in mp4 video files to save space or stored as png files
  │  └ encoding (dict): if video, this documents the main options that were used with ffmpeg to encode the videos
  ├ videos_dir (Path): where the mp4 videos or png images are stored/accessed
  └ camera_keys (list of string): the keys to access camera features in the item returned by the dataset (e.g. `["observation.images.cam_high", ...]`)
```

A `LeRobotDataset` is serialised using several widespread file formats for each of its parts, namely:
- hf_dataset stored using Hugging Face datasets library serialization to parquet
- videos are stored in mp4 format to save space
- metadata are stored in plain json/jsonl files

Dataset can be uploaded/downloaded from the HuggingFace hub seamlessly. To work on a local dataset, you can specify its location with the `root` argument if it's not in the default `~/.cache/huggingface/lerobot` location.

### The `MultiLeRobotDataset` class

The `MultiLeRobotDataset` class is a wrapper around the `LeRobotDataset` class that allows you to train on multiple datasets at once. It is used to train pi0 policy on decentralized-collected so100 datasets. Currently only V2/2.1 dataset format is supported.

```python
from lerobot.common.datasets.lerobot_dataset import MultiLeRobotDataset
```
Currently it will use the statistics of one of the datasets to normalize the images. It will also standardize the image keys to `observation.images.{i}` format. Contrary to the conclusion of SmolVLA, we did not find camera order to be crucial for model performance. In opposite, we found that scrambled camera order allows the trained model to be agnostic to non-wrist camera poses.

## Train a pi0 policy

To train a policy to control your robot, use the [`python lerobot/scripts/accelerate_train.py`](./lerobot/scripts/accelerate_train.py) script. Here is an example command:
```bash
accelerate launch --num_processes=${n_gpus} lerobot/scripts/accelerate_train.py \
    --policy.path=sengi/pi0_hd20_uncompiled \
    --num_datasets=1500 \
    --batch_size=8 \
    --steps=5000 \
    --save_freq=2000 \
    --strict=true \
    --num_workers=4 \
    --log_freq=5 \
    --gradient_accumulation_steps=1 \
    --policy.scheduler_decay_lr=1e-5 \
    --policy.scheduler_decay_steps=1000000 \
    --policy.optimizer_lr=1e-4 \
    --dataset.use_annotated_tasks=false \
    --job_name=pi0_training\
    --wandb.enable=true \
    --wandb.project=lerobot \
    --wandb.entity=${wandb_entity}
```

Let's explain the command:
1. We provided 3 ways to specify the datasets to train on:
    - `--num_datasets=$N` to train on first N decentralized-collected so100 datasets with huggingface ids from the list in [`data_ids/dataset_list_valid.json`](./data_ids/dataset_list_valid.json).
    - `--dataset.repo_ids=custome_data_id.json` to train on the dataset ids specified in the `custome_data_id.json` file.
    - `--dataset.repo_id=${HF_USER}/so101_test` to train on the dataset with single huggingface repo id.
2. We provided 2 ways to specify the policy to train on:
    - `--policy.path=sengi/pi0_hd20_uncompiled` to train on the pretrained pi0 policy with checkpoint.
    - `--policy.type=pi0` to train on randomly initialized pi0 policy with default encoder. This loads configurations from [`configuration_pi0.py`](./lerobot/common/policies/pi0/configuration_pi0.py). Importantly, this policy will automatically adapt to the number of motor states, motor actions and cameras of your robot (e.g. `laptop` and `phone`) which have been saved in your dataset.
4. We provided `policy.device=cuda` since we are training on a Nvidia GPU, but you could use `policy.device=mps` to train on Apple silicon.
5. We provided `wandb.enable=true` to use [Weights and Biases](https://docs.wandb.ai/quickstart) for visualizing training plots. This is optional but if you use it, make sure you are logged in by running `wandb login`.
6. We provided `strict=true` to raise an error if the policy's state dictionary is not strict.
7. We provided `gradient_accumulation_steps=1` to accumulate gradients over 1 batch. This is useful to train on a larger batch size without running out of memory.
8. We provided `policy.scheduler_decay_lr=1e-5` to decay the learning rate to 1e-5 with cosine decay.
9. We provided `policy.scheduler_decay_steps=1000000` to decay the learning rate in 1000000 steps.
10. We provided `policy.optimizer_lr=1e-4` to set the max learning rate to 1e-4.
11. We provided `dataset.use_annotated_tasks=false` to use VLM-annotated task in the dataset.
12. We provided `job_name=pi0_training` to set the job name to `pi0_training`.
13. We provided `wandb.project=lerobot` to set the project to `lerobot`.
14. We provided `wandb.entity=${wandb_entity}` to set the wandb entity.

To resume training from a checkpoint, below is an example command to resume from `last` checkpoint of the `pi0_so101_test` policy:
```bash
accelerate launch --num_processes=${n_gpus} lerobot/scripts/accelerate_train.py \
  --config_path=outputs/train/pi0_so101_test/checkpoints/last/pretrained_model/train_config.json \
  --resume=true
```

#### Upload policy checkpoints

Once training is done, upload the latest checkpoint with:
```bash
huggingface-cli upload ${HF_USER}/pi0_so101_test \
  outputs/train/pi0_so101_test/checkpoints/last/pretrained_model
```

You can also upload intermediate checkpoints with:
```bash
CKPT=010000
huggingface-cli upload ${HF_USER}/pi0_so101_test${CKPT} \
  outputs/train/pi0_so101_test/checkpoints/${CKPT}/pretrained_model
```

## Evaluate pi0 policy

You can use the `record` function from [`lerobot/scripts/control_robot.py`](../lerobot/scripts/control_robot.py) but with a policy checkpoint as input. For instance, run this command to record 10 evaluation episodes:
```bash
python lerobot/scripts/control_robot.py \
  --robot.type=so101 \
  --control.type=record \
  --control.fps=30 \
  --control.single_task="Pick up the marker." \
  --control.repo_id=${HF_USER}/eval_pi0_so101_test \
  --control.tags='["tutorial"]' \
  --control.warmup_time_s=5 \
  --control.episode_time_s=30 \
  --control.reset_time_s=30 \
  --control.num_episodes=10 \
  --control.push_to_hub=true \
  --control.policy.path=sengi/DePi0
```

As you can see, it's almost the same command as previously used to record your training dataset. Two things changed:
1. There is an additional `--control.policy.path` argument which indicates the path to your policy checkpoint with  (e.g. `outputs/train/eval_pi0_so101_test/checkpoints/last/pretrained_model`). You can also use the model repository if you uploaded a model checkpoint to the hub (e.g. `${HF_USER}/pi0_so101_test`).
2. The name of dataset begins by `eval` to reflect that you are running inference (e.g. `${HF_USER}/eval_pi0_so101_test`).

<!-- ### Add a new dataset

To add a dataset to the hub, you need to login using a write-access token, which can be generated from the [Hugging Face settings](https://huggingface.co/settings/tokens):
```bash
huggingface-cli login --token ${HUGGINGFACE_TOKEN} --add-to-git-credential
```

Then point to your raw dataset folder (e.g. `data/aloha_static_pingpong_test_raw`), and push your dataset to the hub with:
```bash
python lerobot/scripts/push_dataset_to_hub.py \
--raw-dir data/aloha_static_pingpong_test_raw \
--out-dir data \
--repo-id lerobot/aloha_static_pingpong_test \
--raw-format aloha_hdf5
```

See `python lerobot/scripts/push_dataset_to_hub.py --help` for more instructions.

If your dataset format is not supported, implement your own in `lerobot/common/datasets/push_dataset_to_hub/${raw_format}_format.py` by copying examples like [pusht_zarr](https://github.com/huggingface/lerobot/blob/main/lerobot/common/datasets/push_dataset_to_hub/pusht_zarr_format.py), [umi_zarr](https://github.com/huggingface/lerobot/blob/main/lerobot/common/datasets/push_dataset_to_hub/umi_zarr_format.py), [aloha_hdf5](https://github.com/huggingface/lerobot/blob/main/lerobot/common/datasets/push_dataset_to_hub/aloha_hdf5_format.py), or [xarm_pkl](https://github.com/huggingface/lerobot/blob/main/lerobot/common/datasets/push_dataset_to_hub/xarm_pkl_format.py). -->


### Add a pretrained policy

Once you have trained a policy you may upload it to the Hugging Face hub using a hub id that looks like `${hf_user}/${repo_name}` (e.g. [lerobot/diffusion_pusht](https://huggingface.co/lerobot/diffusion_pusht)).

You first need to find the checkpoint folder located inside your experiment directory (e.g. `outputs/train/2024-05-05/20-21-12_aloha_act_default/checkpoints/002500`). Within that there is a `pretrained_model` directory which should contain:
- `config.json`: A serialized version of the policy configuration (following the policy's dataclass config).
- `model.safetensors`: A set of `torch.nn.Module` parameters, saved in [Hugging Face Safetensors](https://huggingface.co/docs/safetensors/index) format.
- `train_config.json`: A consolidated configuration containing all parameters used for training. The policy configuration should match `config.json` exactly. This is useful for anyone who wants to evaluate your policy or for reproducibility.

To upload these to the hub, run the following:
```bash
huggingface-cli upload ${hf_user}/${repo_name} path/to/pretrained_model
```

See [eval.py](https://github.com/huggingface/lerobot/blob/main/lerobot/scripts/eval.py) for an example of how other people may use your policy.


### Improve your code with profiling

An example of a code snippet to profile the evaluation of a policy:
```python
from torch.profiler import profile, record_function, ProfilerActivity

def trace_handler(prof):
    prof.export_chrome_trace(f"tmp/trace_schedule_{prof.step_num}.json")

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(
        wait=2,
        warmup=2,
        active=3,
    ),
    on_trace_ready=trace_handler
) as prof:
    with record_function("eval_policy"):
        for i in range(num_episodes):
            prof.step()
            # insert code to profile, potentially whole body of eval_policy function
```

## Citation

If you want, you can cite this work with:
```bibtex
@misc{chen2025depi,
    author = {Chen, Shirui},
    title = {DePi: Pi0 training with large decentralized-collected so100 dataset},
    howpublished = "\url{https://github.com/chinsengi/depi}",
    year = {2025}
}
```
