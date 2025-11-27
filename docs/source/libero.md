# LIBERO

**LIBERO** is a benchmark for **lifelong robot learning**, emphasizing how well policies transfer knowledge to new tasks over time.

- 📄 [LIBERO paper](https://arxiv.org/abs/2306.03310)
- 💻 [Original LIBERO repository](https://github.com/Lifelong-Robot-Learning/LIBERO)

LIBERO bundles **five task suites** (130 tasks total):

- **LIBERO-Spatial (`libero_spatial`)** – spatial reasoning tasks.
- **LIBERO-Object (`libero_object`)** – object-centric manipulation tasks.
- **LIBERO-Goal (`libero_goal`)** – goal-conditioned tasks that shift targets.
- **LIBERO-90 (`libero_90`)** – short-horizon tasks from LIBERO-100.
- **LIBERO-Long (`libero_10`)** – the long-horizon subset of LIBERO-100.

![An overview of the LIBERO benchmark](https://libero-project.github.io/assets/img/libero/fig1.png)

## Evaluate a trained policy

Install the LIBERO extra after setting up LeRobot:

```bash
pip install "lerobot[libero]"
```

Run single-suite evaluation:

```bash
lerobot-eval \
  --policy.path="outputs/train/2025-11-14/19-44-40_DePi0_meta_world" \
  --env.type=libero \
  --env.task=libero_object \
  --eval.batch_size=2 \
  --eval.n_episodes=3
```

Evaluate across multiple suites at once:

```bash
lerobot-eval \
  --policy.path="your-policy-id" \
  --env.type=libero \
  --env.task=libero_object,libero_spatial \
  --eval.batch_size=1 \
  --eval.n_episodes=2
```

Notes:

- Pass any comma-separated suite list to `--env.task` (e.g., `libero_spatial,libero_goal,libero_10`).
- `--eval.batch_size` controls parallel environments; `--eval.n_episodes` sets total rollouts.
- Set `MUJOCO_GL=egl` on headless servers before running evaluations.

## Policy inputs and actions

LIBERO observations follow the LeRobot multi-modal convention:

- `observation.state` – proprioceptive features.
- `observation.images.image` – main camera view (`agentview_image`).
- `observation.images.image2` – wrist camera view (`robot0_eye_in_hand_image`).

Actions use a continuous `Box(-1, 1, shape=(7,))` control space.

## Training reference

LeRobot provides a preprocessed LIBERO dataset that matches the expected keys:

- Ready-to-use dataset: [HuggingFaceVLA/libero](https://huggingface.co/datasets/HuggingFaceVLA/libero)
- Original release: [physical-intelligence/libero](https://huggingface.co/datasets/physical-intelligence/libero)

Example SmolVLA training command on a single suite:

```bash
lerobot-train \
  --policy.type=smolvla \
  --policy.repo_id=${HF_USER}/libero-test \
  --policy.load_vlm_weights=true \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --env.type=libero \
  --env.task=libero_10 \
  --output_dir=./outputs/ \
  --steps=100000 \
  --batch_size=4 \
  --eval.batch_size=1 \
  --eval.n_episodes=1 \
  --eval_freq=1000
```

Evaluation-specific flags (e.g., `--env.max_parallel_tasks=1` or `--policy.n_action_steps=10`) can be added to match benchmark baselines such as π₀.₅.
