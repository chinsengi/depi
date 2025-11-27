# Meta-World

Meta-World is a well-designed, open-source simulation benchmark for multi-task and meta reinforcement learning in continuous-control robotic manipulation. It gives researchers a shared playground to test whether algorithms can **learn many different tasks** and **generalize quickly to new ones**.

- 📄 [Meta-World paper](https://arxiv.org/pdf/1910.10897)
- 💻 [Original Meta-World repository](https://github.com/Farama-Foundation/Metaworld)

![Meta-World MT10 demo](https://meta-world.github.io/figures/ml45.gif)

## Why Meta-World matters

- **Diverse, realistic tasks.** The MT50 suite bundles 50 tabletop manipulation tasks using everyday objects with a common Sawyer arm setup.
- **Focus on generalization.** MT10/MT50 use a one-hot task vector so policies learn transferable skills instead of overfitting to a single scene.
- **Standardized protocol.** Difficulty splits (easy/medium/hard/very_hard) make it straightforward to compare methods fairly.

## What it enables in LeRobot

LeRobot ships a Meta-World integration so you can evaluate policies or VLAs with minimal setup:

- A LeRobot-ready dataset on the HF Hub: [`lerobot/metaworld_mt50`](https://huggingface.co/datasets/lerobot/metaworld_mt50), formatted for the MT50 split with fixed object/goal positions and a one-hot task indicator.
- A `metaworld` environment type that exposes task descriptions alongside pixels and proprioception.
- Multi-task evaluation in `lerobot-eval`, returning per-task and per-difficulty success rates and optional episode videos.

## Evaluate a trained policy

Install the Meta-World extra once:

```bash
pip install "lerobot[metaworld]"
```

Then run evaluation on the medium split (11 tasks):

```bash
lerobot-eval \
  --policy.path="your-policy-id" \
  --env.type=metaworld \
  --env.task=medium \
  --eval.batch_size=1 \
  --eval.n_episodes=2
```

Tips:

- Set `--env.task` to a comma-separated list (e.g., `assembly-v3,push-v3`) or a difficulty group (`easy`, `medium`, `hard`, `very_hard`).
- Increase `--eval.n_episodes` to average over more rollouts per task; results are aggregated per task and per difficulty group.
- Use `--eval.render=true` to save up to ten episode videos in `outputs/eval/.../videos`.
- If Gymnasium raises `AssertionError: ['human', 'rgb_array', 'depth_array']`, install `gymnasium==1.1.0` to match Meta-World.

## Practical tips

- For generalization studies, target the full MT50 suite (`--env.task=mt50`) rather than a handful of tasks.
- Keep the one-hot task conditioning from MT10/MT50 conventions so policies have explicit task context.
- Inspect the dataset task descriptions and the `info["is_success"]` signals when post-processing metrics to align with the benchmark.
