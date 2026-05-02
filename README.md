# panda-gym × Stable-Baselines3 SAC + HER on Jetson AGX Orin

Train a 7-DoF Franka Panda arm to reach goal positions with sparse reward, using
[panda-gym](https://github.com/qgallouedec/panda-gym) (PyBullet-based simulator)
and [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) SAC with
Hindsight Experience Replay.

The whole pipeline runs locally on a Jetson AGX Orin in a Docker container.
No cloud, no AWS cost.

## Result (PandaReach-v3, 25 000 steps)

| Metric | Value |
|---|---|
| Hardware | Jetson AGX Orin (Orin GPU, CUDA 12.6, JetPack 6.2.1) |
| Training time | 15 min 12 s |
| Final rolling success rate | 100 % |
| Deterministic eval (50 episodes) | 100 % (50/50) |
| Mean reward (eval) | -1.80 |
| Mean steps to goal | 2.81 |

`videos/eval.mp4` shows the trained policy reaching the goal in 1-3 steps.
`videos/random.mp4` shows a random policy flailing around for comparison.

## Demo (YouTube)

[![panda-gym × Stable-Baselines3 SAC + HER demo](https://img.youtube.com/vi/E1dQemO-mn0/maxresdefault.jpg)](https://www.youtube.com/watch?v=E1dQemO-mn0)

Side-by-side: random policy (left) vs SAC + HER trained policy (right) on PandaReach-v3.

## Layout

```
panda-gym-sb3-rl-starter/
├── Dockerfile             # dustynv/pytorch:2.7-r36.4.0 base + panda-gym + SB3
├── requirements.txt
├── smoke_test.py           # PandaReach-v3 smoke test → PNG
├── train.py               # SAC + HER training (25 000 steps default)
├── eval.py                # Deterministic eval + MP4
├── demo_random.py         # Random-policy MP4 for "before training" comparison
├── plot_curves.py         # TensorBoard logs → combined learning-curve PNG (one per run)
├── plot_compare.py        # TensorBoard logs → 2-panel comparison PNG (success_rate + ep_rew_mean)
├── README.md
└── README.ja.md
```

## Setup

### Prerequisites

- Jetson AGX Orin with JetPack 6.2.1 (L4T R36.4) and Docker + nvidia runtime
- Or any aarch64 Linux machine with Docker (CUDA optional)

### Clone

```bash
git clone https://github.com/<your-account>/panda-gym-sb3-rl-starter.git
cd panda-gym-sb3-rl-starter
```

### Build the Docker image

```bash
docker build -t panda-gym-sb3-rl-starter:latest .
```

The first build pulls `dustynv/pytorch:2.7-r36.4.0` (~10 GB) and source-builds
PyBullet inside the image (no aarch64 wheel exists on PyPI). Total time on a
Jetson AGX Orin: about 10-15 minutes.

### Run the container

Training takes ~15 min, so start the container as a daemon (`sleep infinity`)
to survive SSH disconnects, then `exec` into it:

```bash
docker run -d --name panda_gym_sb3 \
  --runtime nvidia \
  --network host \
  -v "$(pwd):/workspace/panda-gym-sb3-rl-starter" \
  -w /workspace/panda-gym-sb3-rl-starter \
  panda-gym-sb3-rl-starter:latest \
  sleep infinity

docker exec -it panda_gym_sb3 bash
```

Stop and remove when done:

```bash
docker stop panda_gym_sb3 && docker rm panda_gym_sb3
```

## Workflow

### 1. Smoke test

```bash
python3 smoke_test.py
```

Loads `PandaReach-v3`, runs 50 random steps, saves `smoke_first.png` and
`smoke_last.png`. Both should show the Franka Panda arm and a green goal sphere.

### 2. Train

```bash
python3 train.py
```

Defaults: `PandaReach-v3`, 25 000 timesteps, SAC + `HerReplayBuffer`,
`net_arch=[64, 64]`, `n_critics=1`. Model saved to `logs/sac_her_pandareach.zip`.
TensorBoard log under `logs/tb/`.

### 3. Evaluate

```bash
python3 eval.py --episodes 50 --video-episodes 20
```

Deterministic rollout over 50 fresh seeds, prints success rate and mean reward,
saves a video to `videos/eval.mp4`.

### 4. Random-policy comparison

```bash
python3 demo_random.py --episodes 10
```

Saves `videos/random.mp4` for "before training" baseline.

### 5. Learning curves

```bash
python3 plot_curves.py
```

For each TensorBoard run under `logs/tb/`, saves a single PNG to `plots/`
with 5 vertically stacked subplots (success rate / episode reward / critic loss /
actor loss / entropy coefficient).

### 6. Ablation: train without HER

```bash
python3 train.py --no-her --out logs/sac_pandareach_noher
```

Same SAC + same network, with HER replaced by SB3's default `DictReplayBuffer`.
Useful for showing how much HER actually contributes on goal-conditioned sparse-reward tasks.

### 7. HER on/off comparison plot

```bash
python3 plot_compare.py --runs SAC_1:HER SAC_2:no-HER --name reach_her_vs_noher
```

Saves a single PNG to `plots/` with two vertically stacked subplots (success_rate
on top, ep_rew_mean on bottom), each overlaying the curves from the two TensorBoard
runs. Requires both `SAC_1` (HER on, from step 2) and `SAC_2` (HER off, from step 6)
to exist under `logs/tb/`. Used to produce the comparison chart shown in the blog post.

## Versions

| Component | Version |
|---|---|
| Base image | `dustynv/pytorch:2.7-r36.4.0` |
| Python | 3.10 |
| PyTorch | 2.7 (CUDA 12.6) |
| panda-gym | 3.0.7 |
| stable-baselines3 | 2.7.0 |
| gymnasium | 0.29.1 |
| pybullet | source-built from sdist |

## License

MIT.
