import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


METRICS = [
    ("rollout/success_rate", "Success Rate"),
    ("rollout/ep_rew_mean", "Episode Reward Mean"),
    ("train/critic_loss", "Critic Loss"),
    ("train/actor_loss", "Actor Loss"),
    ("train/ent_coef", "Entropy Coefficient"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", default="logs/tb")
    parser.add_argument("--out", default="plots")
    args = parser.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)

    runs = sorted(p for p in Path(args.logdir).glob("*") if p.is_dir())

    for run_dir in runs:
        ea = EventAccumulator(str(run_dir))
        ea.Reload()
        scalar_tags = ea.Tags()["scalars"]
        for tag, label in METRICS:
            if tag not in scalar_tags:
                continue
            events = ea.Scalars(tag)
            xs = [e.step for e in events]
            ys = [e.value for e in events]
            plt.figure(figsize=(8, 4))
            plt.plot(xs, ys, linewidth=1.2)
            plt.xlabel("training step")
            plt.ylabel(label)
            plt.title(f"{run_dir.name}: {label}")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            safe = tag.replace("/", "_")
            out = Path(args.out) / f"{run_dir.name}_{safe}.png"
            plt.savefig(out, dpi=120)
            plt.close()
            print(f"saved: {out}")


if __name__ == "__main__":
    main()
