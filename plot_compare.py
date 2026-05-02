import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


METRICS = [
    ("rollout/success_rate", "Success Rate"),
    ("rollout/ep_rew_mean", "Episode Reward Mean"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="Pairs of <tb_subdir>:<label>, e.g. SAC_1:HER SAC_2:no-HER",
    )
    parser.add_argument("--logdir", default="logs/tb")
    parser.add_argument("--out", default="plots")
    parser.add_argument("--name", default="compare", help="Prefix for output files")
    args = parser.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)

    runs = []
    for spec in args.runs:
        sub, label = spec.split(":", 1)
        path = Path(args.logdir) / sub
        ea = EventAccumulator(str(path))
        ea.Reload()
        runs.append((label, ea))

    for tag, ylabel in METRICS:
        plt.figure(figsize=(8, 4))
        for label, ea in runs:
            if tag not in ea.Tags()["scalars"]:
                continue
            events = ea.Scalars(tag)
            xs = [e.step for e in events]
            ys = [e.value for e in events]
            plt.plot(xs, ys, linewidth=1.4, label=label)
        plt.xlabel("training step")
        plt.ylabel(ylabel)
        plt.title(ylabel)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        safe = tag.replace("/", "_")
        out = Path(args.out) / f"{args.name}_{safe}.png"
        plt.savefig(out, dpi=120)
        plt.close()
        print(f"saved: {out}")


if __name__ == "__main__":
    main()
