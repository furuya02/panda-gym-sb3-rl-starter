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

    fig, axes = plt.subplots(
        len(METRICS), 1, figsize=(8, 4 * len(METRICS)), sharex=True
    )
    for ax, (tag, ylabel) in zip(axes, METRICS):
        for label, ea in runs:
            if tag not in ea.Tags()["scalars"]:
                continue
            events = ea.Scalars(tag)
            xs = [e.step for e in events]
            ys = [e.value for e in events]
            ax.plot(xs, ys, linewidth=1.4, label=label)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.set_title(ylabel)
        ax.legend()
    axes[-1].set_xlabel("training step")
    fig.suptitle(args.name, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = Path(args.out) / f"{args.name}.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
