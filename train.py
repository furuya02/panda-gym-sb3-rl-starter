import argparse
from pathlib import Path

import gymnasium as gym
import panda_gym  # noqa: F401  registers Panda* envs
from stable_baselines3 import SAC
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="PandaReach-v3")
    parser.add_argument("--timesteps", type=int, default=25000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="logs/sac_her_pandareach")
    parser.add_argument("--tb", default="logs/tb")
    parser.add_argument("--no-her", action="store_true", help="Disable HER (ablation)")
    args = parser.parse_args()

    Path(args.tb).mkdir(parents=True, exist_ok=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    env = gym.make(args.env)

    sac_kwargs = dict(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=1e-3,
        buffer_size=1_000_000,
        batch_size=256,
        gamma=0.95,
        tau=0.05,
        learning_starts=1000,
        policy_kwargs=dict(net_arch=[64, 64], n_critics=1),
        verbose=1,
        seed=args.seed,
        tensorboard_log=args.tb,
    )
    if not args.no_her:
        sac_kwargs["replay_buffer_class"] = HerReplayBuffer
        sac_kwargs["replay_buffer_kwargs"] = dict(
            n_sampled_goal=4,
            goal_selection_strategy="future",
        )

    model = SAC(**sac_kwargs)
    model.learn(total_timesteps=args.timesteps)
    model.save(args.out)
    env.close()
    print(f"saved: {args.out}.zip")


if __name__ == "__main__":
    main()
