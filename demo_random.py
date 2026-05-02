import argparse
from pathlib import Path

import gymnasium as gym
import imageio
import numpy as np
import panda_gym  # noqa: F401  registers Panda* envs

from eval import (
    annotate,
    annotate_lines,
    draw_border,
    draw_goal_marker,
    get_camera_matrices,
    project_3d_to_2d,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="PandaReach-v3")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--out", default="videos/random.mp4")
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--intro-frames", type=int, default=10)
    parser.add_argument("--freeze-frames", type=int, default=15)
    args = parser.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    env = gym.make(args.env, render_mode="rgb_array")
    env.reset(seed=0)
    view, proj, cam_w, cam_h = get_camera_matrices(env)

    successes = 0
    frames: list[np.ndarray] = []

    for episode_index in range(args.episodes):
        observation, info = env.reset(seed=2000 + episode_index)
        episode_success = False
        ep_label = f"Episode {episode_index + 1}/{args.episodes}"
        goal_3d = observation["desired_goal"]

        raw_intro = env.render()
        raw_motion: list[np.ndarray] = []
        last_step_index = 0

        for step in range(env.spec.max_episode_steps):
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            raw_motion.append(env.render())
            if info.get("is_success"):
                episode_success = True
            last_step_index = step
            if terminated or truncated:
                break

        success_step = last_step_index + 1
        end_status = "REACHED" if episode_success else "MISSED"
        end_text_color = (140, 255, 140) if episode_success else (255, 140, 140)
        end_border_color = (40, 200, 40) if episode_success else (220, 60, 60)
        intro_border_color = (255, 200, 60)

        goal_xy = project_3d_to_2d(goal_3d, view, proj, cam_w, cam_h)

        intro_frame = draw_border(draw_goal_marker(raw_intro, goal_xy, "GOAL"), intro_border_color)
        for _ in range(args.intro_frames):
            frames.append(annotate(intro_frame, f"{ep_label} - START", color=(255, 220, 100)))

        for step_index, raw in enumerate(raw_motion):
            framed = draw_goal_marker(raw, goal_xy)
            frames.append(annotate(framed, f"{ep_label} - step {step_index + 1}"))

        end_frame = draw_border(draw_goal_marker(raw_motion[-1], goal_xy), end_border_color)
        end_lines = [
            (f"{ep_label} - step {success_step}", (255, 255, 255)),
            (f"step {success_step} {end_status}", end_text_color),
        ]
        for _ in range(args.freeze_frames):
            frames.append(annotate_lines(end_frame, end_lines))

        if episode_success:
            successes += 1

    env.close()

    imageio.mimsave(args.out, frames, fps=args.fps)
    print(f"random policy: {successes}/{args.episodes} success, frames={len(frames)}, video={args.out}")


if __name__ == "__main__":
    main()
