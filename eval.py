import argparse
from pathlib import Path

import gymnasium as gym
import imageio
import numpy as np
import panda_gym  # noqa: F401  registers Panda* envs
from PIL import Image, ImageDraw, ImageFont
from stable_baselines3 import SAC


def _load_font(size: int) -> ImageFont.ImageFont:
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)


def annotate(frame: np.ndarray, text: str, color: tuple = (255, 255, 255)) -> np.ndarray:
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    font = _load_font(28)
    draw.text((20, 20), text, fill=color, font=font, stroke_width=2, stroke_fill=(0, 0, 0))
    return np.array(img)


def annotate_lines(frame: np.ndarray, lines: list[tuple[str, tuple]]) -> np.ndarray:
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    font = _load_font(28)
    y = 20
    for text, color in lines:
        draw.text((20, y), text, fill=color, font=font, stroke_width=2, stroke_fill=(0, 0, 0))
        y += 40
    return np.array(img)


def draw_border(frame: np.ndarray, color: tuple, thickness: int = 12) -> np.ndarray:
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    h, w = frame.shape[:2]
    for i in range(thickness):
        draw.rectangle([i, i, w - 1 - i, h - 1 - i], outline=color, width=1)
    return np.array(img)


def get_camera_matrices(env: gym.Env) -> tuple[np.ndarray, np.ndarray, int, int]:
    ru = env.unwrapped
    pc = ru.sim.physics_client
    width = ru.render_width
    height = ru.render_height
    view = pc.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=tuple(ru.render_target_position),
        distance=ru.render_distance,
        yaw=ru.render_yaw,
        pitch=ru.render_pitch,
        roll=ru.render_roll,
        upAxisIndex=2,
    )
    proj = pc.computeProjectionMatrixFOV(
        fov=60, aspect=float(width) / height, nearVal=0.1, farVal=100.0,
    )
    return (
        np.array(view).reshape(4, 4, order="F"),
        np.array(proj).reshape(4, 4, order="F"),
        width,
        height,
    )


def project_3d_to_2d(point_3d: np.ndarray, view: np.ndarray, proj: np.ndarray, width: int, height: int) -> tuple[int, int]:
    p = np.array([point_3d[0], point_3d[1], point_3d[2], 1.0])
    clip = proj @ view @ p
    ndc = clip[:3] / clip[3]
    sx = (ndc[0] + 1) * 0.5 * width
    sy = (1 - ndc[1]) * 0.5 * height
    return int(sx), int(sy)


def draw_goal_marker(frame: np.ndarray, screen_xy: tuple[int, int], label_text: str | None = None) -> np.ndarray:
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    x, y = screen_xy
    r = 28
    draw.ellipse([x - r, y - r, x + r, y + r], outline=(255, 50, 50), width=3)
    draw.ellipse([x - r - 6, y - r - 6, x + r + 6, y + r + 6], outline=(255, 200, 50), width=2)
    if label_text:
        font = _load_font(20)
        draw.text((x + r + 8, y - r), label_text, fill=(255, 220, 50), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
    return np.array(img)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="PandaReach-v3")
    parser.add_argument("--model", default="logs/sac_her_pandareach")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--video", default="videos/eval.mp4")
    parser.add_argument("--video-episodes", type=int, default=30)
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--intro-frames", type=int, default=10)
    parser.add_argument("--freeze-frames", type=int, default=15)
    args = parser.parse_args()

    Path(args.video).parent.mkdir(parents=True, exist_ok=True)

    env = gym.make(args.env, render_mode="rgb_array")
    model = SAC.load(args.model, env=env)

    env.reset(seed=0)
    view, proj, cam_w, cam_h = get_camera_matrices(env)

    successes = 0
    rewards = []
    frames: list[np.ndarray] = []

    for episode_index in range(args.episodes):
        observation, info = env.reset(seed=1000 + episode_index)
        episode_reward = 0.0
        episode_success = False
        record_video = episode_index < args.video_episodes
        ep_label = f"Episode {episode_index + 1}/{args.video_episodes}"
        goal_3d = observation["desired_goal"]

        raw_intro = env.render() if record_video else None
        raw_motion: list[np.ndarray] = []
        last_step_index = 0

        for step in range(env.spec.max_episode_steps):
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += float(reward)
            if record_video:
                raw_motion.append(env.render())
            if info.get("is_success"):
                episode_success = True
            last_step_index = step
            if terminated or truncated:
                break

        if record_video:
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

        rewards.append(episode_reward)
        if episode_success:
            successes += 1

    env.close()

    success_rate = successes / args.episodes
    mean_reward = float(np.mean(rewards))

    print(f"episodes:     {args.episodes}")
    print(f"success_rate: {success_rate:.2%}  ({successes}/{args.episodes})")
    print(f"mean_reward:  {mean_reward:.2f}")

    imageio.mimsave(args.video, frames, fps=args.fps)
    print(f"video saved: {args.video} ({len(frames)} frames @ {args.fps} fps)")


if __name__ == "__main__":
    main()
