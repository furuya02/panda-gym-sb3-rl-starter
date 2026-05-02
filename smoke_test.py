import gymnasium as gym
import imageio
import panda_gym  # noqa: F401  registers Panda* envs

env = gym.make("PandaReach-v3", render_mode="rgb_array")

print("env:", env.spec.id)
print("observation_space:", env.observation_space)
print("action_space:", env.action_space)

observation, info = env.reset(seed=0)
print("observation keys:", list(observation.keys()))
print("achieved_goal:", observation["achieved_goal"])
print("desired_goal:", observation["desired_goal"])

frames = []
for step in range(50):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    frame = env.render()
    frames.append(frame)
    if terminated or truncated:
        observation, info = env.reset()

env.close()

print("frames captured:", len(frames), "shape:", frames[0].shape)

imageio.imwrite("smoke_first.png", frames[0])
imageio.imwrite("smoke_last.png", frames[-1])
print("saved: smoke_first.png, smoke_last.png")
