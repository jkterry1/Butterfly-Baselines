import argparse

import supersuit as ss
from pettingzoo.butterfly import pistonball_v4
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor

parser = argparse.ArgumentParser()

env_name = "pistonball_v4"

env = pistonball_v4.parallel_env()

env = ss.color_reduction_v0(env, mode="B")
env = ss.pad_action_space_v0(env)
env = ss.pad_observations_v0(env)
env = ss.resize_v0(env, x_size=96, y_size=96)
env = ss.frame_stack_v1(env, 4)
env = ss.pettingzoo_env_to_vec_env_v0(env)
env = ss.concat_vec_envs_v0(env, 8, num_cpus=4, base_class="stable_baselines3")
env = VecMonitor(env)
kwargs = {
    "batch_size": 256,
    "n_steps": 256,
    "gamma": 0.98,
    "learning_rate": 0.000597417,
    "n_epochs": 10,
    "clip_range": 0.1,
    "gae_lambda": 0.8,
    "max_grad_norm": 0.9,
    "vf_coef": 0.967044,
    "ent_coef": 0.0993558,
}
model = PPO("CnnPolicy", env, verbose=3, **kwargs)
model.learn(total_timesteps=2000000)

# Rendering

env = pistonball_v4.env()
env = ss.color_reduction_v0(env, mode="B")
env = ss.resize_v0(env, x_size=96, y_size=96)
env = ss.frame_stack_v1(env, 4)

model = PPO.load("policy")

env.reset()
for agent in env.agent_iter():
    obs, reward, done, info = env.last()
    act = model.predict(obs, deterministic=True)[0] if not done else None
    env.step(act)
    env.render()
