import argparse
import json
import os
import sys
import logging

import supersuit as ss
from pettingzoo.butterfly import (
    cooperative_pong_v3,
    pistonball_v4,
    knights_archers_zombies_v7,
    prospector_v4,
)
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecMonitor
from torch import nn as nn

from utils import (
    image_transpose,
    AgentIndicatorWrapper,
    BinaryIndicator,
    GeometricPatternIndicator,
    InvertColorIndicator,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--env-name",
    help="Butterfly Environment to use from PettingZoo",
    type=str,
    default="pistonball_v4",
    choices=[
        "pistonball_v4",
        "cooperative_pong_v3",
        "knights_archers_zombies_v7",
        "prospector_v4",
    ],
)
parser.add_argument("--n-runs", type=int, default=5)
parser.add_argument("--n-evaluations", type=int, default=100)
parser.add_argument("--timesteps", type=int, default=0)
parser.add_argument("--num-cpus", type=int, default=8)
parser.add_argument("--num-vec-envs", type=int, default=4)
args = parser.parse_args()

param_file = "./config/" + str(args.env_name) + ".json"
with open(param_file) as f:
    params = json.load(f)

print("Hyperparameters:")
print(params)
muesli_obs_size = 96
muesli_frame_size = 4
evaluations = args.n_evaluations
timesteps = args.timesteps

net_arch = {
    "small": [dict(pi=[64, 64], vf=[64, 64])],
    "medium": [dict(pi=[256, 256], vf=[256, 256])],
}[params["net_arch"]]

activation_fn = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "elu": nn.ELU,
    "leaky_relu": nn.LeakyReLU,
}[params["activation_fn"]]

policy_kwargs = dict(
    net_arch=net_arch,
    activation_fn=activation_fn,
    ortho_init=False,
)
agent_indicator_name = params["agent_indicator"]

del params["net_arch"]
del params["activation_fn"]
del params["agent_indicator"]
params["policy_kwargs"] = policy_kwargs
params["policy"] = "CnnPolicy"

# Generate env
if args.env_name == "prospector_v4":
    env = prospector_v4.parallel_env()
    agent_type = "prospector"
elif args.env_name == "knights_archers_zombies_v7":
    env = knights_archers_zombies_v7.parallel_env()
    agent_type = "archer"
elif args.env_name == "cooperative_pong_v3":
    env = cooperative_pong_v3.parallel_env()
    agent_type = "paddle_0"
elif args.env_name == "pistonball_v4":
    env = pistonball_v4.parallel_env()

env.reset()
num_agents = env.num_agents
env = ss.color_reduction_v0(env)
env = ss.pad_action_space_v0(env)
env = ss.pad_observations_v0(env)
env = ss.resize_v0(
    env, x_size=muesli_obs_size, y_size=muesli_obs_size, linear_interp=True
)
env = ss.frame_stack_v1(env, stack_size=muesli_frame_size)

# Enable black death
if args.env_name == "knights-archers-zombies-v7":
    env = ss.black_death_v2(env)

# Agent indicator wrapper
if agent_indicator_name == "invert":
    agent_indicator = InvertColorIndicator(env, agent_type)
    agent_indicator_wrapper = AgentIndicatorWrapper(agent_indicator)
elif agent_indicator_name == "invert-replace":
    agent_indicator = InvertColorIndicator(env, agent_type)
    agent_indicator_wrapper = AgentIndicatorWrapper(agent_indicator, False)
elif agent_indicator_name == "binary":
    agent_indicator = BinaryIndicator(env, agent_type)
    agent_indicator_wrapper = AgentIndicatorWrapper(agent_indicator)
elif agent_indicator_name == "geometric":
    agent_indicator = GeometricPatternIndicator(env, agent_type)
    agent_indicator_wrapper = AgentIndicatorWrapper(agent_indicator)
if agent_indicator_name != "identity":
    env = ss.observation_lambda_v0(
        env, agent_indicator_wrapper.apply, agent_indicator_wrapper.apply_space
    )

env = ss.pettingzoo_env_to_vec_env_v0(env)
env = ss.concat_vec_envs_v0(
    env,
    num_vec_envs=args.num_vec_envs,
    num_cpus=args.num_cpus,
    base_class="stable_baselines3",
)
env = VecMonitor(env)
env = image_transpose(env)

eval_freq = timesteps // evaluations

all_mean_rewards = []
log_dir = "./data/" + args.env_name + "/"
os.makedirs(log_dir, exist_ok=True)

for i in range(args.n_runs):
    model = PPO(
        env=env,
        tensorboard_log=None,
        # We do not seed the trial
        seed=None,
        verbose=3,
        **params,
    )

    run_log_dir = log_dir + "run_" + str(i)

    n_eval_episodes = 5 * num_agents

    eval_callback = EvalCallback(
        env,
        n_eval_episodes=n_eval_episodes,
        log_path=run_log_dir,
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
    )
    model.learn(total_timesteps=timesteps, callback=eval_callback)
