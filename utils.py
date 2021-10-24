import math
from typing import Dict, List

import numpy as np
from gym.spaces import Box
from stable_baselines3.common.preprocessing import (
    is_image_space,
    is_image_space_channels_first,
)
from stable_baselines3.common.vec_env import VecTransposeImage


def image_transpose(env):
    if is_image_space(env.observation_space) and not is_image_space_channels_first(
        env.observation_space
    ):
        env = VecTransposeImage(env)
    return env


def convert_three_dim(obs, channels_first=True):
    if len(obs.shape) == 2:
        return obs[None, :, :] if channels_first else obs[:, :, None]
    else:
        return obs


# Base class of agent indicators
# @ type: This class basically handles cases where there are only two types of agents in the env.
#   We can classify them by investigating if this [type] string is substring of an agent's name.
class AgentIndicator:
    def __init__(self, env, type):
        self.env = env
        self.type = type

    def apply(self, obs, obs_space, agent):
        assert (
            len(obs.shape) == 3
        ), "Agent indicator can only handle three-dimensional observations"

    def apply(self, obs_space, agent, use_original_obs):
        pass


# Invert an agent's observation by subtracting it from the maximum observable value.
class InvertColorIndicator(AgentIndicator):
    def __init__(self, env, type):
        super().__init__(env, type)

    def apply(self, obs, obs_space, agent):
        super().apply(obs, obs_space, agent)
        high = convert_three_dim(obs_space.high.copy())
        return high - obs if self.type in agent else obs

    def apply_space(self, obs_space, agent, use_original_obs):
        new_low = convert_three_dim(obs_space.low.copy(), True)
        new_high = convert_three_dim(obs_space.high.copy(), True)
        low = new_low.min()
        high = new_high.max()
        channel_low = convert_three_dim(
            np.full(obs_space.low.shape, low), is_image_space_channels_first(new_low)
        )
        channel_high = convert_three_dim(
            np.full(obs_space.high.shape, high), is_image_space_channels_first(new_high)
        )
        new_low = (
            np.concatenate(
                [new_low, channel_low],
                axis=0 if is_image_space_channels_first(new_low) else 2,
            )
            if use_original_obs
            else channel_low
        )
        new_high = (
            np.concatenate(
                [new_high, channel_high],
                axis=0 if is_image_space_channels_first(new_high) else 2,
            )
            if use_original_obs
            else channel_high
        )
        return Box(low=new_low, high=new_high, dtype=new_low.dtype)


# There is a single channel that has same size with the observation.
# The channel is filled with the highest value if [type] is in the agent's name.
# Else, the channel is filled with the lowest value.
class BinaryIndicator(AgentIndicator):
    def __init__(self, env, type):
        super().__init__(env, type)

    def apply(self, obs, obs_space, agent):
        super().apply(obs, obs_space, agent)
        high = obs_space.high.max()
        low = obs_space.low.min()
        if is_image_space_channels_first(obs):
            channel_shape = obs.shape[1:]
        else:
            channel_shape = obs.shape[:2]
        channel = np.full(channel_shape, high if self.type in agent else low)
        return convert_three_dim(channel, is_image_space_channels_first(obs))

    def apply_space(self, obs_space, agent, use_original_obs):
        new_low = convert_three_dim(obs_space.low.copy(), True)
        new_high = convert_three_dim(obs_space.high.copy(), True)
        low = new_low.min()
        high = new_high.max()
        channel_shape = (
            new_low.shape[1:]
            if is_image_space_channels_first(new_low)
            else new_low.shape[:2]
        )
        channel_low = convert_three_dim(
            np.full(channel_shape, low), is_image_space_channels_first(new_low)
        )
        channel_high = convert_three_dim(
            np.full(channel_shape, high), is_image_space_channels_first(new_high)
        )
        new_low = (
            np.concatenate(
                [new_low, channel_low],
                axis=0 if is_image_space_channels_first(new_low) else 2,
            )
            if use_original_obs
            else channel_low
        )
        new_high = (
            np.concatenate(
                [new_high, channel_high],
                axis=0 if is_image_space_channels_first(new_high) else 2,
            )
            if use_original_obs
            else channel_high
        )
        return Box(low=new_low, high=new_high, dtype=new_low.dtype)


# Use different geometric pattern to represent different type of agent
#        Type 0 : 1 0 1 0 1 0 1 0 ...
#        Type 1 : 0 1 0 1 0 1 0 1 ...
class GeometricPatternIndicator(AgentIndicator):
    def __init__(self, env, type):
        super().__init__(env, type)
        self.build_patterns(env)

    def build_patterns(self, env):
        self.patterns = {}
        for agent in env.possible_agents:
            high = env.observation_spaces[agent].high.max()
            low = env.observation_spaces[agent].low.min()
            shape = env.observation_spaces[agent].shape
            if len(shape) == 3:
                if is_image_space_channels_first(env.observation_spaces[agent]):
                    shape = shape[1:]
                else:
                    shape = shape[:2]
            pattern = np.zeros(shape)

            cnt = 0
            for i in range(shape[0]):
                for j in range(shape[1]):
                    if cnt % 2 == 0:
                        pattern[i][j] = high if self.type in agent else low
                    else:
                        pattern[i][j] = low if self.type in agent else high
                    cnt += 1
            self.patterns[agent] = pattern

    def apply(self, obs, obs_space, agent):
        super().apply(obs, obs_space, agent)
        pattern = self.patterns[agent]
        return (
            pattern[None, :, :]
            if is_image_space_channels_first(obs)
            else pattern[:, :, None]
        )

    def apply_space(self, obs_space, agent, use_original_obs):
        new_low = convert_three_dim(obs_space.low.copy(), True)
        new_high = convert_three_dim(obs_space.high.copy(), True)
        low = new_low.min()
        high = new_high.max()
        channel_shape = (
            new_low.shape[1:]
            if is_image_space_channels_first(new_low)
            else new_low.shape[:2]
        )
        channel_low = convert_three_dim(
            np.full(channel_shape, low), is_image_space_channels_first(new_low)
        )
        channel_high = convert_three_dim(
            np.full(channel_shape, high), is_image_space_channels_first(new_high)
        )
        new_low = (
            np.concatenate(
                [new_low, channel_low],
                axis=0 if is_image_space_channels_first(new_low) else 2,
            )
            if use_original_obs
            else channel_low
        )
        new_high = (
            np.concatenate(
                [new_high, channel_high],
                axis=0 if is_image_space_channels_first(new_high) else 2,
            )
            if use_original_obs
            else channel_high
        )
        return Box(low=new_low, high=new_high, dtype=new_low.dtype)


class AgentIndicatorWrapper:
    def __init__(self, indicator, use_original_obs=True):
        self.use_original_obs = use_original_obs
        self.indicator = indicator

    def apply(self, obs, obs_space, agent):
        nobs = convert_three_dim(obs.copy(), True)
        ind = self.indicator.apply(nobs, obs_space, agent)
        return (
            np.concatenate(
                [nobs, ind], axis=0 if is_image_space_channels_first(nobs) else 2
            )
            if self.use_original_obs
            else ind
        )

    def apply_space(self, obs_space, agent):
        return self.indicator.apply_space(obs_space, agent, self.use_original_obs)
