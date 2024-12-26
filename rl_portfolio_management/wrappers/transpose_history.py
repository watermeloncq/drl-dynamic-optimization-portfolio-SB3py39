# import gym.spaces
import numpy as np
import gymnasium as gym

class TransposeHistory(gym.Wrapper):
    """Transpose history."""

    def __init__(self, env, axes=(2, 1, 0)):
        super().__init__(env)
        self.axes = axes

        hist_space = self.observation_space.spaces["history"]
        hist_shape = hist_space.shape
        self.observation_space = gym.spaces.Dict({
            'history': gym.spaces.Box(
                hist_space.low.min(),
                hist_space.high.max(),
                (hist_shape[axes[0]], hist_shape[axes[1]], hist_shape[axes[2]]),
                dtype=np.float32
            ),
            'weights': self.observation_space.spaces["weights"]
        })

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs["history"] = np.transpose(obs["history"], self.axes)
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        obs["history"] = np.transpose(obs["history"], self.axes)
        return obs, info 
