import gymnasium as gym
import numpy as np


def concat_states(state):
    history = state["history"]
    weights = state["weights"]
    weight_insert_shape = (history.shape[0], 1, history.shape[2])
    if len(weights) - 1 == history.shape[0]:
        weight_insert = np.ones(
            weight_insert_shape) * weights[1:, np.newaxis, np.newaxis]
    elif len(weights) - 1 == history.shape[2]:
        weight_insert = np.ones(
            weight_insert_shape) * weights[np.newaxis, np.newaxis, 1:]
    else:
        weight_insert = np.ones(
            weight_insert_shape) * weights[np.newaxis, 1:, np.newaxis]
    state = np.concatenate([weight_insert, history], axis=1)
    return state


class ConcatStates(gym.Wrapper):
    """
    Concat both state arrays for models that take a single inputs.

    Usage:
        env = ConcatStates(env)

    Ref: https://github.com/openai/gym/blob/master/gym/wrappers/README.md
    """

    def __init__(self, env):
        super().__init__(env)
        hist_space = self.observation_space.spaces["history"]
        hist_shape = hist_space.shape
        self.observation_space = gym.spaces.Box(
            low=-10,
            high=10,
            shape=(hist_shape[0], hist_shape[1] + 1, hist_shape[2]),
            dtype=np.float32  # 明确指定数据类型
        )

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = concat_states(obs)
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return concat_states(obs), info 