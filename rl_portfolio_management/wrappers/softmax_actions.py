# import gym.wrappers
import gymnasium as gym

from ..util import softmax


class SoftmaxActions(gym.Wrapper):
    """
    Environment wrapper to softmax actions.

    Usage:
        env = gym.make('Pong-v0')
        env = SoftmaxActions(env)

    Ref: https://github.com/openai/gym/blob/master/gym/wrappers/README.md

    """

    def step(self, action):
        # also it puts it in a list
        if isinstance(action, list):
            action = action[0]

        if isinstance(action, dict):
            action = list(action[k] for k in sorted(action.keys()))

        action = softmax(action, t=1)
        
        # 确保返回5个值
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        return self.env.reset(seed=seed, options=options) 
