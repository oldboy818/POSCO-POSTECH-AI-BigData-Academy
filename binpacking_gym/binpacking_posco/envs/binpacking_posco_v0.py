import numpy as np
import gym
from gym.spaces import Discrete, MultiDiscrete

class binpacking_posco_v0(gym.Env):

    def __init__(self, **kwargs):
        super().__init__()

        # Product's Size
        self.width = kwargs.get('width', 3)
        self.height = kwargs.get('height', 3)
        #self.Map

        self.action_space = MultiDiscrete([5,5])
        self.observation_space = MultiDiscrete([5,5])

    def step(self, **kwargs):
        obs = kwargs.get('Map', 3)
        reward = -1
        done = True
        info = {}

        return obs, reward, done, info

    def reset(self):
        ct = 0
        Map = 0


    def render(self, mode):
        pass
