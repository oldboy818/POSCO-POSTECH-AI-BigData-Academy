import numpy as np
import gym
from gym import spaces
from random import choice
from copy import copy
"""
Loop를 통해 겹쳐지는 물건의 수를 제거하는 버전
= 일정 수 이상 수를 못찾으면 종료되는 것으로.

For stable-baseline,
No masking
When agent pick unavailable action, resample with While Loop.
"""

class binpacking_posco_v0(gym.Env):
    """
    Custom Env for 2D-bin Packing
    
    Reward = [0 - 100]
    """
    # Class Variables 
    products = [(4, 4), (2, 2), (3, 3), (1, 1), (2, 2), (3, 3), (4, 4), (1, 1), (2, 2), (1, 1),
                (4, 4), (2, 2), (3, 3), (1, 1), (2, 2), (3, 3), (4, 4), (1, 1), (2, 2), (1, 1),
                (4, 4), (2, 2), (3, 3), (1, 1), (2, 2), (3, 3), (4, 4), (1, 1), (2, 2), (1, 1)]
    reward_range = (0, 100)
    metadata = {'mode': ['human']}
    spec = "EnvSpec"
    
    
    def __init__(self):
        super(binpacking_posco_v0, self).__init__()
        # Product's Size
        # 여기서는 애초에 ㄱ자, ㄴ자 정의가 안됨
        self.ct = 0 # index of products
        self.width = self.products[self.ct][0]
        self.length = self.products[self.ct][1]
        self.ct2 = 0
        # Map of warehouse
        self.Map = np.zeros([10, 10])
        self.max_x = self.Map.shape[0]-1
        self.max_y = self.Map.shape[1]-1
        
        self.actions_grid = [[i, j] for j in range (self.max_x+1) for i in range(self.max_y+1)]
        #self.action_index = [i for i in range(len(self.actions_grid))] # Action's Index
        self.action_space = spaces.Discrete(len(self.actions_grid))
        #self.action_space = spaces.MultiDiscrete([10, 10])
        self.observation_space = spaces.Discrete(len(self.actions_grid))
        self.threshold = 0.7
    
    def update_product(self):
        self.width = self.products[self.ct][0]
        self.length = self.products[self.ct][1]

    def int_action_to_grid(self, action):
        return (self.actions_grid[action])
    
    def available_act(self, action):
        """
        선택한 Action이 시행 가능한지 확인
        """
        action = self.int_action_to_grid(action)
        
        self.ct2 += 1
        if action[0] + self.length > self.max_x:
            return False
        if action[1] + self.width > self.max_y:
            return False
        if self.Map[action[0]][action[1]] == 1:
            return False
        
        if self.Map[action[0]:(action[0] + self.length), action[1]:(action[1] + self.width)].sum() > 0:
            return False
        
        return True

    def map_action(self, action):
        """
        Randomly sample ACT with available act list.
        
        Actions of Map
        1. Change Map (Drop product)
        2. Change action_space
        """        
        # Drop product (Only for Square)
        self.Map[action[0]:(action[0] + self.length), action[1]:(action[1] + self.width)] = 1

    def step(self, action):
        action = self.int_action_to_grid(action)

        info = {}
        if self.ct2 == 30:
            print ('self.ct2')
            done = True
        else:
            self.ct2 = 0
            done = False

        if not done: # 내려두지 않아야함 done이 True 일 때..
            self.map_action(action)
            self.ct += 1
            self.update_product()
            reward = 0
        else:
            reward = self.calc_reward()
            
        return self.Map.flatten(), reward, done, info

    def reset(self):
        self.ct2 = 0
        self.ct = 0 # index of products
        self.width = self.products[self.ct][0]
        self.length = self.products[self.ct][1]
        
        # Map of warehouse
        self.Map = np.zeros([10, 10])
        # self.max_x = self.Map.shape[0]-1
        # self.max_y = self.Map.shape[1]-1
        
        return (self.Map.flatten().astype(np.int))
    
    def calc_reward(self):
        score = self.Map.sum() / (self.Map.shape[0] * self.Map.shape[1])
        if score > self.threshold:
            return 100*score
        else:
            return score
        
    def render(self, action, reward, mode='human'):
        if mode == 'human':
            print (f'Action :{action}, reward :{reward}, box :{self.products[self.ct-1]}')
            print (self.Map)
    
    def close(self):
        pass