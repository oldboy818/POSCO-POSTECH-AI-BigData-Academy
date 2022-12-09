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
        self.Map = np.zeros([10, 10], dtype=int)
        self.max_x = self.Map.shape[0]-1
        self.max_y = self.Map.shape[1]-1
        self.state = None
        
        self.actions_grid = [[i, j] for j in range (self.max_x+1) for i in range(self.max_y+1)]
        #self.action_index = [i for i in range(len(self.actions_grid))] # Action's Index
        self.action_space = spaces.Discrete(len(self.actions_grid))
        #self.action_space = spaces.MultiDiscrete([10, 10])
        
        # Width == Length
        low = np.array([0 for _ in range(len(self.actions_grid))] + [0])
        high = np.array([1 for _ in range(len(self.actions_grid))] + [4])
        self.observation_space = spaces.Box(low, high, dtype=int)
        self.threshold = 0.6 # 이 비율의 공간을 채웠을 때 더 많은 리워드를 줌
    
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
        # Drop product (Only for Square) / Fill the Map
        self.Map[action[0]:(action[0] + self.length), action[1]:(action[1] + self.width)] = 1

    def step(self, action):
        action = self.int_action_to_grid(action)

        terminated = bool(
            self.ct2 == 30
        )

        if not terminated: # 내려두지 않아야함 terminated이 True 일 때..
            self.map_action(action)
            self.ct += 1
            self.update_product()
            self.state = np.append(self.Map.flatten(), self.width) # 변경된 맵과 물건
            reward = 1 # 물리적으로 올바른 행동을 했을 때 기본점수
            self.ct2 = 0
        else:
            reward = self.calc_reward()
            
        return self.state, reward, terminated, {}

    def reset(self):
        self.ct2 = 0
        self.ct = 0 # index of products
        self.width = self.products[self.ct][0]
        self.length = self.products[self.ct][1]
        # 매번 다른 state를 주고 학습 시킬 수 있음 !
        # Reset 시에 Map을 Sampling 하게 하면 매번 다른 환경에서 학습하게 할 수 있음.
        self.state = np.append(self.Map.flatten(), self.width)
        
        # Map of warehouse
        self.Map = np.zeros([10, 10], dtype=int)
        
        return np.array(self.state)
    
    def calc_reward(self):
        score = self.Map.sum() / (self.Map.shape[0] * self.Map.shape[1])
        if score > self.threshold:
            return 100*score
        else:
            return score*5 # 행동이 끝났을 때 점수
        
    def render(self, action, reward, mode='human'):
        if mode == 'human':
            print (f'Action :{action}, reward :{reward}, box :{self.products[self.ct-1]}')
            print (self.Map)
    
    def close(self):
        pass

class binpacking_posco_v1(binpacking_posco_v0):
    pass