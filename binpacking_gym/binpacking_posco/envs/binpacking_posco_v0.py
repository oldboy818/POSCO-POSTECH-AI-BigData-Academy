import numpy as np
import gym
from gym import spaces
from random import choice
from copy import copy
"""
Model 학습단계에서 Loop를 사용하여 외부적으로 상황을 종료하는 환경.
"""

class binpacking_posco_v0(gym.Env):
    """
    Custom Env for 2D-bin Packing
    """
    # Class Variables 
    products = [(1,1), (2,2), (3,3), (3,3), (2,2), (2,2), (1,1),
                (1,1), (2,2), (3,3), (3,3), (2,2), (2,2), (1,1),
                (1,1), (2,2), (3,3), (3,3), (2,2), (2,2), (1,1),
                (1,1), (2,2), (3,3), (3,3), (2,2), (2,2), (1,1)]
    products_list = [(1,1), (2,2), (3,3)] # for random sampling
    
    # reward_range = (0, 100)
    metadata = {'mode': ['human']}
    spec = "EnvSpec"
    
    
    def __init__(self, **kwargs):
        """
        Kwargs / Default value (type)
        -------------
        ct2_threshold : 불가능한 행동을 몇번까지 허용할 것인가 / 50 (int)
        threshold 몇 %의 맵을 채웠을때 추가점수를 줄 것인가. : / 0.6 (float)
        mapsize : 전체 맵 사이즈 / [10, 10] (list)
        print_Map : Action시 마다 Map 출력 / True (bool)
        """
        super(binpacking_posco_v0, self).__init__()
        # For ending of episode
        self.ct = 0 # index of products
        self.ct2 = 0
        self.ct2_threshold = kwargs.get('ct2_threshold', 50) # 불가능한 행동의 제한 수
        
        # Product's Size
        self.width = self.products[self.ct][0]
        self.length = self.products[self.ct][1]
        
        # Map of warehouse
        self.mapsize = kwargs.get('mapsize', [10, 10])
        self.Map = np.zeros(self.mapsize, dtype=int)
        self.max_x = self.mapsize[0]-1
        self.max_y = self.mapsize[1]-1
        
        self.state = None
        self.actions_grid = [[i, j] for j in range (self.max_x+1) for i in range(self.max_y+1)]
        self.action_space = spaces.Discrete(len(self.actions_grid))
        self.print_Map = kwargs.get('print_Map', True)
        
        # Observation space
        ## Map + Max box size
        low = np.array([0 for _ in range(len(self.actions_grid))] + [0]) 
        high = np.array([1 for _ in range(len(self.actions_grid))] + [4])
        self.observation_space = spaces.Box(low, high, dtype=int)
        self.threshold = kwargs.get('threshold', 0.6) # Default = 0.6 # 이 비율의 공간을 채웠을 때 더 많은 리워드를 줌
    
    def update_product(self): # get next product
        self.width = self.products[self.ct][0]
        self.length = self.products[self.ct][1]

    def int_action_to_grid(self, action):
        return (self.actions_grid[action])
    
    def available_act(self, action):
        """
        선택한 Action이 시행 가능한지 확인
        """        
        self.ct2 += 1 # count unavailable action
        if action[0] + self.length > self.max_x + 1:
            return False
        if action[1] + self.width > self.max_y + 1:
            return False
        if self.Map[action[0]][action[1]] == 1:
            return False
        
        if self.Map[action[0]:(action[0] + self.length), action[1]:(action[1] + self.width)].sum() > 0:
            return False
        
        return True

    def map_action(self, action):
        """
        물건을 내려놓고 Map을 0 -> 1로 변경
        """        
        # Drop product (Only for Square) / Fill the Map
        self.Map[action[0]:(action[0] + self.length), action[1]:(action[1] + self.width)] = 1

    def step(self, action):
        action = self.int_action_to_grid(action)

        terminated = bool(
            self.ct2 == self.ct2_threshold
            or self.ct == 20
        )

        if not terminated: # 내려두지 않아야함 terminated이 True 일 때..
            self.map_action(action)
            self.ct += 1
            self.update_product()
            self.state = np.append(self.Map.flatten(), self.width) # 변경된 맵과 물건
            reward = 10 # 물리적으로 올바른 행동을 했을 때 기본점수
            self.ct2 = 0
            score = 0
        else:
            reward, score = self.calc_reward()
        
        info = {'score' : score}
        
        return self.state, reward, terminated, info

    def reset(self):
        self.ct2 = 0
        self.ct = 0 # index of products
        self.width = self.products[self.ct][0]
        self.length = self.products[self.ct][1]
        # 매번 다른 state를 주고 학습 시킬 수 있음 !
        self.state = np.append(self.Map.flatten(), self.width)
        
        # Map of warehouse
        self.Map = np.zeros(self.mapsize, dtype=int)
        
        return np.array(self.state)
    
    def calc_reward(self):
        """
        Reward when ending of episode        
        """
        score = self.Map.sum() / (self.mapsize[0] * self.mapsize[1]) # Episode 후 남아있는 공간
        if score > self.threshold:
            return 100 + 100*score, score
        else:
            return score*5, score # Reward, Map percent
        
    def render(self, action, reward, mode='human'):
        if mode == 'human':
            print (f'Action :{action}, reward :{reward}, box :{self.width, self.length}')
            if self.print_Map:
                print (self.Map)
    
    def close(self):
        pass

class binpacking_posco_v1(binpacking_posco_v0):
    pass