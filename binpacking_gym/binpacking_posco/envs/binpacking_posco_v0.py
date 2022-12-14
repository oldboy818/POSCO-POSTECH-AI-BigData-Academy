import numpy as np
import gym
from gym import spaces
from random import choice
from copy import copy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
"""
가장 간단한 환경
Product : Random

물리적으로 가능한 행동 : 1
불가능 : -1
"""

class binpacking_posco_v0(gym.Env):
    """
    Custom Env for 2D-bin Packing
    """
    # Class Variables 
    products_list = [(1,1), (2,2), (3,3)] # for random sampling
    
    # reward_range = (0, 100)
    metadata = {'mode': ['human']}
    spec = "EnvSpec"
    
    
    def __init__(self, **kwargs):
        """
        Kwargs / Default value (type)
        -------------
        ct2_threshold : 불가능한 행동을 몇번까지 허용할 것인가 / 50 (int)
        # threshold 몇 %의 맵을 채웠을때 추가점수를 줄 것인가. : / 0.6 (float)
        mapsize : 전체 맵 사이즈 / [10, 10] (list)
        print_Map : Action시 마다 Map 출력 / True (bool)
        """
        super(binpacking_posco_v0, self).__init__()
        # Params
        self.ct2_threshold = kwargs.get('ct2_threshold', 20) # 불가능한 행동의 제한 수
        self.mapsize = kwargs.get('mapsize', [10, 10])
        self.print_Map = kwargs.get('print_Map', True)
        self.threshold = kwargs.get('threshold', 0.6) # Default = 0.6 # 이 비율의 공간을 채웠을 때 더 많은 리워드를 줌
        
        # For ending of episode
        self.filled_map = 0 
        self.ct2 = 0
        
        # Product's Size
        self.random_product()
        
        # Map of warehouse
        self.Map = np.zeros(self.mapsize, dtype=int)
        self.max_x = self.mapsize[0]-1
        self.max_y = self.mapsize[1]-1
        
        self.state = None
        self.actions_grid = [[i, j] for j in range (self.max_x+1) for i in range(self.max_y+1)]
        self.action_space = spaces.Discrete(len(self.actions_grid))
        
        # Observation space
        ## Map + Max box size
        low = np.array([0 for _ in range(len(self.actions_grid))] + [0]) 
        high = np.array([1 for _ in range(len(self.actions_grid))] + [4])
        self.observation_space = spaces.Box(low, high, dtype=int)

    def random_product(self):
        product = self.products_list[np.random.choice(len(self.products_list))]
        self.width = product[0]
        self.length = product[1]
        self.filled_map += self.width * self.length
    
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
            or self.filled_map > 80 # 80% 이상
        )
        
        score = 0
        if not terminated:
            if self.available_act(action):
                self.map_action(action)
                self.random_product()
                self.state = np.append(self.Map.flatten(), self.width)
                reward = 1
                self.ct2 = 0
            else:
                reward = -1
        else:
            reward = 1
        info = {'score' : score}
        
        return self.state, reward, terminated, info

    def reset(self):
        self.ct2 = 0
        self.filled_map = 0
        
        self.random_product()
        
        # 매번 다른 state를 주고 학습 시킬 수 있음 !
        self.state = np.append(self.Map.flatten(), self.width)
        
        # Map of warehouse
        self.Map = np.zeros(self.mapsize, dtype=int)
        
        return np.array(self.state)

    def render(self, mode='human'):
        # if mode == 'human':
        #     print (f'Action :{action}, reward :{reward}, box :{self.width, self.length}')
        #     if self.print_Map:
        #         print (self.Map)
        """
        $ pip install opencv-python==4.1.2.30

        Rendering with matplotlib
        boxes(x, y, width, length)
        boxes는 튜플의 리스트여야 함
        """

        iMode = plt.isinteractive()

        if mode == 'human':
            plt.interactive(True)
        
        # 1. 기본 스타일 설정
        plt.style.use('default')
        plt.rcParams['figure.figsize'] = (4, 4)
        plt.rcParams['font.size'] = 7

        # 2. 튜플의 리스트로 데이터 준비
        # Map에서 1인 부분 찾아 tuple의 list로 저장
        Map = list(map(list, self.Map))
        boxes = []
        for i in range(self.mapsize[0]) :
            for j in range(self.mapsize[1]):
                if Map[i][j] == 1:
                    boxes.append((i, j, 1, 1))

        # 3. 그래프 그리기
        fig, ax = plt.subplots()

        # Map
        ax.set_yticks(np.arange(0, 11))
        ax.set_xticks(np.arange(0, 11))
        plt.grid(True)

        # Plot Boxes
        # TODO : box 크기별로 색상 다르게 하기
        for (x, y, w, h) in boxes:
            ax.add_patch(
                patches.Rectangle(
                    (x, y), w, h,
                    edgecolor='green',
                    facecolor='palegreen',
                    fill=True))
        # HACK : 최후에 한 번만 뜨게 하거나 아무튼 plot이 계속 안 뜨게 해야함
        plt.show()

    def close(self):
        pass

class binpacking_posco_v1(binpacking_posco_v0):
    pass
