import numpy as np
import gym
from gym import spaces
from copy import copy

class binpacking_posco_v0(gym.Env):
    """
    Custom Env for 2D-bin Packing
    """
    # Class Variables 
    products = [(4, 4), (2, 2), (3, 3), (1, 1), (2, 2), (3, 3), (4, 4), (1, 1), (2, 2), (1, 1),
                (4, 4), (2, 2), (3, 3), (1, 1), (2, 2), (3, 3), (4, 4), (1, 1), (2, 2), (1, 1)]
    
    def __init__(self, first = True):
        if first: # for reset
            super().__init__()

        # Product's Size
        # 여기서는 애초에 ㄱ자, ㄴ자 정의가 안됨
        self.ct = 0 # index of products
        self.width = self.products[self.ct][0]
        self.height = self.products[self.ct][1]
        
        # Map of warehouse
        self.Map = np.zeros([10, 10])
        self.max_x = self.Map.shape[0]-1
        self.max_y = self.Map.shape[1]-1
        
        self.actions = [[i, j] for j in range (self.max_x+1) for i in range(self.max_y+1)]
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.MultiBinary([self.max_x+1, self.max_y+1])
    
    def update_product(self, ct):
        self.width = self.products[ct][0]
        self.height = self.products[ct][1]

    def get_action(self, done):
        """
        Get custom action_space
        Select available grid by product's size,
        then Sample randomly.
        
        if no space, End this episode.
        """
        for i in range(len(self.actions)):
            # Max x, y
            if self.actions[i][0] + self.width > self.max_x:
                continue
            if self.actions[i][1] + self.height > self.max_x:
                continue
            
            # 직사각형은 되고 ㄱ 자 는 안됨
            if sum(self.Map[self.actions[i][0]:self.actions[i][0]+self.width][self.actions[i][1]:self.actions[i][1]+self.height]) != 0:
                self.Map[self.actions[i][0]:self.actions[i][0]+self.width][self.actions[i][1]:self.actions[i][1]+self.height] = 1 # drop
                for j in range(self.actions[i][0], self.actions[i][0] + self.width):
                    for k in range(self.actions[i][1], self.actions[i][1] + self.height):
                        self.actions.remove((j, k))
                return None
            else:
                continue
            
        done = True

    def map_action(self, action, done):
        """
        1. 물리적으로 상자를 내려둘 수 있는지 확인
        재귀말고 다 확인해서 새 리스트를 만든 상태에서 random을 하도록.
        
        Actions of Map
        1. Change Map (Drop product)
        2. Change action_space
        """
        for i in range(len(self.actions)):
            # Max x, y
            if self.actions[i][0] + self.width > self.max_x:
                continue
            if self.actions[i][1] + self.height > self.max_x:
                continue
            
            # 직사각형은 되고 ㄱ 자 는 안됨
            if sum(self.Map[self.actions[i][0]:self.actions[i][0]+self.width][self.actions[i][1]:self.actions[i][1]+self.height]) != 0:
                self.Map[self.actions[i][0]:self.actions[i][0]+self.width][self.actions[i][1]:self.actions[i][1]+self.height] = 1 # drop
                self.actions.pop(i)
                break
            else:
                continue
            done = True # 더이상 둘 곳이 없음
        
        # Drop(action) # Drop Product to Map
        pass

    def step(self, action):
        self.map_action(action)

        self.ct += 1
        self.update_product(self.ct)

        obs = self.Map
        done = False
        if done:
            reward = 0#reward function
        else:
            reward = 0       
        
        info = {}

        return obs, reward, done, info

    def reset(self):
        self.__init__(False)
        # self.ct = 0 # index of products
        # self.update_product(self.ct)
        # self.Map = np.zeros([10, 10])
        # self.action_space = spaces.Discrete(5) # number of actions
        
        return (self.Map)
    
    def render(self, mode):
        pass
    


#=================
def main():
    import gym
    import binpacking_posco
    env = gym.make('binpacking_posco-v0')
    
    env.reset()
    env.step(1)
    
    for episode in range(10):
        env.reset() # first
        done = False
        while not done:
            action = env.actions[env.action_space.sample()]
            new_state, reward, done, info = env.step(action)
    
    env.close()
    
    from stable_baselines3.common.env_checker import check_env
    check_env(env)