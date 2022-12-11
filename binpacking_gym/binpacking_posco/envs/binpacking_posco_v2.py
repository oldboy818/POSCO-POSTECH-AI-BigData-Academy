import sys
sys.path.append('./binpacking_posco/envs/')
import numpy as np

from binpacking_posco_v0 import binpacking_posco_v0

class binpacking_posco_v2(binpacking_posco_v0):
    """
    Version 2
    V0 의 모든 함수를 따라감.
    
    input product를 랜덤으로 선별하도록. -> 고정해 놓으니까 Local minimum에 빠지는 듯 함.
    그리고 일정 조건을 넘었을 때 종료 되도록.
    
    
    종료조건을 다시 설정해야할까? ct == 20
    """
    products_list = [(1,1), (2,2), (3,3)] # for random sampling
    
    def __init__(self, **kwargs):
        super(binpacking_posco_v2, self).__init__(**kwargs)
        self.ct = 0 # 모든 Product의 부피 합 (Map을 채울 수 없을 시 학습 종료) / v2에서만..
        self.width, self.length = self.random_product()
    
    def random_product(self):
        product = self.products_list[np.random.choice(len(self.products_list))]
        self.ct += product[0]*product[1]
        return product[0], product[1]
    
    def step(self, action):
        action = self.int_action_to_grid(action)
        
        terminated = bool(
            self.ct2 == self.ct2_threshold
            or self.ct > 80 # 80% 이상
        )
        
        if not terminated:
            info = {}
            if self.available_act(action):
                self.map_action(action)
                #self.ct += 1
                self.width, self.length = self.random_product()
                self.state = np.append(self.Map.flatten(), self.width)
                reward = 5
                self.ct2 = 0
            else:
                reward = -1
        else:
            reward, score = self.calc_reward()
            info = {score}
    
        return self.state, reward, terminated, info
    
    def reset(self):
        self.ct2 = 0
        self.ct = 0
        self.width, self.length = self.random_product()
        # 매번 다른 state를 주고 학습 시킬 수 있음 !
        self.state = np.append(self.Map.flatten(), self.width)
        
        # Map of warehouse
        self.Map = np.zeros(self.mapsize, dtype=int)
        
        return np.array(self.state)