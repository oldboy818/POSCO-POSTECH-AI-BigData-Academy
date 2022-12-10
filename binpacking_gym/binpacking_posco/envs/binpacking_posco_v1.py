import sys
sys.path.append('./binpacking_posco/envs/')
import numpy as np

from binpacking_posco_v0 import binpacking_posco_v0

class binpacking_posco_v1(binpacking_posco_v0):
    """
    Version 1
    V0 의 모든 함수를 따라감.
    
    변경부분만 아래에 적어 두었음.
    
    물리적으로 불가능한 액션 시행시 -1점을 받고 내부적으로 기록하도록.
    """
    def __init__(self, **kwargs):
        super(binpacking_posco_v1, self).__init__(**kwargs)
        
    def step(self, action):
        action = self.int_action_to_grid(action)
        
        terminated = bool(
            self.ct2 == self.ct2_threshold
            or self.ct == 20
        )
        
        if not terminated:
            if self.available_act(action):
                self.map_action(action)
                self.ct += 1
                self.update_product()
                self.state = np.append(self.Map.flatten(), self.width)
                reward = 1
                self.ct2 = 0
            else:
                reward = -1
        else:
            reward = self.calc_reward()
    
        return self.state, reward, terminated, {}