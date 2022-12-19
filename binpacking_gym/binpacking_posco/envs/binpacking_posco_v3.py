import sys
sys.path.append('./binpacking_posco/envs/')
import numpy as np

from binpacking_posco_v1 import binpacking_posco_v1

class binpacking_posco_v3(binpacking_posco_v1):
    """
    Version 3
    V2에서 reward만 수정함
    """
    
    def __init__(self, **kwargs):
        super(binpacking_posco_v1, self).__init__(**kwargs)
        # self.fill_threshold = kwargs.get('fill_threshold', 0.8)

    def available_act(self, action):
        """
        선택한 Action이 시행 가능한지 확인
        """        
        if action[0] + self.length > self.max_x + 1:
            return False
        if action[1] + self.width > self.max_y + 1:
            return False
        if self.Map[action[0]][action[1]] == 1:
            return False
        
        if self.Map[action[0]:(action[0] + self.length), action[1]:(action[1] + self.width)].sum() > 0:
            return False
        
        return True
    
    def step(self, action):
        action = self.int_action_to_grid(action)
        
        terminated = bool(
            self.ct2 == self.ct2_threshold
            or self.prod_idx == 22
        )
        score = 0
        if not terminated:
            if self.available_act(action):
                self.map_action(action)
                reward = self.width * self.length
                self.update_product()
                self.state = np.append(self.Map.flatten(), self.width)
                self.ct2 = 0
            else:
                # print("A")
                self.ct2 += 1
                reward = 0
        else:
            reward = 0

        info = {'score' : score}

        if self.render_mode == "human":
            self._render_frame()
        
        return self.state, reward, terminated, info