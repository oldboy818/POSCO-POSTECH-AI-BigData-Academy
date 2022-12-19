import sys
sys.path.append('./binpacking_posco/envs/')
import numpy as np

from binpacking_posco_v1 import binpacking_posco_v1

class binpacking_posco_v2(binpacking_posco_v1):
    """
    Version 2
    V0 의 모든 함수를 따라감.
    
    input product를 랜덤으로 선별하도록. -> 고정해 놓으니까 Local minimum에 빠지는 듯 함.
    그리고 일정 조건을 넘었을 때 종료 되도록.
    
    현재 들고있는 물건 부피까지해서 전체 맵의 특정 % 이상 점유할 때 종료.
    
    => 우선 물리적으로 가능한 동작만 하게 학습하려고 Reward를 수정함.
    """
    
    def __init__(self, **kwargs):
        super(binpacking_posco_v2, self).__init__(**kwargs)
        self.fill_threshold = kwargs.get('fill_threshold', 0.8)
    
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
                # 중간 리워드 줘야할 것 같은데.. 채운 공간만큼 줘야할듯
                reward = 1
                self.update_product()
                self.state = np.append(self.Map.flatten(), self.width)
                self.ct2 = 0
            else:
                reward = -1
        else:
            if self.filled_map >= 100*self.fill_threshold:
                reward = self.filled_map/5 # 80 - 100
            else:
                reward = -1
        info = {'score' : score}

        if self.render_mode == "human":
            self._render_frame()
        
        return self.state, reward, terminated, info