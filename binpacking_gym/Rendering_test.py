"""
torch, Keras등 일반 모델 러닝을 위해
Env 돌리기
"""
import sys
# sys.path.append('/home/piai/workspace/Reinforcement-Learning-2d-binpacking/binpacking_gym')
sys.path.append('/Reinforcement-Learning-2d-binpacking/binpacking_gym')

import numpy as np
import gym
import binpacking_posco
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker

def get_action_mask(env):
    mask_action = []
    for i in range(len(env.actions_grid)):
        mask_action.append(env.available_act(env.actions_grid[i]))

    return mask_action

env = gym.make('binpacking_posco-v3')
env = ActionMasker(env, get_action_mask)

model = MaskablePPO(MaskableActorCriticPolicy, env, verbose = 1)
model.learn(1, progress_bar = True)

model.save('./model/MP_render_test.zip')

# env.render()
# 모델 결과 Rendering
obs = env.reset()
while True:
    # Retrieve current action mask
    env.render()
    action_masks = get_action_mask(env)
    action, _states = model.predict(obs, action_masks=action_masks)
    obs, rewards, dones, info = env.step(action)
    env.close()
    if dones: 
        env.reset()
        break