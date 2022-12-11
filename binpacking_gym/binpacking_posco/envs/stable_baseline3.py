"""
torch, Keras등 일반 모델 러닝을 위해
Env 돌리기
"""

import gym
import binpacking_posco
env = gym.make('binpacking_posco-v0')
#env = gym.make('CartPole-v1')

# STable_baseline3 Pytorch version
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.dqn import MultiInputPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO, A2C, DQN

# env = gym.make('binpacking_posco-v0')
env = make_vec_env(lambda: env, n_envs=1)
check_env(env)

model = PPO('MlpPolicy', env, verbose=1).learn(5000)
model = MultiInputPolicy("MlpPolicy", env)
model.learn(total_timesteps=10000, log_interval=4)
model.save("binpacking_v0")

del model # remove to demonstrate saving and loading

model = DQN.load("dqn_cartpole")

obs = env.reset()

# from sb3_contrib.common.maskable.utils import get_action_masks
# get_action_masks(env)
