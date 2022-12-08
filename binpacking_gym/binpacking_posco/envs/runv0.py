"""
torch, Keras등 일반 모델 러닝을 위해
Env 돌리기
"""

import gym
import binpacking_posco
env = gym.make('binpacking_posco-v0')
#env = gym.make('CartPole-v1')

# Test environment
for episode in range(10):
    done = False
    env.reset() # first
    while not done:
        for _ in range(30):
            action = env.action_space.sample()
            if env.available_act(action):
                #print (env.ct2)
                break
            else:
                continue
        new_state, reward, done, info = env.step(action)
        env.render(action, reward=reward)
        
env.close()

# STable_baseline3 Pytorch version
# from stable_baselines3.common.env_checker import check_env
# from stable_baselines3.dqn import MultiInputPolicy
# from stable_baselines3.common.env_util import make_vec_env
# from stable_baselines3 import PPO, A2C, DQN

# env = gym.make('binpacking_posco-v0')
# env = make_vec_env(lambda: env, n_envs=1)
# check_env(env)

# model = PPO('MlpPolicy', env, verbose=1).learn(5000)
# model = MultiInputPolicy("MlpPolicy", env)
# model.learn(total_timesteps=10000, log_interval=4)
# model.save("binpacking_v0")

# del model # remove to demonstrate saving and loading

# model = DQN.load("dqn_cartpole")

# obs = env.reset()
# while True:
#     done = False
#     while not done:
#         action, _states = model.predict(obs, deterministic=True)
#         for _ in range(30):
#             action = env.action_space.sample()
#             if env.available_act(action):
#                 break
#             else:
#                 continue
#     obs, reward, done, info = env.step(action)            
#     env.render(action, reward=reward)
#     if done:
#         obs = env.reset()

# # from sb3_contrib.common.maskable.utils import get_action_masks
# # get_action_masks(env)
