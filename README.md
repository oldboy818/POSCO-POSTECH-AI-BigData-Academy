# Reinforcement-Learning-2d-binpacking
From 2022-12-06

To organize products in warehouses.  
we performed reinforcement learning.  

|-|-|
|---|---|
|Input|Products of differnt sizes.|
|Purpose|Minimum Space|
|Model|Reinforcement Learning / PPO (proximal policy optimization)|

![Before]('./docs/v4_before.gif')
![Before]('./docs/v4_before.gif')

## Gym Env

Custom gym Environment name as `binpacking_posco`
```text
v0 : End the episode with external parameter for action masking
v1 : No action masking, If AI take impossible action get minus reward.
v2 : Random Products. - reward = -1 or 1 / 물리적으로 가능한 동작을 하게하려고
v3 : Action Masking by the Rule, The model should take possible action!.
v4 : Stable baseline
```

## Usage

``` python
import gym
import binpacking_gym

env = gym.make('binpacking_posco-v0')
```

## Model

DQN model constructed by [Stable-baselines3]()
