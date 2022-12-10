# Reinforcement-Learning-2d-binpacking

From 2022-12-06

## Gym Env

Custom gym Environment name as `binpacking_posco`
```text
v0 : End the episode with external parameter for action masking
v1 : No action masking, If AI take impossible action get minus reward.
```

## Usage

``` python
import gym
import binpacking_gym

env = gym.make('binpacking_posco-v0')
```

## Model

DQN model constructed by [Stable-baselines3]()
