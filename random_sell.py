import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import torch

from torch import nn
from tqdm import tqdm
import env
from env import SolarEnv
from models import *

def update_rewards(action_arr: list, reward_arr: list, reward: float):
    #While we got a action that has a sell then update the reward of each action.
    #If the thing is cut short, then we would assume the reward to be the current
    #Profit so far in the portfolio
    ptr = len(action_arr)-1
    while ptr >= 0 and action_arr[ptr] == 1:#while the aciton is hold
        reward_arr[ptr] = reward

if __name__ == '__main__':
    torch.set_default_device('cuda')
    #TODO Use farama Vector env instead
    # Create version of Solar Env that's compatible with the spec here
    #https://gymnasium.farama.org/api/experimental/vector/

    # env = gym.make('CartPole-v1', num_envs=2)
    envs = gym.make('SolarEnv-v0')
    # envs_running = 1 #amount of envs running for data collection
    # envs = gym.vector.SyncVectorEnv(
    #     [lambda: gym.make("SolarEnv-v0") for _ in range(envs_running)]
    # )

    # Reset the environment to its initial state and receive the initial observation
    observation = envs.reset()

    # Define the number of episodes (or time steps) you want to run the environment
    num_episodes = 100
    #How often to update the graph.
    #The lower the number, the slower it goes through all the adata
    log_interval = 1000
    interval = 0
    
    torch.set_default_device('cuda')
    params = HyperParameterConfig()
    agent: Agent = AgentT(2, params) #Hold or sell are the ations we will take
    batch_size = 120 if not isinstance(agent, AgentT) else 120 #Transformer is VRAM hungry...
    agent.memory.batch_size = batch_size
    graphx = []
    graphy = []

    test_size = int(365 * 48 * 0.2)
    random.seed(40) #For consistency
    test_inds = random.sample(range(0, 365*48), test_size)
    

    test_tmp = []
    testy = []
    test_actions = []

    train_acitons = []
    train_rewards = []
    is_random = True
    for episode in tqdm(range(num_episodes)):
        observation, _ = envs.reset()
        done = False
        #When not done. This is an array of 
        #dones
        step = 0
        while step < 365*48:
            if is_random:
                actions = random.randint(1, 2) 
            else:
                actions = 2 
            if step in test_inds:
                #Perform calculations without gradients
                with torch.no_grad():
                    next_observation, reward, done,truncated,  _ = envs.step(actions)
                    test_tmp.append(reward)
                    test_actions.append(actions)
            else:

                next_observation, reward, done,truncated,  _ = envs.step(actions)

                train_acitons.append(actions)
                train_rewards.append(reward)

            # Update the current observation with the next observation
            observation = next_observation
            step += 1
        if isinstance(agent, AgentRNN) or isinstance(agent, ActorCNN):
            agent.reset() #Clear the hidden states
        
        
        graphy.append(sum(test_tmp))
        testy = test_tmp.copy()
        test_tmp.clear()
        
    bal_str = f'./per_balance_{"random" if is_random else "sell"}.txt'
    with open(bal_str, 'w') as f:
        f.writelines(str(graphy))

    eps_str = f'./per_episode_{"random" if is_random else "sell"}.txt'
    with open(eps_str, 'w') as f:
        f.writelines(str(testy))

    # Close the environment when done
    # print(sum(agent.memory.rewards[-1]))
    envs.close()