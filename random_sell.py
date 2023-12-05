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
    # envs = gym.make('SolarEnv-v0')
    envs_running = 1 #amount of envs running for data collection
    envs = gym.vector.SyncVectorEnv(
        [lambda: gym.make("SolarEnv-v0") for _ in range(envs_running)]
    )

    # Reset the environment to its initial state and receive the initial observation
    observation = envs.reset()

    # Define the number of episodes (or time steps) you want to run the environment
    num_episodes = 1
    #How often to update the graph.
    #The lower the number, the slower it goes through all the adata
    log_interval = 1000
    interval = 0
    
    torch.set_default_device('cuda')

    #Per episode
    graphy = []
    test_size = int(365 * 48 * 0.25)
    random.seed(55) #For consistency
    test_inds = random.sample(range(0, 365*48), test_size)
    #Per timestep
    testx = []
    test_tmp = []
    testy = []
    is_random = False
    #TODO Test this on a training set of 2017 if possible. Otherwise sample it
    #From data in the current set
    loop = tqdm(range(num_episodes))
    for episode in loop:
        observation, _ = envs.reset()
        done = [False]
        #When not done. This is an array of 
        #dones

        eval_val = 0
        step = 0
        train_val = 0
        balance = 0
        while sum(done) < envs_running:
            actions = [2]
            if step in test_inds:
                #Perform calculations without gradients

                with torch.no_grad():

                    next_observation, reward, done,truncated,  _ = envs.step(actions)
                    test_tmp.append(sum(reward) / envs_running)
                    obs = []
                    for ob in observation:
                        obs.append(ob[-2])
                    testx.append(obs)#Get in the price
                    eval_val = sum(reward) / envs_running
                    balance += sum(reward) 
            else:
               
                next_observation, reward, done,truncated,  _ = envs.step(actions)
                #Try Spare sell rewards
                bias = 2
                i = 0
                train_val = sum(reward)/envs_running
                balance += sum(reward)

                   
            loop.set_description(f"Reward Average: {train_val} Eval: {eval_val}") 
            # Update the current observation with the next observation
            observation = next_observation
            step += 1


        graphy.append(sum(test_tmp) / len(test_tmp))
        testy.extend(test_tmp.copy())
        test_tmp.clear()
        
    
    with open(f'./metrics/rs_per_episode_{is_random}.txt', 'w') as f:
        f.writelines(str(graphy))

    with open(f'./metrics/rs_per_timestep_{is_random}.txt', 'w') as f:
        f.writelines(str(testy))

    with open(f'./metrics/rs_balance_{is_random}.txt', 'w') as f:
        f.writelines(str(balance))

    # Close the environment when done
    # print(sum(agent.memory.rewards[-1]))
    envs.close()