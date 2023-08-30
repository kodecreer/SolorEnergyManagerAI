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
from models import Agent
if __name__ == '__main__':
    #TODO Use farama Vector env instead
    # Create version of Solar Env that's compatible with the spec here
    #https://gymnasium.farama.org/api/experimental/vector/

    # env = gym.make('CartPole-v1', num_envs=2)
    envs = gym.make('SolarEnv-v0')
    # envs = gym.vector.SyncVectorEnv([
    #     lambda: gym.make("SolarEnv-v0"),
    #     lambda: gym.make("SolarEnv-v0"),
    #     lambda: gym.make("SolarEnv-v0")
    # ])
 
    # Reset the environment to its initial state and receive the initial observation
    observation = envs.reset()

    # Define the number of episodes (or time steps) you want to run the environment
    num_episodes = 1
    #How often to update the graph.
    #The lower the number, the slower it goes through all the adata
    log_interval = 1000
    interval = 0
    batch_size = 100
    agent: Agent = Agent(2) #Hold or sell are the ations we will take
    for episode in tqdm(range(num_episodes)):
        observation, _ = envs.reset()
        done = False

        while not done:
            # Replace 'your_action' with the action you want to take in the environment (e.g., 0, 1, 2, ...)
            
            actions = agent.choose_action(observation)

    
            next_observation, reward, done,truncated,  _ = envs.step(actions)

            if interval >= log_interval:
                # You can render the environment at each step if you want to visualize the progress
                envs.render()
                interval = 0

            # Update the current observation with the next observation
            observation = next_observation
            envs.current_step += 1
            interval += 1
            
        
    # Close the environment when done
    print(envs.balance)
    envs.close()