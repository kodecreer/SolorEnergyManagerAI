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
if __name__ == '__main__':
    torch.set_default_device('cuda')
    #TODO Use farama Vector env instead
    # Create version of Solar Env that's compatible with the spec here
    #https://gymnasium.farama.org/api/experimental/vector/

    # env = gym.make('CartPole-v1', num_envs=2)
    # envs = gym.make('SolarEnv-v0')
    envs_running = 2#32 #amount of envs running for data collection
    envs = gym.vector.SyncVectorEnv(
        [lambda: gym.make("SolarEnv-v0") for _ in range(envs_running)]
    )

    # Reset the environment to its initial state and receive the initial observation
    observation = envs.reset()

    # Define the number of episodes (or time steps) you want to run the environment
    num_episodes = 0
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
    testx = []
    test_tmp = []
    testy = []
    #TODO Test this on a training set of 2017 if possible. Otherwise sample it
    #From data in the current set
    for episode in tqdm(range(num_episodes)):
        observation, _ = envs.reset()
        done = [False]
        #When not done. This is an array of 
        #dones
        step = 0
        while step < 365*48:
            
            if step in test_inds:
                #Perform calculations without gradients
                with torch.no_grad():
                    actions, probs, value = agent.choose_action(observation)
                    next_observation, reward, done,truncated,  _ = envs.step(actions)
                    test_tmp.append(sum(reward) / envs_running)
            else:
                actions, probs, value = agent.choose_action(observation)
               
                next_observation, reward, done,truncated,  _ = envs.step(actions)
                for obs, action, prob, val, rew, don in zip(observation, actions, probs, value, reward, done):
                    agent.memory.push( obs, action, prob, val, rew, don)
                if agent.memory.size() >= agent.memory.batch_size:
                    #If we have a large enough data then start learning
                    print(f'Reward: {sum(reward)/envs_running}')
                    print(f"Learning ...")
                    agent.vectorized_clipped_ppo()
                    
                if interval % log_interval == 0:
                    # You can render the environment at each step if you want to visualize the progress
                    # envs.render()
                    graphx.append(interval)
                    graphy.append(sum(reward) / envs_running)
                    # interval = 0
                    interval += 1

            # Update the current observation with the next observation
            observation = next_observation
            step += 1
        if isinstance(agent, AgentRNN) or isinstance(agent, ActorCNN):
            agent.reset() #Clear the hidden states
        
        

        testy = test_tmp
        test_tmp.clear()
        
    agent.actor.save_checkpoint()
    agent.critic.save_checkpoint()
    plt.plot(graphx, graphy)
    plt.savefig('train_metrics.pdf', bbox_inches='tight')   
    plt.cla()
    plt.clf()
    with open('./results.txt', 'w') as f:
        f.writelines(str(testy))
    plt.plot(range(0, len(testy)), testy)
    plt.savefig('test_metrics.pdf', bbox_inches='tight')  
    # Close the environment when done
    # print(sum(agent.memory.rewards[-1]))
    envs.close()