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
import sys
if __name__ == '__main__':
    torch.set_default_device('cuda')
    envs_running = 1 #amount of envs running for data collection
    envs = gym.vector.SyncVectorEnv(
        [lambda: gym.make("SolarEnv-v0") for _ in range(envs_running)]
    )

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
    #Set the model to make it easy to set model
    arg_val = 1 if len(sys.argv) <= 1 else int( sys.argv[1] )
    agent = None
    sell_only = False
    random_only = False
    print(arg_val)
    if arg_val == 1:
        agent = Agent(2, params)
    elif arg_val == 2:
        agent = AgentRNN(2, params)
    elif arg_val == 3:
        agent = AgentT(2, params)
    elif arg_val == 4:
        agent = AgentCNN(2, params)
    elif arg_val == 5:
        agent = Agent(2, params)
        sell_only = True
        num_episodes = 1
    elif arg_val == 6:
        agent = Agent(2, params)
        random_only = True
    print(agent)
    batch_size = 128  #Transformer is VRAM hungry...
    agent.memory.batch_size = batch_size
    #Per episode
    graphy = []
    test_size = int(365* 0.2)
    random.seed(40) #For consistency
    test_inds = random.sample(range(0, 365), test_size)
    #Per timestep
    testx = []
    test_tmp = []
    testy = []
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
        balences = []
        while sum(done) < envs_running:
            
            if step in test_inds:
                actions = [1 for x in range(envs_running)]
                next_observation, reward, done,truncated,  _ = envs.step(actions)
         
            else:
                if sell_only:
                    actions = [2 for x in  range(envs_running)]
                    next_observation, reward, done,truncated,  _ = envs.step(actions)

                    train_val = sum(reward)/envs_running
                    balance += sum(reward)
                elif random_only:
                    actions = [random.randint(1,2) for x in  range(envs_running)]
                    next_observation, reward, done,truncated,  _ = envs.step(actions)

                    train_val = sum(reward)/envs_running
                    balance += sum(reward)
                else:
                    actions, probs, value = agent.choose_action(observation)
                
                    next_observation, reward, done,truncated,  _ = envs.step(actions)
                
                    train_val = sum(reward)/envs_running
                    balance += sum(reward)
                
                    for obs, action, prob, val, rew, don in zip(observation, actions, probs, value, reward, done):
                            agent.memory.push( obs, action, prob, val, rew, don)
                    if agent.memory.size() >= agent.memory.batch_size:
                        #If we have a large enough data then start learning
                        agent.vectorized_clipped_ppo()
                   
            loop.set_description(f"Reward Average: {train_val} Eval: {eval_val}") 
            balences.append(balance)
            balence = 0
            # Update the current observation with the next observation
            observation = next_observation
            step += 1

        if isinstance(agent, AgentRNN) or isinstance(agent, ActorCNN)or isinstance(agent, AgentT):
            agent.reset() #Clear the hidden states
    if not random_only and not sell_only:
        agent.vectorized_clipped_ppo()
    print("Evaluation")
    with torch.no_grad():
        
        loop = tqdm(range(num_episodes))
        for episode in loop:
            observation, _ = envs.reset()
            done = [False]
            step = 0
            while sum(done) < envs_running:
                if step in test_inds:
                    #Perform calculations without gradients
                    
                    if sell_only:
                        actions = [2 for x in  range(envs_running)]
                    elif random_only:
                        actions = [random.randint(1,2) for x in  range(envs_running)]
                    else:
                        actions, probs, value = agent.choose_action_inf(observation)
                    next_observation, reward, done,truncated,  _ = envs.step(actions)
                    eval_val = sum(reward) / envs_running
                    test_tmp.append(eval_val)
                else:
                    actions = [1 for x in range(envs_running)]
                    next_observation, reward, done,truncated,  _ = envs.step(actions)
                observation = next_observation
                step += 1

            graphy.append(sum(test_tmp))
            testy.extend(test_tmp.copy())
            test_tmp.clear()
            if isinstance(agent, AgentRNN) or isinstance(agent, ActorCNN) or isinstance(agent, AgentT):
                agent.reset() #Clear the hidden states
    
    with open(f'./metrics/per_episode_{arg_val}.txt', 'w') as f:
        f.writelines(str(graphy))

    with open(f'./metrics/per_timestep_{arg_val}.txt', 'w') as f:
        f.writelines(str(testy))

    with open(f'./metrics/balance_avg_{arg_val}.txt', 'w') as f:
        f.writelines(str(sum(graphy) / num_episodes) )

    
    envs.close()
