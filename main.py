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
from supervised_models import MOE_Agent
if __name__ == '__main__':
    torch.set_default_device('cuda')
    envs_running = 1 #amount of envs running for data collection
    envs = gym.vector.SyncVectorEnv(
        [lambda: gym.make("SolarEnv-v1") for _ in range(envs_running)]
    )

    # Reset the environment to its initial state and receive the initial observation
    observation = envs.reset()

    # Define the number of episodes (or time steps) you want to run the environment
    num_episodes = 1000
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
    moe_agent = False
    nactions = 2
    is_sparse = True
    
    print(arg_val)
    if arg_val == 1:
        agent = Agent(nactions, params)
    elif arg_val == 2:
        agent = Agent(nactions, params)
        sell_only = True
        num_episodes = 1
    elif arg_val == 3:
        agent = Agent(nactions, params)
        random_only = True
        num_episodes = 5
    elif arg_val == 4:
        moe_agent = True
        agent = MOE_Agent('./shse.mdl', window=4)
    print(agent)
    if arg_val == 1:
        batch_size = 64  #Transformer is VRAM hungry...
        agent.memory.batch_size = batch_size
    #Per episode
    graphy = []
    test_size = int(365* 0.3)
    random.seed(40) #For consistency
    test_inds = range(365-test_size, 365)#random.sample(range(0, 365), test_size)
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
        if moe_agent:
            sell_time_out = agent.window
        while sum(done) < envs_running:
            
            if step in test_inds:
                actions = [0 for x in range(envs_running)]
                next_observation, reward, done,truncated,  _ = envs.step(actions)
         
            else:
                if sell_only:
                    actions = [1 for x in  range(envs_running)]
                    next_observation, reward, done,truncated,  _ = envs.step(actions)

                    train_val = sum(reward)/envs_running
                    balance += sum(reward)
                elif random_only:
                    actions = [random.randint(0,1) for x in  range(envs_running)]
                    next_observation, reward, done,truncated,  _ = envs.step(actions)

                    train_val = sum(reward)/envs_running
                    balance += sum(reward)
                elif moe_agent:
                    with torch.no_grad():
                        if sell_time_out <= 0:
                            action = 1
                            sell_time_out = agent.window+1
                        else: 
                            if sell_time_out > agent.window:
                                
                                date = torch.tensor([observation[0][0]]).unsqueeze(0).float()
                                day_to_sell = agent.choose_action(date)
                                sell_time_out = day_to_sell
                            if sell_time_out <= 0:
                                action = 1 #sell
                                sell_time_out = agent.window + 1
                            else:
                                action = 0
                                sell_time_out -= 1
                    next_observation, reward, done, truncated, _ = envs.step([action])
  
                    
                else:
                    # chance = random.randint(1,10)
                    # if chance == 1:
                    #     action = 0
                    actions, probs, value = agent.choose_action(observation)
                
                    next_observation, reward, done,truncated,  _ = envs.step(actions)
                
                    train_val = sum(reward)/envs_running
                    balance += sum(reward)
                
                    for obs, action, prob, val, rew, don in zip(observation, actions, probs, value, reward, done):
                            if is_sparse and action == 1 or not is_sparse:
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

    if not random_only and not sell_only and not moe_agent:
        if len(agent.memory.actions) > 0:
            agent.vectorized_clipped_ppo()
    print("Evaluation")
    with torch.no_grad():
        
        loop = tqdm(range(30))
        for episode in loop:
            observation, _ = envs.reset()
            done = [False]
            step = 0
            while sum(done) < envs_running:
                if step in test_inds:
                    #Perform calculations without gradients
                    
                    if sell_only:
                        actions = [1 for x in  range(envs_running)]
                    elif random_only:
                        actions = [random.randint(0,3) for x in  range(envs_running)]
                    elif moe_agent:
                        if sell_time_out <= 0:
                            actions = 1
                            sell_time_out = agent.window+1
                        else: 
                            if sell_time_out > agent.window:
                                date = torch.tensor([observation[0][0]]).unsqueeze(0).float()
                                day_to_sell = agent.choose_action(date)
                                sell_time_out = day_to_sell
                            if sell_time_out <= 0:
                                actions = 1 #sell
                                sell_time_out = agent.window
                            else:
                                actions = 0
                                sell_time_out -= 1
                        actions = [actions]
                    else:
                        actions, probs, value = agent.choose_action_inf(observation)
                    next_observation, reward, done,truncated,  _ = envs.step(actions)
                    eval_val = sum(reward) / envs_running
                    test_tmp.append(eval_val)
                else:
                    actions = [0 for x in range(envs_running)]
                    next_observation, reward, done,truncated,  _ = envs.step(actions)
                observation = next_observation
                step += 1

            graphy.append(sum(test_tmp))
            testy.extend(test_tmp.copy())
            test_tmp.clear()

    
    with open(f'./metrics/per_episode_{arg_val}.txt', 'w') as f:
        f.writelines(str(graphy))

    with open(f'./metrics/per_timestep_{arg_val}.txt', 'w') as f:
        f.writelines(str(testy))

    with open(f'./metrics/balance_avg_{arg_val}.txt', 'w') as f:
        f.writelines(str(sum(graphy) / num_episodes) )

    
    envs.close()
