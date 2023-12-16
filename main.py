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
    envs_running = 1#32 #amount of envs running for data collection
    envs = gym.vector.SyncVectorEnv(
        [lambda: gym.make("SolarEnv-v0") for _ in range(envs_running)]
    )

    # Reset the environment to its initial state and receive the initial observation
    observation = envs.reset()

    # Define the number of episodes (or time steps) you want to run the environment
    num_episodes = 5
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
    print(agent)
    batch_size = 120  #Transformer is VRAM hungry...
    agent.memory.batch_size = batch_size
    #Per episode
    graphy = []
    test_size = int(365 * 48 * 0.25)
    random.seed(55) #For consistency
    test_inds = random.sample(range(0, 365*48), test_size)
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
                envs.step(actions)
            else:
                if sell_only:
                    actions = [2 for x in  range(envs_running)]
                    next_observation, reward, done,truncated,  _ = envs.step(actions)

                    train_val = sum(reward)/envs_running
                    balance += sum(reward)
                else:
                    actions, probs, value = agent.choose_action(observation)
                
                    next_observation, reward, done,truncated,  _ = envs.step(actions)

                    train_val = sum(reward)/envs_running
                    balance += sum(reward)
                    for obs, action, prob, val, rew, don in zip(observation, actions, probs, value, reward, done):
                        # if action == 2 or i % bias == 0:
                            agent.memory.push( obs, action, prob, val, rew, don)
                        # else:
                        #     i+= 1
                    if agent.memory.size() >= agent.memory.batch_size:
                        #If we have a large enough data then start learning
                        
                        agent.vectorized_clipped_ppo()
                   
            loop.set_description(f"Reward Average: {train_val} Eval: {eval_val}") 
            balences.append(balance)
            balence = 0
            # Update the current observation with the next observation
            observation = next_observation
            step += 1

        if isinstance(agent, AgentRNN) or isinstance(agent, ActorCNN) or isinstance(agent, AgentT):
            agent.reset() #Clear the hidden states
    print("Evaluation")
    with torch.no_grad():
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
                if step in test_inds:
                    #Perform calculations without gradients
                    
                    if sell_only:
                        actions = [2 for x in  range(envs_running)]
                    else:
                        actions, probs, value = agent.choose_action(observation)
                    next_observation, reward, done,truncated,  _ = envs.step(actions)
                    test_tmp.append(sum(reward) / envs_running)
                    obs = []
                    for ob in observation:
                        obs.append(ob[-2])
                    testx.append(obs)#Get in the price
                    eval_val = sum(reward) / envs_running
                else:
                    actions = [1 for x in range(envs_running)]
                    next_observation, reward, done,truncated,  _ = envs.step(actions)
                observation = next_observation
                step += 1

            graphy.append(sum(test_tmp) / len(test_tmp))
            testy.extend(test_tmp.copy())
            test_tmp.clear()
            if isinstance(agent, AgentRNN) or isinstance(agent, ActorCNN):
                agent.reset() #Clear the hidden states
        
    
    with open(f'./metrics/per_episode_{arg_val}.txt', 'w') as f:
        f.writelines(str(graphy))

    with open(f'./metrics/per_timestep_{arg_val}.txt', 'w') as f:
        f.writelines(str(testy))

    with open(f'./metrics/balance_{arg_val}.txt', 'w') as f:
        f.writelines(str(str(balences)))
    # Close the environment when done
    # print(sum(agent.memory.rewards[-1]))

    #Run a test run
    observation, _ = envs.reset()

    done = [False]
    eprices = []
    egains = []
    #When not done. This is an array of 
    #done
    eval_val = 0
    balance = 0
    while sum(done) < envs_running:
        
        if step in test_inds:
            #Perform calculations without gradients
            with torch.no_grad():
                actions, probs, value = agent.choose_action(observation)
                next_observation, reward, done,truncated,  _ = envs.step(actions)
                eval_val = sum(reward) / envs_running
                egains.append(eval_val)
                obs = []
                for ob in observation:
                    obs.append(ob[-2])
                eprices.append(obs)#Get in the price
        else:

            actions = [1 for _ in range(envs_running)]  
            next_observation, reward, done,truncated,  _ = envs.step(actions)
        loop.set_description(f"Reward Average: {train_val} Eval: {eval_val}") 
        # Update the current observation with the next observation
        observation = next_observation
        step += 1
    with open(f'./metrics/egains_{arg_val}.txt', 'w') as f:
        f.writelines(str(egains))

    with open(f'./metrics/eprices_{arg_val}.txt', 'w') as f:
        f.writelines(str(eprices))

    with open(f'./metrics/ebalance{arg_val}.txt', 'w') as f:
        f.writelines(str(balance))
    
    envs.close()

