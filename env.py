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

#TODO Create simple MLP or RNN to do actions
#TODO incorporate price fluctuations. 

class SolarEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'render_modes': ['human', 'rgb_array'], 'render_fps': 30}  # Add 'render_fps'

    def __init__(self):
        super(SolarEnv, self).__init__()
        data = pd.read_excel('./dummy_data.xlsx')

        self.fig, self.ax = plt.subplots(2, 2)
        self.ax[0][0].set_xlabel('Timestep')
        self.ax[0][0].set_ylabel('Balance')

        self.ax[0][1].set_xlabel('Timestep')
        self.ax[0][1].set_ylabel('VMP')

        self.ax[1][0].set_xlabel('Timestep')
        self.ax[1][0].set_ylabel('IMP')

        self.ax[1][1].set_xlabel('Timestep')
        self.ax[1][1].set_ylabel('Reward')

        plt.ion()
        plt.show(block=False)

        self.df = data
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(12,), dtype=np.float32
        )
        #Key for the AI network
        self.balance = 0
        self.wattage_balance = 0
        self.power_daily = 866
        self.power_sub = self.power_daily / 48 #Kilowatts to substract from wattage_balance per time step
        self.carbon_punishment = 5
        self.current_step = 0
        self.hour = 1
        self.vimp = []
        self.imp = []
        self.actions = []
        self.rewards = []

    def calc_reward(self):
        return self.balance
        
    def get_wattage(self, vmp, imp):
        pmax = vmp * imp
        return pmax

    def step(self, action):
        vimp = self.df['Vmp'][self.current_step]
        imp = self.df['Imp'][self.current_step]
  
        self.vimp.append(vimp)
        self.imp.append(imp)
        WATTAGE_RATE = random.random()

        kilo_watts = self.get_wattage(vimp, imp)#Lets assum Kilo Watts for now

        #Hold the power
        if action == 1 and len(self.actions) > 0:
            #Add it to the balance
            self.wattage_balance += kilo_watts
            #subtract the amount of energy consumed
            self.wattage_balance -= self.power_sub
            if self.wattage_balance < 0:
                #Subtract from the balance
                self.balance -= abs(self.wattage_balance * WATTAGE_RATE) * self.carbon_punishment
            self.actions.append(action)
        #Sell it back to the grid
        #First subtract the wattage consumed from the balane
        #Then add it back toe the balance and sell the excess
        #You get no money for selling money you don't have
        else:
            self.wattage_balance -= self.power_sub
            self.wattage_balance += kilo_watts
            
            if self.wattage_balance > 0:
                self.balance += self.wattage_balance * WATTAGE_RATE
                #clear the wattages
                self.wattage_balance = 0
            else:
                #Subtract from the balance
                self.balance -= abs(self.wattage_balance * WATTAGE_RATE) * self.carbon_punishment
                self.wattage_balance = 0
            self.actions.append(action)
        self.rewards.append(self.balance)
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        observation = np.array(self.df.iloc[self.current_step].values)
        truncated = False
        #for now we will naively set the reward to the balance...
        #Since that is the key statistic we want to maximize.
        #May need to consider something else later
        reward = self.calc_reward()
        self.current_step += 1
        return observation,reward,  done, truncated, {}

    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.actions = []
        self.vimp = []
        self.imp = []
        observation = (np.array(self.df.iloc[0].values))
        return observation, {}

    def render(self, mode='human'):
        log_val = 100
        x_data = range(0, min(len(self.actions), log_val))
        self.ax[0][0].clear()
        self.ax[0][1].clear()
        self.ax[1][0].clear()
        self.ax[1][1].clear()

        self.ax[0][0].set_xlim(min(x_data), max(x_data))
        self.ax[0][0].set_ylim(min(self.actions[-log_val:]), max(self.actions[-log_val:]))
        self.ax[0][0].plot(x_data, self.actions[-log_val:])

        self.ax[0][1].set_xlim(min(x_data), max(x_data))
        self.ax[0][1].set_ylim(min(self.vimp[-log_val:]), max(self.vimp[-log_val:]))
        self.ax[0][1].plot(x_data, self.vimp[-log_val:])

        self.ax[1][0].set_xlim(min(x_data), max(x_data))
        self.ax[1][0].set_ylim(min(self.imp[-log_val:]), max(self.imp[-log_val:]))
        self.ax[1][0].plot(x_data, self.imp[-log_val:])

        self.ax[1][1].set_xlim(min(x_data), max(x_data))
        self.ax[1][1].set_ylim(min(self.rewards[-log_val:]), max(self.rewards[-log_val:]))
        self.ax[1][1].plot(x_data, self.rewards[-log_val:])

        self.fig.canvas.draw()
        plt.pause(1e-40)
        
gym.register(id='SolarEnv-v0', entry_point=SolarEnv)