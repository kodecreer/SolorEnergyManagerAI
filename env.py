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
from datetime import datetime
#TODO Create simple MLP or RNN to do actions
#TODO incorporate price fluctuations. 

class SolarEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'render_modes': ['human', 'rgb_array'], 'render_fps': 30}  # Add 'render_fps'

    def __init__(self, is_dummy=False):
        super(SolarEnv, self).__init__()
        data = pd.read_excel('./Solar data_2016.xlsx')
        self.energy_prices = pd.read_excel('Energy Price_2016.xlsx')
        self.energy_prices = self.energy_prices[self.energy_prices["Price hub"].replace(' ', '') == 'Mid C Peak']
        self.energy_prices = self.energy_prices[1:]#because it has January 30th
        self.energy_dates = self.energy_prices['Trade date'].values
        self.energy_prices = self.energy_prices['Avg price $/MWh'].values

        self.energy_step = 0 
        self.energy_buffer = 0
        self.last_date = np.datetime64("2016-01-01T00:00:00.000000000")
        self.start_date = np.datetime64("2016-01-01T00:00:00.000000000")

        # self.fig, self.ax = plt.subplots(2, 2)
        # self.ax[0][0].set_xlabel('Timestep')
        # self.ax[0][0].set_ylabel('Balance')

        # self.ax[0][1].set_xlabel('Timestep')
        # self.ax[0][1].set_ylabel('VMP')

        # self.ax[1][0].set_xlabel('Timestep')
        # self.ax[1][0].set_ylabel('IMP')

        # self.ax[1][1].set_xlabel('Timestep')
        # self.ax[1][1].set_ylabel('Reward')

        # plt.ion()
        # plt.show(block=False)

        self.df = data
        self.train_sz = len(self.df)-1
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(4,), dtype=np.float32
        )
        #Key for the AI network
        self.balance = 0
        self.wattage_balance = 0
        #Smart home manager, smart home speaker , Electric vehicle charging
        self.power_day = 0
        self.wattage_rate = 0

        self.hour = 1
        self.vimp = []
        self.imp = []
        self.actions = []
        self.rewards = []
  
    def calc_reward(self, aux=1):
        reward = (self.wattage_balance + self.power_day) * self.wattage_rate * aux 
        return reward
        
    def get_wattage(self, vmp, imp):
        pmax = vmp * imp
        return pmax

    def step(self, action):
        reward = 0
        if action == 1:
            #hold for
            self.wattage_balance += self.power_day
        else:
            #sell
            reward = self.calc_reward()
            self.balance += reward
            #reward *= 100 
            reward = self.balance
            self.wattage_balance = 0
        self.power_day = 0
        done = truncated = self.train_sz <= self.current_step
        #Get the day ahead for the observation
       
        self.wattage_rate = 0
        self.power_day = 0
        if not done:
            last_price = self.wattage_rate
            for i in range(48):
                vimp = self.df['Vmp'][self.current_step]
                imp = self.df['Imp'][self.current_step]
        
                #Account for weekend
                yearday = self.current_step
                dayahead = pd.to_datetime( self.energy_dates[self.energy_step+1] )
                start = pd.to_datetime( self.start_date )

                day_diff = (dayahead - start).days
                if day_diff  < yearday / 48:
                    start = dayahead
                    self.energy_step += 1
                self.wattage_rate = self.energy_prices[self.energy_step]/1000

                
                kilo_watts = self.get_wattage(vimp, imp)/1000#Lets assum Kilo Watts for now
                self.power_day += kilo_watts
                self.current_step += 1
                #Day of the year, price, wattage stored, power generated in the day
                observation = np.array([self.current_step//48, self.wattage_rate, self.wattage_balance, self.power_day])
            if last_price < self.wattage_rate and action == 1:
                reward -= 10
        else:
            print(f'Balance: {self.balance:.2f}')
            observation = np.array([0, 0, 0, 0])
            if action == 1:
                reward = -100 #Discourage holding till the end
            
        return observation,reward,  done, truncated, {}

    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.energy_step = 0
        self.balance = 0

        self.wattage_rate = 0
        self.power_day = 0
        for i in range(48):
            vimp = self.df['Vmp'][self.current_step]
            imp = self.df['Imp'][self.current_step]
            #Account for weekend
            yearday = self.current_step
            dayahead = pd.to_datetime( self.energy_dates[self.energy_step+1] )
            start = pd.to_datetime( self.start_date )

            day_diff = (dayahead - start).days
            if day_diff  < yearday / 48:
                start = dayahead
                self.energy_step += 1
            self.wattage_rate = self.energy_prices[self.energy_step]/1000

            
            kilo_watts = self.get_wattage(vimp, imp)/1000#Lets assum Kilo Watts for now
            self.power_day += kilo_watts
            self.current_step += 1
        #Day of the year, price, wattage stored, power generated in the day
        observation = np.array([self.current_step//48, self.wattage_rate, self.wattage_balance, self.power_day])
        return observation,{}


    def render(self, mode='human'):
        log_val = 100
        # x_data = range(0, min(len(self.actions), log_val))
        # self.ax[0][0].clear()
        # self.ax[0][1].clear()
        # self.ax[1][0].clear()
        # self.ax[1][1].clear()

        # self.ax[0][0].set_xlim(min(x_data), max(x_data))
        # self.ax[0][0].set_ylim(min(self.actions[-log_val:]), max(self.actions[-log_val:]))
        # self.ax[0][0].plot(x_data, self.actions[-log_val:])

        # self.ax[0][1].set_xlim(min(x_data), max(x_data))
        # self.ax[0][1].set_ylim(min(self.vimp[-log_val:]), max(self.vimp[-log_val:]))
        # self.ax[0][1].plot(x_data, self.vimp[-log_val:])

        # self.ax[1][0].set_xlim(min(x_data), max(x_data))
        # self.ax[1][0].set_ylim(min(self.imp[-log_val:]), max(self.imp[-log_val:]))
        # self.ax[1][0].plot(x_data, self.imp[-log_val:])

        # self.ax[1][1].set_xlim(min(x_data), max(x_data))
        # self.ax[1][1].set_ylim(min(self.rewards[-log_val:]), max(self.rewards[-log_val:]))
        # self.ax[1][1].plot(x_data, self.rewards[-log_val:])

        # self.fig.canvas.draw()
        # plt.pause(1e-40)
        
gym.register(id='SolarEnv-v0', entry_point=SolarEnv)