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
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(12,), dtype=np.float32
        )

        self.current_step = 0
        self.vimp = []
        self.imp = []
        self.actions = []
        self.rewards = []

    def calc_reward(self, balance):
        return balance
        
    def get_price(self, vmp, imp, wattage_rate):
        pmax = vmp * imp
        reward = wattage_rate * pmax
        return reward

    def step(self, action):
        vimp = self.df['Vmp'][self.current_step]
        imp = self.df['Imp'][self.current_step]
        self.vimp.append(vimp)
        self.imp.append(imp)
        WATTAGE_RATE = 0.5

        reward = self.get_price(vimp, imp, WATTAGE_RATE)
        self.rewards.append(reward)

        if action == 1 and len(self.actions) > 0:
            new_balance = self.actions[-1] + reward
            self.actions.append(new_balance)
        else:
            value = self.actions[-1] if len(self.actions) > 0 else 0
            self.actions.append(value)

        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        observation = np.array(range(12), dtype=np.float32)
        truncated = False
        return observation,reward,  done, truncated, {}

    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.actions = []
        self.vimp = []
        self.imp = []
        observation = (np.array(range(12), dtype=np.float32))
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