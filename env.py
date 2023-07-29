import gymnasium as gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import torch
import torch.nn as nn

#TASK List Sprint 1 ##########
#TODO Create a dummy model   #
#TODO Make sure able to train#
#TODO Fix graph              #
##############################
#TASK List Sprint 2 ##########
#TODO Incorporate into TorchRL for parrallel envs
#TODO Make graphing more efficient. Slows down for some reason after a while for no reason
##############################

class SolarAI(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SolarAI, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.activation = nn.ReLU()  # You can choose any activation function you prefer
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Define the forward pass of your model
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x
class SolarEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, data, initial_balance=10000):
        super(SolarEnv, self).__init__()

        self.fig, self.ax = plt.subplots(2,2)
        self.action_line,  = self.ax[0][0].plot([0], [0])
        self.ax[0][0].set_xlabel('Timestep')
        self.ax[0][0].set_ylabel('Balance')

        self.vmp_line,  = self.ax[0][1].plot([0], [0])
        self.ax[0][1].set_xlabel('Timestep')
        self.ax[0][1].set_ylabel('VMP')

        self.imp_line,  = self.ax[1][0].plot([0], [0])
        self.ax[1][0].set_xlabel('Timestep')
        self.ax[1][0].set_ylabel('IMP')

        self.reward_line,  = self.ax[1][1].plot([0], [0])
        self.ax[1][1].set_xlabel('Timestep')
        self.ax[1][1].set_ylabel('Reward')
        plt.ion()
        plt.show(block=False)

        # Define your stock data, e.g., OHLC (Open, High, Low, Close) prices
        self.df = data

        # Define the action and observation spaces
        self.action_space = spaces.Discrete(3)  # Example: Sell, Hold, Buy
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(len(self.df),), dtype=np.float32
        )

        self.current_step = 0
        self.vimp = []
        self.imp = []
        self.actions = []
        self.rewards = []

    
    def calc_reward(self, balance):
        #Price for rewards calcualtion
        #Constant rate = 1.0
       
        return balance
        
    def get_price(self, vmp, imp, wattage_rate):
         #Convert VIMP into wattage
        # Wattage * Constant = reward
        #Wattage (Pmax) = Vmp × Imp
        #Implement logit for profit. This is good for now though
        pmax = vmp * imp
        reward = wattage_rate * pmax
        return reward
    def step(self, action):
        # Implement the logic for one step in the environment
        # Perform the action (Buy, Sell, Hold), update balance, stock owned, etc.
        # Calculate reward and whether the episode is done
        
        vimp = self.df['Vmp'][self.current_step]
        imp = self.df['Imp'][self.current_step]
        self.vimp.append(vimp)
        self.imp.append(imp)
        WATTAGE_RATE = 0.5 #Random number for 50 cents per wattage hour. EXPENSIVE!!!!

        reward = self.get_price(vimp, imp, WATTAGE_RATE )
        self.rewards.append(reward)

        if action == 1:
            #Sell
            new_balance = self.actions[-1] + reward
            self.actions.append(new_balance)
        else:
            #hold
            value = self.actions[-1] if len(self.actions) > 0 else 0
            self.actions.append(value)

        #next observation
        return (), reward, self.current_step >= len(self.df)-1, 0

    def reset(self):
        # Reset the environment to the initial state and return the observation
        self.current_step = 0
        self.actions = []
        self.vimp = []
      
        return self._next_observation()

    def _next_observation(self):
        # Generate the observation for the current step
        # return np.array([self.data[self.current_step]])
        first_value = self.df.iloc[0]
        return np.array(first_value)

    def render(self, mode='human'):
        log_val = 100
        x_data = range(0, min(len(self.actions), log_val))
        self.action_line.set_data(x_data, self.actions[-log_val:])
        self.vmp_line.set_data(x_data, self.vimp[-log_val:])
        self.imp_line.set_data(x_data, self.imp[-log_val:])
        self.reward_line.set_data(x_data, self.rewards[-log_val:])

        # Manually set the x-axis and y-axis limits
        self.ax[0][0].set_xlim(min(x_data), max(x_data))
        self.ax[0][0].set_ylim(min(self.actions[-log_val:]), max(self.actions[-log_val:]))

        self.ax[0][1].set_xlim(min(x_data), max(x_data))
        self.ax[0][1].set_ylim(min(self.vimp[-log_val:]), max(self.vimp[-log_val:]))

        self.ax[1][0].set_xlim(min(x_data), max(x_data))
        self.ax[1][0].set_ylim(min(self.imp[-log_val:]), max(self.imp[-log_val:]))

        self.ax[1][1].set_xlim(min(x_data), max(x_data))
        self.ax[1][1].set_ylim(min(self.rewards[-log_val:]), max(self.rewards[-log_val:]))

        self.fig.canvas.draw()
        plt.pause(1e-40)
        


if __name__ == '__main__':
    df = pd.read_excel('dummy_data.xlsx')
    print(df.head())
    env = SolarEnv(df, 0)
    # Reset the environment to its initial state and receive the initial observation
    observation = env.reset()

    # Define the number of episodes (or time steps) you want to run the environment
    num_episodes = 1
    log_interval = 90
    interval = 0

    for episode in range(num_episodes):
        observation = env.reset()
        done = False
        while not done:
            # Replace 'your_action' with the action you want to take in the environment (e.g., 0, 1, 2, ...)
            action =  env.action_space.sample()

            # Perform the chosen action in the environment
            next_observation, reward, done, _ = env.step(action)
            interval += 1
            if interval >= log_interval:
                # You can render the environment at each step if you want to visualize the progress
                env.render()
                interval = 0

            # Update the current observation with the next observation
            observation = next_observation
            env.current_step += 1
        

    # Close the environment when done
    env.close()
