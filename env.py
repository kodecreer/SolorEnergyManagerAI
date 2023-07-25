import gymnasium as gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
#TASK List Sprint 1 ##########
#TODO Create a dummy model   #
#TODO Make sure able to train#
#TODO Fix graph              #
##############################

class SolarEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, data, initial_balance=10000):
        super(SolarEnv, self).__init__()

        # Define your stock data, e.g., OHLC (Open, High, Low, Close) prices
        self.df = data

        # Define the action and observation spaces
        self.action_space = spaces.Discrete(3)  # Example: Sell, Hold, Buy
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(len(self.df),), dtype=np.float32
        )

        self.current_step = 0
        self.vimp = []
        self.actions = []

    
    def calc_reward(self):
        #Price for rewards calcualtion
        #Constant rate = 1.0
        #Convert VIMP into wattage
        # Wattage * Constant = reward
        pass
    def step(self, action):
        # Implement the logic for one step in the environment
        # Perform the action (Buy, Sell, Hold), update balance, stock owned, etc.
        # Calculate reward and whether the episode is done

        self.actions.append(random.randint(1, 22))
        vimp = self.df['Vmp'][self.current_step]
        self.vimp.append(vimp)
        return None, 0, False, 0

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
        # Implement the visualization of the environment using matplotlib
        # plt.figure(figsize=(10, 6))
        plt.subplot(1,2, 1)
        # Plot rewards data
        x = np.arange(len(self.actions))[-100:]
        
        plt.plot(x, self.actions[-100:], color='b')
        plt.xlabel('Time')
        plt.ylabel('Reward')
        plt.title(f'Solar Management Environment (Step: {self.current_step})')
        
        plt.pause(0.000001)
        #Show energy production
        plt.subplot(1,2,2)
        plt.plot(x, self.vimp[-100:], color='g')
        plt.xlabel('Time')
        plt.ylabel('Vimp Production')
        plt.draw()
        #Add IMP production
        #Every possible graph relating to the rewards

        plt.pause(0.000001)



if __name__ == '__main__':
    df = pd.read_excel('dummy_data.xlsx')
    print(df.head())
    env = SolarEnv(df, 0)
    # Reset the environment to its initial state and receive the initial observation
    observation = env.reset()

    # Define the number of episodes (or time steps) you want to run the environment
    num_episodes = 1

    for episode in range(num_episodes):
        observation = env.reset()
        done = False
        while not done:
            # Replace 'your_action' with the action you want to take in the environment (e.g., 0, 1, 2, ...)
            action =  env.action_space.sample()

            # Perform the chosen action in the environment
            next_observation, reward, done, _ = env.step(action)

            # You can render the environment at each step if you want to visualize the progress
            env.render()

            # Update the current observation with the next observation
            observation = next_observation
            env.current_step += 1
        

    # Close the environment when done
    env.close()
