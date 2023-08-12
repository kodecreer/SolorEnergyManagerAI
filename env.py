import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import torch
import torch.nn as nn
from collections import defaultdict

import matplotlib.pyplot as plt
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn

import torchrl
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (
    Compose,
    DoubleToFloat,
    ObservationNorm,
    StepCounter,
    TransformedEnv,
)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm

#TASK List Sprint 1 ##########
#TODO Create a dummy model   #
#TODO Make sure able to train#
#TODO Fix graph              #
##############################
#TASK List Sprint 2 ##########
#TODO Incorporate into TorchRL for parrallel envs
#TODO Make graphing more efficient. Slows down for some reason after a while for no reason
##############################
device = "cpu" if not torch.has_cuda else "cuda:0"
num_cells = 256  # number of cells in each layer i.e. output dim.
lr = 3e-4
max_grad_norm = 1.0
frame_skip = 1
frames_per_batch = 1000 // frame_skip
# For a complete training, bring the number of frames up to 1M
total_frames = 10_000 // frame_skip
sub_batch_size = 64  # cardinality of the sub-samples gathered from the current data in the inner loop
num_epochs = 10  # optimisation steps per batch of data collected
clip_epsilon = (
    0.2  # clip value for PPO loss: see the equation in the intro for more context.
)
gamma = 0.99
lmbda = 0.95
entropy_eps = 1e-4


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

    def __init__(self):
        super(SolarEnv, self).__init__()
        data = pd.read_excel('dummy_data.xlsx')
        # print(df.head())
        self.seed = torch.random.initial_seed()
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
            low=0, high=np.inf, shape=(1,12), dtype=np.float32
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

        if action == 1 and len(self.actions) > 0:
            #Sell
            new_balance = self.actions[-1] + reward
            self.actions.append(new_balance)
        else:
            #hold
            value = self.actions[-1] if len(self.actions) > 0 else 0
            self.actions.append(value)

        #next observation
        return self._next_observation(), reward, self.current_step >= len(self.df)-1, 0

    def reset(self, seed=0):
        # Reset the environment to the initial state and return the observation
        self.current_step = 0
        self.actions = []
        self.vimp = []
        self.seed = seed
      
        return self._next_observation()

    def _next_observation(self):
        obs_data = self.df.iloc[self.current_step]
        observation = TensorDict({
            'observation' : torch.tensor(obs_data.values).unsqueeze(0)
            },
            batch_size=1  # Set the batch size to 1 for a single observation
        )
        return observation


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
    
    gym.register(id='SolarEnv-v0', entry_point='env:SolarEnv')
    env = GymEnv('SolarEnv-v0', device=device, frame_skip=frame_skip)
    env = TransformedEnv(
        env,
        Compose(
            # normalize observations
            ObservationNorm(in_keys=["observation"]),
            DoubleToFloat(
                in_keys=["observation"],
            ),
            StepCounter(),
        ),
    )
    env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)
    print("normalization constant shape:", env.transform[0].loc.shape)
    print("observation_spec:", env.observation_spec)
    print("reward_spec:", env.reward_spec)
    print("done_spec:", env.done_spec)
    print("action_spec:", env.action_spec)
    # print("state_spec:", env.state_spec)
    check_env_specs(env)

    actor_net = nn.Sequential(
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(2 * env.action_spec.shape[-1], device=device),
    NormalParamExtractor(),
    )
    policy_module = TensorDictModule(
    actor_net, in_keys=["observation"], out_keys=["loc", "scale"]
    )
    policy_module = ProbabilisticActor(
    module=policy_module,
    spec=env.action_spec,
    in_keys=["loc", "scale"],
    distribution_class=TanhNormal,
    # distribution_kwargs={
    #     "min": env.action_spec.,
    #     "max": env.action_spec.space.maximum,
    # },
    return_log_prob=True,
    # we'll need the log-prob for the numerator of the importance weights
    )
    value_net = nn.Sequential(
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(1, device=device),
    )

    value_module = ValueOperator(
        module=value_net,
        in_keys=["observation"],
    )
    print("Running policy:", policy_module(env.reset()))
    print("Running value:", value_module(env.reset()))
    collector = SyncDataCollector(
    env,
    policy_module,
    frames_per_batch=frames_per_batch,
    total_frames=total_frames,
    split_trajs=False,
    device=device,
    )
    replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(frames_per_batch),
    sampler=SamplerWithoutReplacement(),
    )
    advantage_module = GAE(
    gamma=gamma, lmbda=lmbda, value_network=value_module, average_gae=True
)

    loss_module = ClipPPOLoss(
    actor=policy_module,
    critic=value_module,
    clip_epsilon=clip_epsilon,
    entropy_bonus=bool(entropy_eps),
    entropy_coef=entropy_eps,
    # these keys match by default but we set this for completeness
    value_target_key=advantage_module.value_target_key,
    critic_coef=1.0,
    gamma=0.99,
    loss_critic_type="smooth_l1",
    )

    optim = torch.optim.Adam(loss_module.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, total_frames // frames_per_batch, 0.0
    )
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
