import torch.nn as nn
import os
import torch.optim as optim
import torch
from torch.distributions import Categorical
import numpy as np

class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha,
            fc1_dims=256, fc2_dims=256, chkpt_dir='tmp'):
        super(ActorNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo')
        self.actor = nn.Sequential(
                nn.Linear(*input_dims, fc1_dims),
                nn.ReLU(),
                nn.Linear(fc1_dims, fc2_dims),
                nn.ReLU(),
                nn.Linear(fc2_dims, n_actions),
                nn.Softmax(dim=-1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        dist = self.actor(state)
        dist = Categorical(dist)
        
        return dist

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, fc1_dims=256, fc2_dims=256,
            chkpt_dir='tmp'):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo')
        self.critic = nn.Sequential(
                nn.Linear(*input_dims, fc1_dims),
                nn.ReLU(),
                nn.Linear(fc1_dims, fc2_dims),
                nn.ReLU(),
                nn.Linear(fc2_dims, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)

        return value

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))


class Agent:
    def __init__(self, n_actions) -> None:
        self.actor = ActorNetwork(n_actions, 256, 0.0001)
        self.critic = CriticNetwork(256, 0.0001)
        self.memory = MemoryBuffer(100)


# Implement the memory buffer
class MemoryBuffer:
    def __init__(self, BSize):
    
        self.states = np.array([])
        self.actions = np.array([])
        self.probs = np.array([])
        self.values = np.array([])
        self.rewards = np.array([])
        self.dones = np.array([])
        self.batch_size = BSize


    def push(self, state, action, prob, value, reward, done):
        self.states = np.append(self.states, state)
        self.actions = np.append(self.actions, action)
        self.probs = np.append(self.probs, prob)
        self.values = np.append(self.values, value)
        self.rewards = np.append(self.rewards, reward)
        self.dones = np.append(self.dones, done)
        

    def sample(self):
        #Get the indicies sampled. 
        #Do not mutate the original array
        #TODO force it to make it make batch size match the values stored size
        indicies = np.random.choice(np.arange(len(self.states)), size=self.batch_size)
        return self.states, self.actions, self.probs, self.values, self.rewards, self.dones, indicies
    def clear(self):
        self.states = np.array([])
        self.actions = np.array([])
        self.probs = np.array([])
        self.values = np.array([])
        self.rewards = np.array([])
        self.dones = np.array([])
        
#TODO Impliment PPO Algorithm in Pytorch. Vectorize if possible without referencing my Pokemon project!!!
def clipped_ppo(actor: torch.nn.Module, critic: torch.nn.Module, buffer: MemoryBuffer, optimizer_actor, optimizer_critic, config: HyperParameterConfig = None):
    #Run PPO Algorithm
    for _ in range(config.num_epochs):
        # Calculate General Advantage Estimate
        state_arr, action_arr, old_prob_arr, vals_arr,\
        reward_arr, dones_arr, batches = \
                buffer.sample()

        values = vals_arr
        advantage = np.zeros(len(reward_arr), dtype=np.float32)

        for t in range(len(reward_arr)-1):
            discount = 1
            a_t = 0
            for k in range(t, len(reward_arr)-1):
                a_t += discount*(reward_arr[k] + config.gamma*values[k+1]*\
                        (1-int(dones_arr[k])) - values[k])
                discount *= config.gamma* config.gae_lambda
            advantage[t] = a_t
        advantage = torch.tensor(advantage)

        for batch in batches:
            states = torch.tensor(state_arr[batch], dtype=torch.float)
            old_probs = torch.tensor(old_prob_arr[batch])
            actions = torch.tensor(action_arr[batch])
            
            distributions: torch.distributions.Categorical = actor(states)
            critic_value = critic(states)
            new_probs = distributions.log_prob(actions)
            prob_ratio = new_probs.exp() / old_probs.exp()
            weighted_probs = prob_ratio * advantage[batch]

            weighted_clip_probs = torch.clamp(prob_ratio, 1 - config.clip_epsilon, 1 + config.clip_epsilon) * advantage[batch]
            actor_loss = -torch.min(weighted_probs, weighted_clip_probs) #negative so it can get added together
            gains = advantage[batch] + values[batch]
            critic_loss = (gains - critic_value)**2
            critic_loss = critic_loss.mean()

            total_loss = critic_loss - actor_loss *0.5 #Add a weight 
            optimizer_actor.zero_grad()
            optimizer_critic.zero_grad()
            total_loss.backward()
            optimizer_actor.step()
            optimizer_critic.step()
        buffer.clear() #TODO define a clear method from the buffer.