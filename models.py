import torch.nn as nn
import os
import torch.optim as optim
import torch
from torch.distributions import Categorical
import numpy as np
import torch.nn.functional as F
import math
class HyperParameterConfig():
    num_epochs = 10
    num_steps = 2048
    batch_size = 64
    clip_epsilon = 0.2
    value_coeff = 0.5
    entropy_coeff = 0.01
    learning_rate = 0.1
    #For GAE
    gamma = 0.99
    gae_lambda = 0.95



class AIModel(nn.Module):
    def __init__(self, alpha, 
            chkpt_dir='tmp'):
        super(AIModel, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo.mdl')
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self = self.to(self.device)
        self.alpha = alpha
    def init(self):
        self.optimizer = optim.Adam(self.parameters(), lr=self.alpha)

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

class ActorNetwork(AIModel):
    def __init__(self, n_actions, input_dims, alpha,
            fc1_dims=256, fc2_dims=256, chkpt_dir='tmp'):
        super(ActorNetwork, self).__init__(alpha)

        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo.mdl')
        self.actor = nn.Sequential(
                nn.Linear(input_dims, fc1_dims),
                nn.ReLU(),
                nn.Linear(fc1_dims, fc2_dims),
                nn.ReLU(),
                nn.Linear(fc2_dims, n_actions),
                nn.Softmax(dim=-1)
        )
        self.init()

    def forward(self, state):
        dist = self.actor(state)
        dist = Categorical(dist)
        
        return dist



class CriticNetwork(AIModel):
    def __init__(self, input_dims, alpha, fc1_dims=256, fc2_dims=256,
            chkpt_dir='tmp'):
        super(CriticNetwork, self).__init__(alpha)

        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo.mdl')
        self.critic = nn.Sequential(
                nn.Linear(input_dims, fc1_dims),
                nn.ReLU(),
                nn.Linear(fc1_dims, fc2_dims),
                nn.ReLU(),
                nn.Linear(fc2_dims, 1)
        )
        self.init()


    def forward(self, state):
        value = self.critic(state)

        return value


class ActorRNN(AIModel):
    def __init__(self, n_actions, input_dims, alpha, chkpt_dir='tmp'):
        super(ActorRNN, self).__init__(alpha)

        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo.mdl')
        self.actor = nn.GRU(input_dims, n_actions)
        self.hidden = None
        self.init()

    def forward(self, state):

        dist, hidden = self.actor(state, self.hidden)
        dist = Categorical(F.softmax(dist, dim=-1))
        self.hidden = hidden.detach().clone()

        return dist


class CriticRNN(AIModel):
    def __init__(self, input_dims, alpha, fc1_dims=256, fc2_dims=256,
            chkpt_dir='tmp'):
        super(CriticRNN, self).__init__(alpha)

        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo.mdl')
        self.critic = nn.GRU(input_dims, 1)
        self.hidden = None
        self.init()

    def forward(self, state):
        value, hidden = self.critic(state, self.hidden)
        self.hidden = hidden.detach().clone()
        return value


class ActorCNN(AIModel):
    def __init__(self, n_actions, input_dims, alpha, fc1_dims=256, fc2_dims=256, chkpt_dir='tmp'):
        super(ActorCNN, self).__init__(alpha)
        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo.mdl')
        self.actor = nn.Sequential( 
            nn.Conv1d(1, 32, kernel_size=1),
            nn.ConvTranspose1d(32, 1, kernel_size=1)
        )
        self.fc = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            nn.Linear(fc1_dims, fc2_dims),
            nn.Linear(fc2_dims, n_actions)
        ) 

        self.init()
        


    def forward(self, state):
        obs = state.unsqueeze(1) #To avoid accidental modification
        out = self.actor(obs )
        out = out.squeeze(1)
        out = self.fc(out )
        
        dist = Categorical(F.softmax(out, dim=-1))


        return dist



class CriticCNN(AIModel):
    def __init__(self, input_dims, alpha, fc1_dims=256, fc2_dims=256,
            chkpt_dir='tmp'):
        super(CriticCNN, self).__init__(alpha)

        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo.mdl')
        self.critic = nn.Sequential( 
            nn.Conv1d(1, 32, kernel_size=1),
            nn.ConvTranspose1d(32, 1, kernel_size=1)
        )
        self.net = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            nn.Linear(fc1_dims, fc2_dims),
            nn.Linear(fc2_dims, 1)
        )

        self.init()

    def forward(self, state):
        obs = state.unsqueeze(1)
        value = self.critic(obs)
        value = value.squeeze(1)
        value = self.net(value )
        return value

class PositionalEncoding(nn.Module):
    #Citation of encoder from. Minorly adjusted to not use math library 
    # https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
class ActorT(AIModel):
    def __init__(self, n_actions, input_dims, alpha,fc1_dims=256, fc2_dims=256, num_heads=2, chkpt_dir='tmp'):
        super(ActorT, self).__init__(alpha)

        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo.mdl')
        self.actor = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(fc2_dims, n_actions)
        ) #Feed forward
        self.pos_shape = nn.Linear(input_dims, input_dims*2)
        self.reform = nn.Linear(input_dims*2, n_actions)
        self.pos = PositionalEncoding(input_dims*2)
        self.ln1 = nn.LayerNorm(n_actions) #Layer norm 1
        self.attn = nn.MultiheadAttention(n_actions, num_heads )
        self.ln2  = nn.LayerNorm(n_actions)
        self.fc = nn.Linear(n_actions, n_actions, bias=False)#Final layer to emulate diagram
        self.memory = [] #Keep a memory context for future use and generate auto-regressively
        self.ctx_len = 32
        self.init()


    def forward(self, state):
        obs = state
        pos =  self.pos(self.pos_shape(obs)) 
        pos = self.reform(pos)
        x1 = self.actor(obs) * pos
        x2, _ = self.attn(x1, x1, x1)
        x3 = self.ln1(x1 + x2)
        
        if len(self.memory) > 0:
            mem = torch.stack(self.memory, dim=0)
            x3 = self.ln2(mem+x3)
        dist = self.fc(x3)
        dist = Categorical(F.softmax(dist, dim=-1))
        if len(self.memory) + 1 >= self.ctx_len:
            del self.memory[0]
        self.memory.append(x2[-1])

        return dist


class CriticT(AIModel):
    def __init__(self, input_dims, alpha, fc1_dims=256, fc2_dims=256, num_heads=2,
            chkpt_dir='tmp'):
        super(CriticT, self).__init__(alpha)
        self.actor = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(fc1_dims, fc2_dims),
           
        ) #Feed forward
        self.pos_shape = nn.Linear(input_dims, input_dims*2)
        self.reform = nn.Linear(input_dims*2, fc2_dims)
        self.pos = PositionalEncoding(input_dims*2)
        self.ln1 = nn.LayerNorm(fc2_dims) #Layer norm 1
        self.attn = nn.MultiheadAttention(fc2_dims, num_heads )
        self.ln2  = nn.LayerNorm(fc2_dims)
        self.fc = nn.Linear(fc2_dims, 1, bias=False)#Final layer to emulate diagram
        self.memory = [] #Keep a memory context for future use and generate auto-regressively
        self.ctx_len = 32
        self.init()


    def forward(self, state):
        obs = state
        pos =  self.pos(self.pos_shape(obs)) 
        pos = self.reform(pos)
        x1 = self.actor(obs) * pos
        x2, _ = self.attn(x1, x1, x1)
        x3 = self.ln1(x1 + x2)
        
        if len(self.memory) > 0:
            mem = torch.stack(self.memory, dim=0)
            x3 = self.ln2(mem+x3)
        value = self.fc(x3)
        if len(self.memory) + 1 >= self.ctx_len:
            del self.memory[0]
        self.memory.append(x2[-1])

        return value
    

class Agent:
    def __init__(self, n_actions, HyperParams: HyperParameterConfig) -> None:
        self.actor = ActorNetwork(n_actions, 13, 0.0001)
        print(self.actor.device)
        self.device = self.actor.device
        self.critic = CriticNetwork(13, 0.0001)
        self.memory = MemoryBuffer(100)
        self.config = HyperParams
    
    def choose_action(self, observation):
        observation = torch.tensor(observation, dtype=torch.float).to(self.device)
        dist = self.actor(observation)
        value = self.critic(observation).to(self.device)
        action = dist.sample()
        probs = dist.log_prob(action).to(self.device)
        return action, probs, value
    
    #TODO Impliment PPO Algorithm in Pytorch. Vectorize if possible without referencing my Pokemon project!!!
    def clipped_ppo(self):
        #Run PPO Algorithm
        for _ in range(self.config.num_epochs):
            # Calculate General Advantage Estimate
            state_arr, action_arr, old_prob_arr, vals_arr,\
            reward_arr, dones_arr, batches = \
                    self.memory.sample()
            
            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.config.gamma*values[k+1]*\
                            (1-int(dones_arr[k])) - values[k])
                    discount *= self.config.gamma * self.config.gae_lambda
                advantage[t] = a_t
            advantage = torch.tensor(advantage)
            values = torch.tensor(values)
            for batch in batches:
                states = torch.tensor(state_arr[batch], dtype=torch.float).clone().detach().requires_grad_(True)
                old_probs = torch.tensor(old_prob_arr[batch], dtype=torch.float).clone().detach().requires_grad_(True)
                actions = torch.tensor(action_arr[batch], dtype=torch.float).clone().detach().requires_grad_(True)
                
                distributions: torch.distributions.Categorical = self.actor(states)
                critic_value = self.critic(states)
                new_probs = distributions.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                weighted_probs = prob_ratio * advantage[batch]

                weighted_clip_probs = torch.clamp(prob_ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * advantage[batch]
                actor_loss = -torch.min(weighted_probs, weighted_clip_probs) #negative so it can get added together
                gains = advantage[batch] + values[batch]
                critic_loss = (gains - critic_value)**2
                critic_loss = critic_loss.mean()

                total_loss = critic_loss - actor_loss *0.5 #Add a weight 
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()
        self.memory.clear() 
    def vectorized_clipped_ppo(self):
        #Run PPO Algorithm
        for _ in range(self.config.num_epochs):
            # Calculate General Advantage Estimate
            state_arr, action_arr, old_prob_arr, vals_arr,\
            reward_arr, dones_arr, batches = \
                    self.memory.sample()
            
            values = vals_arr
            advantage = torch.zeros(len(reward_arr)).to(self.device)
            reward_arr = torch.tensor(reward_arr).to(self.device)
            dones_arr = torch.tensor(dones_arr, dtype=torch.int8).to(self.device)
            values = torch.tensor(values).to(self.device)
            
            for t in range(reward_arr.size(0)-1):
                init_discount = 1.0
                exps = torch.arange(t+1, reward_arr.size(0)).to(self.device)
                discount = torch.pow( init_discount * self.config.gamma * self.config.gae_lambda, exps - t - 1).to(self.device)

                a_t = torch.zeros(t, len(reward_arr)-1).to(self.device)
                intermediate = reward_arr[t:-1] + self.config.gamma * values[t+1:] * (1-dones_arr[t:-1]) - values[t:-1]
                a_t = torch.sum(discount *intermediate).to(self.device)
                advantage[t] = a_t

            # advantage = torch.tensor(advantage)
            # values = torch.tensor(values).to(self.device)
            states = []
            old_probs = []
            actions = []
            for batch in batches:
                states.append( state_arr[batch] )
                old_probs.append( old_prob_arr[batch] )
                actions.append(action_arr[batch])
            states = torch.tensor(states).to(self.device)
            old_probs = torch.tensor(old_probs).view(-1, 1).to(self.device)
            actions = torch.tensor(actions).view(-1, 1).to(self.device)

            distributions: torch.distributions.Categorical = self.actor(states)
            critic_value = self.critic(states).to(self.device)
            new_probs = distributions.log_prob(actions).to(self.device)
            prob_ratio = new_probs.exp() / old_probs.exp()
            weighted_probs = prob_ratio * advantage[batch]

            weighted_clip_probs = torch.clamp(prob_ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon).to(self.device)\
                  * advantage[batch]
            actor_loss = -torch.min(weighted_probs, weighted_clip_probs).mean().to(self.device) #negative so it can get added together
            gains = advantage + values
            critic_loss = (gains - critic_value)**2
            critic_loss = critic_loss.mean()

            total_loss = critic_loss - actor_loss *0.5 #Add a weight 
            self.actor.optimizer.zero_grad()
            self.critic.optimizer.zero_grad()

            total_loss.backward()
            self.actor.optimizer.step()
            self.critic.optimizer.step()
            if isinstance(self.actor, ActorT):
                self.actor.memory.clear()
                self.critic.memory.clear()
        self.memory.clear() 

class AgentRNN(Agent):
    def __init__(self, n_actions, HyperParams: HyperParameterConfig) -> None:
        self.actor = ActorRNN(n_actions, 13, 0.0001)
        print(self.actor.device)
        self.device = self.actor.device
        self.critic = CriticRNN(13, 0.0001)
        self.memory = MemoryBuffer(100)
        self.config = HyperParams
    def reset(self):
        self.actor.hidden = None
        self.critic.hidden = None
class AgentCNN(Agent):
    def __init__(self, n_actions, HyperParams: HyperParameterConfig) -> None:
        self.actor = ActorCNN(n_actions, 13, 0.0001)
        print(self.actor.device)
        self.device = self.actor.device
        self.critic = CriticCNN(13, 0.0001)
        self.memory = MemoryBuffer(100)
        self.config = HyperParams
    def reset(self):
        self.actor.hidden = None
        self.critic.hidden = None
class AgentT(Agent):
    def __init__(self, n_actions, HyperParams: HyperParameterConfig) -> None:
        self.actor = ActorT(n_actions, 13, 0.0001)
        print(self.actor.device)
        self.device = self.actor.device
        self.critic = CriticT(13, 0.0001)
        self.memory = MemoryBuffer(100)
        self.config = HyperParams
    def vectorized_clipped_ppo(self):
        self.actor.memory.clear()
        self.critic.memory.clear()
        super().vectorized_clipped_ppo()


#TODO define a clear method from the buffer.
# Implement the memory buffer
class MemoryBuffer:
    def __init__(self, BSize):
    
        self.states = []
        self.actions = []
        self.probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.batch_size = BSize


    def push(self, state, action, prob, value, reward, done):
        self.states.append( state)
        self.actions.append( action)
        self.probs.append( prob)
        self.values.append( value)
        self.rewards.append( reward)
        self.dones.append( done)
        

    def sample(self):
        #Get the indicies sampled. 
        #Do not mutate the original array
        #TODO force it to make it make batch size match the values stored size
        indicies = np.random.choice(np.arange(len(self.states)), size=self.batch_size)
        return self.states, self.actions, self.probs, self.values, self.rewards, self.dones, indicies
    def clear(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.values = []
        self.rewards = []
        self.dones = []
    def size(self):
        return len(self.states)
    
        
