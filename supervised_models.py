import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np 
from datetime import datetime
import pandas as pd
import random
class SolarDataset(Dataset):
    '''Dataset for autorgressive manager'''
    def __init__(self, dates, prices):
        self.x_data = dates
        self.y_data = prices
        # self.ctx_len = ctx_len
    def __len__(self):
        return len(self.x_data)
    def __getitem__(self, index):
        # Create a numpy datetime64 object
        np_date = self.x_data[index]

        # Convert to a Python datetime object
        py_date = pd.Timestamp(np_date).to_pydatetime()

        # Extract the day of the year
        day_of_year = py_date.timetuple().tm_yday + min(random.random(), 0.99)
        return torch.tensor([day_of_year]).float(), torch.tensor([self.y_data[index]]).float()
    
#TODO add attention to make it a transformer
class Expert(nn.Module):
    def __init__(self, in_dims, out_dim, mid=64): #Default is 512

        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dims, mid),
            nn.ReLU(),
            nn.LayerNorm(mid),
            nn.Dropout(0.2),
            nn.Linear(mid, mid),
            nn.ReLU(),
            nn.LayerNorm(mid),
            nn.Linear(mid, mid),
            nn.ReLU(),
            nn.LayerNorm(mid),
            nn.Dropout(0.2),
            nn.Linear(mid, out_dim)
        )
    def forward(self, x):
        return self.mlp(x)

class GateNetwork(nn.Module):
    def __init__(self, in_dims, mid=2048,num_experts=2):
        super().__init__()

        self.gate_net = nn.Sequential(
            nn.Linear(in_dims, mid ),
            nn.ReLU(),
            nn.LayerNorm(mid),
            nn.Linear(mid, mid),
            nn.ReLU(),
            nn.LayerNorm(mid),
            nn.Linear(mid, num_experts),
            nn.Dropout(0.5),
            nn.Softmax(dim=-1)
        )


    def forward(self, x):
        return self.gate_net(x)
class DAE(nn.Module):
    def __init__(self, in_dims, out_dims):
        super().__init__()
        self.a = nn.Linear(in_dims, out_dims)
        self.vt = nn.Linear(in_dims, out_dims, bias=False)
    def forward(self, x):
        bxvt = F.tanh(self.vt(x))
        return self.a(x) * (1/ torch.cos(bxvt))**2 + torch.sin(bxvt/torch.cos(bxvt))
class MOE(nn.Module):
    def __init__(self, in_dims, out_dims=1, num_experts=6):
        super().__init__()
        mid_dim = 128
        self.embed = DAE(in_dims, mid_dim)#nn.Linear(in_dims, mid_dim)#nn.Embedding(366, mid_dim)#
        self.experts = nn.ModuleList([Expert(mid_dim, in_dims) for _ in range(num_experts)])
        self.gate = GateNetwork(mid_dim, num_experts=num_experts)
        self.fc = nn.Linear(1, out_dims, bias=False)
    def forward(self, x):
        expert_ans = []
        x = self.embed(x)
        # b, m, d = x.size()
        # x = x.view(b, -1)
        for expert in self.experts:
            ans = expert(x)
            expert_ans.append(ans)
        
        gate_ans = self.gate(x)
        # x = expert_ans[0] * gate_ans[:, 0] + expert_ans[1] * gate_ans[:, 1]
        x =  [expert_ans[i] * gate_ans[:, i].unsqueeze(1) for i in range(len(self.experts))]
        
        x = sum(x)
        x = x.sum(dim=-1).unsqueeze(1)
        x = self.fc(x)
        return x

class MOE_Agent():
    def __init__(self, state_dict_pth, window = 10):
        self.agent = MOE(1)
        self.window = window
        self.agent.load_state_dict(torch.load(state_dict_pth))
    @torch.no_grad
    def choose_action(self, day):
        prices = []
        for i in range(self.window):
            price = self.agent(day+i)
            prices.append(price[0].item())
        bp = max(prices)
        idx = prices.index(bp)
        return idx
        
