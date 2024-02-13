import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm

import sys
import env
from models import HyperParameterConfig, Agent

from supervised_models import *
if __name__ == '__main__':

    observation_ds = []
    # Define the number of episodes (or time steps) you want to run the environment
    num_episodes = 1


    
    # torch.set_default_device('cuda')
    energy_prices = pd.read_excel('Energy Price_2016.xlsx')
    energy_prices = energy_prices[energy_prices["Price hub"].replace(' ', '') == 'Mid C Peak']
    energy_prices = energy_prices[1:]#because it has January 30th
    energy_dates = energy_prices['Trade date'].values
    energy_prices = energy_prices['Avg price $/MWh'].values
    #Now put it into a dataset
    total_set = SolarDataset(energy_dates, energy_prices)
    total_size = len(total_set)
    test_size = int(total_size * 0.2)

    # train_set = Subset(total_set, range(total_size-test_size))
    # test_set = Subset(total_set, range(total_size-test_size, total_size))
    train_set, test_set = random_split(total_set, [total_size-test_size, test_size])
    bsize = 5
    train_loader = DataLoader(train_set, bsize, drop_last=True)
    test_loader = DataLoader(test_set, bsize, drop_last=True)
    
    date, price = test_set[0]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    assert device == 'cuda'

    epochs = 300
    lr = 1e-4
    criterion = nn.MSELoss()
    obs_size = 1
    model = MOE(obs_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, 0.1)
    loop = tqdm(range(epochs))
    for epoch in loop:
        total_loss = 0
        total_passes = 0

        val_loss = 0
        val_passes = 1
        for x_data, y_data in train_loader:
            optimizer.zero_grad()
            x_data = x_data.to(device)
            y_data = y_data.to(device)
            y_pred = model(x_data)
            
            loss = criterion(y_pred, y_data)
            loss.backward()
            optimizer.step()
     
            total_loss += loss.item()
            total_passes += 1
        scheduler.step()
        loop.set_description_str(f'Train loss: {total_loss/total_passes} RMSE loss: {val_loss / val_passes}')
        with torch.no_grad():
            for x_data, y_data in test_loader:

                x_data = x_data.to(device)
                y_data = y_data.to(device)

                y_pred = model(x_data)
                loss = torch.sqrt( F.mse_loss(y_pred, y_data) )
                val_loss += loss.item()
                loop.set_description_str(f'Train loss: {total_loss/total_passes} RMSE loss: {val_loss / val_passes}')
                val_passes += 1

    torch.save(model.state_dict(), './shse_emb.mdl' )


