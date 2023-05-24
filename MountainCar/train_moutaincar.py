#%%
import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as option

import numpy as np
import gymnasium as gym

import settings
from replay_buffer import ReplayBuffer

#make Generator
rng = np.random.default_rng()

#make environment
env = gym.make('MountainCarContinuous-v0')
obs_num = env.observation_space.shape[0]
act_num = env.action_space.shape[0]

#make Neural Network
class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(obs_num, settings.HIDDEN_SIZE)
        self.fc2 = nn.Linear(settings.HIDDEN_SIZE, settings.HIDDEN_SIZE)
        self.fc3 = nn.Linear(settings.HIDDEN_SIZE, settings.HIDDEN_SIZE)
        self.fc4 = nn.Linear(settings.HIDDEN_SIZE, act_num)
        
    def forward(self, x):
        x.to(settings.device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        y = F.relu(self.fc4(x))
        return y
        
def main():
    Q_train = NN()
    Q_target = NN()
    optimizer = option.RMSprop(Q_train.parameters(), lr=0.00015, alpha=0.95, eps=0.01)  #最適化
    
    toral_step = 0
    memory = ReplayBuffer(settings.BAFFER_SIZE)
# %%
