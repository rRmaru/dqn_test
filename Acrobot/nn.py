import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
from torch.autograd import Variable

HIDDEN_SIZE = 100

class NN(nn.Module):
    def __init__(self, obs_num, action_num):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(obs_num, HIDDEN_SIZE)  #線形に変換、一般的には全結合層
        self.fc2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc3 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc4 = nn.Linear(HIDDEN_SIZE, action_num)
        
    def forward(self, input):       #活性化関数の定義
        h = F.relu(self.fc1(input))     #出力＝活性化関数（第ｎ層（入力））の形式で記述
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        output = F.relu(self.fc4(h))
        return output
    
class DQN(NN):
    def __init__(self, obs_num, action_num, optimizer=None, criterion=None):
        super(DQN, self).__init__(obs_num, action_num)
        self.optimizer = optimizer
        self.criterion = criterion
        