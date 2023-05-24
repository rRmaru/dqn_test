#%%
import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as option
from torch.autograd import Variable

import numpy as np
import gymnasium as gym
import time

import settings
from replay_buffer import ReplayBuffer

#make Generator
rng = np.random.default_rng()

#make environment
env = gym.make('MountainCar-v0')
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
        return y.to(settings.device)
        
def main():
    Q_train = NN()
    Q_target = NN()
    optimizer = option.RMSprop(Q_train.parameters(), lr=0.00015, alpha=0.95, eps=0.01)  #最適化
    
    toral_step = 0
    memory = ReplayBuffer(settings.BAFFER_SIZE)
    tortal_reward = []
    
    #学習開始
    print("\t".join(["episode", "epsilon", "reward", "total_step", "time"]))
    start = time.time()
    
    for episode in range(settings.EPISODE_NUM):
        pobs, _ = env.reset()
        step = 0        #step
        done = False    #judge end game
        tortal_reward = 0   #累積報酬
        print(pobs)
        while not done and (step < settings.STEP_MAX):
            #行動選択(適当な行動値)
            act = env.action_space.sample()
            # ε-greedy法
            if rng.random() < settings.EPSILON:
                pobs_ = np.array(pobs, dtype="float32").reshape((1, obs_num))
                pobs_ = Variable(torch.from_numpy(pobs_))
                act = Q_train(pobs_)
                max, indices = torch.max(act.data, 1)  #valueとindicesが返ってくる
                act = indices.detach().cpu().numpy()[0]

            #実行
            obs, reward, done, _, _ = env.step(act)

            break

if __name__ == "__main__":
    main()            
# %%
