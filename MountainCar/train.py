#%%
import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as option
from torch.autograd import Variable

import numpy as np
import gymnasium as gym
import time
import copy

import settings
from replay_buffer import ReplayBuffer

#make Generator
rng = np.random.default_rng()

#make environment
env = gym.make('MountainCar-v0')
obs_num = env.observation_space.shape[0]
act_num = env.action_space.n

#make Neural Network
class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(obs_num, settings.HIDDEN_SIZE)
        self.fc2 = nn.Linear(settings.HIDDEN_SIZE, settings.HIDDEN_SIZE)
        self.fc3 = nn.Linear(settings.HIDDEN_SIZE, settings.HIDDEN_SIZE)
        self.fc4 = nn.Linear(settings.HIDDEN_SIZE, act_num)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        y = F.relu(self.fc4(x))
        return y
        
def main():
    Q_train = NN()
    Q_target = NN()
    optimizer = option.RMSprop(Q_train.parameters(), lr=0.00015, alpha=0.95, eps=0.01)  #最適化
    
    total_step = 0
    memory = ReplayBuffer(settings.BAFFER_SIZE)
    total_rewards = []
    
    #学習開始
    print("\t".join(["episode", "epsilon", "reward", "total_step", "time"]))
    start = time.time()
    
    for episode in range(settings.EPISODE_NUM):
        pobs, _ = env.reset()
        step = 0        #step
        done = False    #judge end game
        total_reward = 0   #累積報酬
        while not done and (step < settings.STEP_MAX):
            #行動選択(適当な行動値)
            act = env.action_space.sample()
            # ε-greedy法
            if rng.random() < settings.EPSILON:
                pobs_ = np.array(pobs, dtype="float32").reshape((1, obs_num))
                pobs_ = Variable(torch.from_numpy(pobs_))
                act = Q_train(pobs_)
                max, indices = torch.max(act.data, 1)  #valueとindicesが返ってくる
                act = indices.numpy()[0]

            #実行
            obs, reward, done, _, _ = env.step(act)
            
            #add memory
            memory.add((pobs, act, reward, obs, done))  #次状態、行動、報酬、状態、エピソード終了判定をbufferに格納
            
            
            #学習
            if len(memory) == settings.BAFFER_SIZE:
                if total_step % settings.TRAIN_FREQ == 0:
                    for i in range(int(settings.BAFFER_SIZE/settings.BATCH_SIZE)):
                        batch = memory.sample(settings.BATCH_SIZE)
                        pobss = np.array([b[0] for b in batch], dtype="float32").reshape((settings.BATCH_SIZE, obs_num))
                        acts = np.array([b[1] for b in batch], dtype="float32")
                        rewards = np.array([b[2] for b in batch], dtype="float32")
                        obss = np.array([b[3] for b in batch], dtype="float32").reshape((settings.BATCH_SIZE, obs_num))
                        dones = np.array([b[4] for b in batch], dtype="float32")
                        
                        #set y
                        pobss_ = Variable(torch.from_numpy(pobss))
                        q = Q_train(pobss_)
                        obss_ = Variable(torch.from_numpy(obss))
                        maxs, indices = torch.max(Q_target(obss_).data, 1)
                        maxq = maxs.numpy() #maxQ
                        target = copy.deepcopy(q.data.numpy())
                        for j in range(settings.BATCH_SIZE):
                            target[j, acts[j]] = rewards[j]+settings.GAMMA*maxq[j]*(not dones[j])    #教師信号
                        optimizer.zero_grad()
                        loss = nn.MSELoss()(q, Variable(torch.from_numpy(target)))
                        loss.backward()
                        optimizer.step()
                #Q関数の更新
                if total_step % settings.UPDATE_TARGET_Q_FREQ == 0:
                    Q_target = copy.deepcopy(Q_train)
            #εの減少
            if settings.EPSILON > settings.EPISODE_NUM and total_step > settings.START_REDUCE_EPSIOLON:
                settings.EPSILON -= settings.EPSILON_DECREASE
                
            #次の行動へ
            total_reward += reward
            step += 1
            total_step += 1
            pobs = obs       
            
        total_rewards.append(total_reward)  #累積報酬を記録
        
        if(episode + 1) % settings.LOG_FREQ == 0:
            r = sum(total_rewards[((episode + 1) - settings.LOG_FREQ):(episode + 1)])/settings.LOG_FREQ
            elapsed_time = time.time() - start
            print("\t".join(map(str, [episode + 1, settings.EPSILON, r, total_step, str(elapsed_time)+"[sec]"])))
            start = time.time()
            

            
                        



if __name__ == "__main__":
    main()            
# %%
