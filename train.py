#%%
import copy
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import gymnasium as gym
from gymnasium import wrappers

import settings

#Generator作成
rng = np.random.default_rng()

#環境
# 環境
MONITOR = False
env = gym.make("CartPole-v1")   #環境作成
if MONITOR:     #display
    env = wrappers.Monitor(env, "./result", force=True)

obs_num = env.observation_space.shape[0]
acts_num = env.action_space.n
HIDDEN_SIZE = 100   #隠し層

class NN(nn.Module):
    
    def __init__(self):
        
        super(NN, self).__init__()
        self.fc1 = nn.Linear(obs_num, HIDDEN_SIZE)  #4 input
        self.fc2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc3 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc4 = nn.Linear(HIDDEN_SIZE, acts_num)   #2 output
    
    def __call__(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        y = F.relu(self.fc4(h))
        return y

# モデル
Q = NN() # 近似Q関数
Q_ast = copy.deepcopy(Q)
optimizer = optim.RMSprop(Q.parameters(), lr=0.00015, alpha=0.95, eps=0.01) #最適化関数

total_step = 0 # 総ステップ（行動）数
memory = [] # メモリ
total_rewards = [] # 累積報酬記録用リスト

# 学習開始
print("\t".join(["epoch", "epsilon", "reward", "total_step", "elapsed_time"]))
start = time.time()
for epoch in range(settings.EPOCH_NUM): #replay for epoch_num
    
    pobs, _ = env.reset() # 環境初期化
    step = 0 # ステップ数
    done = False # ゲーム終了フラグ
    total_reward = 0 # 累積報酬
    
    while not done and step < settings.STEP_MAX:
        if MONITOR:
            env.render()
        # 行動選択
        pact = env.action_space.sample()
        # ε-greedy法
        if np.random.rand() > settings.EPSILON:
            # 最適な行動を予測
            pobs_ = np.array(pobs, dtype="float32").reshape((1, obs_num))
            pobs_ = Variable(torch.from_numpy(pobs_))
            pact = Q(pobs_)
            maxs, indices = torch.max(pact.data, 1) #valueとindicesが返ってくる
            pact = indices.numpy()[0]
            
        # 行動
        obs, reward, done, _, _ = env.step(pact)
        if done:
            reward = -1
            
        # メモリに蓄積
        memory.append((pobs, pact, reward, obs, done)) # 状態、行動、報酬、行動後の状態、ゲーム終了フラグ
        if len(memory) > settings.MEMORY_SIZE: # メモリサイズを超えていれば消していく
            memory.pop(0)       #古いものから消す
            
        # 学習
        if len(memory) == settings.MEMORY_SIZE: # メモリサイズ分溜まっていれば学習
            # 経験リプレイ
            if total_step % settings.TRAIN_FREQ == 0:
                memory_ = copy.deepcopy(memory)
                rng.shuffle(memory_)        #memoryの順番をランダムに入れ替える
                memory_idx = range(len(memory_))
                for i in memory_idx[::settings.BATCH_SIZE]:
                    batch = memory_[i:i+settings.BATCH_SIZE] # 経験ミニバッチ
                    pobss = np.array([b[0] for b in batch], dtype="float32").reshape((settings.BATCH_SIZE, obs_num))
                    pacts = np.array([b[1] for b in batch], dtype="int32")
                    rewards = np.array([b[2] for b in batch], dtype="int32")
                    obss = np.array([b[3] for b in batch], dtype="float32").reshape((settings.BATCH_SIZE, obs_num))
                    dones = np.array([b[4] for b in batch], dtype="bool")
                    # set y
                    pobss_ = Variable(torch.from_numpy(pobss))
                    q = Q(pobss_)
                    obss_ = Variable(torch.from_numpy(obss))
                    maxs, indices = torch.max(Q_ast(obss_).data, 1)
                    maxq = maxs.numpy() # maxQ
                    target = copy.deepcopy(q.data.numpy())
                    for j in range(settings.BATCH_SIZE):
                        target[j, pacts[j]] = rewards[j]+settings.GAMMA*maxq[j]*(not dones[j]) # 教師信号
                    # Perform a gradient descent step
                    optimizer.zero_grad()
                    loss = nn.MSELoss()(q, Variable(torch.from_numpy(target)))
                    loss.backward()
                    optimizer.step()
            # Q関数の更新
            if total_step % settings.UPDATE_TARGET_Q_FREQ == 0:
                Q_ast = copy.deepcopy(Q)
                
        # εの減少
        if settings.EPSILON > settings.EPSILON_MIN and total_step > settings.START_REDUCE_EPSILON:
            settings.EPSILON -= settings.EPSILON_DECREASE
            
        # 次の行動へ
        total_reward += reward
        step += 1
        total_step += 1
        pobs = obs
        
    total_rewards.append(total_reward) # 累積報酬を記録
    
    if (epoch+1) % settings.LOG_FREQ == 0:
        r = sum(total_rewards[((epoch+1)-settings.LOG_FREQ):(epoch+1)])/settings.LOG_FREQ # ログ出力間隔での平均累積報酬
        elapsed_time = time.time()-start
        print("\t".join(map(str,[epoch+1, settings.EPSILON, r, total_step, str(elapsed_time)+"[sec]"]))) # ログ出力
        start = time.time()
        
if MONITOR:
    env.render(close=True)
# %%
