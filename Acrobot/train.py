#%%
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchviz import make_dot
import torch.optim as optim
import copy
import time

from nn import NN, DQN
from replay_buffer import Replay_buffer
import settings

#randomのジェネレーター作成
rng = np.random.default_rng()


def init_parameters(layer):
    if type(layer) == nn.Linear:
        nn.init.xavier_uniform_(layer.weight)   #重みを「一様のランダム値」に初期化
        layer.bias.data.fill_(0.0)              #バイアスを「0」に初期化

def main():
    env = gym.make("CartPole-v1")    #環境の構築
    obs_num = env.observation_space.shape[0]    #環境から観測のtensorの型を取得
    act_num = env.action_space.n                #環境から行動のtensorの型を取得
    #NNを作成
    Q_train = NN(obs_num, act_num)              
    Q_target = copy.deepcopy(Q_train)
    
    #損失関数、最適化関数を指定
    optimizer = optim.RMSprop(Q_train.parameters(), lr=settings.LEANING_RATE, alpha=0.95, eps=0.01)     #最適化
    criterion = nn.MSELoss()    #損失関数：平均二乗誤差
    
    #バッファを構築
    memory = Replay_buffer(settings.BAFFER_SIZE)
    
    #勾配をフラットな状態にする
    Q_train.zero_grad()

    #エピソードごとの報酬を格納するリスト
    total_rewards = []

    #学習全体のstep数を管理する
    total_step = 0
    
    #epsilon
    epsilon = settings.EPSILON
    
    #エピソード開始
    print("\t".join(["episode", "epsilon", "reward", "episode", "time"]))
    start = time.time()
    
    for episode in range(settings.MAX_EPISODE_NUM):
        pobs, _ = env.reset()     #新しいエピソードを始めるために環境をリセットする
        sum_reward = 0      #エピソードの報酬を格納する
        step = 0
        done = False
        
        for _ in range(settings.MAX_STEP_NUM):
            #行動選択
            act = env.action_space.sample()
            #ε-greedy法
            if rng.random() > epsilon:
                pobs_ = np.array(pobs, dtype="float32").reshape((1, obs_num))     #2次元ベクトル
                pobs_ = Variable(torch.from_numpy(pobs_))                         #計算グラフの機能を持たせれる
                act = Q_train(pobs_)                              #行動の計算
                max, indices = torch.max(act.data, 1)
                act = indices.numpy()[0]
            
            obs, reward, done, _, _ = env.step(act)
            
            if done:
                reward = -1
                break
            memory.add([pobs, act, reward, obs, done])
            
            
            #学習
            if len(memory) == settings.BAFFER_SIZE:      #大体、~エピソード目から学習が始まる
                if total_step % settings.TRAIN_FREQ == 0:
                    batchs = memory.sample(settings.BATCH_SIZE)
                    for batch in batchs:
                        pobss = np.array([b[0] for b in batch], dtype="float32").reshape((settings.BATCH_SIZE, obs_num))
                        acts = np.array([b[1] for b in batch], dtype="int32")
                        rewards = np.array([b[2] for b in batch], dtype="int32")
                        obss = np.array([b[3] for b in batch], dtype="float32").reshape((settings.BATCH_SIZE, obs_num))
                        dones = np.array([b[4] for b in batch], dtype="bool")
                        
                        #set y 
                        pobss_ = Variable(torch.from_numpy(pobss))
                        q = Q_train(pobss_)
                        obss_ = Variable(torch.from_numpy(obss))
                        maxs, indices = torch.max(Q_target(obss_).data, 1)
                        maxq = maxs.numpy()
                        target = copy.deepcopy(q.data.numpy())
                        for j in range(settings.BATCH_SIZE):
                            target[j, acts[j]] = rewards[j] + settings.GAMMA*maxq[j]*(not dones[j])  #教師信号
                        optimizer.zero_grad()
                        loss = criterion(q, Variable(torch.from_numpy(target)))
                        loss.backward()
                        optimizer.step()
                if total_step % settings.UPDATE_TARGET_Q_FREQ == 0:
                    Q_target = copy.deepcopy(Q_train)
            if epsilon > settings.EPSILON_MIN and total_step > settings.START_REDUCE_EPSILON:
                epsilon -= settings.EPSILON_DECREASE
            
            
            #次の行動へ
            sum_reward += reward
            total_step += 1
            pobs = obs
        
        total_rewards.append(sum_reward)        #累積報酬を記録
        
        #報酬値の記録
        if (episode + 1) % settings.LOG_FREQ == 0:
            r = sum(total_rewards[((episode + 1) - settings.LOG_FREQ):(episode + 1)])/settings.LOG_FREQ
            elapsed_time = time.time() - start
            print("\t".join(map(str, [episode + 1, epsilon, r, total_step, str(elapsed_time)+"[sec]"])))
            start = time.time()
    env.close()

if __name__ == "__main__":
    main()


# %%
