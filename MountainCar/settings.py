#settings
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#定数
HIDDEN_SIZE = 100
BAFFER_SIZE = 200
EPISODE_NUM = 5000
STEP_MAX = 200
EPSILON = 1.0
START_REDUCE_EPSILON = 200
EPSILON_DECREASE = 0.001
LOG_FREQ = 1000
UPDATE_TARGET_Q_FREQ = 20
GAMMA = 0.97
TRAIN_FREQ = 10
BATCH_SIZE = 50

