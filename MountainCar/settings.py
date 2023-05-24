#settings
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#定数
HIDDEN_SIZE = 100
BAFFER_SIZE = 200
EPISODE_NUM = 5000
STEP_MAX = 200
