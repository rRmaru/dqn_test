#settings
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#定数
HIDDEN_SIZE = 100
BAFFER_SIZE = 200
