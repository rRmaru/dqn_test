import torch
import numpy as np

class ReplayBuffer(object):
    def __init__(self, size):
        self.size = size
        self.memory_ = []
    def add(self, input):
        if len(self) >= self.size:
            self.memory_.pop(0)
        self.memory_.append(input)
        
    def sample(self, sample_size):
        index = np.random.randint(0, self.size, size=sample_size)
        sample_list = []
        for i in index:
            sample_list.append(self.memory_[i])
        return sample_list
    
    def __len__(self):
        return len(self.memory_)