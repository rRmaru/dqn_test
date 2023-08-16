import numpy as np
import random

class Replay_buffer(object):
    def __init__(self, size):
        self.size = size
        self._memory = []
        
    def add(self, input):
        if len(self._memory) >= self.size:
            self._memory.pop(0)
        self._memory.append(input)
        
    def sample(self, batch_size):
        sample_list = []
        random.shuffle(self._memory)
        for i in range(int(self.size/batch_size)):
            sample_list.append(self._memory[i*batch_size:i*batch_size+batch_size])
        return sample_list
    
    def __len__(self):
        return len(self._memory)
    
    