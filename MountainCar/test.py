#%%
import torch

from replay_buffer import ReplayBuffer

def main():
    memory = ReplayBuffer(5)
    memory.add((1,2))
    memory.add((3,4))
    memory.add((5,6))
    memory.add((7,8))
    memory.add((9,10))
    memory.add((11,12))
    
    print(len(memory))
    
    sample = memory.sample(3)
    print(sample)
    
    print("memory:{}".format(memory.memory_))
    
if __name__ == "__main__":
    main()
# %%
