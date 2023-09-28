#%%
import numpy as np
import matplotlib.pyplot as plt

import csv

def main():
    with open("./result/test/cartpole_test.csv", "r") as f:
        reader = csv.reader(f)
        for row in reader:
            row = [float(v) for v in row]
            print(row)
            
    #グラフ描画領域
    save_step = 50
    max_episode = 2000
    x = range(save_step, max_episode+save_step, save_step)
    y = row
    
    fig, ax = plt.subplots()
    
    ax.plot(x,y)
    ax.set_xlim(save_step, max_episode)
    plt.show()
        
        
if __name__ == "__main__":
    main()
# %%
