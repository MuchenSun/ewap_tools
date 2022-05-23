from reader import reader
import numpy as np


eth = reader()

def get_count(thred):
    count = 0

    for ped in eth.ped_idx_all:
        # print('ped: ', ped)
        frames = eth.ped_dict[ped]["frames"]
        flag = 0
        for frame in frames:
            ped_pos = eth.ped_dict[ped][frame]
            
            neighbors = eth.frame_dict[frame]["pedestrians"]
            # print(neighbors)
            for nb in neighbors:
                if nb == ped:
                    continue
                nb_pos = eth.ped_dict[nb][frame]
                
                dist = np.linalg.norm(ped_pos[0:2] - nb_pos[0:2])
                if dist <= thred:
                    count += 1
                    flag = 1
                    break
            if flag == 1:
                break
    
    return count

thred_list = np.arange(0.1, 1.1, 0.1)
count_list = []
for thred in thred_list:
    count = get_count(thred)
    count_list.append(count)
    print("{}: {}".format(thred, count))

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.plot(thred_list, count_list, "-*")

plt.show()
plt.close()
