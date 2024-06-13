import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 

from reader import reader


dataset = reader('eth')
all_frames = dataset.get_all_frames()

fig, ax = plt.subplots(1, 1, figsize=(5,3), dpi=150)

for frame in tqdm(all_frames):
    peds, states = dataset.get_frame(frame)

    if len(states) > 0:
        ax.cla()
        ax.set_title("Frame: {0:05d}".format(frame))
        ax.set_aspect('equal')
        ax.set_xlim(-5.0, 15.0)
        ax.set_ylim(-0.0, 10.0)
        for state in states:
            if state[2] > 0.0:
                ax.plot(state[0], state[1], linestyle='', marker='o', color='C0')
                ax.plot([state[0], state[0] + dataset.dt * state[2]], [state[1], state[1] + dataset.dt * state[3]], linestyle='-', marker='', color='C0')
            else:
                ax.plot(state[0], state[1], linestyle='', marker='o', color='C1')
                ax.plot([state[0], state[0] + dataset.dt * state[2]], [state[1], state[1] + dataset.dt * state[3]], linestyle='-', marker='', color='C1')

    plt.pause(0.01)

plt.show()
plt.close()
