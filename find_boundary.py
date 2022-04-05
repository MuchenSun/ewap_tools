import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# read in raw data
raw_obs = np.loadtxt("./ewap_dataset/seq_eth/obsmat.txt")
print(raw_obs.shape)

# read in destinations
destinations = np.loadtxt("./ewap_dataset/seq_eth/destinations.txt")

# count how many pedestrians in total
ped_idx_all = np.sort(np.unique(raw_obs[:, 1].astype(int)))
print("ped_idx_all: ", ped_idx_all[0:10])

# count how many frames in total
frame_idx_all = np.sort(np.unique(raw_obs[:, 0]).astype(int))
print("frame_idx_all: ", frame_idx_all[0:10])

# dict for all pedestrians
ped_dict = {}
for ped in ped_idx_all:
    ped_dict[ped] = dict()

# dict for all frames
frame_dict = {}
for frame in frame_idx_all:
    frame_dict[frame] = dict()

# proccess raw data to fill in pedestrian & frame dict
rot = np.array([
    [np.cos(-np.pi/2), -np.sin(-np.pi/2)],
    [np.sin(-np.pi/2) ,  np.cos(-np.pi/2)]
])
for obs in raw_obs:
    frame_idx = int(obs[0])
    ped_idx = int(obs[1])
    ped_state = np.array([obs[2], obs[4], obs[5], obs[7]])
    # ped_state[0:2] = rot @ ped_state[0:2]
    # ped_state[2:4] = rot @ ped_state[2:4]
    
    # fill in pedestrian data to pedestrian dict
    ped_dict[ped_idx][frame_idx] = ped_state

    # fill in frame data to frame dictionary
    frame_dict[frame_idx][ped_idx] = ped_state

# post-proccess pedestrian dict to add frame list for every pedestrian
for ped in ped_idx_all:
    frame_list = []
    for frame in ped_dict[ped]:
        frame_list.append(frame)
    ped_dict[ped]["frames"] = frame_list

# post-process frame dict to add pedestrian list for every frame
for frame in frame_idx_all:
    ped_list = []
    for ped in frame_dict[frame]:
        ped_list.append(ped)
    frame_dict[frame]["pedestrians"] = ped_list

# interpolate missing frames for each pedestrian
for ped in ped_idx_all:
    frame_list = np.sort(ped_dict[ped]["frames"])
    for i in range(len(frame_list) - 1):
        curr_frame = frame_list[i]
        next_frame = frame_list[i+1]

        curr_x = ped_dict[ped][curr_frame][0]
        curr_y = ped_dict[ped][curr_frame][1]
        curr_vx = ped_dict[ped][curr_frame][2]
        curr_vy = ped_dict[ped][curr_frame][3]
    
        next_x = ped_dict[ped][next_frame][0]
        next_y = ped_dict[ped][next_frame][1]
        next_vx = ped_dict[ped][next_frame][2]
        next_vy = ped_dict[ped][next_frame][3]

        dx = float(next_x - curr_x) / (next_frame - curr_frame)
        dy = float(next_y - curr_y) / (next_frame - curr_frame)
        dvx = float(next_vx - curr_vx) / (next_frame - curr_frame)
        dvy = float(next_vy - curr_vy) / (next_frame - curr_frame)

        for j in range(1, next_frame-curr_frame):
            # generate interpolated state
            inter_frame = curr_frame + j

            inter_x = curr_x + j * dx
            inter_y = curr_y + j * dy
            inter_vx = curr_vx + j * dvx
            inter_vy = curr_vy + j * dvy
            inter_state = np.array([inter_x, inter_y, inter_vx, inter_vy])

            # add to pedestrian dict
            ped_dict[ped][inter_frame] = inter_state
            ped_dict[ped]["frames"].append(inter_frame)

            # add to frame dict
            if inter_frame not in frame_dict:
                frame_dict[inter_frame] = dict()
                frame_dict[inter_frame][ped] = inter_state
                
                frame_dict[inter_frame]["pedestrians"] = [ped]
            else:
                frame_dict[inter_frame][ped] = inter_state

                frame_dict[inter_frame]["pedestrians"].append(ped)

    ped_dict[ped]["frames"] = np.sort(ped_dict[ped]["frames"])

# post-process to add full pedestrian list to each frame
full_frame_list = []
for frame in frame_dict:
    frame_dict[frame]["pedestrians"] = np.sort(frame_dict[frame]["pedestrians"])
    
    full_frame_list.append(frame)
full_frame_list = np.sort(full_frame_list).astype(int)

# post-process to add full trajectory to each pedestrian
for ped in ped_dict:
    frame_list = ped_dict[ped]["frames"]
    traj = []
    for frame in frame_list:
        traj.append(ped_dict[ped][frame])
    traj = np.array(traj)
    ped_dict[ped]["traj"] = traj.copy()

# start visualization
fig, ax = plt.subplots(1, 1, tight_layout=True, figsize=(6, 6), dpi=100)

for ped in ped_dict:
    traj = ped_dict[ped]["traj"]
    ax.plot(traj[:,1], traj[:,0])

ax.plot(destinations[:,1], destinations[:,0], linestyle="", marker="x", markersize=10, color="k")

ax.set_xlim(-6.0, 18.0)
ax.set_ylim(15.0, -9.0)
ax.set_aspect("equal")


coords = []

def onclick(event):
    global ix, iy
    ix, iy = event.xdata, event.ydata
    print("x={}, y={}".format(ix, iy))

    global coords
    coords.append((iy, ix))

    if len(coords) == 1000:
        fig.canvas.mpl_disconnect(cid)

    return coords
cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()
plt.close()

coords = np.array(coords)
np.savetxt("./ewap_dataset/seq_eth/boundary.txt", coords)

print(len(coords))
