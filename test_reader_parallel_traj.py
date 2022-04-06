import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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

# load boundary 
boundary = np.loadtxt("./ewap_dataset/seq_eth/boundary.txt")
boundary_left = []
boundary_right = []
for pt in boundary:
    if pt[1] < 5.0:
        boundary_left.append(pt)
    else:
        boundary_right.append(pt)
boundary_left = np.array(boundary_left)
boundary_right = np.array(boundary_right)

# start visualization
fig, ax = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={'width_ratios': [1.38, 1]}, tight_layout=True)

traj_lag = 40
traj_ahead = 30
for frame in tqdm(full_frame_list):
    ax[0].cla()
    frame_img = cv2.imread("./ewap_dataset/seq_eth/frames/frame{0:08}.png".format(frame))
    ax[0].imshow(frame_img[...,::-1])
    ax[0].set_aspect("equal")
    ax[0].xaxis.set_major_locator(plt.NullLocator())
    ax[0].yaxis.set_major_locator(plt.NullLocator())
    
    ax[1].cla()
    ax[1].set_xlim(-6.0, 18.0)
    ax[1].set_ylim(15.0, -10.0)
    ax[1].set_aspect('equal')
    ax[1].set_title("Frame: {}".format(frame), fontsize=20)
    ax[1].plot(boundary_left[:,1], boundary_left[:,0], linestyle="-", linewidth=5, color="k")
    ax[1].plot(boundary_right[:,1], boundary_right[:,0], linestyle="-", linewidth=5, color="k")
    ax[1].plot(destinations[:,1], destinations[:,0], linestyle="", marker="X", markersize=20, color="k")
    for axis in ['top', 'bottom', 'left', 'right']:
        ax[1].spines[axis].set_linewidth(5)
        ax[1].spines[axis].set_zorder(0)
    ax[1].tick_params(axis='x', labelsize=20, width=5., length=7.)
    ax[1].tick_params(axis='y', labelsize=20, width=5., length=7.)

    if frame not in frame_dict:
        print("missing frame: ", frame)
        continue

    for ped in frame_dict[frame]["pedestrians"]:
        state = frame_dict[frame][ped]
        ax[1].plot(state[1], state[0], linestyle='', marker='o', markersize=15, color='C'+str(ped), label="Ped {}".format(ped))

        curr_frame_idx = np.where(frame == ped_dict[ped]["frames"])[0][0]
        
        if curr_frame_idx < traj_lag:
            traj = ped_dict[ped]["traj"][:curr_frame_idx]
        else:
            traj = ped_dict[ped]["traj"][curr_frame_idx-traj_lag:curr_frame_idx]
        ax[1].plot(traj[:,1], traj[:,0], linestyle=":", linewidth=5, color='C'+str(ped))

        if curr_frame_idx + traj_ahead < len(ped_dict[ped]["frames"]):
            plan = ped_dict[ped]["traj"][curr_frame_idx: curr_frame_idx+traj_ahead]
        else:
            plan = ped_dict[ped]["traj"][curr_frame_idx:]
        ax[1].plot(plan[:,1], plan[:,0], linestyle="-", linewidth=5, color='C'+str(ped))
    ax[1].legend(fontsize=15, loc=4)

    plt.pause(0.01)
    # plt.savefig("./ewap_dataset/seq_eth/img/eth_parallel_frame{0:08}.png".format(frame))
