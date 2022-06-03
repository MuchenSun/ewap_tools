import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm


class reader:
    def __init__(self):
        # read in raw data
        self.raw_obs = np.loadtxt("./ewap_dataset/seq_eth/obsmat.txt")

        # read in destinations
        self.destinations = np.loadtxt("./ewap_dataset/seq_eth/destinations.txt")

        # count how many pedestrians in total
        self.ped_idx_all = np.sort(np.unique(self.raw_obs[:, 1].astype(int)))
        print("read in {} pedestrians in total.".format(len(self.ped_idx_all)))

        # count how many frames in total
        self.frame_idx_all = np.sort(np.unique(self.raw_obs[:, 0]).astype(int))

        # dict for all pedestrians
        self.ped_dict = {}
        for ped in self.ped_idx_all:
            self.ped_dict[ped] = dict()

        # dict for all frames
        self.frame_dict = {}
        for frame in self.frame_idx_all:
            self.frame_dict[frame] = dict()

        # proccess raw data to fill in pedestrian & frame dict
        for obs in self.raw_obs:
            frame_idx = int(obs[0])
            ped_idx = int(obs[1])
            ped_state = np.array([obs[2], obs[4], obs[5], obs[7]])
            
            # fill in pedestrian data to pedestrian dict
            self.ped_dict[ped_idx][frame_idx] = ped_state

            # fill in frame data to frame dictionary
            self.frame_dict[frame_idx][ped_idx] = ped_state

        # post-proccess pedestrian dict to add frame list for every pedestrian
        for ped in self.ped_idx_all:
            frame_list = []
            for frame in self.ped_dict[ped]:
                frame_list.append(frame)
            self.ped_dict[ped]["frames"] = frame_list

        # post-process frame dict to add pedestrian list for every frame
        for frame in self.frame_idx_all:
            ped_list = []
            for ped in self.frame_dict[frame]:
                ped_list.append(ped)
            self.frame_dict[frame]["pedestrians"] = ped_list

        # interpolate missing frames for each pedestrian
        for ped in self.ped_idx_all:
            frame_list = np.sort(self.ped_dict[ped]["frames"])
            for i in range(len(frame_list) - 1):
                curr_frame = frame_list[i]
                next_frame = frame_list[i+1]

                curr_x = self.ped_dict[ped][curr_frame][0]
                curr_y = self.ped_dict[ped][curr_frame][1]
                curr_vx = self.ped_dict[ped][curr_frame][2]
                curr_vy = self.ped_dict[ped][curr_frame][3]
            
                next_x = self.ped_dict[ped][next_frame][0]
                next_y = self.ped_dict[ped][next_frame][1]
                next_vx = self.ped_dict[ped][next_frame][2]
                next_vy = self.ped_dict[ped][next_frame][3]

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
                    self.ped_dict[ped][inter_frame] = inter_state
                    self.ped_dict[ped]["frames"].append(inter_frame)

                    # add to frame dict
                    if inter_frame not in self.frame_dict:
                        self.frame_dict[inter_frame] = dict()
                        self.frame_dict[inter_frame][ped] = inter_state
                        
                        self.frame_dict[inter_frame]["pedestrians"] = [ped]
                    else:
                        self.frame_dict[inter_frame][ped] = inter_state

                        self.frame_dict[inter_frame]["pedestrians"].append(ped)

            self.ped_dict[ped]["frames"] = np.sort(self.ped_dict[ped]["frames"])

        # post-process to add full pedestrian list to each frame
        self.full_frame_list = []
        for frame in self.frame_dict:
            self.frame_dict[frame]["pedestrians"] = np.sort(self.frame_dict[frame]["pedestrians"])
            
            self.full_frame_list.append(frame)
        self.full_frame_list = np.sort(self.full_frame_list).astype(int)
        print("read in/process {} frames in total.".format(len(self.full_frame_list)))

        # post-process to add full trajectory to each pedestrian
        for ped in self.ped_dict:
            frame_list = self.ped_dict[ped]["frames"]
            traj = []
            for frame in frame_list:
                traj.append(self.ped_dict[ped][frame])
            traj = np.array(traj)
            self.ped_dict[ped]["traj"] = traj.copy()

        # load boundary 
        self.boundary = np.loadtxt("./ewap_dataset/seq_eth/boundary.txt")
        self.boundary_left = []
        self.boundary_right = []
        for pt in self.boundary:
            if pt[1] < 5.0:
                self.boundary_left.append(pt)
            else:
                self.boundary_right.append(pt)
        self.boundary_left = np.array(self.boundary_left)
        self.boundary_right = np.array(self.boundary_right)


    def animate_all(self, savefig=False):
        # start visualization
        fig, ax = plt.subplots(1, 2, figsize=(15.5, 7.5), gridspec_kw={'width_ratios': [1.5, 1]}, tight_layout=True)

        traj_lag = 40
        traj_ahead = 30
        for frame in tqdm(self.full_frame_list):
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
            ax[1].set_title("Frame: {}\n(Dots: past path; Solid line: future path)".format(frame), fontsize=20)
            ax[1].plot(self.boundary_left[:,1], self.boundary_left[:,0], linestyle="-", linewidth=5, color="k")
            ax[1].plot(self.boundary_right[:,1], self.boundary_right[:,0], linestyle="-", linewidth=5, color="k")
            ax[1].plot(self.destinations[:,1], self.destinations[:,0], linestyle="", marker="X", markersize=20, color="k")
            for axis in ['top', 'bottom', 'left', 'right']:
                ax[1].spines[axis].set_linewidth(5)
                ax[1].spines[axis].set_zorder(0)
            ax[1].tick_params(axis='x', labelsize=20, width=5., length=7.)
            ax[1].tick_params(axis='y', labelsize=20, width=5., length=7.)

            if frame not in self.frame_dict:
                print("missing frame: ", frame)
                continue

            for ped in self.frame_dict[frame]["pedestrians"]:
                state = self.frame_dict[frame][ped]
                ax[1].plot(state[1], state[0], linestyle='', marker='o', markersize=15, color='C'+str(ped), label="Ped {}".format(ped))

                curr_frame_idx = np.where(frame == self.ped_dict[ped]["frames"])[0][0]
                
                if curr_frame_idx < traj_lag:
                    traj = self.ped_dict[ped]["traj"][:curr_frame_idx]
                else:
                    traj = self.ped_dict[ped]["traj"][curr_frame_idx-traj_lag:curr_frame_idx]
                ax[1].plot(traj[:,1], traj[:,0], linestyle=":", linewidth=5, color='C'+str(ped))

                if curr_frame_idx + traj_ahead < len(self.ped_dict[ped]["frames"]):
                    plan = self.ped_dict[ped]["traj"][curr_frame_idx: curr_frame_idx+traj_ahead]
                else:
                    plan = self.ped_dict[ped]["traj"][curr_frame_idx:]
                ax[1].plot(plan[:,1], plan[:,0], linestyle="-", linewidth=5, color='C'+str(ped))
            ax[1].legend(fontsize=10, loc=4, markerscale=0.5, borderpad=0.3) 

            plt.pause(0.01)
            if savefig == True:
                plt.savefig("./ewap_dataset/seq_eth/images/eth_parallel_frame{0:08}.png".format(frame))

        plt.close()


    def get_ped_traj(self, ped):
        """Get the trajectory and corresponding frames of a pedetrian"""
        traj = self.ped_dict[ped]["traj"]
        frames = self.ped_dict[ped]["frames"]

        return traj, frames

    def get_frame_states(self, frame):
        """Get all pedestrians and their state given a frame"""
        if frame in self.full_frame_list:
            peds_idx = self.frame_dict[frame]["pedestrians"]
            states = []
            for ped in peds_idx:
                states.append(self.frame_dict[frame][ped])
            states = np.array(states)

            return peds_idx, states
        else:
            print("The input frame is not processed in the dataset.")
            print("Check avaiable frame indices at: reader.full_frame_list")

    def config_ax(self, ax):
        ax.set_xlim(-6.0, 18.0)
        ax.set_ylim(15.0, -10.0)
        ax.set_aspect('equal')
        ax.plot(self.boundary_left[:,1], self.boundary_left[:,0], linestyle="-", linewidth=5, color="k")
        ax.plot(self.boundary_right[:,1], self.boundary_right[:,0], linestyle="-", linewidth=5, color="k")
        ax.plot(self.destinations[:,1], self.destinations[:,0], linestyle="", marker="X", markersize=20, color="k")
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(5)
            ax.spines[axis].set_zorder(0)
        ax.tick_params(axis='x', labelsize=20, width=5., length=7.)
        ax.tick_params(axis='y', labelsize=20, width=5., length=7.)
