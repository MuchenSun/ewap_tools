import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm


class reader:
    def __init__(self, dataset, swap_xy=False):
        # read in raw data
        self.dt = 0.4
        self.header = f"./ewap_dataset/seq_{dataset}/"
        self.raw_obs = np.loadtxt(self.header + "obsmat.txt")
        self.swap_xy = swap_xy

        # count how many pedestrians in total
        self.ped_idx_all = np.sort(np.unique(self.raw_obs[:, 1].astype(int)))

        # count how many frames in total
        self.frame_idx_all = np.sort(np.unique(self.raw_obs[:, 0]).astype(int))

        # process dataset 
        self.process()

    def process(self):
        # dict for all pedestrians
        self.ped_dict = {}
        for ped in self.ped_idx_all:
            self.ped_dict[ped] = {
                "frames": [],
                "traj": []
            }

        # dict for all frames
        self.frame_dict = {}
        for frame in self.frame_idx_all:
            self.frame_dict[frame] = {
                "peds": [],
                "states": []
            }

        # proccess raw data to fill in pedestrian & frame dict
        for oid, obs in enumerate(self.raw_obs):
            frame_idx = int(obs[0])
            ped_idx = int(obs[1])
            ped_state = np.array([obs[2], obs[4], obs[5], obs[7]])
            if self.swap_xy == True:
                ped_state = np.array([obs[4], obs[2], obs[7], obs[5]])
            
            # fill in pedestrian data to pedestrian dict
            self.ped_dict[ped_idx]["frames"].append(frame_idx)
            self.ped_dict[ped_idx]["traj"].append(ped_state.copy())

            # fill in frame data to frame dictionary
            self.frame_dict[frame_idx]["peds"].append(ped_idx)
            self.frame_dict[frame_idx]["states"].append(ped_state.copy())

        # post-process, convert all to numpy array
        for ped in self.ped_idx_all:
            self.ped_dict[ped]["frames"] = np.array(self.ped_dict[ped]["frames"], dtype=int) 
            self.ped_dict[ped]["traj"] = np.array(self.ped_dict[ped]["traj"]) 

        for frame in self.frame_idx_all:
            self.frame_dict[frame]["peds"] = np.array(self.frame_dict[frame]["peds"], dtype=int)
            self.frame_dict[frame]["states"] = np.array(self.frame_dict[frame]["states"])


<<<<<<< HEAD
    def get_ped(self, ped):
=======
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
            else:
                plt.close()


    def get_ped_traj(self, ped):
        """Get the trajectory and corresponding frames of a pedetrian"""
>>>>>>> 335db2e17bb928b8bf91ce615c2a99011785e3b0
        traj = self.ped_dict[ped]["traj"]
        frames = self.ped_dict[ped]["frames"]
        return traj, frames
<<<<<<< HEAD
    

    def get_frame(self, frame):
        peds = self.frame_dict[frame]["peds"]
        states = self.frame_dict[frame]["states"]
        return peds, states
    

    def get_all_frames(self):
        return self.frame_idx_all.copy()
    
    def get_all_peds(self):
        return self.ped_idx_all.copy()
=======

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
>>>>>>> 335db2e17bb928b8bf91ce615c2a99011785e3b0
