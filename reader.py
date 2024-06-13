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


    def get_ped(self, ped):
        traj = self.ped_dict[ped]["traj"]
        frames = self.ped_dict[ped]["frames"]
        return traj, frames
    

    def get_frame(self, frame):
        peds = self.frame_dict[frame]["peds"]
        states = self.frame_dict[frame]["states"]
        return peds, states
    

    def get_all_frames(self):
        return self.frame_idx_all.copy()
    
    def get_all_peds(self):
        return self.ped_idx_all.copy()
