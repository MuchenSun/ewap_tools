from reader import reader
import numpy as np


# Initialize dataset reader
eth = reader()

# Get trajectory of the specified pedestrian
traj, frames = eth.get_ped_traj(18)
print("Pedestrian 8's trajectory:")
print(traj)
print("The corresponding frames are: ", frames)

# Get pedestrian states of the specified frame
peds_idx, states = eth.get_frame_states(3000)
print("Pedestrians in frame 800: ", peds_idx)
print("Their corresponding states: ")
print(states)

# Animate the original video with processed visualization side by side
eth.animate_all(savefig=True)  # with savefig=True, each side-by-side frame will be saved at "ewap_dataset/seq_eth/images"
