from reader import reader
import numpy as np


# Initialize dataset reader
eth = reader()

# Get trajectory of the specified pedestrian
traj, frames = eth.get_ped_traj(8)
print("Pedestrian 8's trajectory:")
print(traj)
print("The corresponding frames are: ", frames)

