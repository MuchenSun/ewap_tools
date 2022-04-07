from reader import reader
import matplotlib.pyplot as plt

eth = reader()
traj, frames = eth.get_ped_traj(8)

fig1, ax1 = plt.subplots(1, 1, tight_layout=True, figsize=(7, 7))
eth.config_ax(ax1)
ax1.plot(traj[:,1], traj[:,0], linestyle="-", linewidth=5, color="C0")

fig2, ax2 = plt.subplots(1, 1, tight_layout=True)
ax2.plot(frames*0.04, traj[:,0], color="C0", label="y")
ax2.plot(frames*0.04, traj[:,1], color="C1", label="x")
ax2.legend()
plt.show()

