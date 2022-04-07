from reader import reader
import matplotlib.pyplot as plt

eth = reader()
traj, frames = eth.get_ped_traj(8)

fig, ax = plt.subplots(1, 1, tight_layout=True, figsize=(7, 7))
for i in range(len(traj)):
    eth.config_ax(ax)
    ax.plot(traj[i,1], traj[i,0], linestyle="", marker="o", markersize=5, color="C0")

    plt.pause(0.01)

plt.show()
plt.close()
