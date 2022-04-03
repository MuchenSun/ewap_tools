import cv2
import numpy as np
import matplotlib.pyplot as plt


map_img = cv2.imread('/home/msun/workspace/ewap_tools/ewap_dataset/seq_eth/map.png', cv2.IMREAD_GRAYSCALE)
map_coord = []
for j in range(map_img.shape[1]):
    for i in range(map_img.shape[0]):
        if map_img[i,j] > 0:
            map_coord.append([i, j])

map_coord = np.array(map_coord)
print(map_coord.shape)
print(map_coord)


fig, ax = plt.subplots(1, 1)
ax.set_aspect('equal')
ax.set_xlim(0, map_img.shape[1])
ax.set_ylim(0, map_img.shape[0])

ax.scatter(map_coord[:,1], map_coord[:,0])

plt.show()
plt.close()



