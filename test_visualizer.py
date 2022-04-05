import numpy as np
import matplotlib.pyplot as plt
import cv2

mask_img = cv2.imread("/home/msun/workspace/ewap_tools/ewap_dataset/seq_eth/frames/frame00000002_mask.png", cv2.IMREAD_GRAYSCALE)
print(mask_img.shape)

mask = []
for i in range(1, mask_img.shape[0]+1):
    for j in range(1, mask_img.shape[1]+1):
        if mask_img[i-1, j-1] != 0:
            mask.append([mask_img.shape[1]-j, mask_img.shape[0]-i])

mask = np.array(mask).T
mask_homo = np.vstack([mask, np.ones(mask.shape[1])])
print(mask_homo.shape)

H = np.loadtxt("/home/msun/workspace/ewap_tools/ewap_dataset/seq_eth/H.txt")
print("Homograph matrix: ")
print(H)

mask_world = (H @ mask_homo)[0:2]

fig, ax = plt.subplots(1, 1)
ax.set_aspect("equal")

ax.plot(mask_world[0], -mask_world[1], linestyle="", marker="o", markersize=5)
plt.show()
