import matplotlib.pyplot as plt
import matplotlib
import os
import cv2
import torch
os.chdir("/home/gzr/workspace/Template_Matching_v5_loftr")
for rk in range(1,5):
    path0 = './s_0_'+str(rk)+'.png'
    path1 = './s_1_'+str(rk)+'.png'
    image0 = cv2.imread(path0)
    image1 = cv2.imread(path1)
    image0 = cv2.cvtColor(image0, cv2.COLOR_RGB2BGR)
    image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2BGR)

    image0 = torch.from_numpy(image0)
    image1 = torch.from_numpy(image1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=100)
    axes[0].imshow(image0)
    axes[1].imshow(image1)
    for i in range(2):  # clear all frames
        axes[i].get_yaxis().set_ticks([])
        axes[i].get_xaxis().set_ticks([])
        for spine in axes[i].spines.values():
            spine.set_visible(False)
    plt.tight_layout(pad=1)
    fig.canvas.draw()
    fig.tight_layout()  # 调整整体空白
    plt.subplots_adjust(wspace=0.005, hspace=0)  # 调整子图间距
    plt.savefig('./self_atten_' + str(rk) + '.png', bbox_inches='tight', dpi=100, pad_inches=0.0)
    plt.close()
