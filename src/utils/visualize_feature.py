import numpy
import torch
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import math

class BilinearInterpolation(object):
    def __init__(self, w_rate: float, h_rate: float, *, align='center'):
        if align not in ['center', 'left']:
            align = 'center'
        self.align = align
        self.w_rate = w_rate
        self.h_rate = h_rate

    def set_rate(self,w_rate: float, h_rate: float):
        self.w_rate = w_rate    # w 的缩放率
        self.h_rate = h_rate    # h 的缩放率

    # 由变换后的像素坐标得到原图像的坐标    针对高
    def get_src_h(self, dst_i,source_h,goal_h) -> float:
        if self.align == 'left':
            # 左上角对齐
            src_i = float(dst_i * (source_h/goal_h))
        elif self.align == 'center':
            # 将两个图像的几何中心重合。
            src_i = float((dst_i + 0.5) * (source_h/goal_h) - 0.5)
        src_i += 0.001
        src_i = max(0.0, src_i)
        src_i = min(float(source_h - 1), src_i)
        return src_i
    # 由变换后的像素坐标得到原图像的坐标    针对宽
    def get_src_w(self, dst_j,source_w,goal_w) -> float:
        if self.align == 'left':
            # 左上角对齐
            src_j = float(dst_j * (source_w/goal_w))
        elif self.align == 'center':
            # 将两个图像的几何中心重合。
            src_j = float((dst_j + 0.5) * (source_w/goal_w) - 0.5)
        src_j += 0.001
        src_j = max(0.0, src_j)
        src_j = min((source_w - 1), src_j)
        return src_j

    def transform(self, img):
        source_h, source_w, source_c = img.shape  # (235, 234, 3)
        goal_h, goal_w = round(
            source_h * self.h_rate), round(source_w * self.w_rate)
        new_img = np.zeros((goal_h, goal_w, source_c), dtype=np.uint8)

        for i in range(new_img.shape[0]):       # h
            src_i = self.get_src_h(i,source_h,goal_h)
            for j in range(new_img.shape[1]):
                src_j = self.get_src_w(j,source_w,goal_w)
                i2 = math.ceil(src_i)
                i1 = int(src_i)
                j2 = math.ceil(src_j)
                j1 = int(src_j)
                x2_x = j2 - src_j
                x_x1 = src_j - j1
                y2_y = i2 - src_i
                y_y1 = src_i - i1
                new_img[i, j] = img[i1, j1]*x2_x*y2_y + img[i1, j2] * \
                    x_x1*y2_y + img[i2, j1]*x2_x*y_y1 + img[i2, j2]*x_x1*y_y1
        return new_img
#使用方法



def visualize_feature_map(img, img_batch,out_path, type="drone",BI=None):
    # save raw image
    img = torch.squeeze(img).detach().cpu().numpy()
    img = np.expand_dims(img, axis=2)
    plt.imshow(img)
    plt.savefig(out_path + "raw.jpg".format(type))
    plt.xticks()
    plt.yticks()
    plt.axis('off')

    if BI is None:
        BI = BilinearInterpolation(2, 2)

    feature_map = torch.squeeze(img_batch)
    feature_map = feature_map.detach().cpu().numpy()

    feature_map_sum = feature_map[0, :, :]
    feature_map_sum = np.expand_dims(feature_map_sum, axis=2)
    for i in range(0, feature_map.shape[0]):
        print(i)
        feature_map_split = feature_map[i,:, :]
        feature_map_split = np.expand_dims(feature_map_split,axis=2)
        if i > 0:
            feature_map_sum += feature_map_split
        # feature_map_split = BI.transform(feature_map_split)

        plt.imshow(feature_map_split)
        plt.savefig(out_path + str(i) + "_{}.jpg".format(type) )
        plt.xticks()
        plt.yticks()
        plt.axis('off')

    # feature_map_sum = BI.transform(feature_map_sum)
    plt.imshow(feature_map_sum)
    plt.savefig(out_path + "sum_{}.jpg".format(type))
    print("save sum_{}.jpg".format(type))

if __name__ == '__main__':
    BI = BilinearInterpolation(2, 2)
    imgs_path = "/path/to/imgs/"
    save_path = "/save/path/to/output/"


    # visualize_feature_map