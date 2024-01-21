import cv2
import os
from torch import nn #导入对应库
import torch
import matplotlib.pyplot as plt
from qianru import s,p
from coder_model import model
e, dp,ds= model.forward((s, p))

figure_save_path = "data/nerf_synthetic/lego/test"
if not os.path.exists(figure_save_path):
    os.makedirs(figure_save_path) # 如果不存在目录figure_save_path，则创建
plt.savefig(os.path.join(figure_save_path , 'r_0.png'))#第一个是指存储路径，第二个是图片名字
plt.show()