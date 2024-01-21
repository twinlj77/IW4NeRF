import matplotlib.pyplot as plt
import torch
import cv2
import os
from PIL import Image
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from qianru import s,p
from encoder import modelE
with torch.no_grad():
    modelE.eval()
    e, dp= modelE.forward((s, p))
e, dp = e.cpu(), dp.cpu()

i=0
eimage = e.contiguous().view((-1, 800, 800, 3)).numpy()
eimage=eimage[i]
plt.figure(figsize=(8, 8))  # 绘图尺寸
plt.subplot(1, 1, 1)  # 画布分割2行3列取第一块
plt.imshow(cv2.cvtColor(eimage, cv2.COLOR_BGR2RGB))
"""figure_save_path = "data/nerf_synthetic/lego/test"
if not os.path.exists(figure_save_path):
    os.makedirs(figure_save_path) # 如果不存在目录figure_save_path，则创建
plt.savefig(os.path.join(figure_save_path , 'r_0.png'))#第一个是指存储路径，第二个是图片名字"""
plt.show()