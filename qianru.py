import cv2
import os
from torch import nn #导入对应库
import torch
from encoder import modelE
from MS_SSIM_L1_loss import MS_SSIM_L1_LOSS
import matplotlib.pyplot as plt
from PIL import Image
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
img_path1="data/nerf_synthetic/lego/test/r_0.png"
a=cv2.imread(img_path1)
#plt.figure(figsize=(15, 15))  # 绘图尺寸
#plt.subplot(1, 1, 1)
#plt.imshow(cv2.cvtColor(a, cv2.COLOR_BGR2RGB))
#plt.show()
img_path2="img/fenjing.png"
m=cv2.imread(img_path2)
b=cv2.resize(m,(800,800))
a = torch.from_numpy(a).to(device).float()
b = torch.from_numpy(b).to(device).float()

modelE.to(device)
criterion1 = nn.MSELoss() #标准是损失函数均方误差
criterion2 = MS_SSIM_L1_LOSS()
optimizer = torch.optim.Adam(modelE.parameters(), lr=0.5)#优化算法，学习率为0.001
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)#第一个参数就是所使用的优化器对象
metric = nn.L1Loss()#创建一个标准来测量输入x和目标y中每个元素之间的平均绝对误差（MAE）
# 训练
epochs =100# 迭代10000次
train_losses, val_losses = [], []
flat_source_size = 800 * 800 * 3
flat_payload_size = 800 * 800 * 3
for epoch in range(epochs):  # 一个循环
    modelE.train()  # 如果模型中有BN层(Batch Normalization）和Dropout，需要在训练时添加model.train()，在测试时添加model.eval()。
    # 其中model.train()是保证BN层用每一批数据的均值和方差，而model.eval()是保证BN用全部训练数据的均值和方差；
    # 而对于Dropout，model.train()是随机取一部分网络连接来训练更新参数，而model.eval()是利用到了所有网络连接。
    train_loss = 0.
    s, p = a,b
    s.to(device)
    p.to(device)
    optimizer.zero_grad()
    e_out, dp_out=modelE.forward((s, p))  # 获取卷积层和relu层的结果encoder_output, decoded_payload, decoded_source
    e_loss = criterion1(e_out.contiguous().view((-1, flat_source_size)), s.contiguous().view((-1, flat_source_size)))
    dp_loss = criterion1(dp_out.contiguous().view((-1, flat_payload_size)),p.contiguous().view((-1, flat_payload_size)))
    #loss1=criterion2(e_out.contiguous().view((-1,3, 800,800)), s.contiguous().view((-1, 3, 800,800)))
    loss = e_loss+dp_loss
    loss.backward()  # 反向传播求梯度
    optimizer.step()
    train_loss += loss.item()
    print("Train loss: ", train_loss )






