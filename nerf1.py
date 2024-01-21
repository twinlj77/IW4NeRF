import torch
import numpy as np
import matplotlib.pyplot as plt

# 定义nerf模型
class NerfModel(torch.nn.Module):
    def __init__(self):
        super(NerfModel, self).__init__()
        self.layer1 = torch.nn.Linear(3, 128)
        self.layer2 = torch.nn.Linear(128, 128)
        self.layer3 = torch.nn.Linear(128, 128)
        self.layer4 = torch.nn.Linear(128, 128)
        self.layer5 = torch.nn.Linear(128, 4)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = torch.relu(self.layer4(x))
        x = self.layer5(x)
        return x

# 读取要嵌入的图片（这里假设图片已经转化为点云数据）
img_data = np.load('img/wujing.png', allow_pickle=True)
img_tensor = torch.from_numpy(img_data).float()

# 定义nerf模型并载入训练好的权重
model = NerfModel()
model.load_state_dict(torch.load('logs/blender_paper_lego/006000.tar'))

# 将图片嵌入到nerf模型中
with torch.no_grad():
    img_embed = model(img_tensor.reshape(1, -1))

# 在nerf模型中从前向后和后向前数次采样，确保嵌入效果更加真实
with torch.no_grad():
    ray_dir = torch.rand(3)
    ray_pos = torch.rand(3)
    pos = ray_pos
    for i in range(10):
        value = model(pos.reshape(1, -1))
        pos += ray_dir * value[0][0]

    ray_dir = -ray_dir
    ray_pos = torch.rand(3)
    pos = ray_pos
    for i in range(10):
        value = model(pos.reshape(1, -1))
        pos += ray_dir * value[0][0]

# 提取出嵌入在nerf模型中的图片
with torch.no_grad():
    img_data_extract = img_embed.numpy().reshape(64, 64, -1)
    plt.imshow(img_data_extract)
    plt.show()
