from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt

train_data = FashionMNIST(root='./data', train=True, 
                          transform=transforms.Compose([transforms.Resize(size=224), transforms.ToTensor()]), 
                          download=True)

train_loader = Data.DataLoader(dataset=train_data, batch_size=64, shuffle=True, num_workers=0)

# 获取一个批次的数据
for step, (b_x, b_y) in enumerate(train_loader): # b_x是一个批次的图像数据，b_y是对应的标签数据
    # 只获取第一个批次的数据
    if step == 0:
        break
batch_x = b_x.squeeze().numpy() # 将数据转换为numpy数组   squeeze()函数用于去除维度为1的维度
batch_y = b_y.numpy() # 将张量转换为numpy数组
class_label = train_data.classes # 训练集的标签
print("The size of batch in train data:", batch_x.shape) # 输出批次数据的大小

# 可视化图像一个批次的图像
plt.figure(figsize=(12, 5))
for i in np.arange(len(batch_y)):
    plt.subplot(4, 16, i + 1) # 创建4行16列的子图
    plt.imshow(batch_x[i], cmap=plt.cm.gray) # 显示图像，使用灰度色图
    plt.title(class_label[batch_y[i]]) # 设置标题为对应的标签
    plt.axis('off') # 关闭坐标轴显示
    plt.subplots_adjust(wspace=0.1)
plt.show() # 显示图像