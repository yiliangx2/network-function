import torch
import torch.nn as nn
import torch.utils.data as Data
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import copy
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from model import GoogLeNet, Inception

def train_val_data_process():
    train_data = FashionMNIST(root='./data', train=True, 
                              transform=transforms.Compose([transforms.Resize(size=224), transforms.ToTensor()]),
                              download=True)
    
    train_data, val_data = Data.random_split(train_data, [round(len(train_data) * 0.8), round(len(train_data) * 0.2)])

    train_dataloader = Data.DataLoader(dataset=train_data, batch_size=64, shuffle=True, num_workers=6)
    val_dataloader = Data.DataLoader(dataset=val_data, batch_size=64, shuffle=True, num_workers=6)


    return train_dataloader, val_dataloader

def train_model_process(model, train_dataloader, val_dataloader, num_epochs):
    # 设定训练设备，如果有GPU则使用GPU，否则使用CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 使用Adam优化器和交叉熵损失函数， 学习率为0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    # 将模型放入训练设备中
    model = model.to(device)
    # 复制当前模型的参数
    best_model_wts = copy.deepcopy(model.state_dict())

    # 初始化参数
    # 最高准确度
    best_acc = 0.0
    # 训练和验证的损失和准确度列表
    train_loss_all = []
    val_loss_all = []
    train_acc_all = []
    val_acc_all = []
    # 训练开始时间
    since = time.time()

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        print("-" * 10)

        # 初始化参数
        # 训练集、验证集的损失和准确度
        train_loss = 0.0
        train_corrects = 0
        val_loss = 0.0
        val_corrects = 0
        # 训练集、验证集的样本数量
        train_num = 0
        val_num = 0

        # 训练阶段
        for step, (b_x, b_y) in enumerate(train_dataloader): # enumerate()函数用于将一个可遍历的数据对象组合为一个索引序列，同时列出数据和数据下标
            # 将数据放入训练设备中
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            # 设置模型为训练模式
            model.train()

            # 前向传播 输入为当前批次的图像数据，输出为模型的预测结果
            output = model(b_x)
            pre_lab = torch.argmax(output, dim=1) # torch.argmax()函数返回输入张量中每行的最大值的索引，dim=1表示对第一维进行操作
            loss = criterion(output, b_y) # 计算当前批次的损失值，criterion()函数根据模型的输出和真实标签计算损失值

            # 将梯度置零，进行反向传播，更新模型参数
            optimizer.zero_grad()
            loss.backward() # loss.backward()函数用于计算损失函数的梯度
            optimizer.step() # optimizer.step()函数用于更新模型参数

            # 对当前批次的损失值和正确预测的数量进行累加
            train_loss += loss.item() * b_x.size(0) # loss.item()函数用于获取当前批次的损失值，b_x.size(0)表示当前批次的样本数量
            train_corrects += torch.sum(pre_lab == b_y) 

            # 对当前批次的样本数量进行累加
            train_num += b_x.size(0)

        # 验证阶段
        for step, (b_x, b_y) in enumerate(val_dataloader):
            # 将数据放入验证设备中
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            model.eval() # 设置模型为评估模式

            # 前向传播 输入为当前批次的图像数据，输出为模型的预测结果
            output = model(b_x)
            pre_lab = torch.argmax(output, dim=1)
            loss = criterion(output, b_y)

            # 对当前批次的损失值和正确预测的数量进行累加
            val_loss += loss.item() * b_x.size(0)
            val_corrects += torch.sum(pre_lab == b_y)
            # 对当前批次的样本数量进行累加
            val_num += b_x.size(0)

        # 将当前epoch的训练集和验证集的平均损失值和准确度添加到列表中
        train_loss_all.append(train_loss / train_num)
        train_acc_all.append(train_corrects.double().item() / train_num)
        val_loss_all.append(val_loss / val_num)
        val_acc_all.append(val_corrects.double().item() / val_num)

        # 输出当前epoch的训练集和验证集的平均损失值和准确度
        print("{} train loss: {:.4f} acc: {:.4f}".format(epoch + 1, train_loss_all[-1], train_acc_all[-1]))
        print("{} val loss: {:.4f} acc: {:.4f}".format(epoch + 1, val_loss_all[-1], val_acc_all[-1]))

        # 如果当前epoch的验证集准确度高于之前的最高准确度，则更新最高准确度和最佳模型参数
        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())

        # 训练结束时间
        time_use = time.time() -since
        print("训练和验证的时间: {:.0f}m {:.0f}s".format(time_use // 60, time_use % 60))

    # 将最佳模型参数保存到文件中
    torch.save(best_model_wts, 'best_model.pth') 

    train_process = pd.DataFrame(data={"epoch": range(num_epochs),
                                        "train_loss_all": train_loss_all,
                                        "val_loss_all": val_loss_all,
                                        "train_acc_all": train_acc_all,
                                        "val_acc_all": val_acc_all})
    
    return train_process

def matplot_acc_loss(train_process):
    # 绘制训练集和验证集的损失值曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_process["epoch"], train_process.train_loss_all, 'ro-', label="train_loss")
    plt.plot(train_process["epoch"], train_process.val_loss_all, 'bs-', label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()

    # 绘制训练集和验证集的准确度曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_process["epoch"], train_process["train_acc_all"], 'ro-', label="train_acc")
    plt.plot(train_process["epoch"], train_process["val_acc_all"], 'bs-', label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()

    plt.show()

if __name__ == "__main__":
    GoogLeNet= GoogLeNet(Inception) # 将模型实例化
    train_dataloader, val_dataloader = train_val_data_process() # 获取训练集和验证集的数据
    train_process = train_model_process(GoogLeNet, train_dataloader, val_dataloader, num_epochs=20)
    matplot_acc_loss(train_process)
