import os

import torch
from torchvision import transforms
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models


def resnet18_model(n_classes):
    """
    加载resnet18模型
    :param n_classes:
    :return:
    """
    resnet18 = models.resnet18(pretrained=True)

    # 调整Linear层
    num_ftrs = resnet18.fc.in_features
    resnet18.fc = nn.Sequential(nn.Linear(num_ftrs, n_classes),
                                nn.LogSoftmax(dim=1))
    return resnet18


def vgg11_model(n_classes):
    """
    加载vgg11模型
    :param n_classes:
    :return:
    """
    googlenet = models.googlenet(pretrained=True)
    num_ftrs = googlenet.fc.in_features
    # 调整Linear层
    googlenet.fc = nn.Sequential(nn.Linear(num_ftrs, n_classes),
                                nn.LogSoftmax(dim=1))
    return googlenet


def train(train_data, test_data, model_name, dataset_name, n_classes):
    """
    模型训训练，最后将模型结果存储在./models下
    :param train_data:
    :param test_data:
    :param model_name:
    :param dataset_name:
    :param n_classes:
    :return:
    """
    train_data_size = len(train_data)
    test_data_size = len(test_data)
    print("训练集的数量： {}".format(train_data_size))
    print("测试集的数量： {}".format(test_data_size))

    # dataloder进行数据集的加载
    train_dataloader = DataLoader(train_data, batch_size=16)
    test_dataloader = DataLoader(test_data, batch_size=16)
    if model_name == "resnet18":
        model = resnet18_model(n_classes)
    elif model_name == "googlenet":
        model = vgg11_model(n_classes)
    else:
        return None
    # 网络模型cuda
    if torch.cuda.is_available():
        model = model.cuda()

    # 定义损失函数loss function
    loss_fn = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        loss_fn = loss_fn.cuda()
    # 定义优化器optimizer
    learning_rate = 0.01
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # 设置网络训练的一些参数
    # 记录训练的次数
    total_train_step = 0
    # 记录测试的次数
    total_test_step = 0
    # 训练的轮数
    epoch = 10
    # 开始迭代训练
    for i in range(epoch):
        print("-------第{}轮训练开始-------".format(i + 1))
        model.train()
        # 训练步骤开始
        for data in train_dataloader:
            imgs, targets = data
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()
            outputs = model(imgs)
            loss = loss_fn(outputs, targets)
            # 优化器优化模型
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 输出训练误差
            total_train_step = total_train_step + 1
            if total_train_step % 100 == 0:
                print("训练次数：{}, Loss: {}".format(total_train_step, loss.item()))

        # 测试集
        total_test_loss = 0
        with torch.no_grad():
            for data in test_dataloader:
                imgs, targets = data
                if torch.cuda.is_available():
                    # 图像cuda；标签cuda
                    # 训练集和测试集都要有
                    imgs = imgs.cuda()
                    targets = targets.cuda()
                outputs = model(imgs)
                loss = loss_fn(outputs, targets)
                total_test_loss += loss.item()
                total_test_step += 1
                # 输出测试误差
                if total_test_step % 100 == 0:
                    print("测试次数：{}，Loss：{}".format(total_test_step, loss))
    # 保存模型
    if not os.path.exists("./models"):
        os.mkdir("./models")
    torch.save(model, f"./models/{model_name}_{dataset_name}.pth")
