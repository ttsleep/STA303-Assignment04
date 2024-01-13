import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import os
from torchvision import transforms
from training import train
from torchcp.classification.predictors import ClusterPredictor, ClassWisePredictor, SplitPredictor
from torchcp.classification.scores import THR, APS, SAPS
import pandas as pd
import matplotlib.pyplot as plt
os.chdir(os.path.dirname(os.path.abspath(__file__)))

"""
Conformal prediction 是一种置信度预测方法，给定测试示例和经过训练的分类器，CP会生成具有用户指定覆盖范围的预测结果，不同于一般的分类或回归任务，在分类任务上CP预测的是一个集合，在回归任务上CP预测的是一个区间。
根据基础模型的好坏和指定的显著性水平alpha，在分类任务上，较小的显著性水平，即允许的误差较小，会产生较大的预测集；在回归任务上，较小的显著性水平，会产生较宽的区间并且不太具体，反之亦然，允许的误差越大，预测区间就越窄。

ClusterPredictor: 将具有“相似”保形分数的类聚类在一起，并在聚类级别执行保形预测
ClassWisePredictor: 可以对每个类别的预测误差率进行更精确的控制
SplitPredictor: 适用于数据分布在不同子集中的情况

THR: 最小化模糊性
APS: 除了提供边际覆盖外，还完全适应复杂的数据分布
SAPS: 该算法丢弃了除最大softmax概率外的所有概率值。SAPS背后的关键思想是在保留不确定性信息的同时，最大限度地减少不合格分数对概率值的依赖性。

1. 数据集和模型
    1.1. 选择的数据集：cifar（5000， 10000）、mnist（60000， 10000），（TrainSet， TestSet）
    1.2. 选择的算法：resnet18、googlenet
2. 过程
   2.1 分别使用数据集cifar、mnist和模型resnet18、googlenet训练4个模型
   2.2 计算4个训练好的模型在ClusterPredictor, ClassWisePredictor, SplitPredictor及THR, APS下的结果
   2.3 在4个训练好的模型下，使用SplitPredictor，当SAPS的weights取不同值的结果
   2.3 保存结果并画图
"""


def load_data(dataset_name, train_flag):
    """
    加载数据集，包括mnist和cifar10，每个数据集分为训练集和测试集，数据集会存储在./images文件夹下
    :param dataset_name:
    :param train_flag:
    :return:
    """
    data = None
    data_dir = "./images/"
    if train_flag:
        data_dir = data_dir + "/train"
    else:
        data_dir = data_dir + "/test"
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    if dataset_name == "mnist":
        transform = transforms.Compose([transforms.Grayscale(num_output_channels=3),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                             (0.2023, 0.1994, 0.2010))])
        data = datasets.MNIST(data_dir, train=train_flag, download=True,
                              transform=transform)
    elif dataset_name == "cifar10":
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                             (0.2023, 0.1994, 0.2010))])
        data = datasets.CIFAR10(data_dir, train=train_flag, download=True,
                                transform=transform)
    return data


def score(model, dataloader):
    """
    用于计算准确率
    :param model:
    :param dataloader:
    :return:
    """
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            imgs, targets = data
            if torch.cuda.is_available():
                # 图像cuda；标签cuda
                # 训练集和测试集都要有
                imgs = imgs.cuda()
                targets = targets.cuda()
            outputs = model(imgs)
            numbers, predicted = torch.max(outputs.data, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
    return correct/total


def cp_evaluation(model, dataset_name, model_name):
    """
    计算不同predictor下不同score_function的结果
    :param model:
    :param dataset_name:
    :param model_name:
    :return:
    """
    params = {
        "predictor": [ClusterPredictor, ClassWisePredictor, SplitPredictor],
        "score": [THR, APS]
    }
    result = dict()
    alpha = 0.1
    for predictor_ in params["predictor"]:
        for score_ in params["score"]:
            score_function = score_()
            predictor = predictor_(score_function, model)
            print(
                f"Experiment--Data : {dataset_name}, Model : {model_name}, Score : {score_.__name__}, Predictor : {predictor_.__name__}, Alpha : {alpha}")
            print(f"The size of calibration set is {len(train_dataloader)}.")
            predictor.calibrate(train_dataloader, alpha)
            res_test = predictor.evaluate(test_dataloader)
            print(f"test: {res_test}")
            result[predictor_.__name__ + "-" + score_.__name__] = (res_test["Coverage_rate"], res_test["Average_size"])
    return result


def cp_evaluation_saps(model):
    """
    计算SAPS方法在不同weights下的结果
    :param model:
    :param dataset_name:
    :param model_name:
    :return:
    """

    params = {
        "weights": [0.1, 0.5, 0.9, 1, 1.1, 1.5]
        # "weights": [0.9, 1]
    }
    result = dict()
    alpha = 0.1
    for weight_ in params["weights"]:
        score_function = SAPS(weight_)
        predictor = SplitPredictor(score_function, model)
        print(f"The size of calibration set is {len(train_dataloader)}.")
        predictor.calibrate(train_dataloader, alpha)
        res_test = predictor.evaluate(test_dataloader)
        print(f"test: {res_test}")
        result[f"weight-{weight_}"] = (res_test["Coverage_rate"], res_test["Average_size"])
    return result


def visualize(data, dataset):
    """
    画不同weights下的saps变化折线图
    :param data:
    :param dataset:
    :return:
    """
    x1 = [float(item.split("-")[-1]) for item in data.columns]
    fig, ax1 = plt.subplots(figsize=(8, 6))

    for idx in data.index:
        y = data.loc[idx].values
        y1 = [item[0] for item in y]
        ax1.plot(x1, y1, ls='--', label=idx.split("-")[0] + "-CoverageRate")
    ax1.set_xlabel('Weights', fontsize=16)
    ax1.set_ylabel('CoverageRate', color='b', fontsize=16)
    ax1.legend(loc='upper left', shadow=True,)

    ax2 = ax1.twinx()
    for idx in data.index:
        y = data.loc[idx].values
        y2 = [item[1] for item in y]
        ax2.plot(x1, y2, ls=":", label=idx.split("-")[0] + "-AverageSize")
    ax2.legend(loc='upper right', shadow=True)

    # ax.tick_params('y', colors='b')
    ax2.set_ylabel('AverageSize', color='r', fontsize=16)
    # ax2.tick_params('y', colors='r')

    plt.legend()
    plt.title('SAPS', fontsize=16)
    plt.savefig(f"{dataset}-SAPS.png", dpi=300)
    # plt.show()


if __name__ == '__main__':
    index_lst = []
    result = []
    result_saps_dic = dict()
    index_saps = []
    params = {"models": ["googlenet", "resnet18"],
             "datasets": ["cifar10", "mnist"]}
    # params = {"models": ["googlenet"],
    #          "datasets": ["cifar10"]}
    for model_name in params["models"]:
        for dataset_name in params["datasets"]:
            # 遍历模型及数据集

            n_classes = 10
            train_data = load_data(dataset_name, True)
            test_data = load_data(dataset_name, False)
            # dataloder进行数据集的加载
            train_dataloader = DataLoader(train_data, batch_size=1024, pin_memory=True)
            test_dataloader = DataLoader(test_data, batch_size=1024, pin_memory=True)

            # train(train_data, test_data, model_name, dataset_name, n_classes)
            model = torch.load(f"./models/{model_name}_{dataset_name}.pth")
            if torch.cuda.is_available():
                model = model.to("cuda:0")
            model.eval()
            # acc = score(model, test_dataloader)
            # print(acc)
            index = model_name + "-" + dataset_name
            index_lst.append(index)
            result.append(cp_evaluation(model, dataset_name, model_name))
            if dataset_name not in result_saps_dic.keys():
                result_saps_dic[dataset_name] = []
            result_saps_dic[dataset_name].append(cp_evaluation_saps(model))

    result = pd.DataFrame(result, index=index_lst)
    result.to_csv("./result.csv")
    print(result)
    result_saps = None
    for dataset_name in result_saps_dic.keys():
        index_saps = [item for item in index_lst if dataset_name in item]
        result_saps_ = pd.DataFrame(result_saps_dic[dataset_name], index=index_saps)
        visualize(result_saps_, dataset_name)
        if result_saps is not None:
            result_saps = pd.concat([result_saps, result_saps_])
        else:
            result_saps = result_saps_
    result_saps.to_csv("./result_saps.csv")
    print(result_saps)
