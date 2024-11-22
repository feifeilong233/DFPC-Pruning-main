import os

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
#0623更新，将数据输入通道加到了6个
#0706更新，将数据通道数增加到10，并尝试加入数据增强——数据对称
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_absolute_error, \
    mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter

import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.dataset as Dataset
import torch.utils.data.dataloader as DataLoader
import torch.nn.functional as F
#from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

from data_combine1021Alpha_great_combineAll import data_combine
from subDataset import subDataset
# import hiddenlayer as hl
#import tensorboard
from try_resnet_1003 import ResNet
from try_resnet_1003 import BasicBlock
#from try_resnet import Bottleneck
from loss_function_0712Alpha import loss_soft_add, test_recall
from loss_function_0712Alpha import test_soft_add
from save_model import save_model
from save_model import load_model
# from torchviz import make_dot

'''step1 定义参数'''

# 定义超参数
input_size = 5  # 棋盘总尺寸5*5
num_classes = 25  # 标签的种类数
num_epochs = 300  # 训练的总循环周期
batch_size = 64  # 一个撮（批次）的大小，64张图片
build_graph_structure = False  # 是否画出网络结构
use_tensorboard = False  # 是否使用tensorboard
use_test = True
'''step 2 获取数据'''
# 训练集
# Data = np.load("data/input2.npy")
# Label = np.load("data/output2.npy")
# record = np.load("data/record2.npy")
# result = np.load("data/result2.npy")

Data, Label, record, result = data_combine(False,False,5,True,True,True)
count1=0
count2=0
for i in result:
    count1 = count1 + 1
    if i > 0:
        count2 = count2 + 1

X_train, X_test, y_train, y_test = train_test_split(Data, Label, test_size=0.05, random_state=40)

# scale = StandardScaler()
# X_train_s = scale.fit_transform(X_train)
# X_test_s = scale.fit_transform(X_test)
train_xt = torch.from_numpy(X_train.astype(np.float32))
train_yt = torch.from_numpy(y_train.astype(np.float32))
test_xt = torch.from_numpy(X_test.astype(np.float32))
test_yt = torch.from_numpy((y_test.astype(np.float32)))

trainData = subDataset(train_xt, train_yt)
testData = subDataset(test_xt, test_yt)
train_loader = DataLoader.DataLoader(dataset=trainData, batch_size=batch_size, shuffle=True)
test_loader = DataLoader.DataLoader(dataset=testData, batch_size=batch_size, shuffle=True)
'''step3 构建网络模型'''

### 卷积网络模块构建
# - 一般卷积层，relu层，池化层可以写成一个套餐
# - 注意卷积最后结果还是一个特征图，需要把图转换成向量才能做分类或者回归任务
# - 注意从上到下棋子顺序为右，下，右下


# %% md

### 准确率作为评估标准

# %%


'''step4 训练模型'''
# 实例化
current_time = time.strftime("%Y-%m-%dT%H:%M", time.localtime())
device = torch.device("cuda")
net = ResNet(BasicBlock, [1, 1, 1, 1])
net=net.cuda(device)
#     print(name)
#     print(param.data)
#     print("requires_grad:", param.requires_grad)
#     print("-----------------------------------")

# 损失函数
criterionnew = nn.MSELoss().cuda(device)
# criterion = nn.CrossEntropyLoss()
criterion = loss_soft_add().cuda(device)
test_cal = test_soft_add().cuda(device)
test_recall_cal = test_recall().cuda(device)
learning_rate = 0.01
#optimizer = torch.optim.SGD(resnet50.parameters(), lr=learning_rate, )
optimizer=torch.optim.Adam(net.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
# 优化器
#optimizer = optim.SGD(net.parameters(), lr=0.00001, weight_decay=0.1)
# optimizer = optim.Adagrad(net.parameters(), lr=0.001, weight_decay=0.1)  # 定义优化器
# if use_tensorboard is False:
#     history1 = hl.History()
#     canvas1 = hl.Canvas()

# if build_graph_structure is True:
#     x = torch.randn(1, 2, 5, 5).requires_grad_(True)  # 定义一个网络的输入值
#     x = x.cuda(device)
#     y = net(x)  # 获取网络的预测值
#     MyConvNetVis = make_dot(y, params=dict(list(net.named_parameters()) + [('x', x)]))
#     MyConvNetVis.format = "png"
#     # 指定文件生成的文件夹
#     MyConvNetVis.directory = "output/structure"
#     # 生成文件
#     MyConvNetVis.view()

# if use_tensorboard is True:
#     writer = SummaryWriter(log_dir='logs')

# 开始训练循环
for epoch in range(num_epochs):
    file1 = open('0402_1111_Alpha_train.txt', 'a+')

    # 当前epoch的结果保存下来
    print("we are in ", epoch)
    train_rights = []
    train_loss = 0
    test_recall_v = 0
    my_right_recall = 0
    for batch_idx, (data, target) in enumerate(train_loader):  # 针对容器中的每一个批进行循环
        net.train()

        if torch.cuda.is_available():
            data = data.cuda(device)
            target = target.cuda(device)
        output = net(data)
        # my_right_recall = my_right_recall + test_recall_cal(output,target)

        # print(output)
        # target = target.reshape(batch_size, -1)
        if (epoch * len(train_loader) + batch_idx + 1) % 20 == 0:
            accuracy_2 = test_cal(output, target)
            accuracy_2 = accuracy_2 ** 0.5
            print('the accuracy is ', accuracy_2)
        Before = list(net.parameters())[1].clone()
        loss = criterion(output, target)
        optimizer.zero_grad()
        train_loss = loss.item() * data.size(0)
        # print(train_loss)
        loss.backward()
        optimizer.step()
        # After = list(net.parameters())[1].clone()
        niter = epoch * len(train_loader) + batch_idx + 1
        # print('模型的第1层更新幅度：', torch.sum(After - Before))
        torch.cuda.empty_cache()

        # accuracy_2 = test_cal(output, target)


    # accuracy_1 = my_right_recall / count_recall_train
    # file3.write(str(accuracy_1)+'\n')
    # print('the recall is ', accuracy_1)
    accuracy_2 = test_cal(output, target)
    accuracy_2 = accuracy_2 ** 0.5
    file1.write(str(accuracy_2) + '\n')
    print('the accuracy is ', accuracy_2)
    file1.close()
    if epoch % 5 == 0:
        save_model('0402model_dict_Alpha.pth', epoch, optimizer, net)
        torch.save(net.state_dict(), '0402_1111_Alpha2.pt')
    if use_test is True:
        if epoch % 5 == 0:
            file2 = open('0402_1111_Alpha_test.txt', 'a+')
            net.eval()
            test_accuracy = 0
            test_recall_v = 0
            for test_batch, (data_test, data_target) in enumerate(test_loader):
                if torch.cuda.is_available():
                    data_test = data_test.cuda(device)
                    data_target = data_target.cuda(device)
                    output = net(data_test)
                    accuracy_1 = test_cal(output, data_target)
                    test_accuracy += accuracy_1**0.5

                    # 计算自定义召回率
                    # accuracy_2 = test_recall_cal(output, data_target)
                    # test_recall_v += accuracy_2
            # output=torch.from_numpy(output.astype(np.float32))
            test_accuracy /= len(test_loader)
            print(test_accuracy)
            file2.write(str(test_accuracy) + '\n')
            file2.close()
            # test_recall_v /= len(test_loader)
            # print('the test recall is ', test_recall_v)

            torch.cuda.empty_cache()

            # if epoch >= 2 and use_tensorboard is True:
            #     writer = SummaryWriter(log_dir='logs')
            #     writer.add_scalar('train_loss', train_loss, epoch)
            #     writer.add_scalar('test_loss', test_accuracy, epoch)
            #     writer.add_scalar('my_recall', test_recall_v, epoch)

            # if epoch >= 2 and use_tensorboard is False:
            #     history1.log(niter, train_loss=train_loss,
            #                  test_accuracy=test_accuracy)
            #
            #     with canvas1:
            #         canvas1.draw_plot(history1["train_loss"])
            #         canvas1.draw_plot(history1["test_accuracy"])
# writer.close()
# save_model('0716model_dict_Alpha.pth',epoch, optimizer, net)
torch.save(net.state_dict(), '0402_1111_Alpha2.pt')
# tensorboard --logdir C:\Users\Elessar\Desktop\Game_theory\chess\logs
# nvidia-smi

# for name, param in mymodel.named_parameters():
#     print(name)
#     print(param.data)
#     print("requires_grad:", param.requires_grad)
#     print("-----------------------------------")

