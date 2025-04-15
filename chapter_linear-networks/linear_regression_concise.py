import random
import torch
from torch.utils import data # utils是 utilities 的缩写，Pytorch的一个子包，它包含了一些通用工具和模块，用于简化数据加载、模型管理和其他任务
from d2l import torch as d2l

#w, b的真实值
true_w = torch.tensor([5.2, 4.1]) #expected svalar type Float but found long, 这里的参数必须是float
true_b = 3.2

#调用d2l库中的syncthetic_data函数(见linear_regression_scratch.py)生成数据集
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

def load_array(data_arrays, batch_size, is_train=True):
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train) # 布尔值is_train表示是否希望数据迭代器每个迭代周期内将数据打乱

batch_size = 10
data_iter = load_array((features, labels), batch_size)

print(next(iter(data_iter))) #data_iter的使用方法与上一节相同，与上节不同，这里我们使用iter构造Python迭代器，并使用next从迭代器中获取第一项

from torch import nn
net = nn.Sequential(nn.Linear(2, 1)) #此处仅含一层fully connected层; (2, 1)指定输入输出尺寸

#初始化网络参数
net[0].weight.data.normal_(0, 0.01) # 直接用net[0]访问第一个网络(此处仅存的一个网络)
net[0].bias.data.fill_(0)
#损失函数
loss = nn.MSELoss() #平方L2范数，返回所有样本损失的平均值
#优化算法
trainer = torch.optim.SGD(net.parameters(), lr=0.03) #需要指定网络参数, 以及超参数：学习率(在小批量随机梯度下降中，仅有这一个超参数需要手动设置)
#################以上简洁地完成了所有基本组件，剩下的与_scratch中的操作类似##############################

num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')

w = net[0].weight.data
print('w的估计误差: ', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差: ', true_b - b)