import torch
from torch import nn
from d2l import torch as d2l

# 与scratch同样的方法得到训练/测试集
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
# nn.Flatten()层将输入的(batch_size, 28, 28)的MNIST图像展平为(batch_size, 28 \times 28)的矩阵,从而可以传入全连接层(进行矩阵乘法)

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)
net.apply(init_weights)

loss = nn.CrossEntropyLoss(reduction='none')
# reduction='none'表示在计算交叉熵损失时，不会对各个样本的损失值进行求和或取均值，而是返回一个每个样本对应的损失值张量
# 这在需要对损失值进行自定义处理或进一步细粒度分析时非常有用
trainer = torch.optim.SGD(net.parameters(), lr=0.1)
# 训练器，与线性回归相同
num_epochs = 10

d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)