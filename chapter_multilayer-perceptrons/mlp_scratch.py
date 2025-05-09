import torch
from d2l import torch as d2l
from torch import nn

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 手动实现relu(x)
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(a, X)

# 隐藏层：含256个参数(一般选择2的幂次)
num_inputs, num_hiddens, num_outputs = 784, 256, 10

# 模型参数
# W1 = nn.parameter(torch.randn(num_inputs, num_hiddens, requires_grad=True))
# TypeError: 'module' object is not callable 大写字母被达成了小写
W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens, requires_grad=True) * 0.01)
# * 0.01的作用：防止从标准正态分布中生成随机数时初始权重过大(否则容易在sigmoid/tanh中直接进入梯度消失/饱和状态)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))
params = [W1, b1, W2, b2]

# 定义模型
def net(X):
    X = X.reshape(-1, num_inputs)
    H = relu(X @ W1 + b1)
    # @表示矩阵乘法，相当于torch.mm(相较于torch.matmul, torch.mm限定于二维矩阵乘法)
    return H @ W2 + b2

# 损失函数，与ch3_softmax_concise.py相同
loss = nn.CrossEntropyLoss(reduction='none')

# 训练轮数与学习率
num_epochs , lr = 10, 0.1
updater = torch.optim.SGD(params, lr=lr)

# 调用上一章定义的训练函数
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)

d2l.predict_ch3(net, test_iter)