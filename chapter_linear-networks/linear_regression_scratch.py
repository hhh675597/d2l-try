import random
import torch
from d2l import torch as d2l

#生成一个数据集，包含1000个样本，每个样本含2个特征，
def synthetic_data(w, b, num_examples):
    """生成y = X w + b + 噪声"""
    X = torch.normal(0, 1, (num_examples, len(w))) # 这两个特征服从\mu = 0, \sigma = 1的正态分布，形状为 num_examples \times len(w) 的矩阵
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape) #噪声服从\mu = 0, \sigma = 0.01的正态分布
#y.reshape((-1, 1))将y重塑为二维张量，其中行数自动推断(用-1表示)，列数固定为1. 也就是说将y视为一个列向量
    return X, y.reshape((-1, 1))

#真正的参数，使用该参数生成真实数据集
true_w = torch.tensor([3, -2.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000) #features共1000行，每一行(一个样本)含2个特征，2列; labels 1000行1列, 每一行含一个标记
print('features: ', features[0], '\nlabels: ', labels[0])

#从数据矩阵中随机选取batch_size个数据作为训练集
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples)) #生成索引列表
    random.shuffle(indices) #随机打乱索引
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i : min(i + batch_size, num_examples)]) #每次取出batch_size个索引并转换为张量
        yield features[batch_indices], labels[batch_indices] #利用这些索引从特征张量和标签张量中提取对应批次的数据，用yield返回
#在 Python 中，yield 用于定义一个生成器函数。这种函数返回一个生成器对象，允许你逐步生成序列中的值，而不是一次性创建整个序列。生成器在迭代过程中会“记住”上一次执行结束的位置，并在下一次调用时从该位置继续执行。
#以下是 yield 工作原理的细节：
#暂停函数执行：**当生成器函数调用到 yield 时，会暂停执行，并将当前的值返回给调用者。**
#保存状态：生成器会保存当前局部变量的状态（如循环中的变量），下次迭代时，从暂停的地方继续执行，而不重新初始化。
#内存效率：由于一次只生成一个数据项，不会一次性加载所有数据，所以它在处理大型数据集时更高效。 


#读取第一个小批量数据集并输出(for test)
batch_size = 10
for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break
#问题：空间效率较低，需要执行大量随机内存访问
#解决：使用框架提供的内置迭代器

#初始化模型参数
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

#定义模型
def linreg(X, w, b):
    """线性回归模型"""
    return torch.matmul(X, w) + b #Recall: broadcast机制的使用

#定义损失函数
def squared_loss(y_hat, y):
    """均方损失函数"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2 #需要将y(真实值)的形状转换为与y_hat(预测值)

#定义优化算法
def sgd(params, learning_rate, batch_size):
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= learning_rate * param.grad / batch_size
            param.grad.zero_()
# 在 Python 中，with 语句用于管理上下文资源，确保在进入和退出代码块时自动执行特定操作。
# torch.no_grad() 是一个上下文管理器，用于临时关闭自动求导机制
# 两者综合起来：进入代码块时临时关闭求梯度

#两个超参数，需要手动调整
learning_rate = 0.3
num_epochs = 3 #迭代周期个数

net = linreg
loss = squared_loss

for epoch in range(num_epochs):
#############
    for X, y in data_iter(batch_size, features, labels): #data_iter遍历数据集产生小批量训练集
        l = loss(net(X, w, b), y) #小批量样本X, y的损失

        l.sum().backward() #反向传播存储每个参数的梯度g
        sgd([w, b], learning_rate, batch_size) #(w, b)<-(w, b) - \frac{\eta}{|B|} g, where \eta = learning_rate, |B| = batch_size 
##############
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch: {epoch + 1}, loss:{float(train_l.mean()):f}')

print(w, '\n', b)

print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')
