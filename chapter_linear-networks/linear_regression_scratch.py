# %matpoltlib inline
# 这是一个 Jupyter Notebook 的魔法命令（magic command），用于控制 Matplotlib 图表的显示方式. 在vsc中没有作用
# 作用：
# %matplotlib inline 的作用是让 Matplotlib 绘制的图表直接嵌入在 Jupyter Notebook 的单元格输出中，而不是弹出一个单独的窗口显示图表。这对于数据分析和可视化非常方便, 这样可以在 Notebook 中直接看到图表结果.
import random
import torch
from d2l import torch as d2l

#生成一个数据集，包含1000个样本，每个样本含2个特征，
def synthetic_data(w, b, num_examples):
    """生成y = X w + b + 噪声"""
    X = torch.normal(0, 1, (num_examples, len(w))) # 这两个特征服从\mu = 0, \sigma = 1的正态分布
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape) #噪声服从\mu = 0, \sigma = 0.01的正态分布
#y.shape((-1, 1))将y重塑为二维张量，其中行数自动推断(用-1表示)，列数固定为1. 也就是说将y视为一个列向量
    return X, y.reshape((-1, 1))

true_w = torch.tensor([3, -2.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000) #features共1000行，每一行(一个样本)含2个特征，2列; labels 1000行1列, 每一行含一个标记
print('features: ', features[0], '\nlabels: ', labels[0])

#从数据矩阵中随机选取batch_size个数据作为训练集
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    #随机选取
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i : min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]

batch_size = 10
for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break
