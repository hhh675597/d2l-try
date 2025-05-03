import torch
from IPython import display
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

num_inputs = 784 #输入为28 \times 28像素的灰度图
num_outputs = 10 #共计10个类别

w = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True) #权重构成一个784 \times 10的矩阵 
b = torch.zeros(num_outputs, requires_grad=True) #偏置为1 \times 10的行向量

# Recall: 张量加法
# X = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
# print(X.sum(0, keepdim=True)) #限定求和的'方向'：对每一列求和得行向量 tensor([[5., 7., 9.]])
# print(X.sum(1, keepdim=True)) #对每一行求和得到列向量 tensor([[ 6.],
#                               #                          [15.]])
# print(X.sum(1, keepdim=False)) #tensor([ 6., 15.])

def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True) #小批量中每一行是一条样本
    return X_exp / partition #此处运用了广播机制
#测试：
#X = torch.normal(0, 1, (2, 5))
#print(f'X: {X} \nX_exp: {torch.exp(X)}')
#X_prob = softmax(X)
#print(X_prob)
#print(X_prob.sum(1, keepdim=True))
