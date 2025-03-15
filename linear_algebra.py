import torch

x, y = torch.tensor(1.0), torch.tensor(2.0)
print(x + y, x - y, x * y) #一个元素的张量为标量

x = torch.arange(12)
print(x) #向量(数字数组)
print(x[0], x[11]) #通过张量的索引访问任一元素
print(len(x)) #py内置函数len()来访问张量/向量的长度/维数
#特别注意！dimension的具体意义需要关注上下文context
print(x.shape) #shape是一个元素组，列出张量沿每个轴的维数.对于只有一个轴的张量，shape仅含一个元素
linear_relevant = torch.tensor([[2, 3, 5, 7], [4, 6, 10, 14]]) #猜shape会作矩阵行/列变换吗？会,输出[1, 4],不会，输出[2, 4]
print(linear_relevant.shape) #果然不会