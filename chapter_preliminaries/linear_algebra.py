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

A = torch.arange(20, dtype=torch.float32).reshape(4, 5)
print(A)
print(A.T) #矩阵与矩阵的转置
print(A[1]) #通过下标访问矩阵A的第(1 + 1)行
print(A[1][2]) #A(2, 3)位于第二行第三列的元素
print(A.shape[0]) #输出5, 表示行向量(沿0轴)的长度

B = torch.tensor([[1, 0, 3], [0, 2, 4], [3, 4, 5]])
print(B == B.T) #symmetric matrix 满足B == B.T
#对于矩阵而言，尽管列向量是单个向量默认的方向，但实际中以行向量表示一条数据样本，列对应于不同的属性
X = torch.arange(24).reshape(2, 3, 4)
print(X) #张量
#按元素二元运算
B = A.clone() #分配新的内存，将A拷贝给B
A += B #给定任意两个相同形状的张量， 任何 按元素二元运算的结果仍是相同形状的张量
print(A, B) #B不会跟着A一起变
print(A * B) #儿童乘法，Hadamard Product
#标量与张量
alpha = 2
print(alpha + X) #标量与张量相加:每个元素都与标量相加
print((alpha * X).shape) #数乘，不改变标量的形状
#降维
A = torch.arange(20, dtype=torch.float32).reshape(4, 5)

A_sum_axis0 = A.sum(axis=0) #指定沿哪一个轴求和来实现降维
print(A_sum_axis0) #tensor([30., 34., 38., 42., 46.]), 0轴的形状在输出形状中消失
print(A_sum_axis0.shape) #torch.Size([5])

A_sum_axis1 = A.sum(axis=1)
print(A_sum_axis1) #tensor([10., 35., 60., 85.])

print(A.sum(axis=[0, 1])) #沿着所有行和列对矩阵求和，等价于A.sum()

print(A.mean(), A.sum() / A.numel()) #求所有元素的平均值mean()
print(A.mean(axis=1), A.sum(axis=1) / A.shape[1]) #指定沿哪个轴求平均值
