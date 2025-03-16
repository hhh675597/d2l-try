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
X = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
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
print(A.mean(axis=1), A.sum(axis=1) / len(A[1])) #指定沿哪个轴求平均值，前后两种方式等价
print(len(A)) #输出4？？？这是一个4*5的矩阵
#非降维求和
sum_A = A.sum(axis=1, keepdims=True)
print(sum_A) #tensor([[10.],
             #        [35.],
             #        [60.],
             #        [85.]])在对每行进行求和后仍保持两个轴
print(A / sum_A) #某个元素在所在该行中占的'比例', 用到了广播机制

print(A.cumsum(axis=0)) #tensor([[ 0.,  1.,  2.,  3.,  4.], cumsum函数不会沿任何轴降低输入张量的维数
                        #[ 5.,  7.,  9., 11., 13.],
                        #[15., 18., 21., 24., 27.],
                        #[30., 34., 38., 42., 46.]]) 沿某个轴计算A元素的累计总和.axis=0表示按**行**计算
#点积：相同位置的按元素乘积之和
x = torch.arange(4, dtype=torch.float32) #不加类型：RuntimeError: dot : expected both vectors to have same dtype, but found Long and Float
y = torch.ones_like(x, dtype=torch.float32)
print(torch.dot(x, y))
#矩阵-向量积 Recall: 设A m行n列, 从R^n到R^m的线性映射
print("matrix-vector product")
A = A.reshape(5, 4)
print(A.shape, x.shape)
print(torch.mv(A, x)) #remark:x的长度必须与A的列数相等
#矩阵乘法
print("matrix-matrix multiplication")
B = torch.tensor([[1, 2, 3],
                  [0, 1, 0],
                  [0, 0, 1], 
                  [1, 0, 0]], dtype=torch.float32)
print(torch.mm(A, B)) #要求B的行数等于A的列数
#范数 向量&矩阵
print("norms")
u = torch.tensor([3.0, -4.0])
print(torch.norm(u)) #L2范数，向量所有元素平方和的平方根
print(torch.abs(u).sum()) #L1范数
#矩阵的Frobenius范数：矩阵元素平方和的平方根
print(torch.norm(torch.ones(4, 9))) #输出是tensor(6.) 4行9列的全1矩阵

#####Exercises#####
print(len(X)) #recall:X是2*3*4的矩阵
#结论：len(X)总是返回张量X在轴0上的长度
#tensor([[[ 0,  1,  2,  3],
#         [ 4,  5,  6,  7],
#         [ 8,  9, 10, 11]],
#       [[12, 13, 14, 15],
#         [16, 17, 18, 19],
#          [20, 21, 22, 23]]])
#观察下列按轴求和函数的输出，感觉类似压缩饼干/二向箔/液压机
print(X.sum(axis=0))
#tensor([[12, 14, 16, 18],
#        [20, 22, 24, 26],
#        [28, 30, 32, 34]])
print(X.sum(axis=1))
#tensor([[12, 15, 18, 21],0+4+8,1+5+9,2+6+10,3+7+11
#        [48, 51, 54, 57]])
print(X.sum(axis=2))
#tensor([[ 6, 22, 38], hint:0+1+2+3,4+5+6+7,15+19+23
 #       [54, 70, 86]])
print(A / A.sum(axis=1, keepdim=True)) 
#没有keepdim=True会报错，因为广播机制无法将(5，)变为(5,4)，但可以将(5, 1)变为(5, 4)

print(torch.linalg.norm(X)) #输出65.7571
#RuntimeError: linalg.vector_norm: Expected a floating point or complex tensor as input. Got Long
###################