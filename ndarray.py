import torch

x = torch.arange(12).reshape(3, 4)
print(x)

y = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1], [1, 2, 3, 3]])
print(y)
print(y.shape) #查询张量的形状

print(x + y, x * y, x ** y) #按元素运算
print(x > y) #按元素比较大小，输出的也是tensor
print(torch.ones(2, 3, 4)) #全为1的张量，形状为2*3*4
print(torch.zeros_like(y)) #形状与张量y相同、元素全为0的张量

a = torch.arange(3).reshape(3, 1)
b = torch.arange(2).reshape(1, 2)
print(a + b) #广播机制，各张量先复制自身至形状相同，再进行运算
#尝试三维张量
#A = torch.ones(2, 3, 4)
#B = torch.zeros(3, 2, 4)
#print(A + B) #RuntimeError: The size of tensor a (3) must match the size of tensor b (2) at non-singleton dimension 1
A = torch.ones(2, 3, 4)
B = torch.zeros(2, 1, 1)
print(A + B)
#广播原则要求对应维度要么相同，要么其中一个就是1，无论数值之间是否有整除关系都无效。这不是数学上的“整除”或“倍数”关系，而是专门设定的一项规则。

X = torch.arange(12, dtype=torch.float32).reshape(3, 4) #dtype表示datatype, float32为32-bit(单精度)浮点数，每个数占用4个字节(32bits)
print(torch.cat((X, y), dim=0))
print(torch.cat((X, y), dim=1)) #concatenate 按不同轴连结两个张量

X[0:2, :] = 12
print(X)
print("X[2] = ", X[2])
print("X[2][0] = ", X[2][0]) #索引与切片

T1 = torch.tensor([1, 2, 1, 1])
T2 = torch.tensor([3, 4, 5, 6])
before = id(T2)
T2 = T2 + T1
print(before == id(T2)) #输出false, python先做计算再将计算结构赋给一个新的张量，这不是我们需要的(需要原地计算)
#解决方案：切片表示法
T3 = torch.zeros_like(T2)
print("id(T3):", id(T3))
T3[:] = T1 + T2
print("id(T3):", id(T3))
#等效的有
#T2[:] = T2 + T1切片，或者
before = id(T2)
T2 += T1
print(before == id(T2)) #输出True
