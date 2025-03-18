import torch

x = torch.arange(4.0)
#print(x)
#不会每次求导都分配新的内存
x.requires_grad_(True) #等价的写法：x = torch.arange(4.0, requires_grad=True)

##########grad#############
#当 requires_grad=True 时，PyTorch 会记录所有在该张量上的操作
#这些操作会构建一个计算图，用于后续的反向传播计算梯度
#.grad 属性会存储反向传播时计算的梯度值
#注意：只有浮点类型张量支持求导操作！！！
#默认情况下，requires_grad=False
#叶子节点（如输入张量）的 .grad 属性会在反向传播后被填充
#中间结果的梯度会被自动释放以节省内存
###########################

print(x.grad) #此时输出为None
#Let y = 2 \mathbf{x}^\top \mathbf{x}
y = 2 * torch.dot(x, x) #向量x的转置乘x，相当于x与自身点积, 最后得到一个标量
print(y) #tensor(28., grad_fn=<MulBackward0>)
#调用反向传播函数自动计算y关于x每个分量的梯度
y.backward() #backpropagate
print(x.grad) #tensor([ 0.,  4.,  8., 12.])
print(x.grad == 4 * x)

######backward()##############
#计算图构建:
#当执行前向计算时（如 y = 2 * torch.dot(x, x)），PyTorch 自动构建计算图
#每个操作都被记录为图中的一个节点
#计算图记录了从输入到输出的完整计算路径
#梯度计算:
#调用 backward() 时，PyTorch 从输出节点（这里是 y）开始
#使用链式法则，逐层计算每个变量的梯度
#梯度按照计算图的反向顺序传播
#结果存储:
#对于 requires_grad=True 的叶子节点（如 x），梯度被存储在其 .grad 属性中
#中间节点的梯度计算完后会被释放以节省内存
###########################

x.grad.zero_() #在默认情况下，pytorch会累计梯度，需要清除之前的值
y = x.sum()
y.backward()
print(x.grad) #tensor([1, 1, 1, 1])
#非标量变量的backpropagate
x.grad.zero_()
y = x * x #Recall: * 按元素乘法，得到一个与x形状相同的向量
y.backward(torch.ones_like(x)) #本例中求每个样本的偏导数之和
#理解：这里的“偏导数之和”应指对向量𝑥每个分量𝑥𝑖求偏导数的和
print(x.grad)
#####对向量/标量调用backward()#######
# 对非标量调用backward需要传入一个gradient参数，该参数指定微分函数关于self的梯度。
# 本例只想求偏导数的和，所以传递一个1的梯度是合适的
# 详见hhh/d2l-zh/pytorch:Jupyter notebook
##############################

#注意：原来求导结果应该是一个矩阵，这些现象会在以后深度学习中出现
#[0, 0, 0, 0]
#[0, 2, 0, 0]
#[0, 0, 4, 0]
#[0, 0, 0, 6]

#分离计算: 考虑y = y(x),z = z(x, y),需要计算z关于x的梯度，有时希望将y视作常数，即只考虑x在y被计算后发挥的作用
x.grad.zero_()
y = x * x
u = y.detach() #分离y返回一个新的变量u.该变量与y具有相同的值,但丢弃计算图中如何计算y的任何信息.即梯度不会向后流经u到x
z = u * x
z.sum().backward()
print(x.grad) #输出tensor([0, 1, 4, 9]) ,即向量u
print(x.grad == u) #tensor([True, True, True, True])

x.grad.zero_()
y.sum().backward()
print(x.grad) #tensor([0., 2., 4., 6.])

#对比：不使用u, z = y * x = x * x * x
x.grad.zero_()
#z = y * x
#z.sum().backward()
#print(x.grad) #tensor([ 0.,  3., 12., 27.])即3 x^{2}

#######报错#############报错原因：上三行中y已经被反向传播过了
#RuntimeError: Trying to backward through the graph a second time 
#(or directly access saved tensors after they have already been freed).
#Saved intermediate values of the graph are freed when you call 
#.backward() or autograd.grad(). 
#Specify retain_graph=True if you need to backward through the graph 
#a second time or if you need to access saved tensors
#after calling backward.
#######################

y = x * x #解决：重新写一遍
z = y * x
z.sum().backward()
print(x.grad)

#Python控制流(条件，循环或任意函数调用)中的梯度计算
def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    
    return c

a = torch.randn(size=(2, 3), requires_grad=True) #randn服从标准正态分布
#size=?控制形状()空括号，标量;(2,)长度为2;(2, 3)2 * 3矩阵;(2, 3, 4)依此类推
d = f(a) #f在输入a中是分段线性的，对任意a，存在k,f(a) = k * a
d.backward(torch.ones(2, 3))
print(a.grad == d / a)

#######Exercises##########
def Euclidean(a):
    for i in range(2):
       a = a * a
    return a

matrix_2 = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.float32, requires_grad=True)
matrix_3 = torch.tensor([[1], [2], [3]], dtype=torch.float32, requires_grad=True)
d = Euclidean(matrix_2)
d.backward(torch.ones_like(matrix_2))
print(matrix_2.grad)
d = Euclidean(matrix_3)
d.backward(torch.ones_like(matrix_3))
print(matrix_3.grad) #4 x^{3}, x为分量

##########################