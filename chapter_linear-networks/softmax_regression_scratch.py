import torch
from IPython import display
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

num_inputs = 784 #输入为28 \times 28像素的灰度图
num_outputs = 10 #共计10个类别

W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True) #权重构成一个784 \times 10的矩阵 
b = torch.zeros(num_outputs, requires_grad=True) #偏置为1 \times 10的行向量
#print(W.shape[0]) #784

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

####注意########
# 虽然这在数学上看起来是正确的，但我们在代码实现中有点草率
# exp矩阵中的非常大或非常小的元素可能造成数值上溢或下溢，但我们没有采取措施来防止这点。
################

#测试：
#X = torch.normal(0, 1, (2, 5))
#print(f'X: {X} \nX_exp: {torch.exp(X)}')
#X_prob = softmax(X)
#print(X_prob)
#print(X_prob.sum(1, keepdim=True))

#定义模型
def net(X):
    return softmax(torch.matmul(X.reshape(-1, W.shape[0]), W) + b) #先用reshape函数将图像转换为向量(行数-1自动判断，列数W.shape[0]为参数矩阵W的行数)

#定义损失函数
 #测试
 #y_hat = torch.tensor([[0.1, 0.3, 0.6], 
 #                    [0.1, 0.4, 0.5]]) #两行各表示一个样本，三个数值表示模型对该样本属于哪个类别的估计.
 #y = torch.tensor([0, 2])              #y为正确标记，即第一个样本属于第0 + 1 = 1类，第二个样本属于第2 + 1 = 3类
 #print(y_hat[[0, 1], y]) #输出：tensor([0.1000, 0.5000]) #(书上注释：然后使用y作为y_hat中概率的索引， 我们选择第一个样本中第一个类的概率和第二个样本中第三个类的概率)
 #高级索引(花式索引)：第一个[0, 1]指定了选取该 2 \times 3张量的第0行和第1行; 第二个索引y即[0, 2]指定了与前面选取的行对应的列索引 \therefore 整体选取第0行第0列 & 第1行第2列
