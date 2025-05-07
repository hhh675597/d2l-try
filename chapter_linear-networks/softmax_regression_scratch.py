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
# 虽然这在数学上看起来是正确的，但在代码实现中有点草率
# exp矩阵中的非常大或非常小的元素可能造成数值上溢或下溢，但我们没有采取措施来防止这点
################

#测试：
#X = torch.normal(0, 1, (2, 5))
#print(f'X: {X} \nX_exp: {torch.exp(X)}')
#X_prob = softmax(X)
#print(X_prob)
#print(X_prob.sum(1, keepdim=True))

#定义模型
def net(X):
    return softmax(torch.matmul(X.reshape(-1, W.shape[0]), W) + b) 
    # 先用reshape函数将图像转换为向量，其中：行数-1根据元素总数自动判断
    # 列数W.shape[0]为参数矩阵W的行数，使之能够执行矩阵乘法

#定义损失函数
 #举例
y_hat = torch.tensor([[0.1, 0.3, 0.6], 
                     [0.1, 0.4, 0.5]]) #两行各表示一个样本，三个数值表示模型对该样本属于哪个类别的估计.
y = torch.tensor([0, 2])              #y为正确标记，即第一个样本属于第0 + 1 = 1类，第二个样本属于第2 + 1 = 3类
 #print(y_hat[[0, 1], y]) #输出：tensor([0.1000, 0.5000]) #(书上注释：然后使用y作为y_hat中概率的索引， 我们选择第一个样本中第一个类的概率和第二个样本中第三个类的概率)
 #高级索引(花式索引)：第一个[0, 1]指定了选取该 2 \times 3张量的第0行和第1行，即全部的两个样本; 第二个索引y即[0, 2]指定了与前面选取的行对应的列索引 \therefore 整体选取第0行第0列 & 第1行第2列
#交叉熵损失
def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y)), y]) #直接提取真实类别的概率预测
print(cross_entropy(y_hat, y)) #tensor([2.3026, 0.6931])

 #Recall:
print(y_hat[1].shape) #torch.Size[3]
print(y_hat.shape[0]) #2
print(y_hat.shape[1]) #3
print(y_hat.shape) #torch.Size[2, 3]
print(len(y_hat.shape)) #2，等效于y_hat.shape[0]


#精度:
def accuracy(y_hat, y):
    """模型预测正确的数量"""
    if  y_hat.shape[0]> 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1) #选取 每一行中最大概率对应的项 作为 模型预测给出的结果
    cmp = y_hat.type(y.dtype) == y # ==运算符对数据类型敏感,故此处强制转换类型保证不影响比较的结果
    return float(cmp.type(y.dtype).sum())
    #结果是一个n行的列向量，bool类型，0错1对. 最后将n个元素相加，算出正确预测的数目
 # 测试
 # print(accuracy(y_hat, y)) #输出1.0
 # 在上例中,真实标签向量y=[0, 2]第一个样本属于第1类，第二个样本属于第3类;
 # 模型预测矩阵y_hat=[[0.1,0.3,0.6],[0.1,0.4,0.5]]显示第一个样本属于第3类，第二个样本属于第3类

class Accumulator:
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)] # 同时累加多个变量时(n>=2)，利用*args能接受任意数量的输入并逐一累加

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def evaluate_accuracy(net, data_iter):
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 一个累加器类，内含两个变量：正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel()) # self.data:accuracy(net(X), y).....args:y.numel()返回张量y中元素的总个数
    return metric[0] / metric[1] # 正确预测数 / 预测总数 = 预测精度

print(evaluate_accuracy(net, test_iter)) #随机初始化，共10个类，故精度为0.1左右

def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)

def train_epoch_ch3(net, train_iter, loss, updater):
# 请注意，updater是更新模型参数的常用函数，它接受批量大小作为参数
# 它可以是d2l.sgd函数，也可以是框架的内置优化函数
    """训练模型一个迭代周期(定义见第3章)"""
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module): # isinstance( , )函数:用于判断某个对象是否是指定类(或其子类)的示例
        net.train()
    # 含3个变量的累加器：训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])

        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    
    return metric[0] / metric[2], metric[1] / metric[2] # 返回训练损失和训练精度

def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    """训练模型(定义见第3章)"""
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    # 断言检查：若提供条件为真，则继续执行程序;否则报错AssertionError并提供错误信息(逗号后面给出的参数)
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc

lr = 0.1
num_epochs = 10
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)

def predict_ch3(net, test_iter, n=6):
    """预测标签(定义见第3章)"""
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    print(titles[0:15])
    #d2l.show_images(
    #    X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n]) #在Jupyter notebook中有用

predict_ch3(net, test_iter)