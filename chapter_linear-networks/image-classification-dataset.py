import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l

d2l.use_svg_display()

trans = transforms.ToTensor() ## 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式，并除以255使得所有像素的数值均在0～1之间
mnist_train = torchvision.datasets.FashionMNIST(
    root = "~/d2l-try/data", train=True, transform=trans, download=False) #新下载时 download=True
mnist_test = torchvision.datasets.FashionMNIST(
    root = "~/d2l-try/data", train=False, transform=trans, download=False)

print(len(mnist_train), len(mnist_test)) #输出60000，10000:Fashion-MNIST共有10各类别的图像，每个类别含6000张图像的训练数据集和1000张图像的测试数据集
print(mnist_train[0][0].shape) #输出torch.Size([1, 28, 28]) 三个参数分别为[通道数, 高度像素数, 宽度像素数].该数据集均为灰度图像(通道数为1)，每个输入图像高度、宽度均为28像素

#两个可视化数据集的函数此处省略

#使用子包内置的数据迭代器小批量读取
batch_size = 256

def get_dataloader_workers():
    """使用4个进程来读取数据"""
    return 4

train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,
                             num_workers=get_dataloader_workers())
#查看读取训练数据集所需的时间
timer = d2l.Timer()
for X, y in train_iter:
    continue
print(f'{timer.stop():.2f} sec')

#整合上述所有组件：获取并读取Fashion-MNIST数据集；返回训练集和测试集的数据迭代器；接受一个可选参数resize用来将图像大小调整为另一种大小
def load_data_fashion_mnist(batch_size, resize=None):
    """下载Fashion-MNIST数据集，然后将其加载到内存中"""
    trans = [transforms.ToTensor()]

    if resize:
        trans.insert(0, transforms.Resize(resize))

    trans = transforms.Compose(trans)

    mnist_train = torchvision.datasets.FashionMNIST(
        root="~/d2l-try/data", train=True, transform=trans, download=False)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="~/d2l-try/data", train=False, transform=trans, download=False)
    
    return (data.DataLoader(mnist_train, batch_size, shuffle=False, #真实情况下改为True
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False, #真实情况下改为True
                            num_workers=get_dataloader_workers()))

train_iter, test_iter = load_data_fashion_mnist(32, resize=64)
for X, y in train_iter:
    print(X.shape, X.dtype, y.shape, y.dtype)
    break