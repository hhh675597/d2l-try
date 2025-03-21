import time
import math
import torch
import numpy as np
from d2l import torch as d2l
#矢量化代码带来的数量级加速
n = 10000
a = torch.ones([n])
b = torch.ones([n]) #实例化两个全为1的10000维向量，一种方法使用py for loop遍历向量，另一种方法调用重载后的+

class Timer:  #@save
    """记录多次运行时间"""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """启动计时器"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并将时间记录在列表中"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间"""
        return np.array(self.times).cumsum().tolist()
    
c = torch.zeros(n)
timer = Timer()
for i in range(n):
    c[i] = a[i] + b[i]
print(f'{timer.stop():.5f} sec') #for循环做加法

timer.start()
d = a + b
print(f'{timer.stop():.5f}sec') #利用重载过的+运算符
#正态分布
#设随机变量均值为x, 方差为\sigma^{2}, 则正态分布概率密度函数为
#p(x) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp\left(-\frac{1}{2 \sigma^2} (x - \mu)^2\right)
def normal(x, mu, sigma):
    p = 1 / math.sqrt(2 * math.pi * sigma**2)
    return p * np.exp(-0.5 / sigma**2 * (x - mu)**2)

