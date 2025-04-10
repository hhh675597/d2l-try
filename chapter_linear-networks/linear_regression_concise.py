import random
import torch
from d2l import torch as d2l

#w, b的真实值
true_w = torch.tensor([5.2, 4.1]) #expected svalar type Float but found long, 这里的参数必须是float
true_b = 3.2

#调用d2l库中的syncthetic_data函数(见linear_regression_scratch.py)生成数据集
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

