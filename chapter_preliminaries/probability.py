import torch
from torch.distributions import multinomial #distribution分布， multinomial多项分布
from d2l import torch as d2l
fair_probs = torch.ones(6) / 6 #这个中括号[]在这里有意义吗？起什么作用？
#经实验，去掉[]暂时不影响结果
#为了抽取一个样本，只需传入一个概率向量，输出是另一个长度相同的向量
#该概率向量描述的是各个结果的概率分布， sum must= 1
for i in range(5):
    print(multinomial.Multinomial(1, fair_probs).sample()) #使用py中的for循环来寻找同一个分布中的多个样本，速度会慢得惊人

m = multinomial.Multinomial(100, torch.tensor([1, 1, 1, 1]))
x = m.sample()
print(x) #equal probability of the 0th, 1st, 2nd, 3rd object

log_prob = m.log_prob(x)
print(log_prob)
#log_prob allows different total_count for each parameter and sample

counts = multinomial.Multinomial(10, fair_probs).sample((500,))
cum_counts = counts.cumsum(dim=0)
estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True)
print(estimates)
