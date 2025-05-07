import torch
from d2l import torch as d2l

x = torch.tensor([1.1, 2.3, 3.2, -1.4], requires_grad=True)

# relu(x) = \max(x, 0)
print(torch.relu(x))
# tensor([1.1000, 2.3000, 3.2000, 0.0000], grad_fn=<ReluBackward0>)

# sigmoid(x) = \frac{1}{1 + e ^{-x}}
print(torch.sigmoid(x))
# tensor([0.7503, 0.9089, 0.9608, 0.1978], grad_fn=<SigmoidBackward0>)

# tanh(x = \frac{1 - e ^{-2x}}{1 + e ^{-2x}}
print(torch.tanh(x))
# tensor([ 0.8005,  0.9801,  0.9967, -0.8854], grad_fn=<TanhBackward0>)