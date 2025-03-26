import torch

x = torch.tensor([[0, 1, 2],
                  [1, 3, 2],
                  [2, 2, 4]
                ], dtype=torch.float)
y = torch.tensor([[0, 1, 0],
                  [1, -1 / 3, 0],
                  [4, -2, 1]
                ])
z = torch.mm(y, x)
print(torch.mm(z, y.T))

x = torch.tensor([[1, 2],
                  [-2, 1]
])
y = torch.tensor([[2, 0],
                  [0, 2]
])

print(torch.mm(x, torch.mm(y, x.T)))
#print(torch.mm(x, y))