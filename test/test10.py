import torch

a = torch.ones(3, 3, 3)
print(a)

b = a.view(3*3, 3)
print(b)

b_1 = torch.zeros(3*3, 3)

b_2 = torch.cat([b, b_1], dim=1)

c = b_2.view(3, 3, -1)
print(c)
