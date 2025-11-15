import torch


a = torch.tensor([1., 2., 3.])
b = torch.tensor([4., 5., 6.])

a = a.unsqueeze(dim=0)
b = b.unsqueeze(dim=0)

c = torch.cat([a, b], dim=1)

print(c)
print(c.shape)
