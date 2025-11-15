import torch


a = torch.tensor([[1., 2., 3., 4.], [2., 3., 4., 5.], [3., 4., 5., 6.]])
b = a[:, 2:]

print(b)
print(b.shape)
