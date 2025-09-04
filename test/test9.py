import torch

steps_output = [
    torch.tensor([1., 2.]),
    torch.tensor([1., 2.]),
    torch.tensor([3., 4.])
]

b = torch.stack(steps_output, dim=0)
print(b)
print(b.shape)


res = torch.sum(b, dim=0)
print(res)
