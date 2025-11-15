import torch


B, S, F = 1, 1, 8

index_list = [1, 4, 6, 7]


x_torch = torch.rand(B, S, F)

print(x_torch)

missing_x = torch.ones(B, S, F)
missing_x[:, :, index_list] = 0

print(missing_x)


print(missing_x * x_torch)
