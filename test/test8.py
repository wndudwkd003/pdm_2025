import torch

epsilon = 1e-15
batch, features = 3, 4

# Uniform mask (각 feature=0.25)
M_uniform = torch.full((batch, features), 1/features)

a = torch.zeros([3, 4])

print(a)

print(M_uniform)

gamma = 1.0

prior = torch.ones([3, 4])

prior2 = torch.mul(gamma - a, prior)
print(prior2)
