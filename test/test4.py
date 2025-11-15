import torch

epsilon = 1e-15
batch, features = 2, 4

# Uniform mask (각 feature=0.25)
M_uniform = torch.full((batch, features), 1/features)
loss_uniform = torch.mean(torch.sum(M_uniform * torch.log(M_uniform + epsilon), dim=1))

# One-hot mask (한 feature=1, 나머지=0)
M_onehot = torch.tensor([[1.,0.,0.,0.],[0.,0.2,0., 1]])
loss_onehot = torch.mean(torch.sum(M_onehot * torch.log(M_onehot + epsilon), dim=1))

print("Uniform mask M_loss:", loss_uniform.item())
print("One-hot mask   M_loss:", loss_onehot.item())
