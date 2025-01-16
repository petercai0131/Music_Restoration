import torch.nn.functional as F

# Calculate MSE Loss
def compute_mse(output, target):
    return F.mse_loss(output, target)