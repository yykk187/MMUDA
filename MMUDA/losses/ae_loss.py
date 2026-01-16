import torch
import torch.nn as nn
import torch.nn.functional as F

class AELoss(nn.Module):
    def __init__(self):
        super(AELoss, self).__init__()

    def forward(self, x, recon):
        # KL loss
        # kl_loss = (0.5 * (log_var.exp() + mu ** 2 - 1 - log_var)).mean()

        # recon loss
        recon_loss = F.mse_loss(recon, x, reduction="mean")

        # print(kl_loss, recon_loss)
        return recon_loss


class VAELoss(nn.Module):
    def __init__(self):
        super(VAELoss, self).__init__()

    def forward(self, x, recon, mu, log_var):
        recon_loss = F.mse_loss(recon, x, reduction="mean")
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        total_loss = recon_loss + kl_loss
        return total_loss
