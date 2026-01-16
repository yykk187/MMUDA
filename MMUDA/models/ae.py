import torch
import torch.nn as nn
from models.decoder import Decoder
from models.encoder import MultiScaleCNNLSTMEncoder
# from models.encoder import Encoder


class AE(nn.Module):
    def __init__(self, params):
        super(AE, self).__init__()
        self.encoder = MultiScaleCNNLSTMEncoder(params)
        self.decoder = Decoder(params)


    def sample_z(self, mu, log_var):
        """sample z by reparameterization trick"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu = self.encoder(x)
        # z = self.sample_z(mu, log_var)
        recon = self.decoder(mu)
        return recon, mu
    
class VAE(nn.Module):
    def __init__(self, params):
        super(VAE, self).__init__()
        self.params = params
        self.encoder = MultiScaleCNNLSTMEncoder(params)
        self.fc_mu = nn.Linear(params.encoder_output_dim, params.latent_dim)
        self.fc_logvar = nn.Linear(params.encoder_output_dim, params.latent_dim)
        self.decoder = Decoder(params)

    def sample_z(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, return_z=False):
        mu, log_var = self.encoder(x)
        z = self.sample_z(mu, log_var)
        recon = self.decoder(z)
        return recon, mu, log_var,z





