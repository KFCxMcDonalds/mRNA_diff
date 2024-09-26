import torch
import numpy as np

class LatentDistribution():
    """
    latent distribution for VAE
    """
    def __init__(self, mean, logvar, device):
        self.mean = mean
        self.logvar = torch.clamp(logvar, -30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar).to(device)

        self.device = device
    
    def sample(self):
        """
        sample from latent distribution, equivalent to reparameterization trick
        """
        self.eps = torch.randn_like(self.std).to(self.device)
        return self.mean + self.std * self.eps

        