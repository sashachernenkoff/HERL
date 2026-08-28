"""
Configurable Categorical Variational Autoencoder (VAE) for genomic data.

Composes the Encoder and Decoder architectures to compress allele dosages into a 
continuous latent space, and decodes them back into a categorical probability distribution.
This module defines the full-rank Categorical VAE architecture with activation checkpointing for GPU memory management.
are handled externally.

Example usage:
    from vae.model import VAE
    vae = VAE(input_dim=84077, config=json_config)
    reconstructed, mu, log_var, z = vae(x)
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .encoder import Encoder
    from .decoder import Decoder
except ImportError:  # pragma: no cover - script-style import fallback
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from encoder import Encoder
    from decoder import Decoder


class VAE(nn.Module):
    def __init__(self, input_dim, config):
        super().__init__()
        self.config = config
        self.latent_dim = config["encoder"]["latent_dim"]

        self.encoder = Encoder(input_dim, config["encoder"])
        spatial_sizes = getattr(self.encoder, "spatial_sizes", None)
        self.decoder = Decoder(input_dim, self.latent_dim, config["decoder"], spatial_sizes)

    def encode(self, x):
        return self.encoder(x)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(mu)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstructed = self.decode(z)
        return reconstructed, mu, log_var, z

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path, map_location=None):
        self.load_state_dict(torch.load(path, map_location=map_location))
