import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, config):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = config["latent_dim"]
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, self.latent_dim * 2)
        )
        
        # Explicit Initializations
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                # Output layer mapping to latent space
                if m.out_features == self.latent_dim * 2:
                    nn.init.xavier_uniform_(m.weight)
                # Hidden ReLU layer
                else:
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.float()
        h = self.net(x)
        mu, log_var = torch.split(h, self.latent_dim, dim=1)
        
        return mu, log_var
