import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, output_dim, latent_input_dim, config, spatial_sizes=None):
        super().__init__()
        self.output_dim = output_dim
        
        self.net = nn.Sequential(
            nn.Linear(latent_input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, self.output_dim * 3)
        )
        
        # Explicit Initializations
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                # Output layer mapping to categorical logits
                if m.out_features == self.output_dim * 3:
                    nn.init.xavier_uniform_(m.weight)
                # Hidden ReLU layer
                else:
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, z):
        h = self.net(z)
        # Reshape to (batch_size, num_snps, 3 classes)
        h = h.view(h.size(0), self.output_dim, 3)
        return h
