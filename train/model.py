import torch.nn as nn
from mamba_ssm.modules.mamba_simple import Mamba

class TinyMambaHAR(nn.Module):
    def __init__(self, input_dim=57, hidden_dim=64, seq_len=10, num_classes=6):
        super().__init__()
        self.linear_in = nn.Linear(input_dim, hidden_dim)
        #self.mamba = Mamba(d_model=hidden_dim)
        self.mamba = nn.Linear(hidden_dim, hidden_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.linear_in(x)      # [B, T, H]
        x = self.mamba(x)          # [B, T, H]
        x = x.transpose(1, 2)      # [B, H, T]
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)
