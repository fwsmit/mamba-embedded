import torch
import torch.nn as nn
from mamba_ssm.modules.mamba_simple import Mamba


class ResidualMamba(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand):
        super().__init__()
        self.mamba = Mamba(
            d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand
        )

    def forward(self, x):
        return x + self.mamba(x)  # residual keeps gradients healthy


class Net(nn.Module):
    def __init__(
        self,
        output_size: int = 10,
        d_model: int = 8,
        d_state: int = 4,
        n_layers: int = 5,
    ):
        super().__init__()
        self.input_proj = nn.Linear(28, d_model, bias=False)  # row → d_model
        self.mamba_layers = nn.Sequential(
            *[
                ResidualMamba(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=3,
                    expand=2,
                )
                for _ in range(n_layers)
            ]
        )
        self.classifier = nn.Linear(d_model, output_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, 28, 28)   — standard torchvision MNIST format
        Returns:
            logits: (B, 10)
        """
        x = x.squeeze(1)
        x = self.input_proj(x)
        x = self.mamba_layers(x)
        x = x.mean(dim=1)
        return self.classifier(x)


class Net2(nn.Module):
    def __init__(
        self,
        output_size: int = 6,
        d_model: int = 8,
        d_state: int = 4,
        n_layers: int = 5,
    ):
        super().__init__()
        self.input_proj = nn.Linear(561, d_model, bias=False)  # row → d_model
        self.mamba_layers = nn.Sequential(
            *[
                ResidualMamba(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=3,
                    expand=2,
                )
                for _ in range(n_layers)
            ]
        )
        self.classifier = nn.Linear(d_model, output_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 561, 1)   — standard torchvision MNIST format
        Returns:
            logits: (B, 6)
        """
        x = x.transpose(1, 2)  # B, 561, 1 → B, 1, 561
        x = self.input_proj(x)
        x = self.mamba_layers(x)
        x = x.mean(dim=1)
        return self.classifier(x)


class TinyMambaHAR(nn.Module):
    def __init__(self, input_dim=57, hidden_dim=64, output_size=6):
        super().__init__()
        self.linear_in = nn.Linear(input_dim, hidden_dim)
        self.mamba = Mamba(d_model=hidden_dim)
        # self.mamba = nn.Linear(hidden_dim, hidden_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        x = self.linear_in(x)  # [B, T, H]
        x = self.mamba(x)  # [B, T, H]
        x = x.transpose(1, 2)  # [B, H, T]
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)
