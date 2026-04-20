import torch
import torch.nn as nn
from mamba_ssm import Mamba, Mamba3
from .mamba_cpu_funcs import mamba3_siso_combined_ref


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


class TinyMamba(nn.Module):
    def __init__(self, input_dim=57, hidden_dim=64, output_size=6):
        super().__init__()
        self.linear_in = nn.Linear(input_dim, hidden_dim)
        self.mamba = Mamba(d_model=hidden_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        x = self.linear_in(x)  # [B, T, H]
        x = self.mamba(x)  # [B, T, H]
        x = x.transpose(1, 2)  # [B, H, T]
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)


class TinyMamba3(nn.Module):
    def __init__(self, input_dim=57, hidden_dim=64, output_size=6):
        super().__init__()
        self.linear_in = nn.Linear(input_dim, hidden_dim)
        self.mamba = Mamba3(
            d_model=hidden_dim,
            d_state=8,
            headdim=16,
            is_mimo=False,
            chunk_size=8,
            is_outproj_norm=False,
            dtype=torch.float32,
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(hidden_dim, output_size)

        import mamba_ssm.ops.triton.mamba3.mamba3_siso_combined as _combined_mod

        _combined_mod.mamba3_siso_combined = mamba3_siso_combined_ref

        import mamba_ssm.modules.mamba3 as _mamba3_mod

        _mamba3_mod.mamba3_siso_combined = mamba3_siso_combined_ref

    def forward(self, x):
        x = self.linear_in(x)  # [B, T, H]
        x = self.mamba(x)  # [B, T, H]
        x = x.transpose(1, 2)  # [B, H, T]
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)
