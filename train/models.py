import torch
import torch.nn as nn
from mamba_ssm import Mamba, Mamba3
from mamba_ssm.modules.mamba2_simple import Mamba2Simple
from .mamba_cpu_funcs import mamba3_siso_combined_ref


class ResidualMamba(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand):
        super().__init__()
        self.mamba = Mamba(
            d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand
        )

    def forward(self, x):
        return x + self.mamba(x)  # residual keeps gradients healthy


class ResidualMamba2(nn.Module):
    def __init__(self, d_model, d_state, headdim, expand):
        super().__init__()
        self.mamba = Mamba2Simple(
            d_model=d_model,
            d_state=d_state,
            headdim=headdim,
            expand=expand,
            dtype=torch.float32,
        )

    def forward(self, x):
        return x + self.mamba(x)  # residual keeps gradients healthy


class ResidualMamba3(nn.Module):
    def __init__(self, d_model, d_state, headdim, expand, is_mimo):
        super().__init__()
        self.mamba = Mamba3(
            d_model=d_model,
            d_state=d_state,
            headdim=headdim,
            expand=expand,
            is_mimo=is_mimo,
            dtype=torch.float32,
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
    def __init__( self, input_dim=57, d_model=64, d_state=16, d_conv=4, expand=2, output_size=6):
        super().__init__()
        self.linear_in = nn.Linear(input_dim, d_model)
        self.mamba = Mamba(
            d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(d_model, output_size)

    def forward(self, x):
        x = self.linear_in(x)  # [B, T, H]
        x = self.mamba(x)  # [B, T, H]
        x = x.transpose(1, 2)  # [B, H, T]
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)


class TinyMambaMulti(nn.Module):
    def __init__( self, input_dim=57, d_model=64, d_state=16, d_conv=4, expand=2, n_layers=1, output_size=6):
        super().__init__()
        self.linear_in = nn.Linear(input_dim, d_model)
        self.mamba_layers = nn.Sequential(
            *[
                ResidualMamba(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
                for _ in range(n_layers)
            ]
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(d_model, output_size)

    def forward(self, x):
        x = self.linear_in(x)  # [B, T, H]
        x = self.mamba_layers(x)  # [B, T, H]
        x = x.transpose(1, 2)  # [B, H, T]
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)


class TinyMamba3Multi(nn.Module):
    def __init__(self, input_dim=57, d_model=64, headdim=32, d_state=32, expand=2, output_size=6, is_mimo=False, n_layers=1):
        super().__init__()
        self.linear_in = nn.Linear(input_dim, d_model)

        self.mamba_layers = nn.Sequential(
            *[
                ResidualMamba3(
                    d_model=d_model,
                    d_state=d_state,
                    headdim=headdim,
                    expand=expand,
                    is_mimo=is_mimo,
                )
                for _ in range(n_layers)
            ]
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(d_model, output_size)
        import mamba_ssm.ops.triton.mamba3.mamba3_siso_combined as _combined_mod

        _combined_mod.mamba3_siso_combined = mamba3_siso_combined_ref

        import mamba_ssm.modules.mamba3 as _mamba3_mod

        _mamba3_mod.mamba3_siso_combined = mamba3_siso_combined_ref

    def forward(self, x):
        x = self.linear_in(x)  # [B, T, H]
        x = self.mamba_layers(x)  # [B, T, H]
        x = x.transpose(1, 2)  # [B, H, T]
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)


class TinyMamba2Multi(nn.Module):
    def __init__(self, input_dim=57, d_model=64, headdim=32, d_state=32, expand=2, output_size=6, is_mimo=False, n_layers=1):
        super().__init__()
        self.linear_in = nn.Linear(input_dim, d_model)

        self.mamba_layers = nn.Sequential(
            *[
                ResidualMamba2(
                    d_model=d_model,
                    d_state=d_state,
                    headdim=headdim,
                    expand=expand,
                )
                for _ in range(n_layers)
            ]
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(d_model, output_size)

        # import mamba_ssm.ops.triton.mamba3.mamba3_siso_combined as _combined_mod
        #
        # _combined_mod.mamba3_siso_combined = mamba3_siso_combined_ref
        #
        # import mamba_ssm.modules.mamba3 as _mamba3_mod
        #
        # _mamba3_mod.mamba3_siso_combined = mamba3_siso_combined_ref

    def forward(self, x):
        x = self.linear_in(x)  # [B, T, H]
        x = self.mamba_layers(x)  # [B, T, H]
        x = x.transpose(1, 2)  # [B, H, T]
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)
