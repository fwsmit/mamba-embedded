import torch
import torch.nn as nn
from mamba_ssm import Mamba, Mamba3
from mamba_ssm.modules.mamba2_simple import Mamba2Simple
from .mamba_cpu_funcs import mamba3_siso_combined_ref


class MambaWrapper(nn.Module):
    def __init__(self, input_dim, output_dim, n_layers, mamba_model, dropout=0.1, **mamba_kwargs):
        super().__init__()
        print(mamba_kwargs)
        self.linear_in = nn.Linear(input_dim, mamba_kwargs["d_model"])
        self.mamba_layers = nn.ModuleList([
            mamba_model(**mamba_kwargs)
            for _ in range(n_layers)
        ])
        self.norm_layers = nn.ModuleList(
            [nn.LayerNorm(mamba_kwargs["d_model"]) for _ in range(n_layers)]
        )
        self.classifier = nn.Linear(mamba_kwargs["d_model"], output_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.norm_out = nn.LayerNorm(mamba_kwargs["d_model"])

        # Mamba 3 pascal GPU compatibility
        import mamba_ssm.ops.triton.mamba3.mamba3_siso_combined as _combined_mod
        _combined_mod.mamba3_siso_combined = mamba3_siso_combined_ref
        import mamba_ssm.modules.mamba3 as _mamba3_mod
        _mamba3_mod.mamba3_siso_combined = mamba3_siso_combined_ref

    def forward(self, x):
        x = self.linear_in(x)  # [B, T, H]
        for mamba_layer, norm_layer in zip(self.mamba_layers, self.norm_layers):
            res = x
            x = norm_layer(x)   # pre-norm
            x = mamba_layer(x)
            x = self.dropout(x)
            x = res + x
        x = self.norm_out(x[:, -1, :])
        return self.classifier(x)
