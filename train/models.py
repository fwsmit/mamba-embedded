from typing import Optional

import torch
import torch.nn as nn
from mamba_ssm import Mamba, Mamba3
from mamba_ssm.modules.mamba2_simple import Mamba2Simple
from .mamba_cpu_funcs import mamba3_siso_combined_ref


class MambaWrapper(nn.Module):
    def __init__(self, input_dim, output_dim, n_layers, mamba_model, dropout=0.1,
                 bidirectional: bool = False, bidirectional_strategy: Optional[str] = "add",
                 **mamba_kwargs):
        super().__init__()
        print(mamba_kwargs)
        if bidirectional and bidirectional_strategy is None:
            bidirectional_strategy = "add"
        if bidirectional and bidirectional_strategy not in ["add", "ew_multiply"]:
            raise NotImplementedError(
                f"{bidirectional_strategy} strategy for bi-directionality is not implemented!"
            )
        self.bidirectional = bidirectional
        self.bidirectional_strategy = bidirectional_strategy

        self.linear_in = nn.Linear(input_dim, mamba_kwargs["d_model"])
        self.mamba_layers = nn.ModuleList([
            mamba_model(**mamba_kwargs)
            for _ in range(n_layers)
        ])
        if bidirectional:
            self.mamba_layers_rev = nn.ModuleList([
                mamba_model(**mamba_kwargs)
                for _ in range(n_layers)
            ])
        else:
            self.mamba_layers_rev = None
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
        for i, (mamba_layer, norm_layer) in enumerate(zip(self.mamba_layers, self.norm_layers)):
            res = x
            x = norm_layer(x)   # pre-norm
            out = mamba_layer(x)
            if self.bidirectional:
                out_rev = self.mamba_layers_rev[i](x.flip(dims=(1,))).flip(dims=(1,))
                if self.bidirectional_strategy == "add":
                    out = out + out_rev
                elif self.bidirectional_strategy == "ew_multiply":
                    out = out * out_rev
            x = self.dropout(out)
            x = res + x
        x = self.norm_out(x)
        x = x.mean(dim=1)           # [B, H]
        return self.classifier(x)
