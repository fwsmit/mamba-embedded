import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

def _selective_scan_vectorized(
    u,
    delta,
    A,
    B,
    C,
    D=None,
    z=None,
    delta_bias=None,
    delta_softplus=False,
    return_last_state=False,
):
    if delta_bias is not None:
        delta = delta + delta_bias.unsqueeze(-1)
    if delta_softplus:
        delta = F.softplus(delta)

    # log_dA = delta * A  (skipping the exp entirely — it would cancel with log anyway)
    # A has shape (d_inner, d_state), A is negative by construction (-exp(A_log))
    log_dA = delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(2)  # (B, d, L, N)

    # dB = delta * B,  shape (B, d, L, N)
    # B arrives as (B, d_state, L) per selective_scan_fn convention
    dB = delta.unsqueeze(-1) * B.permute(
        0,
        2,
        # (B, d, L, N)
        1,
    ).unsqueeze(1)

    A = torch.exp(log_dA)                  # (B, d, L, N)
    bu = dB * u.unsqueeze(-1)              # (B, d, L, N)

    h = torch.zeros_like(bu)
    state = torch.zeros_like(bu[:, :, 0])   # (B, d, N)

    for t in range(bu.size(2)):
        state = A[:, :, t] * state + bu[:, :, t]
        h[:, :, t] = state

    # Output projection: y_t = sum_N(h_t * C_t)
    y = (h * C.permute(0, 2, 1).unsqueeze(1)).sum(-1)  # (B, d, L)

    if D is not None:
        y = y + D.unsqueeze(0).unsqueeze(-1) * u
    if z is not None:
        y = y * F.silu(z)

    if return_last_state:
        return y, h[:, :, -1, :]
    return y
