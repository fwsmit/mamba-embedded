import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from einops import rearrange, repeat

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
    """
    u: r(B D L)
    delta: r(B D L)
    A: c(D N) or r(D N)
    B: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
    C: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    out: r(B D L)
    last_state (optional): r(B D dstate) or c(B D dstate)
    """
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)
    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    is_variable_B = B.dim() >= 3
    is_variable_C = C.dim() >= 3
    if A.is_complex():
        if is_variable_B:
            B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
        if is_variable_C:
            C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
    else:
        B = B.float()
        C = C.float()
    x = A.new_zeros((batch, dim, dstate))
    ys = []
    # deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    deltaA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(2))
    if not is_variable_B:
        # deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
        deltaB_u = (delta * u).unsqueeze(-1) * B.unsqueeze(0).unsqueeze(2)
    else:
        if B.dim() == 3:
            # Burn doesn't support einsum with 3 inputs
            # deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
            deltaB_u = (delta * u).unsqueeze(2) * B.unsqueeze(1)
            deltaB_u = deltaB_u.permute(0, 1, 3, 2)
        else:
            B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
            # deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
            deltaB_u = (delta * u).unsqueeze(-1) * B
    if is_variable_C and C.dim() == 4:
        C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
    last_state = None
    for i in range(u.shape[2]):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        if not is_variable_C:
            # y = torch.einsum('bdn,dn->bd', x, C)
            y = (x * C.unsqueeze(0)).sum(dim=-1)
        else:
            if C.dim() == 3:
                # y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
                y = (x * C[:, :, i].unsqueeze(0)).sum(dim=-1)
            else:
                # y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
                y = (x * C[:, :, :, i]).sum(dim=-1)
        if i == u.shape[2] - 1:
            last_state = x
        if y.is_complex():
            y = y.real * 2
        ys.append(y)
    y = torch.stack(ys, dim=2) # (batch dim L)
    out = y if D is None else y + u * rearrange(D, "d -> d 1")
    if z is not None:
        out = out * F.silu(z)
    out = out.to(dtype=dtype_in)
    return out if not return_last_state else (out, last_state)
