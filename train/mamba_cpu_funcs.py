"""
Adapted from various mamba sources

Copyright (c) 2025, Dao AI Lab, Goombalab
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange, repeat


# Reference Implementations
def _segsum(x: torch.Tensor) -> torch.Tensor:
    """Segment sum helper for attention computation."""
    T = x.size(-1)
    x = repeat(x, "... d -> ... d e", e=T)
    # mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=-1)
    mask = torch.tril(torch.ones(T, T, device=x.device), diagonal=-1).bool()
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    # mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=0)
    mask = torch.tril(torch.ones(T, T, device=x.device), diagonal=0).bool()
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum


def mamba3_siso_step_ref(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    ADT: torch.Tensor,
    DT: torch.Tensor,
    Trap: torch.Tensor,
    Q_bias: torch.Tensor,
    K_bias: torch.Tensor,
    Angles: torch.Tensor,
    D: Optional[torch.Tensor] = None,
    Z: Optional[torch.Tensor] = None,
    Input_States: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Reference implementation of Mamba-3 in recurrent (step) mode.
    
    Args:
        Input_States: Optional tuple of (Angle_State, SSM_State, K_State, V_State)
    
    Returns:
        out: Output tensor (batch, seqlen, nheads, headdim_v)
        Final_States: Tuple of (Angle_State, SSM_State, K_State, V_State)
    """
    batch, seqlen, nheads_qk, headdim_qk = Q.shape
    _, _, nheads, headdim_v = V.shape
    headdim_angles = Angles.shape[-1]
    device = Q.device
    assert seqlen > 0
    Angles = torch.tanh(Angles) * math.pi

    # Expand Q/K for GQA
    if Q.shape[2] != V.shape[2]:
        Q = repeat(Q, "b s h_bc d -> b s (h_bc g) d", g=V.shape[2] // Q.shape[2])
    if K.shape[2] != V.shape[2]:
        K = repeat(K, "b s h_bc d -> b s (h_bc g) d", g=V.shape[2] // K.shape[2])

    def apply_rotary_emb(tensor, cos, sin):
        tensor_reshaped = tensor.view(*tensor.shape[:-1], -1, 2)
        tensor_0 = tensor_reshaped[..., 0]
        tensor_1 = tensor_reshaped[..., 1]
        if cos.shape[-1] < tensor_0.shape[-1]:
            pad_size = tensor_0.shape[-1] - cos.shape[-1]
            # cos = F.pad(cos, (0, pad_size), value=1.0)
            # sin = F.pad(sin, (0, pad_size), value=0.0)
            cos = torch.cat([cos, cos.new_ones(*cos.shape[:-1], pad_size)], dim=-1)
            sin = torch.cat([sin, sin.new_zeros(*sin.shape[:-1], pad_size)], dim=-1)
        rotated_0 = tensor_0 * cos - tensor_1 * sin
        rotated_1 = tensor_0 * sin + tensor_1 * cos
        rotated = torch.stack([rotated_0, rotated_1], dim=-1).view_as(tensor)
        return rotated
    
    # Initialize states
    if Input_States is not None:
        Angle_State, SSM_State, K_State, V_State = Input_States
        Angle_State = Angle_State.clone()
        SSM_State = SSM_State.clone().to(torch.float32)
        K_State = K_State.clone()
        V_State = V_State.clone()
    else:
        Angle_State = torch.zeros((batch, nheads, headdim_angles), dtype=torch.float32, device=device)
        SSM_State = torch.zeros((batch, nheads, headdim_v, headdim_qk), dtype=torch.float32, device=device)
        K_State = torch.zeros((batch, nheads, headdim_qk), dtype=Q.dtype, device=device)
        V_State = torch.zeros((batch, nheads, headdim_v), dtype=V.dtype, device=device)
    
    TWO_PI = 2 * math.pi
    out_arr = []

    for idx in range(seqlen):
        q = Q[:, idx, :, :] + Q_bias.unsqueeze(0)
        k = K[:, idx, :, :] + K_bias.unsqueeze(0)
        v = V[:, idx, :, :]
        adt = ADT[:, :, idx]
        dt = DT[:, :, idx]
        trap = Trap[:, :, idx]
        z = Z[:, idx, :, :] if Z is not None else None
        angles = Angles[:, idx, :, :]

        # Update angle state with cumsum: Angle_State = (Angle_State + Angles * DT) mod 2π
        Angle_State = Angle_State + angles * dt.unsqueeze(-1)
        Angle_State = Angle_State - TWO_PI * torch.floor(Angle_State / TWO_PI)

        # Apply rotary embeddings to Q and K using cumulative angles
        cos_angles = torch.cos(Angle_State)
        sin_angles = torch.sin(Angle_State)
        q_rot = apply_rotary_emb(q, cos_angles, sin_angles)
        k_rot = apply_rotary_emb(k, cos_angles, sin_angles)

        trap = torch.sigmoid(trap)
        alpha = torch.exp(adt)
        beta = (1 - trap) * dt * alpha
        gamma = trap * dt

        # Update SSM state using previous K_State and V_State
        SSM_State = alpha.unsqueeze(-1).unsqueeze(-1) * SSM_State 
        SSM_State = SSM_State + beta.unsqueeze(-1).unsqueeze(-1) * (K_State.unsqueeze(-2) * V_State.unsqueeze(-1))
        SSM_State = SSM_State + gamma.unsqueeze(-1).unsqueeze(-1) * (k_rot.unsqueeze(-2) * v.unsqueeze(-1))

        # Compute output
        # out = torch.einsum("bhdD, bhD -> bhd", SSM_State, q_rot.to(SSM_State.dtype))
        out = torch.matmul(SSM_State, q_rot.to(SSM_State.dtype).unsqueeze(-1)).squeeze(-1)
        
        if D is not None:
            out = out + D[None, :, None] * v
        
        if Z is not None:
            out = out * z * torch.sigmoid(z)
        
        out_arr.append(out)
        
        # Update K and V states for next step
        K_State = k_rot
        V_State = v
    
    out = torch.stack(out_arr, dim=1)
    Final_States = (Angle_State, SSM_State, K_State, V_State)
    return out, Final_States


def mamba3_siso_fwd_ref(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    ADT: torch.Tensor,
    DT: torch.Tensor,
    Trap: torch.Tensor,
    Q_bias: torch.Tensor,
    K_bias: torch.Tensor,
    Angles: torch.Tensor,
    D: Optional[torch.Tensor] = None,
    Z: Optional[torch.Tensor] = None,
    Initial_States: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    chunk_size: int = 64,
    dtype: torch.dtype = torch.float32,
    cu_seqlens: Optional[torch.Tensor] = None,
):
    """Reference implementation of Mamba-3 forward pass.
    
    Args:
        Initial_States: Optional tuple of (Angle_State, SSM_State, K_State, V_State)
    
    Returns:
        out_z: Output with Z gating applied
        final_states: (Final_Angle_State, Final_SSM_State, Final_K_State, Final_V_State)
    """
    batch, total_seqlen, nheads_qk, headdim_qk = Q.shape
    _, _, nheads, headdim_v = V.shape
    headdim_angles = Angles.shape[-1]
    device = Q.device
    
    is_varlen = cu_seqlens is not None
    if is_varlen:
        assert batch == 1
    
    # Cast inputs
    Q = Q.to(dtype)
    K = K.to(dtype)
    V = V.to(dtype)
    ADT = ADT.to(torch.float32)
    DT = DT.to(torch.float32)
    Trap = Trap.to(dtype)
    Q_bias = Q_bias.to(dtype)
    K_bias = K_bias.to(dtype)
    Angles = Angles.to(dtype)
    if D is not None:
        D = D.to(dtype)
    if Z is not None:
        Z = Z.to(dtype)
    if Initial_States is not None:
        Initial_Angle_State, Initial_SSM_State, Initial_K_State, Initial_V_State = Initial_States

    Angles = torch.tanh(Angles) * math.pi
    # Expand Q/K for GQA
    if Q.shape[2] != V.shape[2]:
        Q = repeat(Q, "b s h_bc d -> b s (h_bc g) d", g=V.shape[2] // Q.shape[2])
    if K.shape[2] != V.shape[2]:
        K = repeat(K, "b s h_bc d -> b s (h_bc g) d", g=V.shape[2] // K.shape[2])

    out_zs = []
    Final_Angle_States = []
    Final_SSM_States = []
    Final_K_States = []
    Final_V_States = []

    TWO_PI = 2 * math.pi

    def _rotary(tensor, cos, sin):
        tensor_reshaped = tensor.view(*tensor.shape[:-1], -1, 2)
        tensor_0 = tensor_reshaped[..., 0]
        tensor_1 = tensor_reshaped[..., 1]
        if cos.shape[-1] < tensor_0.shape[-1]:
            pad_size = tensor_0.shape[-1] - cos.shape[-1]
            # cos = F.pad(cos, (0, pad_size), value=1.0)
            # sin = F.pad(sin, (0, pad_size), value=0.0)
            cos = torch.cat([cos, cos.new_ones(*cos.shape[:-1], pad_size)], dim=-1)
            sin = torch.cat([sin, sin.new_zeros(*sin.shape[:-1], pad_size)], dim=-1)
        rotated_0 = tensor_0 * cos - tensor_1 * sin
        rotated_1 = tensor_0 * sin + tensor_1 * cos
        return torch.stack([rotated_0, rotated_1], dim=-1).view_as(tensor)

    def compute_one_sequence(seq_idx):
        if is_varlen:
            start_idx, end_idx = cu_seqlens[seq_idx].item(), cu_seqlens[seq_idx + 1].item()
            Q_curr = Q[0, start_idx:end_idx, :, :]
            K_curr = K[0, start_idx:end_idx, :, :]
            V_curr = V[0, start_idx:end_idx, :, :]
            ADT_curr = ADT[0, :, start_idx:end_idx]
            DT_curr = DT[0, :, start_idx:end_idx]
            Trap_curr = Trap[0, :, start_idx:end_idx]
            Angles_curr = Angles[0, start_idx:end_idx, :, :]
            Z_curr = Z[0, start_idx:end_idx, :, :] if Z is not None else None
        else:
            Q_curr = Q[seq_idx]
            K_curr = K[seq_idx]
            V_curr = V[seq_idx]
            ADT_curr = ADT[seq_idx]
            DT_curr = DT[seq_idx]
            Trap_curr = Trap[seq_idx]
            Angles_curr = Angles[seq_idx]
            Z_curr = Z[seq_idx] if Z is not None else None

        Trap_curr = torch.sigmoid(Trap_curr)
        seqlen_curr = Q_curr.shape[0]

        Angles_scaled = Angles_curr.float() * DT_curr.transpose(0, 1).unsqueeze(-1)
        Angles_Cumsum = torch.cumsum(Angles_scaled, dim=0)
        if Initial_States is not None:
            Initial_Angle_State_curr = Initial_Angle_State[seq_idx]
            Angles_Cumsum = Angles_Cumsum + Initial_Angle_State_curr.unsqueeze(0)
        Angles_Cumsum = Angles_Cumsum - TWO_PI * torch.floor(Angles_Cumsum / TWO_PI)
        Final_Angle_States.append(Angles_Cumsum[-1])

        # Initialize acc_states
        if Initial_States is not None:
            Initial_SSM_State_curr = Initial_SSM_State[seq_idx]
            Initial_K_State_curr = Initial_K_State[seq_idx]
            Initial_V_State_curr = Initial_V_State[seq_idx]

            scalar = DT_curr[:, 0] * (1 - Trap_curr[:, 0])
            acc_states = Initial_SSM_State_curr + Initial_V_State_curr[:, :, None] * Initial_K_State_curr[:, None, :] * scalar[:, None, None]
        else:
            acc_states = torch.zeros((nheads, headdim_v, headdim_qk), device=device, dtype=torch.float32)

        # Compute shifted gamma and scale
        # DT_shifted = F.pad(DT_curr[:, 1:], (0, 1))
        # Trap_shifted = F.pad(Trap_curr[:, 1:], (0, 1))
        DT_shifted   = torch.cat([DT_curr[:, 1:],   DT_curr.new_zeros(DT_curr.shape[0],   1)], dim=1)
        Trap_shifted = torch.cat([Trap_curr[:, 1:], Trap_curr.new_zeros(Trap_curr.shape[0], 1)], dim=1)
        shifted_gamma = DT_shifted * (1 - Trap_shifted)
        scale = DT_curr * Trap_curr + DT_shifted * (1 - Trap_shifted)

        # Add biases
        Q_curr = Q_curr + Q_bias.unsqueeze(0)
        K_curr = K_curr + K_bias.unsqueeze(0)

        # Compute QK dot for skip connection
        QK_dot = torch.sum(K_curr * Q_curr, dim=-1) * shifted_gamma.transpose(0, 1)

        # Rotary embeddings using Angles_Cumsum
        cos_angles_curr = torch.cos(Angles_Cumsum).to(Q_curr.dtype)
        sin_angles_curr = torch.sin(Angles_Cumsum).to(Q_curr.dtype)
        Q_curr = _rotary(Q_curr, cos_angles_curr, sin_angles_curr)
        K_curr = _rotary(K_curr, cos_angles_curr, sin_angles_curr)

        Final_K_States.append(K_curr[-1])
        Final_V_States.append(V_curr[-1])

        K_curr_scaled = K_curr * scale.transpose(0, 1).unsqueeze(-1).to(K_curr.dtype)

        # Compute output via quadratic attention
        # QK = torch.einsum("thd,shd->hts", Q_curr, K_curr_scaled)
        QK = torch.matmul(Q_curr.permute(1, 0, 2),          # (h, t, d)
                  K_curr_scaled.permute(1, 2, 0))    # (h, d, s)  →  (h, t, s)
        QK_causal = torch.tril(QK)
        QK_causal = (QK_causal * torch.exp(_segsum(ADT_curr))).to(QK_causal.dtype)
        # out = torch.einsum("hts,shd->thd", QK_causal, V_curr)
        out = torch.matmul(QK_causal,                        # (h, t, s)
                   V_curr.permute(1, 0, 2)           # (h, s, d)  →  (h, t, d)
                  ).permute(1, 0, 2)                 # → (t, h, d)

        if Initial_States is not None:
            da_cs = torch.cumsum(ADT_curr, dim=-1)
            exp_da_cs = torch.exp(da_cs)
            # out = out + torch.einsum("hDd,thd,ht->thD", acc_states.to(Q_curr.dtype), Q_curr, exp_da_cs.to(Q_curr.dtype))
            out = out + (torch.matmul(Q_curr.to(acc_states.dtype).permute(1, 0, 2),   # (h, t, d)
                            acc_states.permute(0, 2, 1)                      # (h, d, D)  →  (h, t, D)
                ).permute(1, 0, 2)                                            # → (t, h, D)
                * exp_da_cs.to(Q_curr.dtype).permute(1, 0).unsqueeze(-1)     # (t, h, 1)
            )

        if D is not None:
            out = out + D[None, :, None] * V_curr

        out = out - V_curr * QK_dot.unsqueeze(-1)

        if Z_curr is not None:
            out = out * Z_curr * torch.sigmoid(Z_curr)
        out_zs.append(out)

        # Compute final state
        da_cs_last = torch.exp(torch.sum(ADT_curr, dim=-1))
        da_cs_rev = torch.exp(torch.sum(ADT_curr, dim=-1, keepdim=True) - torch.cumsum(ADT_curr, dim=-1))
        V_curr_scaled = V_curr * da_cs_rev.permute(1, 0).unsqueeze(-1).to(V_curr.dtype)
        # final_acc_states = acc_states * da_cs_last.unsqueeze(-1).unsqueeze(-1) + torch.einsum(
        #     "thd,thD->hDd", K_curr_scaled, V_curr_scaled.to(K_curr_scaled.dtype))
        final_acc_states = (
            acc_states * da_cs_last.unsqueeze(-1).unsqueeze(-1)
            + torch.matmul(V_curr_scaled.to(K_curr_scaled.dtype).permute(1, 2, 0),  # (h, D, t)
                        K_curr_scaled.permute(1, 0, 2))                           # (h, t, d)  →  (h, D, d)
        )
        Final_SSM_States.append(final_acc_states)

    num_sequences = cu_seqlens.size(0) - 1 if is_varlen else batch
    for seq_idx in range(num_sequences):
        compute_one_sequence(seq_idx)

    if not is_varlen:
        out_zs = torch.stack(out_zs, dim=0)
        Final_Angle_States = torch.stack(Final_Angle_States, dim=0)
        Final_SSM_States = torch.stack(Final_SSM_States, dim=0)
        Final_K_States = torch.stack(Final_K_States, dim=0)
        Final_V_States = torch.stack(Final_V_States, dim=0)
    else:
        out_zs = torch.cat(out_zs, dim=0).unsqueeze(0)
        Final_Angle_States = torch.stack(Final_Angle_States, dim=0)
        Final_SSM_States = torch.stack(Final_SSM_States, dim=0)
        Final_K_States = torch.stack(Final_K_States, dim=0)
        Final_V_States = torch.stack(Final_V_States, dim=0)

    return out_zs, (Final_Angle_States, Final_SSM_States, Final_K_States, Final_V_States)


# ================================================================== 
# Test Utilities
# ================================================================== 

def detach_clone(*args):
    """Detach and clone tensors, preserving None values."""
    return tuple([arg.detach().clone().requires_grad_() if arg is not None else None for arg in args])

@torch.no_grad()
def relative_error(
    ker: torch.Tensor,
    ref: torch.Tensor,
    eps: float = 1e-6,
    ref_mag_mask: float = 1e-2,
    p: float = 0.95,
    name: str = "",
    print_top_errors: bool = True,
    angle: bool = False,   # if True: use circular absolute error; else: relative error
) -> float:
    assert ker.shape == ref.shape

    ker_xx = ker.detach().to(torch.float32)
    ref_xx = ref.detach().to(torch.float32)

    abs_ref = ref_xx.abs()

    if angle:
        delta = ker_xx - ref_xx
        delta = torch.remainder(delta + math.pi, 2 * math.pi) - math.pi
        abs_diff = delta.abs()
    else:
        abs_diff = (ker_xx - ref_xx).abs()

    mask = abs_ref >= ref_mag_mask
    if not mask.any():
        return 0.0

    vals = abs_diff[mask].flatten() if angle else (abs_diff[mask] / (abs_ref[mask] + eps)).flatten()

    n = vals.numel()
    k = max(1, min(n, int(math.ceil(p * n))))
    err = vals.kthvalue(k).values.item()

    if print_top_errors and err > 0.01:
        print(f"\n  Top 10 errors for {name}:")
        diff_flat = abs_diff.flatten()
        ref_flat = ref_xx.flatten()
        ker_flat = ker_xx.flatten()
        topk = diff_flat.topk(min(10, diff_flat.numel()))
        for i, idx in enumerate(topk.indices):
            idx = idx.item()
            r = ref_flat[idx].item()
            k_val = ker_flat[idx].item()
            d = diff_flat[idx].item()
            if angle:
                # For angles, show absolute angular error (radians)
                print(f"    {i}: ref={r:.6e}, ker={k_val:.6e}, ang_err={d:.6e} rad")
            else:
                rel_e = d / (abs(r) + eps) if abs(r) >= ref_mag_mask else float('nan')
                print(f"    {i}: ref={r:.6e}, ker={k_val:.6e}, diff={d:.6e}, rel={rel_e:.2%}")

    return err


def create_mamba3_siso_inputs(
    batch: int,
    seqlen: int,
    nheads: int,
    nheads_qk: int,
    headdim_qk: int,
    headdim_v: int,
    dtype: torch.dtype,
    device: str,
    has_D: bool,
    has_Z: bool,
    has_input_states: bool,
    cu_seqlens: Optional[torch.Tensor] = None,
    requires_grad: bool = False,
):
    num_sequences = cu_seqlens.size(0) - 1 if cu_seqlens is not None else batch
    
    Q = torch.randn((batch, seqlen, nheads_qk, headdim_qk), device=device, dtype=dtype)
    Q = F.rms_norm(Q, normalized_shape=(headdim_qk,)).clone()
    K = torch.randn((batch, seqlen, nheads_qk, headdim_qk), device=device, dtype=dtype)
    K = F.rms_norm(K, normalized_shape=(headdim_qk,)).clone()
    V = torch.randn((batch, seqlen, nheads, headdim_v), device=device, dtype=dtype)

    dt_max, dt_min = 0.1, 0.001
    a_init = -torch.empty(batch, nheads, seqlen, device=device, dtype=torch.float32).uniform_(1.0, 16.0)
    dt = torch.exp(
        torch.rand(batch, nheads, seqlen, device=device, dtype=torch.float32) 
        * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
    )
    ADT = (a_init * dt).contiguous()
    DT = dt.contiguous()
    Trap = torch.empty(batch, nheads, seqlen, dtype=dtype, device=device).uniform_(0.0, 1.0).clone()
    Q_bias = torch.randn(nheads, headdim_qk, dtype=dtype, device=device)
    K_bias = torch.randn(nheads, headdim_qk, dtype=dtype, device=device)
    
    # headdim_angles constraint: 2*headdim_angles <= headdim_qk
    headdim_angles = headdim_qk // 4
    Angles = torch.randn(batch, seqlen, nheads, headdim_angles, dtype=torch.float32, device=device)

    D = torch.ones((nheads,), device=device, dtype=torch.float32) if has_D else None
    Z = torch.randn((batch, seqlen, nheads, headdim_v), device=device, dtype=dtype) if has_Z else None
    
    if has_input_states:
        Input_Angle_State = torch.randn((num_sequences, nheads, headdim_angles), device=device, dtype=torch.float32)
        Input_SSM_State = torch.randn((num_sequences, nheads, headdim_v, headdim_qk), device=device, dtype=torch.float32)
        Input_K_State = torch.randn((num_sequences, nheads, headdim_qk), device=device, dtype=torch.float32)
        Input_V_State = torch.randn((num_sequences, nheads, headdim_v), device=device, dtype=torch.float32)
        Input_States = (Input_Angle_State, Input_SSM_State, Input_K_State, Input_V_State)
    else:
        Input_States = None
    
    if requires_grad:
        Q.requires_grad_(True)
        K.requires_grad_(True)
        V.requires_grad_(True)
        ADT.requires_grad_(True)
        DT.requires_grad_(True)
        Trap.requires_grad_(True)
        Q_bias.requires_grad_(True)
        K_bias.requires_grad_(True)
        Angles.requires_grad_(True)
        if D is not None:
            D.requires_grad_(True)
        if Z is not None:
            Z.requires_grad_(True)
        if Input_States is not None:
            for state in Input_States:
                state.requires_grad_(True)
    
    return {
        'Q': Q, 'K': K, 'V': V,
        'ADT': ADT, 'DT': DT, 'Trap': Trap,
        'Q_bias': Q_bias, 'K_bias': K_bias, 'Angles': Angles,
        'D': D, 'Z': Z, 'Input_States': Input_States,
    }


# API compatibility
def mamba3_siso_combined_ref(
    Q, K, V, ADT, DT, Trap, Q_bias, K_bias, Angles,
    D=None, Z=None, Input_States=None,
    chunk_size=64,
    return_final_states=False,
    cu_seqlens=None,
    dtype=torch.float32,
):
    out, (Final_Angle_State, Final_SSM_State, Final_K_State, Final_V_State) = mamba3_siso_fwd_ref(
        Q, K, V, ADT, DT, Trap, Q_bias, K_bias, Angles,
        D=D, Z=Z,
        Initial_States=Input_States,   # renamed param
        chunk_size=chunk_size,
        dtype=dtype,
        cu_seqlens=cu_seqlens,
    )
    if return_final_states:
        return out, Final_Angle_State, Final_SSM_State, Final_K_State, Final_V_State
    return out


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
