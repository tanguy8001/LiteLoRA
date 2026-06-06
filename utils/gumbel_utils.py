"""
Gumbel-Sparsemax and Gumbel-Sigmoid utilities for CL-LoRA task selection.

Implements:
- Gumbel noise sampling
- Sparsemax projection
- Gumbel-Sparsemax gating
- Gumbel-Sigmoid binary gating (Hard-Concrete style with STE)
- Sparsity regularization (Entropy or L1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def sample_gumbel(shape, device='cuda', eps=1e-20):
    # Sample from Gumbel(0, 1) distribution: -log(-log(Uniform(0,1)))
    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)


def sparsemax(logits, dim=-1):
    """
    Sparsemax activation function (Martins & Astudillo, 2016).

    Projects logits onto the probability simplex, yielding sparse probabilities
    with exact zeros.

    Args:
        logits: Input logits, shape [..., num_classes]
        dim: Dimension to apply sparsemax over

    Returns:
        Sparse probability distribution with exact zeros

    Reference:
        "From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification"
        https://arxiv.org/abs/1602.02068
    """
    # Replace -inf with very large negative number for numerical stability
    logits = torch.where(torch.isinf(logits), torch.full_like(logits, -1e9), logits)

    # Sort logits in descending order
    logits_sorted, _ = torch.sort(logits, dim=dim, descending=True)

    # Compute cumulative sum
    cumsum = torch.cumsum(logits_sorted, dim=dim)

    # Compute k(z) - the number of non-zero elements
    # k(z) = max{k : 1 + k * z_k > sum_{j=1}^{k} z_j}
    arange = torch.arange(1, logits.shape[dim] + 1, device=logits.device, dtype=logits.dtype)

    # Expand arange to match logits shape
    shape = [1] * len(logits.shape)
    shape[dim] = -1
    arange = arange.view(*shape)

    threshold = (cumsum - 1) / arange

    # Find support: where sorted logits > threshold
    support = (logits_sorted > threshold).float()

    # Compute k: number of elements in support
    k = support.sum(dim=dim, keepdim=True)

    # Compute tau(z): the threshold value
    tau_sum = (logits_sorted * support).sum(dim=dim, keepdim=True)
    tau = (tau_sum - 1) / (k + 1e-8)

    # Apply sparsemax transformation
    output = torch.clamp(logits - tau, min=0.0)

    # Normalize to ensure probabilities sum to 1
    output_sum = output.sum(dim=dim, keepdim=True)
    output = output / (output_sum + 1e-8)

    return output


def gumbel_sparsemax(logits, tau=1.0, dim=-1, training=True):
    """Gumbel-Sparsemax for sparse differentiable selection."""
    if training:
        gumbel_noise = sample_gumbel(logits.shape, device=logits.device)
        noisy_logits = (logits + gumbel_noise) / tau
    else:
        noisy_logits = logits / tau
    soft_beta = sparsemax(noisy_logits, dim=dim)
    return soft_beta


def gumbel_binary_gate(logits, tau=1.0, hard=True, training=True):
    """Gumbel-Sigmoid binary gating with Straight-Through Estimator."""
    if training:
        g_noise = sample_gumbel(logits.shape, device=logits.device)
        y_soft = torch.sigmoid((logits + g_noise) / tau)
    else:
        y_soft = torch.sigmoid(logits / tau)

    if hard:
        y_hard = (y_soft > 0.5).float()
        beta = y_hard - y_soft.detach() + y_soft
    else:
        beta = y_soft

    return beta


def sparsity_loss(beta, mode='entropy', eps=1e-8):
    # beta has shape [nb_tasks_seen_so_far] because of slice in forward pass
    if mode == 'l1':
        return torch.norm(beta, p=1, dim=-1).mean()

    entropy = -torch.sum(beta * torch.log(beta + eps), dim=-1)
    return entropy.mean()


def hard_selection(beta, threshold=0.1):
    return (beta > threshold).float()
