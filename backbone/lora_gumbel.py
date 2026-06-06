import math
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.vision_transformer import VisionTransformer as timm_ViT
from torch import Tensor
from torch.nn.parameter import Parameter

from backbone.base_vit import ViT
import os
from backbone.linears import SimpleLinear
import gc
import torch.nn.utils as utils
import copy
from utils.gumbel_utils import gumbel_sparsemax, gumbel_binary_gate

class _LoRALayer(nn.Module):
    def __init__(self, w: nn.Module, w_a: nn.Module, w_b: nn.Module):
        super().__init__()
        self.w = w
        self.w_a = w_a
        self.w_b = w_b

    def forward(self, x):
        x = self.w(x) + self.w_b(self.w_a(x))
        return x


class LoRA_ViT(nn.Module):
    """
    Args:
        vit_model: a vision transformer model, see base_vit.py
        r: rank of LoRA
        num_classes: how many classes the model output, default to the vit model
        lora_layer: which layer we apply LoRA.
    """
    def __init__(self, vit_model: ViT, r: int, num_classes: int = 0, lora_layer=None):
        super(LoRA_ViT, self).__init__()

        assert r > 0
        base_vit_dim = vit_model.transformer.blocks[0].attn.proj_q.in_features
        dim = base_vit_dim
        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(range(len(vit_model.transformer.blocks)))
        # create for storage, then we can init them or load weights
        self.w_As = []  # These are linear layers
        self.w_Bs = []
        # lets freeze first
        for param in vit_model.parameters():
            param.requires_grad = False

        # Here, we do the surgery
        for t_layer_i, blk in enumerate(vit_model.transformer.blocks):
            # If we only want few lora layer instead of all
            if t_layer_i not in self.lora_layer:
                continue
            w_q_linear = blk.attn.proj_q
            w_v_linear = blk.attn.proj_v
            w_a_linear_q = nn.Linear(dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, dim, bias=False)
            w_a_linear_v = nn.Linear(dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, dim, bias=False)
            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)
            blk.attn.proj_q = _LoRALayer(w_q_linear, w_a_linear_q, w_b_linear_q)
            blk.attn.proj_v = _LoRALayer(w_v_linear, w_a_linear_v, w_b_linear_v)

        self.reset_parameters()
        self.lora_vit = vit_model
        if num_classes > 0:
            self.lora_vit.fc = nn.Linear(vit_model.fc.in_features, num_classes)

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.lora_vit(x)


class _LoRA_qkv_timm(nn.Module):
    """
    In timm it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    """
    def __init__(
        self,
        qkv: nn.Module,
        linear_a_q: nn.Module,
        linear_b_q: nn.Module,
        linear_a_v: nn.Module,
        linear_b_v: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)

    def forward(self, x):
        qkv = self.qkv(x)  # B,N,3*org_C
        new_q = self.linear_b_q(self.linear_a_q(x)) #* self.scaling_factor
        new_v = self.linear_b_v(self.linear_a_v(x)) #* self.scaling_factor
        qkv[:, :, : self.dim] += new_q
        qkv[:, :, -self.dim :] += new_v
        return qkv

class _LoRA_qkv_timm_train(nn.Module):
    """
    Training-mode LoRA layer with Gumbel-Sparsemax gating.

    Implements Equation 1: ΔW_t = Σ β_i α_i × (A_i B_i / ||A_i B_i||_F)

    Key design:
    - ALL tasks (including current) go through the same gating mechanism
    - Normalization by Frobenius norm: ||AB||_F = sqrt(sum((A @ B)^2))
    - Previous tasks: frozen adapters loaded from disk
    - Current task: trainable adapters with learnable α, l
    """
    def __init__(self, qkv, linear_a_q, linear_b_q, linear_a_v, linear_b_v,
                 task_id, saved_A, saved_B, t_layer_i, rank, gumbel_gate, tau=1.0):
        super().__init__()
        self.qkv = qkv
        self.dim = qkv.in_features
        self.rank = rank
        self.task_id = task_id
        self.t_layer_i = t_layer_i

        # Current task adapters (trainable)
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v

        # Previous task adapters (frozen, loaded from disk)
        self.saved_A = saved_A
        self.saved_B = saved_B

        # Gumbel gate for task selection
        self.gumbel_gate = gumbel_gate
        self.tau = tau

    def _compute_adapter_out(self, w_a, w_b, x, normalize=True):
        """
        Compute adapter output, optionally normalized by ||B||*||A||.
        
        Original SD-LoRA only normalizes PREVIOUS tasks to maintain 
        training stability for the CURRENT task, whose norm starts at 0.
        """
        adapter_out = w_b(w_a(x))
        if not normalize:
            return adapter_out
            
        # Normalization used for previously learned tasks
        norm = torch.norm(w_b.weight) * torch.norm(w_a.weight) + 1e-8
        return adapter_out / norm

    def forward(self, x):
        """
        Forward pass: ΔW = Σ β_i α_i × normalized_adapter_i

        Args:
            x: Input [batch, seq, dim]

        Returns:
            qkv: Output [batch, seq, 3*dim]
        """
        normalized_adapters_q = []
        normalized_adapters_v = []
        task_indices = []

        # === Load previous tasks (frozen) ===
        for i in range(self.task_id):
            
            if self.gumbel_gate.pruning_mask[i] == 0:
                continue

            task_indices.append(i)

            # Load saved adapters
            saved_A_i = self.saved_A['saved_A_' + str(i)]
            saved_B_i = self.saved_B['saved_B_' + str(i)]

            # Extract Q and V adapters for this layer
            adapters = list(zip(saved_A_i, saved_B_i))
            A_q, B_q = adapters[self.t_layer_i * 2]
            A_v, B_v = adapters[self.t_layer_i * 2 + 1]

            # Create temporary linear layers (no gradient)
            w_a_q = nn.Linear(self.dim, self.rank, bias=False)
            w_b_q = nn.Linear(self.rank, self.dim, bias=False)
            w_a_v = nn.Linear(self.dim, self.rank, bias=False)
            w_b_v = nn.Linear(self.rank, self.dim, bias=False)

            # Load weights (frozen)
            w_a_q.weight = nn.Parameter(A_q.weight.to(x.device), requires_grad=False)
            w_b_q.weight = nn.Parameter(B_q.weight.to(x.device), requires_grad=False)
            w_a_v.weight = nn.Parameter(A_v.weight.to(x.device), requires_grad=False)
            w_b_v.weight = nn.Parameter(B_v.weight.to(x.device), requires_grad=False)

            # Compute normalized outputs for previous tasks
            norm_q = self._compute_adapter_out(w_a_q, w_b_q, x, normalize=True)
            norm_v = self._compute_adapter_out(w_a_v, w_b_v, x, normalize=True)

            normalized_adapters_q.append(norm_q)
            normalized_adapters_v.append(norm_v)

        # === Add current task (trainable) ===
        task_indices.append(self.task_id)

        # Skip normalization for the current task to match SOTA and prevent instability
        norm_q_curr = self._compute_adapter_out(self.linear_a_q, self.linear_b_q, x, normalize=False)
        norm_v_curr = self._compute_adapter_out(self.linear_a_v, self.linear_b_v, x, normalize=False)

        normalized_adapters_q.append(norm_q_curr)
        normalized_adapters_v.append(norm_v_curr)

        # === Apply Gumbel gating ===
        if len(normalized_adapters_q) > 0:
            # Check if we are in Phase 1 (Magnitude) vs Phase 2 (Selection)
            # Default to Phase 2 for safety (standard gating)
            phase = getattr(self.gumbel_gate, 'phase', 2)
            
            delta_q, beta_q = self.gumbel_gate(
                normalized_adapters_q, task_indices, tau=1.0, training=self.training, phase=phase
            )
            delta_v, beta_v = self.gumbel_gate(
                normalized_adapters_v, task_indices, tau=1.0, training=self.training, phase=phase
            )

            # Store beta for sparsity loss computation
            self.last_beta_q = beta_q
            self.last_beta_v = beta_v
        else:
            delta_q, delta_v = 0, 0
            self.last_beta_q = None
            self.last_beta_v = None

        # === Apply to QKV ===
        qkv = self.qkv(x)  # [B, seq, 3*dim]
        qkv[:, :, :self.dim] += delta_q  # Add to Q
        qkv[:, :, -self.dim:] += delta_v  # Add to V

        return qkv

class _LoRA_qkv_timm_eval(nn.Module):
    """
    Eval mode LoRA layer with Gumbel gating.

    Same as training mode, but:
    - No Gumbel noise (deterministic)
    - Lower temperature (sharper selection)
    - No gradient computation
    """
    def __init__(self, task_id, qkv, saved_A, saved_B, t_layer_i, rank, gumbel_gate, save_file):
        super().__init__()
        self.qkv = qkv
        self.dim = qkv.in_features
        self.rank = rank
        self.task_id = task_id
        self.t_layer_i = t_layer_i

        self.saved_A = saved_A
        self.saved_B = saved_B
        self.gumbel_gate = gumbel_gate
        self.save_file = save_file

    def _compute_adapter_out(self, w_a, w_b, x, normalize=True):
        """Compute adapter output, optionally normalized."""
        adapter_out = w_b(w_a(x))
        if not normalize:
            return adapter_out
        norm = torch.norm(w_b.weight) * torch.norm(w_a.weight)
        return adapter_out / norm

    def forward(self, x):
        """Evaluation forward pass (no Gumbel noise)."""
        normalized_adapters_q = []
        normalized_adapters_v = []
        task_indices = []

        # Load all non-pruned tasks (only those that have been saved)
        # Note: self.task_id is incremented after saving, so it represents the next task to train
        # For evaluation, we only use tasks 0 to task_id-1 (completed tasks)
        for i in range(self.task_id):
            # Skip pruned tasks
            if self.gumbel_gate.pruning_mask[i] == 0:
                continue

            # Skip if adapter not saved yet (e.g., current task during training)
            if 'saved_A_' + str(i) not in self.saved_A or 'saved_B_' + str(i) not in self.saved_B:
                continue

            task_indices.append(i)

            # Load saved adapters
            saved_A_i = self.saved_A['saved_A_' + str(i)]
            saved_B_i = self.saved_B['saved_B_' + str(i)]

            adapters = list(zip(saved_A_i, saved_B_i))
            A_q, B_q = adapters[self.t_layer_i * 2]
            A_v, B_v = adapters[self.t_layer_i * 2 + 1]

            # Create temporary layers
            w_a_q = nn.Linear(self.dim, self.rank, bias=False)
            w_b_q = nn.Linear(self.rank, self.dim, bias=False)
            w_a_v = nn.Linear(self.dim, self.rank, bias=False)
            w_b_v = nn.Linear(self.rank, self.dim, bias=False)

            w_a_q.weight = nn.Parameter(A_q.weight.to(x.device), requires_grad=False)
            w_b_q.weight = nn.Parameter(B_q.weight.to(x.device), requires_grad=False)
            w_a_v.weight = nn.Parameter(A_v.weight.to(x.device), requires_grad=False)
            w_b_v.weight = nn.Parameter(B_v.weight.to(x.device), requires_grad=False)

            # Normalize all except the latest task (to match training behavior)
            is_latest = (i == self.task_id - 1)
            norm_q = self._compute_adapter_out(w_a_q, w_b_q, x, normalize=not is_latest)
            norm_v = self._compute_adapter_out(w_a_v, w_b_v, x, normalize=not is_latest)

            normalized_adapters_q.append(norm_q)
            normalized_adapters_v.append(norm_v)

        # Apply gating (no Gumbel noise, low temp)
        if len(normalized_adapters_q) > 0:
            delta_q, _ = self.gumbel_gate(
                normalized_adapters_q, task_indices, tau=0.5, training=False
            )
            delta_v, _ = self.gumbel_gate(
                normalized_adapters_v, task_indices, tau=0.5, training=False
            )
        else:
            delta_q, delta_v = 0, 0

        # Apply to QKV
        qkv = self.qkv(x)
        qkv[:, :, :self.dim] += delta_q
        qkv[:, :, -self.dim:] += delta_v
        return qkv
    


class GumbelGate(nn.Module):
    """
    Gating module for sparse task selection.

    - Learnable magnitude parameters α_i for each task
    - Learnable gate logits l_i for selection
    - Output: Σ β_i α_i × normalized_adapter_i

    Args:
        max_tasks: Maximum number of tasks
        init_alpha: Initial value for magnitude parameters α
        init_logit: Initial value for gate logits l
    """
    def __init__(self, max_tasks=20, init_alpha=0.8, init_logit=0.0, binary_beta=False):
        super().__init__()
        self.binary_beta = binary_beta

        # α: learnable magnitude parameters (one scalar per task)
        self.alpha = nn.ParameterList([
            nn.Parameter(torch.tensor(init_alpha), requires_grad=False) for _ in range(max_tasks)
        ])

        # l: learnable gate logits (one scalar per task)
        self.gate_logits = nn.ParameterList([
            nn.Parameter(torch.tensor(init_logit), requires_grad=False) for _ in range(max_tasks)
        ])

        # Pruning mask: 1 = keep, 0 = pruned (permanent)
        self.register_buffer('pruning_mask', torch.ones(max_tasks))
        self.max_tasks = max_tasks
        self.phase = 2 # Default to selection phase

        # Phase-2 knockout: when True, force β for the current task to 0 in
        # forward (with STE so gradient still flows through the soft sigmoid).
        # Set per-batch by the trainer; ignored outside training/phase 2.
        self.force_drop_current = False

    def load_parameters(self, alphas_list, logits_list, mask):
        """Load trained alphas, logits and mask."""
        for i, val in enumerate(alphas_list):
            if i < self.max_tasks:
                self.alpha[i].data.copy_(val.to(self.alpha[i].device))
        for i, val in enumerate(logits_list):
            if i < self.max_tasks:
                self.gate_logits[i].data.copy_(val.to(self.gate_logits[i].device))
        self.pruning_mask.copy_(mask.to(self.pruning_mask.device))

    def freeze_task_parameters(self, task_id):
        self.alpha[task_id].requires_grad = False
        self.alpha[task_id].grad = None
        self.gate_logits[task_id].requires_grad = False
        self.gate_logits[task_id].grad = None

    def unfreeze_task_parameters(self, task_id):
        self.alpha[task_id].requires_grad = True
        self.gate_logits[task_id].requires_grad = True

    def freeze_alphas(self, task_indices):
        for idx in task_indices:
            self.alpha[idx].requires_grad = False
            self.alpha[idx].grad = None

    def unfreeze_alphas(self, task_indices):
        for idx in task_indices:
            self.alpha[idx].requires_grad = True

    def freeze_logits(self, task_indices):
        for idx in task_indices:
            self.gate_logits[idx].requires_grad = False
            self.gate_logits[idx].grad = None

    def unfreeze_logits(self, task_indices):
        for idx in task_indices:
            self.gate_logits[idx].requires_grad = True

    def forward(self, normalized_adapters, task_indices, tau=1.0, training=True, phase=2):
        """
        Args:
            normalized_adapters: List of normalized adapter outputs [B, seq, dim]
            task_indices: List of task IDs
            tau: Temperature for Gumbel selection
            training: Whether in training mode (adds Gumbel noise)
            phase: 1 for Magnitude, 2 for Selection

        Returns:
            weighted_sum: Σ β_i α_i × normalized_adapter_i
            beta: Selection mask
        """
        if len(task_indices) == 0:
            return 0, None

        if phase == 1:
            training = False

        logits = torch.stack([self.gate_logits[i] for i in task_indices])  # [N]

        # Apply pruning mask (set pruned tasks to -inf)
        mask = torch.stack([self.pruning_mask[i] for i in task_indices])  # [N]
        logits = logits.masked_fill(mask == 0, float('-inf'))

        if training:
            # Apply Gumbel noise ONLY to the current task (the last logit)
            # This ensures stable expert reuse while exploring the necessity of the new expert.
            from utils.gumbel_utils import sample_gumbel
            g_noise = sample_gumbel(logits.shape, device=logits.device)
            
            # Mask out noise for all previous tasks
            noise_mask = torch.zeros_like(g_noise)
            noise_mask[-1] = 1.0
            
            # Apply noise and call gate with training=False (to avoid double noise)
            noisy_logits = logits + (g_noise * noise_mask)
            
            if self.binary_beta:
                beta = gumbel_binary_gate(noisy_logits, tau=tau, hard=True, training=False)
            else:
                beta = gumbel_sparsemax(noisy_logits, tau=tau, hard=True, training=False)
        else:
            # Deterministic selection for inference or Phase 2 evaluation
            if self.binary_beta:
                beta = gumbel_binary_gate(logits, tau=tau, hard=True, training=False)
            else:
                beta = gumbel_sparsemax(logits, tau=tau, hard=True,training=False)

        # Phase-2 knockout: force β for the current task (last index) to 0
        # in forward, while keeping a gradient path through the soft sigmoid
        # so logit_curr receives a "loss-without-current" signal via STE.
        if (training and phase == 2 and self.force_drop_current
                and len(task_indices) > 0):
            soft_last = torch.sigmoid(logits[-1] / tau)
            ste_zero = soft_last - soft_last.detach()  # value 0, grad through soft_last
            beta = torch.cat([beta[:-1], ste_zero.unsqueeze(0)])

        # Gather magnitude parameters: [α_0, α_1, ..., α_t]
        alphas = torch.stack([self.alpha[i] for i in task_indices])  # [N]

        # Weighted sum: Σ β_i α_i × adapter_i
        weighted_sum = 0
        for i, adapter_out in enumerate(normalized_adapters):
            weight = beta[i] * alphas[i]  # β_i * α_i (scalar)
            weighted_sum = weighted_sum + weight * adapter_out

        return weighted_sum, beta

    def get_betas(self, task_indices, tau=1.0,):
        """Compute β values without Gumbel noise (for evaluation/decision-making)."""
        if len(task_indices) == 0:
            return torch.tensor([])

        logits = torch.stack([self.gate_logits[i] for i in task_indices])
        mask = torch.stack([self.pruning_mask[i] for i in task_indices])
        logits = logits.masked_fill(mask == 0, float('-inf'))

        # No Gumbel noise (training=False)
        if self.binary_beta:
            beta = gumbel_binary_gate(logits, tau=tau, training=False)
        else:
            beta = gumbel_sparsemax(logits, tau=tau, training=False)
        return beta

    def prune_task(self, task_id):
        """Permanently prune task (set mask to 0)."""
        self.pruning_mask[task_id] = 0

    def keep_task(self, task_id):
        """Keep task (ensure mask is 1)."""
        self.pruning_mask[task_id] = 1


class LoRA_ViT_timm(nn.Module):
    def __init__(self, vit_model: timm_ViT, r: int, num_classes: int = 0, increment=10, filepath = './', lora_layer=None, eval=False, index=True, cur_task_index=None, args=None):
        super(LoRA_ViT_timm, self).__init__()
        self.args = args

        assert r > 0
        self.rank = r
        self.base_vit = copy.deepcopy(vit_model)

        self.save_file = filepath
        self.increment = increment

        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(range(len(vit_model.blocks)))

        self.w_As, self.w_Bs = [], []

        if index:
            self.task_id, self.cur_id = 0, 0

        if cur_task_index is not None:
            self.task_id = cur_task_index

        # freeze the saved part
        for param in self.base_vit.parameters():
            param.requires_grad = False

        for param in vit_model.parameters():
            param.requires_grad = False

        saved_lora_A, saved_lora_B = {}, {}
        for i in range(self.task_id + 1):
            file_path_a = self.save_file+'lora_w_a_'+str(i)+'.pt'
            file_path_b = self.save_file+'lora_w_b_'+str(i)+'.pt'
            if os.path.exists(file_path_a) and os.path.exists(file_path_b):
                saved_lora_A['saved_A_'+str(i)] = torch.load(file_path_a, weights_only=False)
                saved_lora_B['saved_B_'+str(i)] = torch.load(file_path_b, weights_only=False)

        # Init GumbelGate for task selection
        init_alpha = self.args["init_alpha"]
        binary_beta = self.args["binary_beta"]
        init_logit = self.args["init_logit"]

        self.gumbel_gate = GumbelGate(max_tasks=20, init_alpha=init_alpha, init_logit=init_logit, binary_beta=binary_beta)

        # Load the gating parameters of the most recently completed task,
        # falling back to just the pruning mask if the full state is missing.
        mask_path = self.save_file + 'pruning_mask.pt'
        gate_path = self.save_file + f'gumbel_gate_task_{self.task_id-1}.pt'
        if os.path.exists(gate_path):
            state = torch.load(gate_path, weights_only=False)
            self.gumbel_gate.load_parameters(state['alphas'], state['gate_logits'], state['pruning_mask'])
        elif os.path.exists(mask_path):
            self.gumbel_gate.pruning_mask = torch.load(mask_path, weights_only=False)

        self.tau = 1.0

        # Do the surgery
        for t_layer_i, blk in enumerate(vit_model.blocks):
            # If we only want few lora layer instead of all
            if t_layer_i not in self.lora_layer:
                continue
            w_qkv_linear = blk.attn.qkv
            self.dim = w_qkv_linear.in_features
            w_a_linear_q = nn.Linear(self.dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, self.dim, bias=False)
            w_a_linear_v = nn.Linear(self.dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, self.dim, bias=False)

            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)

            if not eval:
                blk.attn.qkv = _LoRA_qkv_timm_train(
                    w_qkv_linear, w_a_linear_q, w_b_linear_q, w_a_linear_v, w_b_linear_v,
                    self.task_id, saved_lora_A, saved_lora_B, t_layer_i, self.rank, self.gumbel_gate, tau=self.tau
                )
            else:
                blk.attn.qkv = _LoRA_qkv_timm_eval(self.task_id, w_qkv_linear, saved_lora_A, saved_lora_B, t_layer_i, self.rank, self.gumbel_gate, self.save_file)

        self.reset_parameters()
        self.lora_vit = vit_model
        if not eval:
            self.lora_vit.head = torch.nn.Identity()
        else:
            self.reset_lora_vit_head()



    def reset_lora_vit_head(self):
        task_incremental = self.increment
        self.lora_vit.head = self.generate_fc(768, (self.task_id)*task_incremental).cuda()
        temp_weights = torch.load(self.save_file+'CLs_weight'+str(self.task_id-1)+'.pt', weights_only=False) 
        temp_bias = torch.load(self.save_file+'CLs_bias'+str(self.task_id-1)+'.pt', weights_only=False) 

        self.lora_vit.head.weight.data = temp_weights.data.cuda()
        self.lora_vit.head.bias.data = temp_bias.data.cuda()


    # This part is only used during the evaluation
    def reset(self, eval=False):
        self.__init__(self.base_vit, self.rank, increment=self.increment, filepath=self.save_file, lora_layer=None, eval=eval, index=False, args=self.args)

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def save_lora_parameters(self, filename: str, task_id) -> None:
        self.task_id += 1
        if not os.path.exists(filename):
           os.makedirs(filename)
        torch.save(self.w_As, filename + 'lora_w_a_'+str(task_id)+'.pt')
        torch.save(self.w_Bs, filename + 'lora_w_b_'+str(task_id)+'.pt')

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc

    def compute_ortho_loss(self):
        loss = torch.tensor(0).float().cuda()
        for i in range(self.task_id):
            file_path = self.save_file+'lora_w_a_'+str(i)+'.pt'
            if os.path.exists(file_path):
                w_As = torch.load(file_path, weights_only=False)
                num_layer = len(self.w_As)
                for j in range(num_layer):
                    temp = torch.matmul(w_As[j].weight.to(self.w_As[j].weight.device), self.w_As[j].weight.t())
                    temp = torch.sum(torch.square(temp))
                    loss = loss.to(self.w_As[j].weight.device)
                    loss += temp
        return loss
    
    def forward(self, x: Tensor, loss=False, eval=False) -> Tensor:
        if eval:
            if not getattr(self, '_eval_ready', False):
                self.reset(eval=True)
                self._eval_ready = True
                self._train_ready = False

                gate = getattr(self, 'gumbel_gate', None)
                if gate is not None:
                    mode = "BINARY-SIGMOID" if gate.binary_beta else "SPARSEMAX"
                    active = (gate.pruning_mask[:self.task_id] > 0).nonzero().flatten().tolist()
                    inactive = (gate.pruning_mask[:self.task_id] == 0).nonzero().flatten().tolist()
                    print(f"[eval] task {self.task_id-1} | mode={mode} | active adapters={active} | inactive={inactive}")
            
            return self.lora_vit(x)
        else:
            # Switch back to train mode if we were in eval
            if not getattr(self, '_train_ready', True): # Default to true for fresh instances
                self.reset(eval=False)
                self._train_ready = True
                self._eval_ready = False
            
            if loss:
                loss_val = self.compute_ortho_loss()
                return self.lora_vit(x), loss_val
            return self.lora_vit(x)
