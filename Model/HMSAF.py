import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class HMSAF(nn.Module):
    """
    Enhanced DCMHAttention that implements:
    1. Basic attention (A_ij)
    2. Head interaction matrix (Wb) for cross-head information flow
    3. Dynamic gating
    """
    def __init__(self,
                 n_head,
                 input_dim,
                 output_dim,
                 use_Guide=True,
                 use_gating=True,
                 use_head_interaction=True,
                 dropout=0.1):

        super().__init__()
        # Define dimensions
        assert output_dim % n_head == 0

        self.n_head = n_head
        self.output_dim = output_dim
        self.input_dim = input_dim
        # Head dimension uses output_dim // n_head (not input_dim)
        self.head_dim = output_dim // n_head
        self.use_head_interaction = use_head_interaction
        self.use_gating = use_gating
        self.use_Guide = use_Guide
        # self.dropout = nn.Dropout(dropout)

        # Key, query, value projections
        self.wq = nn.Linear(input_dim, output_dim, bias=True)
        self.wk = nn.Linear(input_dim, output_dim, bias=True)
        self.wv = nn.Linear(input_dim, output_dim, bias=True)
        self.wo = nn.Linear(output_dim, input_dim, bias=True)

        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(input_dim)

        # Head interaction matrix (Wb) shaped [n_head, n_head]
        if self.use_Guide:
            # Combine atom-motif attention with base attention
            self.attn_cross_particle = nn.Parameter(torch.randn(n_head, n_head))

            # Weighting parameters with modest initialization
            self.alpha = nn.Parameter(torch.tensor(0.5))  # Weight for attn_scores
            self.beta = nn.Parameter(torch.tensor(0.5))  # Weight for motif-to-atom attn

        # Dynamic gating components
        if self.use_gating:
            # self.gate_proj_k_cross = nn.Parameter(torch.randn(n_head, n_head))
            self.gate_proj_k = nn.Linear(output_dim, n_head, bias=True)

            # self.gate_proj_q_cross = nn.Parameter(torch.randn(n_head, n_head))
            self.gate_proj_q = nn.Linear(output_dim, n_head, bias=True)

        if self.use_head_interaction:
            self.head_interaction = nn.Parameter(
                torch.eye(n_head) + 0.01 * torch.randn(n_head, n_head)
            )

        self.scale_factor = 1 / math.sqrt(self.head_dim)

    # Atom outputs + atom/motif attention -> fusion + gating + head interaction
    def forward(self, x: torch.Tensor, batch, motif_to_atom_attn) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: input node features [num_nodes, input_dim]
            batch: batch indices per node [num_nodes]
            motif_to_atom_attn: motif-to-atom attention [n_head, num_nodes, num_nodes]

        Returns:
            torch.Tensor: updated node features [num_nodes, input_dim]
        """
        num_atom, input_dim = x.shape
        identity = x  # Residual

        # Project into query/key/value spaces
        q = self.wq(x).view(num_atom, self.n_head, self.head_dim)
        k = self.wk(x).view(num_atom, self.n_head, self.head_dim)
        v = self.wv(x).view(num_atom, self.n_head, self.head_dim)

        # Transpose to [n_head, num_atom, head_dim]
        q = q.transpose(0, 1)  # [H, N, d]
        k = k.transpose(0, 1)  # [H, N, d]
        v = v.transpose(0, 1)  # [H, N, d]

        # Base attention scores [H, N, N]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale_factor

        # Apply batch mask so nodes only attend within the same graph
        if batch is not None:
            batch_mask = torch.eq(batch.unsqueeze(-1), batch.unsqueeze(-2))  # [N, N]
            batch_mask = batch_mask.unsqueeze(0).expand(self.n_head, -1, -1)  # [H, N, N]
            # Use -1e9 instead of -inf for stability
            attn_scores = attn_scores.masked_fill(~batch_mask, -1e9)

        attn_final = attn_scores                    #  [H, N, N]
        attn_scores = attn_scores.permute(1, 2, 0)  # [N, N, H]

        # Apply head interaction matrix (Wb)
        if self.use_Guide:
            # Convert motif_to_atom_attn to [N, N, H]
            motif_to_atom_attn = motif_to_atom_attn.permute(1, 2, 0)

            # Weighted fusion of attentions
            # attn_base = self.alpha * attn_scores + self.beta * torch.einsum('stn,nm->stm', motif_to_atom_attn,
            #                                                                 self.attn_cross_particle)
            attn_base = self.alpha * attn_scores + self.beta * motif_to_atom_attn
            # ============================================================================ previous
            # # Apply head interaction: [N, N, H] @ [H, H] -> [N, N, H]
            # attn_base_interaction = torch.einsum('stn,nm->stm', attn_base, self.head_interaction)
            # # Convert back to [H, N, N]
            # attn_base_interaction = attn_base_interaction.permute(2, 0, 1)

            # ============================================================================ current
            attn_base_interaction = attn_base.permute(2, 1, 0)

            attn_final = attn_final + attn_base_interaction


        # Dynamic gating
        if self.use_gating:
            # Flatten q/k
            q_flat = q.transpose(0, 1).reshape(num_atom, -1)  # [N, H*d]
            k_flat = k.transpose(0, 1).reshape(num_atom, -1)  # [N, H*d]

            # Gate values
            gates_q = self.gate_proj_q(q_flat)  # [N, H]
            gates_q = torch.tanh(gates_q)  # [N, H]

            gates_k = self.gate_proj_k(k_flat)  # [N, H]
            gates_k = torch.tanh(gates_k)  # [N, H]

            # Shape gates to match attention scores
            gates_q = gates_q.transpose(0, 1).unsqueeze(-1)  # [H, N, 1]
            gates_k = gates_k.transpose(0, 1).unsqueeze(1)  # [H, 1, N]

            # Retrieve raw attention
            raw_attn = attn_scores.permute(2, 0, 1)  # [H, N, N]

            # Apply gating
            gated_attn = raw_attn * gates_q + raw_attn * gates_k  # [H, N, N]
            attn_final = attn_final + gated_attn


        if self.use_head_interaction:
            attn_head_intercation = torch.einsum('stn,nm->stm', attn_scores, self.head_interaction).permute(2, 0, 1)
            attn_final = attn_final + attn_head_intercation



        # Softmax (optionally subtract max for stability)
        # attn_final = attn_final - attn_final.max(dim=-1, keepdim=True)[0]   # Disabled

        # Softmax to obtain attention probabilities
        attn_probs = F.softmax(attn_final, dim=-1)  # [H, N, N]

        # Optional dropout
        # attn_probs = self.dropout(attn_probs)

        # Apply attention to values
        output = torch.matmul(attn_probs, v)  # [H, N, d]

        # Reshape [H, N, d] -> [N, H*d]
        output = output.transpose(0, 1).contiguous().view(num_atom, -1)

        # Final projection
        output = self.wo(output)

        # Residual + layer norm
        output = self.layer_norm(output + identity)

        return output,attn_final
        # return output

