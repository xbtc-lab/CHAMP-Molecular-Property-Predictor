import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, HeteroData, Batch
from typing import Dict, List, Optional, Tuple, Union

"""
    # Create dummy data
    hetero_data = HeteroData()
    
    # Add atom features
    hetero_data['atom'].x = torch.randn(num_atoms, 9)  # 9-dim atom features
    hetero_data['atom'].batch = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])  # 5 atoms per molecule
    
    # Add motif features
    hetero_data['motif'].x = torch.randn(num_motifs, 5)  # 5-dim motif features
    hetero_data['motif'].batch = torch.tensor([0, 0, 1, 1])  # 2 motifs per molecule
    
    # Add atom-in-motif edges
    atom_idx = torch.tensor([0, 1, 1, 2, 3, 4, 5, 6, 6, 7, 8, 9])
    motif_idx = torch.tensor([0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3])
    hetero_data['atom', 'in', 'motif'].edge_index = torch.stack([atom_idx, motif_idx])
    
"""

class AtomMotifAttention(nn.Module):
    def __init__(self, 
                 atom_dim: int,
                 motif_dim: int,
                 output_dim: int = None,
                 num_heads: int = 4,
                 dropout: float = 0.1):
        """
        Computes attention between atoms and motifs in molecular graphs.
        Motifs are substructures that contain atoms.
        
        Args:
            atom_dim (int): Dimension of atom node features
            motif_dim (int): Dimension of motif node features
            output_dim (int, optional): Output dimension. Defaults to atom_dim if None.
            num_heads (int): Number of attention heads
            dropout (float): Dropout probability
        """
        super(AtomMotifAttention, self).__init__()
        
        self.atom_dim = atom_dim
        self.motif_dim = motif_dim
        self.output_dim = output_dim if output_dim is not None else atom_dim
        self.num_heads = num_heads

        assert self.atom_dim % self.num_heads == 0, "atom_dim must be divisible by num_heads"
        
        self.head_dim = atom_dim // self.num_heads
        
        # Linear projections (same dimensionality)
        self.atom_query = nn.Linear(atom_dim, atom_dim)
        self.motif_key = nn.Linear(motif_dim, motif_dim)
        self.motif_value = nn.Linear(motif_dim, motif_dim)
        
        # Output projection
        self.output_proj = nn.Linear(atom_dim, self.output_dim)
        
        # Optional dropout to prevent overfitting
        # self.dropout = nn.Dropout(dropout)
        
        # Layer norm for stability
        self.layer_norm = nn.LayerNorm(self.output_dim)


    def forward(self,
                hetero_data: HeteroData) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute atom-to-motif attention and update atom features.

        Args:
            hetero_data (HeteroData): contains:
                - 'atom.x': atom features [num_atoms, atom_dim]
                - 'motif.x': motif features [num_motifs, motif_dim]
                - 'atom.batch': atom batch indices
                - 'motif.batch': motif batch indices

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Updated atom features [num_atoms, output_dim]
                - Attention weights [num_atoms, num_heads, num_motifs]
        """
        # Extract features and relationships
        atom_features = hetero_data['atom'].x  # [num_atoms, atom_dim]
        motif_features = hetero_data['motif'].x  # [num_motifs, motif_dim]

        # Batch info
        atom_batch = hetero_data['atom'].batch  # [num_atoms]
        motif_batch = hetero_data['motif'].batch  # [num_motifs]
        
        # Shapes
        num_atoms = atom_features.size(0)
        num_motifs = motif_features.size(0)
        device = atom_features.device
        
        # Batch mask so atoms only attend to motifs from same molecule
        batch_mask = atom_batch.view(-1, 1) == motif_batch.view(1, -1)  # [num_atoms, num_motifs]
        batch_mask = batch_mask.to(device)
        
        # Project features
        q = self.atom_query(atom_features)  # [num_atoms, atom_dim]
        k = self.motif_key(motif_features)  # [num_motifs, atom_dim]
        v = self.motif_value(motif_features)  # [num_motifs, atom_dim]
        
        # Reshape for multi-head attention
        q = q.view(num_atoms, self.num_heads, self.head_dim)  # [num_atoms, num_heads, head_dim]
        k = k.view(num_motifs, self.num_heads, self.head_dim)  # [num_motifs, num_heads, head_dim]
        v = v.view(num_motifs, self.num_heads, self.head_dim)  # [num_motifs, num_heads, head_dim]
        
        # Attention scores: [num_atoms, num_heads, num_motifs]
        scores = torch.einsum('nhd,mhd->nhm', q, k) / (self.head_dim ** 0.5)
        
        # Apply batch mask
        mask_value = -1e9  # Use -1e9 instead of -inf to avoid grad issues
        mask_expanded = ~batch_mask.unsqueeze(1)  # [num_atoms, 1, num_motifs]
        scores = scores.masked_fill(mask_expanded, mask_value)
        
        # Numerically stable softmax (subtract max)
        scores = scores - scores.max(dim=-1, keepdim=True)[0]
        attn = F.softmax(scores, dim=-1)  # [num_atoms, num_heads, num_motifs]
        # attn = self.dropout(attn)
        
        # Weighted combination
        output = torch.einsum('nhm,mhd->nhd', attn, v)  # [num_atoms, num_heads, head_dim]
        
        # Restore shape
        output = output.reshape(num_atoms, self.atom_dim)  # [num_atoms, atom_dim]
        
        # Output projection
        output = self.output_proj(output)  # [num_atoms, output_dim]
        
        # Residual + layer norm
        if atom_features.size(1) >= self.output_dim:
            # Use existing feature slice if wide enough
            residual = atom_features[:, :self.output_dim]
        else:
            # Otherwise pad to match dimension
            residual = F.pad(atom_features, (0, self.output_dim - atom_features.size(1)))
        
        output = self.layer_norm(output + residual)
        
        return output, attn

    def get_atom_to_atom_attention_efficient(self,
                                           hetero_data: HeteroData) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Efficiently compute motif-mediated atom-to-atom attention.

        Args:
            hetero_data (HeteroData): PyTorch Geometric object

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Updated atom features [num_atoms, output_dim]
                - Atom-to-atom attention weights [num_heads, num_atoms, num_atoms]
        """
        # Atom-to-motif attention
        output, atom_to_motif_attn = self.forward(hetero_data)  # [num_atoms, num_heads, num_motifs]

        # Relationship info
        if ('atom', 'in', 'motif') in hetero_data.edge_types:
            edge_index = hetero_data['atom', 'in', 'motif'].edge_index
            atom_indices = edge_index[0]  # Atom indices
            motif_indices = edge_index[1]  # Motif indices
        else:
            raise ValueError("Cannot compute atom attention without atom-motif edges")

        # Batch info
        atom_batch = hetero_data['atom'].batch  # [num_atoms]

        # Shapes
        num_atoms = hetero_data['atom'].x.size(0)
        num_motifs = hetero_data['motif'].x.size(0)
        device = hetero_data['atom'].x.device

        # Binary membership matrix
        membership = torch.zeros((num_atoms, num_motifs), device=device)
        membership[atom_indices, motif_indices] = 1.0

        # Number of motifs per atom
        motifs_per_atom = membership.sum(dim=1)  # [num_atoms]
        # Avoid division by zero
        motifs_per_atom = torch.clamp(motifs_per_atom, min=1.0)

        # Normalize membership
        membership_norm = membership / motifs_per_atom.unsqueeze(1)  # [num_atoms, num_motifs]

        # Atom-to-atom attention
        # Transpose to align dimensions
        atom_to_motif_attn_t = atom_to_motif_attn.permute(1, 0, 2)  # [num_heads, num_atoms, num_motifs]

        # Attention: atom i attends to j if it focuses on a motif containing j
        # [num_heads, num_atoms, num_motifs] @ [num_motifs, num_atoms] -> [num_heads, num_atoms, num_atoms]
        atom_to_atom_attn = torch.matmul(atom_to_motif_attn_t, membership_norm.t())

        # Mask across batches
        batch_mask = atom_batch.view(1, -1, 1) == atom_batch.view(1, 1, -1)  # [1, num_atoms, num_atoms]
        batch_mask = batch_mask.expand(self.num_heads, -1, -1)  # [num_heads, num_atoms, num_atoms]
        atom_to_atom_attn = atom_to_atom_attn * batch_mask

        # Normalize attention rows
        row_sums = atom_to_atom_attn.sum(dim=-1, keepdim=True)
        row_sums = torch.clamp(row_sums, min=1e-10)  # Avoid divide-by-zero
        atom_to_atom_attn = atom_to_atom_attn / row_sums

        return output, atom_to_atom_attn


if __name__ == "__main__":
    # This is an example of how to use the model with explicit edges
    import torch_geometric.transforms as T

    print("Example with atom-in-motif edges:")
    num_atoms = 10
    num_motifs = 4

    # Create dummy data
    hetero_data = HeteroData()

    # Add atom features
    hetero_data['atom'].x = torch.randn(num_atoms, 16)  # 9-dim atom features
    hetero_data['atom'].batch = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])  # 5 atoms per molecule

    # Add motif features
    hetero_data['motif'].x = torch.randn(num_motifs, 16)  # 5-dim motif features
    hetero_data['motif'].batch = torch.tensor([0, 0, 1, 1])  # 2 motifs per molecule

    # Add atom-in-motif edges
    atom_idx = torch.tensor([0, 1, 1, 2, 3, 4, 5, 6, 6, 7, 8, 9])
    motif_idx = torch.tensor([0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3])
    hetero_data['atom', 'in', 'motif'].edge_index = torch.stack([atom_idx, motif_idx])

    # Compute atom-to-motif attention

    model = AtomMotifAttention(atom_dim=16, motif_dim=16)

    # Move model to same device as input data
    device = hetero_data['atom'].x.device
    model = model.to(device)

    # Compute atom-to-motif average attention
    updated_atom_features, global_attention_weights = model(hetero_data)
    print(updated_atom_features.shape)
    print(global_attention_weights.shape)

    # Compute atom-to-atom attention
    atom_to_atom_attn = model.get_atom_to_atom_attention_efficient(hetero_data)
    print(atom_to_atom_attn.shape)




