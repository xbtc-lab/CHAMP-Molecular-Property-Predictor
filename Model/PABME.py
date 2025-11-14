import torch
import torch.nn as nn
from torch.nn import Linear, ReLU
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_add_pool
from torch_geometric.nn import GCNConv, GATConv, GINConv, GINEConv,EGConv,FAConv,FiLMConv,PANConv,PNAConv

class EdgeMLP(torch.nn.Module):
    def __init__(self,hidden_dim):
        super(EdgeMLP, self).__init__()
        self.mlp = torch.nn.Sequential(
            Linear(hidden_dim, hidden_dim)
        )
    def forward(self, edge_attr):
        return self.mlp(edge_attr)


class GINLayer(torch.nn.Module):
    def __init__(self, hidden_dim):
        super(GINLayer, self).__init__()
        nn_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.conv = GINConv(nn_mlp)

    def forward(self, x, edge_index, edge_attr=None):
        return self.conv(x, edge_index)

class GINELayer(torch.nn.Module):
    def __init__(self, hidden_dim):
        super(GINELayer, self).__init__()
        nn_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.conv = GINEConv(nn_mlp)

    def forward(self, x, edge_index, edge_attr):
        return self.conv(x, edge_index, edge_attr)


# TODO: customize message passing module
class CustomGNNLayer(MessagePassing):
    def __init__(self, hidden_dim):
        super(CustomGNNLayer, self).__init__(aggr='add')  # Sum aggregation
        self.node_mlp = nn.Sequential(
            Linear(hidden_dim, hidden_dim)
        )
        self.edge_mlp = EdgeMLP(hidden_dim)
        self.epsilon = torch.nn.Parameter(torch.tensor([0.]))  # Learnable epsilon

    def forward(self, x, edge_index, edge_attr):
        # Run message passing
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        # Update node representations
        out = self.node_mlp(out)
        return out

    def message(self, x_j, edge_attr):
        # Edge feature modulation
        edge_features = self.edge_mlp(edge_attr)
        # Neighbor contribution
        return edge_features * x_j

    def update(self, aggr_out, x):
        # Self-loop contribution
        self_eps = (1 + self.epsilon) * x
        # Combine neighbor and self information
        return aggr_out + self_eps



class GCNLayer(torch.nn.Module):
    def __init__(self, hidden_dim):
        super(GCNLayer, self).__init__()
        self.conv = GCNConv(hidden_dim, hidden_dim)

    def forward(self, x, edge_index, edge_attr=None):
        return self.conv(x, edge_index)

class GATLayer(torch.nn.Module):
    def __init__(self, hidden_dim, heads=4):
        super(GATLayer, self).__init__()
        self.conv = GATConv(hidden_dim, hidden_dim, heads=heads, concat=False)

    def forward(self, x, edge_index, edge_attr=None):
        return self.conv(x, edge_index)



# TODO: readout module
class HierarchicalEdgePooling(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, Pair_MLP):
        super().__init__()

        # Node attention
        self.node_attn_fc = nn.Linear(node_dim + edge_dim, 1)

        # Pair-MLP attention
        self.pair_attn_fc = nn.Linear(2 * node_dim + edge_dim, 1)

        # Pair-MLP encoders
        self.Pair_MLP = Pair_MLP
        self.pair_mlp_sum = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim)
        )

        self.pair_mlp_concat = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim)
        )

        self.beta = nn.Parameter(torch.tensor(1.0))  # Learnable beta

    def forward(self, x, edge_index, edge_attr, motif_atom_edges):
        """
        x: (num_nodes, node_dim) -> node features
        edge_index: (2, num_edges) -> edge indices
        edge_attr: (num_edges, edge_dim) -> edge features
        motif_atom_edges: (2, num_motif_atom_edges) -> [motif_idx, atom_idx] mapping
        """
        # Unpack motif-atom mapping
        motif_idx, atom_idx = motif_atom_edges

        # Duplicate node features per motif-atom pair
        x_motif_atom = torch.index_select(x, 0, atom_idx)

        # Need to remap edges because node indices changed; build mapping
        # from (original atom idx, motif idx) -> new virtual node idx
        #===================================================================================================
        # First create unique (motif_idx, atom_idx) pairs and assign indices:
        # mapping dictionary (atom_idx, motif_idx) -> new_idx
        # node_pairs = torch.stack([motif_idx, atom_idx], dim=1)
        # unique_pairs, inverse_indices = torch.unique(node_pairs, dim=0, return_inverse=True)
        # mapping = {(row[1].item(), row[0].item()): i for i, row in enumerate(unique_pairs)}
        #
        # # Remap original edge_index to the new virtual node indices
        # new_edge_index = torch.zeros_like(edge_index)
        # for i in range(edge_index.size(1)):
        #     #
        #     u, v = edge_index[0, i].item(), edge_index[1, i].item()
        #
        #     # search motif_id of src_node and dis_node
        #     motifs_u = motif_idx[atom_idx == u]
        #     motifs_v = motif_idx[atom_idx == v]
        #
        #     # Find motifs shared by u and v (if any)
        #     common_motifs = torch.tensor([m.item() for m in motifs_u if m in motifs_v])
        #
        #     # When they belong to a shared motif
        #     if len(common_motifs) > 0:
        #         # Use the first shared motif
        #         motif = common_motifs[0].item()
        #         new_u = mapping.get((u, motif), u)
        #         new_v = mapping.get((v, motif), v)
        #         new_edge_index[0, i] = new_u
        #         new_edge_index[1, i] = new_v

        # Build mapping from (motif_idx, atom_idx) -> new node indices
        node_pairs = torch.stack([motif_idx, atom_idx], dim=1)
        unique_pairs, inverse_indices = torch.unique(node_pairs, dim=0, return_inverse=True)
        motif_unique = unique_pairs[:, 0]
        atom_unique = unique_pairs[:, 1]

        # Lookup table: (motif, atom) -> new node index
        num_motifs = int(motif_idx.max().item()) + 1
        num_nodes = x.size(0)
        mapping = torch.full((num_motifs, num_nodes), -1, dtype=torch.long, device=x.device)
        mapping[motif_unique, atom_unique] = torch.arange(unique_pairs.size(0), device=x.device)

        # Original edges
        u, v = edge_index

        # Membership masks per motif
        membership = mapping >= 0  # (num_motifs, num_nodes)
        mask_u = membership[:, u]  # (num_motifs, num_edges)
        mask_v = membership[:, v]
        mask_common = mask_u & mask_v  # Both atoms in motif

        # Identify first shared motif per edge
        has_common = mask_common.any(dim=0)
        first_common_motif = mask_common.float().argmax(dim=0).long()

        # Reindex nodes using shared motif mapping
        new_u = torch.where(has_common,
                            mapping[first_common_motif, u],
                            u)
        new_v = torch.where(has_common,
                            mapping[first_common_motif, v],
                            v)

        new_edge_index = torch.stack([new_u, new_v], dim=0)
        row, col = new_edge_index

        # Sum incoming edge features (sum e_uv)
        e_sum = torch.zeros(size=(x_motif_atom.size(0), edge_attr.size(1)), device=x.device)
        e_sum.index_add_(0, col, edge_attr)

        # Node attention weights (alpha_v)
        node_attn_input = torch.cat([x_motif_atom, e_sum], dim=1)
        node_alpha = torch.sigmoid(self.node_attn_fc(node_attn_input))
        h_pool = global_add_pool(node_alpha * x_motif_atom, motif_idx)


        # Compute Pair-MLP encodings


        # For edge pooling we must know which motif each edge belongs to
        #========================================================================
        # edge_motif = torch.zeros_like(row)
        # for i in range(row.size(0)):
        #     # Edge endpoints
        #     u, v = row[i].item(), col[i].item()
        #
        #     # Identify a shared motif for the two atoms (either endpoint works)
        #     motif = motif_idx[inverse_indices[u]]
        #     edge_motif[i] = motif

        #========================================================================
        node_indices = inverse_indices[row]  # shape: (num_edges,)
        edge_motif = motif_idx[node_indices]  # shape: (num_edges,)

        # h_G = sum(alpha_v * h_v) + beta * mean Pair-MLP outputs
        if self.Pair_MLP == True:
            h_u = x_motif_atom[row]
            h_v = x_motif_atom[col]

            pair_input = torch.cat([h_u, h_v, edge_attr], dim=1)  # [h_u; h_v; e_uv]
            pair_alpha = torch.sigmoid(self.pair_attn_fc(pair_input))  # pair_alpha:[E,1]
            pair_output = self.pair_mlp_concat(pair_input)
            pair_output = pair_alpha * pair_output

            edge_pool = global_add_pool(pair_output, edge_motif)
            # Pad missing motifs if needed
            padding_size = h_pool.size(0) - edge_pool.size(0)
            padding_pool = torch.zeros(size = (padding_size,edge_pool.size(1)), dtype=edge_pool.dtype, device=edge_pool.device)
            edge_pool = torch.cat([edge_pool, padding_pool], dim=0)

            # Different motifs can have varying edge counts
            edge_counts = torch.bincount(edge_motif)
            edge_counts = torch.cat([edge_counts,torch.zeros(padding_size,device=edge_counts.device)],dim=0)
            edge_counts = torch.clamp(edge_counts, min=1)
            expanded_edge_counts = edge_counts.unsqueeze(1).expand(-1, pair_output.size(1))
            edge_pool = edge_pool / expanded_edge_counts.float()
            h_G = h_pool + self.beta * edge_pool
            return node_alpha, pair_alpha, h_G

        else:
            h_G = h_pool
            pair_alpha = 0
            return node_alpha,pair_alpha,h_G

class GNNModel(torch.nn.Module):
    def __init__(self, hidden_dim, num_layers, Pair_MLP=True,gnn_type="our"):
        super(GNNModel, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.gnn_type = gnn_type
        self.Pair_MLP = Pair_MLP

        for _ in range(num_layers):
            if gnn_type == 'our':
                self.layers.append(CustomGNNLayer(hidden_dim))
            elif gnn_type == 'GCN':
                self.layers.append(GCNLayer(hidden_dim))
            elif gnn_type == 'GAT':
                self.layers.append(GATLayer(hidden_dim, heads=1))
            elif gnn_type == 'GIN':
                self.layers.append(GINLayer(hidden_dim))
            elif gnn_type == 'GINE':
                self.layers.append(GINELayer(hidden_dim))

            # ---- Additional convolutional variants ----
            elif gnn_type == 'EGConv':
                # EGConv requires num_bases and num_heads. Using example values.
                self.layers.append(EGConv(hidden_dim, hidden_dim, num_bases=4, num_heads=4))
            elif gnn_type == 'FAConv':
                self.layers.append(FAConv(hidden_dim, hidden_dim))
            elif gnn_type == 'FiLMConv':
                self.layers.append(FiLMConv(hidden_dim, hidden_dim))
            elif gnn_type == 'PANConv':
                # PANConv does not require num_nodes at init, it can infer it from 'x' in forward
                self.layers.append(PANConv(hidden_dim, hidden_dim))
            else:
                raise ValueError(f"Unsupported gnn_type: {gnn_type}")

        self.final_mlp = nn.Sequential(
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim)
        )

        self.Pool = HierarchicalEdgePooling(hidden_dim, hidden_dim, hidden_dim,self.Pair_MLP)

    def forward(self, x, edge_index, edge_attr, motif_atom_edge_index):
        """
        (1) Intra-motif message passing
        (2) Motif-level readout
        (3) Motif encoding
        """
        # (1) Intra-motif message passing
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)

        x = self.final_mlp(x)

        # (2) Motif-level readout
        node_alpha,pair_alpha,h_g = self.Pool(x, edge_index, edge_attr, motif_atom_edge_index)

        return node_alpha,pair_alpha,h_g, x
        # return h_g, x


