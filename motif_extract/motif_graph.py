import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.datasets import MoleculeNet

import os
from rdkit import Chem
from IPython.display import SVG

# Reload the module on every run
import importlib
from motif_extract import mol_motif
importlib.reload(mol_motif)

# Convert motifs to Data objects
# RDKit atom indices exactly match those in torch_geometric Data

import torch
# For a motif, return its edge_index (0-based) and indices (to look up edge_attr)
def motif_in_edge(data, motif):
    # Extract atom indices from the motif
    atom_indices = list(motif)

    # Create subgraph node features and edge indices
    edge_index = []  # Edge indices
    edge_index_indices = []

    # Build edge indices
    for i in range(data.edge_index.size(1)):
        start = data.edge_index[0, i].item()
        end = data.edge_index[1, i].item()
        if start in atom_indices and end in atom_indices:
            # Map to subgraph atom indices
            new_start = atom_indices.index(start)
            new_end = atom_indices.index(end)
            edge_index.append([new_start, new_end])
            edge_index_indices.append(i)

    # Convert to torch.Tensor
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_index_indices = torch.tensor(edge_index_indices, dtype=torch.long)
    # Return the Data object components
    return edge_index,edge_index_indices

# ToDo: ensure motif-level isomorphism (including edges)
class MotifGINLayer(MessagePassing):
    """Isomorphism-aware message passing layer"""

    def __init__(self, hidden_dim):
        super().__init__(aggr="add")  # Sum aggregation preserves WL isomorphism

        # Learn node-level isomorphism information
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.eps = nn.Parameter(torch.Tensor([0]))  # Learnable perturbation

    def forward(self, x, edge_index, edge_attr,is_edge):
        # Message passing and aggregation: message -> update -> aggregate
        if is_edge:
            out = self.propagate(
                edge_index,
                x=x,
                edge_attr=edge_attr
            )
            # Residual connection for the central node
            return self.node_mlp((1 + self.eps) * x + out)
        else:
            return self.node_mlp(x)

    def message(self, x_j, edge_attr):
        """Message computation: edge features dynamically modulate node features"""
        return edge_attr * x_j  # Element-wise product (E, hidden_dim)

# ToDo: refine motif embedding strategy
class MotifEncoder(nn.Module):
    """End-to-end motif encoding model"""
    def __init__(self,
                 # Input dimensions
                 atom_feature_dim=9,  # Atom feature dimension
                 edge_feat_dim=3,  # Edge feature dimension
                 # Output dimensions
                 hidden_dim=16,
                 type_hidden_dim=16,
                 num_layers=2):
        super().__init__()
        self.type_hidden_dim = type_hidden_dim
        self.hidden_dim = hidden_dim

        # 0. Type encoding
        self.type_embedder = torch.nn.Embedding(4, type_hidden_dim)

        # 1. Atom encoder
        self.atom_nn = torch.nn.Linear(atom_feature_dim, hidden_dim)

        # 2. Bond encoder
        self.edge_nn = torch.nn.Linear(edge_feat_dim, hidden_dim)

        # 3. GIN layers to capture motif isomorphism
        self.convs = nn.ModuleList([
            MotifGINLayer(hidden_dim) for _ in range(num_layers)
        ])

        # 5. Node-level attention
        self.node_attn = nn.Sequential(
            nn.Linear(hidden_dim * 2, 1),  # Input: [node feature, aggregated edges]
            nn.Sigmoid()
        )

        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + hidden_dim, hidden_dim),  # Input: [h_i, h_j, e_ij]
            nn.ReLU()
        )

        # 6. Motif encoders
        self.motif_nn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.motif_edge_nn = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, data: Data,motifs_node,motifs_type,motif_edge_index,motif_edge_attr):
        motifs_X = torch.empty(0,self.hidden_dim)
        # Encode each motif first
        for type,motif in zip(motifs_type,motifs_node):
            # 1. Motif type encoding
            motif_type_node = self.type_embedder(torch.tensor(type))
            motif_type_node[self.type_hidden_dim // 2:] = motif_type_node[self.type_hidden_dim // 2:] * data.x.size(0)

            # 2. GIN to ensure isomorphism
            # 1) Atom encoding
            x = self.atom_nn(torch.index_select(data.x,0,torch.tensor(list(motif))).float())
            # 2) Edge encoding
            node_edge_index, node_edge_index_indices = motif_in_edge(data, motif)    # Local motif indices; matches motif atom ordering, shape = [2, E]
            # Motif contains edges
            if node_edge_index.shape[0] != 0:
                edge_attr = torch.index_select(data.edge_attr, 0, node_edge_index_indices).float()
                edge_attr = self.edge_nn(edge_attr)    # (E, edge_feat_dim)
                for conv in self.convs:
                    x = conv(x, node_edge_index, edge_attr, True)

                edge_agg = torch.zeros_like(x)
                row, col = node_edge_index
                # Aggregate edge weights based on edge type
                for i in range(edge_attr.size(0)):
                    edge_agg[row[i]] += edge_attr[i, :]
                    edge_agg[col[i]] += edge_attr[i, :]

                # Node-level attention (edge_agg can be further improved)
                alpha = self.node_attn(torch.cat([x, edge_agg], dim=-1))  # (N,1)
                h_nodes = torch.sum(alpha * x, dim=0)  # (hidden_dim)

                # Joint edge-node encoding
                h_edges = []
                for i in range(node_edge_index.size(1)):
                    src, dst = node_edge_index[:, i]
                    h_edge = self.edge_mlp(
                        torch.cat([x[src], x[dst], edge_attr[i]], dim=-1)
                    )
                    h_edges.append(h_edge)

                h_edges = torch.mean(torch.stack(h_edges), dim=0)  # (hidden_dim)
                motif_embedding = self.motif_nn(torch.concat([h_nodes + h_edges, motif_type_node], dim=-1)).unsqueeze(0)  # (hidden_dim)
            # Motif without edges
            else:
                # Isomorphism encoding
                for conv in self.convs:
                    x = conv(x, None, None, False)
                # Readout
                h_nodes = x.squeeze(0)
                motif_embedding = self.motif_nn(torch.concat([h_nodes, motif_type_node], dim=-1)).unsqueeze(0)  # (hidden_dim)

            motifs_X = torch.cat((motifs_X, motif_embedding), dim=0)

        # Encode motif edges: motif types + participating atoms/edges. motif_edge_index.shape = [2, E], motif_edge_attr.shape = [E, :]
        motifs_edge_attr = torch.empty(0,self.hidden_dim)
        for i,(attr) in enumerate(motif_edge_attr):
            # Look up motif types at both ends
            start = motifs_type[motif_edge_index[:,i][0].item()]
            end   = motifs_type[motif_edge_index[:,i][1].item()]
            motif_type_edge = (self.type_embedder(torch.tensor(start)) + self.type_embedder(torch.tensor(end))).unsqueeze(0)
            # Preserve original nodes/edges: two nodes surrounding the edge
            h_atom_edge = torch.zeros(1, self.hidden_dim)
            h_atom_edge += self.atom_nn(data.x[attr["node"],:].float()).sum(dim=0)
            h_atom_edge += self.edge_nn(data.edge_attr[attr["edge"],:].float()).sum(dim=0)
            # Merge embeddings
            motif_attr_embedding = self.motif_edge_nn(torch.cat((motif_type_edge, h_atom_edge), dim=-1))
            motifs_edge_attr = torch.cat((motifs_edge_attr, motif_attr_embedding), dim=0)

        # Assemble the motif-level Data object
        motif_Data = Data(x = motifs_X,edge_index=motif_edge_index,edge_attr = motifs_edge_attr,smiles = data.smiles,y=data.y)
        return motif_Data

# Find all connecting edges within a node list
def find_unique_edges_with_indices(node_list, edge_index):
    """
    Identify connecting edges among the provided node list and deduplicate them
    (supports PyTorch tensors). Also return the indices of those edges in the
    original edge_index tensor.

    Args:
        node_list (list): Nodes of interest.
        edge_index (torch.Tensor): Edge indices shaped [2, num_edges].

    Returns:
        tuple:
            - list of tuples: unique edges formatted as [(src, dst), ...].
            - list of int: indices of those edges within edge_index.
    """
    # Ensure edge_index is 2 x num_edges
    if edge_index.size(0) != 2:
        raise ValueError("edge_index must have shape [2, num_edges]")

    # Create a set for faster membership checks
    node_set = set(node_list)

    # Extract source and destination rows
    src_nodes = edge_index[0]  # Row 0: sources
    dst_nodes = edge_index[1]  # Row 1: destinations

    # Filter edges where both endpoints are in node_set and normalize ordering
    unique_edges = set()
    edge_indices = []  # Track the indices of qualifying edges

    for i, (src, dst) in enumerate(zip(src_nodes.tolist(), dst_nodes.tolist())):
        if src in node_set and dst in node_set:
            normalized_edge = tuple(sorted((src, dst)))  # Normalize ordering
            if normalized_edge not in unique_edges:
                unique_edges.add(normalized_edge)
                edge_indices.append(i)  # Record the index

    # Convert to list before returning
    return list(unique_edges), edge_indices[0]


# Build inter-motif edges: returned edge_index uses motif ordering
def get_motif_edge(data, motifs_result):
    """
    Construct the motif-level graph structure and return edge_index/edge_attr.
    Assumptions:
        1. No overlap: check edges within motifs.
        2. Two-node overlap: treat as ring-like edge.
        3. Single-node overlap: treat as node connection.

    Args:
        data (torch_geometric.data.Data): Molecule-level graph.
        motifs_result (list of set): Node sets for each motif.

    Returns:
        tuple: edge_index, edge_attr describing motif connectivity.
    """
    # 1. Parse SMILES to recover molecular topology
    mol = Chem.MolFromSmiles(data.smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")

    # Extract bonds
    bonds = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in mol.GetBonds()]

    # 2. Initialize motif-level nodes and edges
    num_motifs = len(motifs_result)
    edge_index = []  # Motif-level edges
    edge_attr = []  # Edge attributes (retain original bond context)
    edge_attr_dim = data.edge_attr.shape[1]
    # Convert motif sets to lists for indexing
    motifs_result = [list(motif) for motif in motifs_result]

    # 3. Build motif-level edges
    for i in range(num_motifs):
        for j in range(i + 1, num_motifs):

            # Check whether motif_i and motif_j intersect
            intersection = set(motifs_result[i]).intersection(set(motifs_result[j]))

            # When overlapping, connect motifs directly using intersection data
            if intersection:

                # Skip invalid cases with >2 overlapping nodes
                if len(intersection) > 2:
                    return None

                edge_index.append([j, i])
                edge_index.append([i, j])

                # One overlapping node -> treat as node-node connector
                if len(intersection) == 1:
                    atom = next(iter(intersection))
                    edge_attr.append([0] * edge_attr_dim + [atom,atom])      # Edge, node, node
                    edge_attr.append([0] * edge_attr_dim + [atom,atom])      # Edge, node, node

                # Two overlapping nodes -> fetch the exact edge
                elif len(intersection) == 2:
                    # Retrieve the corresponding edge index
                    edge,edge_indices = find_unique_edges_with_indices(list(intersection),data.edge_index)
                    edge_attr.append([k.item() for k in data.edge_attr[edge_indices]] + list(intersection))
                    edge_attr.append([k.item() for k in data.edge_attr[edge_indices]] + list(intersection))


            # No overlap: check whether any intra-motif edge links the motifs
            else:
                for u in motifs_result[i]:
                    for v in motifs_result[j]:
                        if (u, v) in bonds or (v, u) in bonds:
                            # Found a connecting bond between the two motifs
                            edge_index.append([i, j])
                            edge_index.append([j, i])  # Undirected graph: add reverse
                            # Locate the bond index
                            index = [k for k in range(data.edge_index.size(1)) if data.edge_index[0, k].item() == u and data.edge_index[1, k].item() == v][0]

                            edge_attr.append([k.item() for k in data.edge_attr[index]] + [u,v])
                            edge_attr.append([k.item() for k in data.edge_attr[index]] + [u,v])
                            break

    # 5. Convert to tensors
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # Shape [2, num_edges]
    edge_attr = torch.tensor(edge_attr, dtype=torch.long)  # Edge attributes

    return edge_index,edge_attr

if __name__ == '__main__':
    # Dataset
    datasets = MoleculeNet(root="./dataset/", name="Lipo")
    Data_motif_list = []
    # motif_graph = MotifEncoder(
    #              # Input dimensions
    #              atom_feature_dim=9,  # Atom feature dim
    #              edge_feat_dim=3,  # Edge feature dim
    #              # Output dimensions
    #              hidden_dim=16,
    #              type_hidden_dim=16,
    #              num_layers=2)


    for i,data in enumerate(datasets[:100]):
        smiles = data.smiles
        mol = Chem.MolFromSmiles(smiles)
        print(smiles)
        # print(i, smiles)
        if mol is not None:
            try:
                # Partition substructures and their types
                motifs_type,motifs_node = mol_motif.mol_get_motif(mol)

                # Build motif edges
                if get_motif_edge(data, motifs_node) is None:
                    continue
                motif_edge_index,motif_edge_attr = get_motif_edge(data, motifs_node)

                motifs_type = torch.tensor(motifs_type, dtype=torch.long)

                print(motifs_node)

                # Save visualization
                svg = mol_motif.visualize_motif(mol, motifs_node,method='save')
                with open(f'./dataset/Lipo_image/{i}.png', 'wb') as file:
                    file.write(svg)
                    print(f"Saved image #{i}")

                # Training placeholder

                # motif_X = []
                # for motif in motifs_node:
                #     # Build motif_Data
                #     motif_type = next(iter([k for k, lst in motifs_type.items() if motif in lst]))
                #     motifs_Data = motif_to_Data(data, motif, motif_type)
                #     # Obtain motif and atom embeddings
                #     h_atom, h_motif = motif_encoder(motifs_Data)
                #     # Inject h_atom into data
                #     motif_X.append(h_motif)
                # # Goal: given functional group sequences/types and their
                # # connecting atoms/edges -> build the motif graph
                # Data_motif = get_motif_edge(data, motifs_node, motif_X)
                # Data_motif_list.append(Data_motif)

            except Exception as e:
                print(print(f"Error processing SMILES '{i}:{data.smiles}': {e}"))


    # Training/evaluation placeholder
    # dataset = GraphDataset(root='./dataset/motif_Tox21', data_list=Data_motif_list)

    # For every molecule
    """
    (1) Molecule -> motif sets
    (2) Learn motif embeddings while caching atom embeddings
    (2) Use motif sets to build motif-level graphs with required info
    """

    # For each molecule we should store both atom_Data and motif_Data


    # motifs_X =





