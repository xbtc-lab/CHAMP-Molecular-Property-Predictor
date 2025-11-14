import torch
from torch_scatter import scatter_add
import torch.nn.functional as F
def compute_ring_contrastive_loss(batch, temperature=0.1, eps=1e-8):
    """
    Structure-aware contrastive loss restricted to ring-5 and ring-6 motifs.
    """
    z = batch["motif"].x  # [N, D]
    labels = batch["mol"].y[batch["motif"].batch]  # [N]
    type_list = batch["motif"].type  # [N]

    # Count atoms per motif
    edge_index = batch["motif", "contains", "atom"].edge_index
    motif_indices = edge_index[0]
    atoms_per_motif = scatter_add(torch.ones_like(motif_indices), motif_indices, dim=0)  # [N]

    # Build motif_type descriptors
    type_prefix = {0: 'ring', 1: 'non-cycle', 2: 'chain', 3: 'other'}
    motif_type = [f"{type_prefix[int(t)]}{int(n.item())}" for t, n in zip(type_list, atoms_per_motif)]   # domain
    type_class = [type_prefix[int(t)] for t in type_list]                                                # type

    # Keep only ring5 and ring6
    allowed_types = {'ring5', 'ring6'}
    valid_mask = [m in allowed_types for m in motif_type]
    valid_mask = torch.tensor(valid_mask, dtype=torch.bool, device=z.device)

    # Insufficient samples -> zero loss
    if valid_mask.sum() < 2:
        return torch.tensor(0.0, device=z.device)

    # Filter valid samples
    z = z[valid_mask]
    labels = labels[valid_mask]
    motif_type = [m for i, m in enumerate(motif_type) if valid_mask[i]]
    type_class = [t for i, t in enumerate(type_class) if valid_mask[i]]

    # Convert to tensors
    labels = labels.view(-1, 1)
    motif_type_tensor = torch.tensor([hash(m) for m in motif_type], device=z.device)
    type_class_tensor = torch.tensor([hash(t) for t in type_class], device=z.device).view(-1, 1)

    # Similarity matrix
    sim_matrix = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=-1)

    # Masks
    N = z.size(0)
    diag_mask = ~torch.eye(N, dtype=torch.bool, device=z.device)

    label_eq = (labels == labels.T)
    label_neq = (labels != labels.T)
    type_eq = (motif_type_tensor.view(-1, 1) == motif_type_tensor.view(1, -1))
    class_eq = (type_class_tensor == type_class_tensor.T)

    pos_mask = label_eq & type_eq & diag_mask     # Same label and same domain
    neg_mask = label_neq & class_eq & diag_mask   # Different label but same superclass

    # Contrastive terms
    pos_sim = sim_matrix.masked_fill(~pos_mask, -1e9)
    neg_sim = sim_matrix.masked_fill(~neg_mask, -1e9)

    # Positive pairs
    numerator = torch.exp(pos_sim / temperature).sum(dim=-1)
    # Negative pairs
    denominator = numerator + torch.exp(neg_sim / temperature).sum(dim=-1)

    valid = (pos_mask.sum(dim=-1) > 0)
    loss = -torch.log((numerator + eps) / (denominator + eps))

    return loss[valid].mean()


import torch
import torch.nn.functional as F
from torch_scatter import scatter_add

def compute_ring_contrastive_loss_multilabel(batch, temperature=0.1, eps=1e-8):
    """
    Multi-label variant that supports NaNs and only considers ring-5/6 motifs.
    """
    z = batch["motif"].x  # [N, D]
    raw_labels = batch["mol"].y[batch["motif"].batch]  # [N, C]
    type_list = batch["motif"].type  # [N]

    # Count atoms per motif
    edge_index = batch["motif", "contains", "atom"].edge_index
    motif_indices = edge_index[0]
    atoms_per_motif = scatter_add(torch.ones_like(motif_indices), motif_indices, dim=0)

    # Build motif descriptors
    type_prefix = {0: 'ring', 1: 'non-cycle', 2: 'chain', 3: 'other'}
    motif_type = [f"{type_prefix[int(t)]}{int(n.item())}" for t, n in zip(type_list, atoms_per_motif)]
    type_class = [type_prefix[int(t)] for t in type_list]

    # Keep only ring5 and ring6
    allowed_types = {'ring5', 'ring6'}
    valid_mask = torch.tensor([m in allowed_types for m in motif_type], dtype=torch.bool, device=z.device)

    if valid_mask.sum() < 2:
        return torch.tensor(0.0, device=z.device)

    z = z[valid_mask]
    raw_labels = raw_labels[valid_mask]      # [N, C]
    motif_type = [m for i, m in enumerate(motif_type) if valid_mask[i]]
    type_class = [t for i, t in enumerate(type_class) if valid_mask[i]]

    # ========================
    # Remove samples whose labels are all NaN
    # ========================
    not_nan_mask = ~torch.isnan(raw_labels)  # [N, C]
    valid_label_mask = not_nan_mask.any(dim=1)  # Need at least one finite label

    if valid_label_mask.sum() < 2:
        return torch.tensor(0.0, device=z.device)

    z = z[valid_label_mask]
    labels = raw_labels[valid_label_mask]
    motif_type = [m for i, m in enumerate(motif_type) if valid_label_mask[i]]
    type_class = [t for i, t in enumerate(type_class) if valid_label_mask[i]]

    # Replace NaNs with zero (only overlap is checked)
    labels = torch.nan_to_num(labels, nan=0.0)

    # Hash motif/category
    motif_type_tensor = torch.tensor([hash(m) for m in motif_type], device=z.device)
    type_class_tensor = torch.tensor([hash(t) for t in type_class], device=z.device).view(-1, 1)

    # Similarity/cosine similarity
    sim_matrix = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=-1)
    N = z.size(0)
    diag_mask = ~torch.eye(N, dtype=torch.bool, device=z.device)

    # Multi-label similarity
    label_overlap = torch.matmul(labels.float(), labels.T.float())  # [N, N]
    label_eq = label_overlap > 0
    label_neq = label_overlap == 0

    type_eq = motif_type_tensor.view(-1, 1) == motif_type_tensor.view(1, -1)
    class_eq = type_class_tensor == type_class_tensor.T

    pos_mask = label_eq & type_eq & diag_mask
    neg_mask = label_neq & class_eq & diag_mask

    pos_sim = sim_matrix.masked_fill(~pos_mask, -1e9)
    neg_sim = sim_matrix.masked_fill(~neg_mask, -1e9)

    numerator = torch.exp(pos_sim / temperature).sum(dim=-1)
    denominator = numerator + torch.exp(neg_sim / temperature).sum(dim=-1)

    valid = (pos_mask.sum(dim=-1) > 0)
    loss = -torch.log((numerator + eps) / (denominator + eps))

    return loss[valid].mean()





def  compute_nonring_contrastive_loss(batch, threshold=0.9, temperature=0.1, eps=1e-8):
    """
        Positive: same domain and same label.
        Negative: same domain but different labels.

    Applies only to non-ring (type != 0) motifs.
    """
    z = batch["motif"].x  # [N, D]
    labels = batch["mol"].y[batch["motif"].batch]  # [N]
    type_list = batch["motif"].type  # [N]
    vectors = batch["motif"].vector  # [N, V]

    # Select non-ring motifs
    nonring_mask = (type_list != 0)
    if nonring_mask.sum() < 2:
        return torch.tensor(0.0, device=z.device)

    # Filter tensors
    z = z[nonring_mask]
    labels = labels[nonring_mask]
    vectors = vectors[nonring_mask]

    # Similarity matrix based on domain vectors
    vec_sim_matrix = F.cosine_similarity(vectors.unsqueeze(1), vectors.unsqueeze(0), dim=-1)
    domain_mask = (vec_sim_matrix >= threshold)

    # Embedding similarity matrix
    sim_matrix = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=-1)

    # Build masks
    N = z.size(0)
    diag_mask = ~torch.eye(N, dtype=torch.bool, device=z.device)
    label_eq = (labels.view(-1, 1) == labels.view(1, -1))
    label_neq = ~label_eq

    # Positive: same label & domain
    pos_mask = label_eq & domain_mask & diag_mask
    # Negative: different label but same domain
    neg_mask = label_neq & domain_mask & diag_mask

    # Not enough positives
    if pos_mask.sum() < 1:
        return torch.tensor(0.0, device=z.device)

    # InfoNCE
    pos_sim = sim_matrix.masked_fill(~pos_mask, -1e9)
    neg_sim = sim_matrix.masked_fill(~neg_mask, -1e9)

    numerator = torch.exp(pos_sim / temperature).sum(dim=-1)
    denominator = numerator + torch.exp(neg_sim / temperature).sum(dim=-1)

    valid = (pos_mask.sum(dim=-1) > 0)
    loss = -torch.log((numerator + eps) / (denominator + eps))

    return loss[valid].mean()


def compute_nonring_contrastive_loss_multilabel(batch, threshold=0.9, label_sim_threshold=0.5, temperature=0.1, eps=1e-8):
    """
    Multi-label + missing-value variant for non-ring motifs:
    - Positive: same domain & label similarity >= threshold
    - Negative: same domain & label similarity < threshold
    - Samples with NaN labels are ignored
    """

    z = batch["motif"].x  # [N, D]
    labels_raw = batch["mol"].y[batch["motif"].batch]  # [N, C] multi-label
    type_list = batch["motif"].type  # [N]
    vectors = batch["motif"].vector  # [N, V]

    # Select non-ring motifs
    nonring_mask = (type_list != 0)
    if nonring_mask.sum() < 2:
        return torch.tensor(0.0, device=z.device)

    # Filter tensors
    z = z[nonring_mask]
    labels_raw = labels_raw[nonring_mask]
    vectors = vectors[nonring_mask]

    # Remove samples containing NaNs
    valid_label_mask = ~torch.isnan(labels_raw).any(dim=1)
    if valid_label_mask.sum() < 2:
        return torch.tensor(0.0, device=z.device)

    z = z[valid_label_mask]
    labels = labels_raw[valid_label_mask]  # [N, C]
    vectors = vectors[valid_label_mask]

    # Domain similarity matrix
    vec_sim_matrix = F.cosine_similarity(vectors.unsqueeze(1), vectors.unsqueeze(0), dim=-1)
    domain_mask = (vec_sim_matrix >= threshold)

    # Embedding similarity
    sim_matrix = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=-1)

    # Label similarity (cosine)
    label_sim_matrix = F.cosine_similarity(labels.unsqueeze(1), labels.unsqueeze(0), dim=-1)
    label_eq = label_sim_matrix >= label_sim_threshold
    label_neq = ~label_eq

    N = z.size(0)
    diag_mask = ~torch.eye(N, dtype=torch.bool, device=z.device)

    # Positive/negative masks
    pos_mask = label_eq & domain_mask & diag_mask
    neg_mask = label_neq & domain_mask & diag_mask

    if pos_mask.sum() < 1:
        return torch.tensor(0.0, device=z.device)

    # InfoNCE
    pos_sim = sim_matrix.masked_fill(~pos_mask, -1e9)
    neg_sim = sim_matrix.masked_fill(~neg_mask, -1e9)

    numerator = torch.exp(pos_sim / temperature).sum(dim=-1)
    denominator = numerator + torch.exp(neg_sim / temperature).sum(dim=-1)

    valid = (pos_mask.sum(dim=-1) > 0)
    loss = -torch.log((numerator + eps) / (denominator + eps))

    return loss[valid].mean()

def compute_ring_contrastive_loss_regression(batch, temperature=0.1, sigma=1.0, label_thresh_ratio=0.1, eps=1e-8):
    """
    Regression-oriented contrastive loss for ring-5/6 motifs.
    """
    z = batch["motif"].x
    labels = batch["mol"].y[batch["motif"].batch]
    type_list = batch["motif"].type

    # Count atoms per motif (used as suffix)
    edge_index = batch["motif", "contains", "atom"].edge_index
    motif_indices = edge_index[0]
    atoms_per_motif = scatter_add(torch.ones_like(motif_indices), motif_indices, dim=0)

    # Build motif_type (domain)
    type_prefix = {0: 'ring', 1: 'non-cycle', 2: 'chain', 3: 'other'}
    motif_type = [f"{type_prefix[int(t)]}{int(n.item())}" for t, n in zip(type_list, atoms_per_motif)]

    # Keep ring5/ring6
    allowed_types = {'ring5', 'ring6'}
    valid_mask = [m in allowed_types for m in motif_type]
    valid_mask = torch.tensor(valid_mask, dtype=torch.bool, device=z.device)

    if valid_mask.sum() < 2:
        return torch.tensor(0.0, device=z.device)

    # Filter tensors
    z = z[valid_mask]
    labels = labels[valid_mask].view(-1, 1)  # [N, 1]
    motif_type = [m for i, m in enumerate(motif_type) if valid_mask[i]]

    # Convert to tensor
    motif_type_tensor = torch.tensor([hash(m) for m in motif_type], device=z.device)

    # Cosine similarity matrix
    sim_matrix = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=-1)  # [N, N]

    # Same domain & exclude diagonal
    N = z.size(0)
    diag_mask = ~torch.eye(N, dtype=torch.bool, device=z.device)
    domain_mask = motif_type_tensor.view(-1, 1) == motif_type_tensor.view(1, -1)
    domain_mask = domain_mask & diag_mask

    # Relative label difference
    label_diff = torch.abs(labels - labels.T)
    label_base = torch.max(labels, labels.T) + eps
    relative_diff = label_diff / label_base  # Percentage difference

    # Positive: same domain + below threshold
    pos_mask = domain_mask & (relative_diff < label_thresh_ratio)
    neg_mask = domain_mask & (relative_diff >= label_thresh_ratio)

    if pos_mask.sum() < 1:
        return torch.tensor(0.0, device=z.device)

    # Gaussian weights over label difference (positives only)
    weight_matrix = torch.exp(- (label_diff ** 2) / (2 * sigma ** 2))
    pos_weight = weight_matrix.masked_fill(~pos_mask, 0.0)

    # Apply masks
    pos_sim = sim_matrix.masked_fill(~pos_mask, -1e9)
    neg_sim = sim_matrix.masked_fill(~neg_mask, -1e9)

    numerator = (torch.exp(pos_sim / temperature) * pos_weight).sum(dim=-1)
    denominator = numerator + torch.exp(neg_sim / temperature).sum(dim=-1)

    valid = (pos_mask.sum(dim=-1) > 0)
    loss = -torch.log((numerator + eps) / (denominator + eps))

    return loss[valid].mean()

def compute_nonring_contrastive_loss_regression(batch, threshold=0.9, label_thresh_ratio=0.1, sigma=1.0, temperature=0.1, eps=1e-8):
    """
    Regression loss for non-ring motifs with structure-aware contrastive terms.
    """
    z = batch["motif"].x  # [N, D]
    labels = batch["mol"].y[batch["motif"].batch].view(-1, 1)  # [N, 1]
    type_list = batch["motif"].type  # [N]
    vectors = batch["motif"].vector  # [N, V]

    # Select non-ring motifs (type != 0)
    nonring_mask = (type_list != 0)
    if nonring_mask.sum() < 2:
        return torch.tensor(0.0, device=z.device)

    z = z[nonring_mask]
    labels = labels[nonring_mask]
    vectors = vectors[nonring_mask]

    # Domain similarity based on atom-type vectors
    vec_sim_matrix = F.cosine_similarity(vectors.unsqueeze(1), vectors.unsqueeze(0), dim=-1)
    domain_mask = (vec_sim_matrix >= threshold)

    # Relative label differences
    label_diff = torch.abs(labels - labels.T)
    label_base = torch.max(labels, labels.T) + eps
    relative_diff = label_diff / label_base

    # Positive when difference is small within same domain
    pos_mask = (relative_diff < label_thresh_ratio) & domain_mask
    neg_mask = (relative_diff >= label_thresh_ratio) & domain_mask
    diag_mask = ~torch.eye(z.size(0), dtype=torch.bool, device=z.device)
    pos_mask = pos_mask & diag_mask
    neg_mask = neg_mask & diag_mask

    if pos_mask.sum() < 1:
        return torch.tensor(0.0, device=z.device)

    # Feature similarity for contrastive loss
    sim_matrix = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=-1)

    # Gaussian weighting (positives only)
    weight_matrix = torch.exp(- (label_diff ** 2) / (2 * sigma ** 2))
    pos_weight = weight_matrix.masked_fill(~pos_mask, 0.0)

    # Weighted InfoNCE
    pos_sim = sim_matrix.masked_fill(~pos_mask, -1e9)
    neg_sim = sim_matrix.masked_fill(~neg_mask, -1e9)

    numerator = (torch.exp(pos_sim / temperature) * pos_weight).sum(dim=-1)
    denominator = numerator + torch.exp(neg_sim / temperature).sum(dim=-1)

    valid = (pos_mask.sum(dim=-1) > 0)
    loss = -torch.log((numerator + eps) / (denominator + eps))

    return loss[valid].mean()




