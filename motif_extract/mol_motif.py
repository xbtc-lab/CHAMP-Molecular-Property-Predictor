from collections import deque
from IPython.display import Image
from rdkit.Chem.Draw import rdMolDraw2D
from IPython.display import SVG       # Used for generating vector graphics
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from rdkit.Chem import Draw, rdmolops
import numpy as np
from rdkit import Chem
# from torch.distributed.rpc.api import method
from torch_geometric.datasets import MoleculeNet


# Visualize functional groups
def visualize_motif(mol, fgs, method = "display"):
    # mol = Chem.MolFromSmiles(smiles)
    # Label each atom
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        atom.SetProp('atomNote', str(idx))  # Use atom index as label

    # Create drawing backends
    # For saving output:
    if method == "save":
        drawer = rdMolDraw2D.MolDraw2DCairo(500, 300)
        drawer.SetFontSize(6)  # Atom index font size

    if method == "display":
        drawer = rdMolDraw2D.MolDraw2DSVG(500, 300)
        drawer.SetFontSize(6)  # Atom index font size


    # Assign a color to each motif
    colors = list(mcolors.TABLEAU_COLORS.values())  # Use the Matplotlib palette
    # Map of atom -> color
    highlight_atoms = {}

    for i, fg in enumerate(fgs):
        color = mcolors.to_rgb(colors[i % len(colors)])
        # Convert set to list for indexing
        fg_list = list(fg)
        # Highlight motif atoms
        for atom in fg_list:
            highlight_atoms[atom] = color

    # Draw the molecule and highlight atoms/bonds
    drawer.DrawMolecule(
        mol,
        highlightAtoms=list(highlight_atoms.keys()),
        highlightAtomColors=highlight_atoms,
    )

    # Finish drawing
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return svg

# TODO: add try/except error handling if needed
# Distinguish aromatic and non-aromatic rings
def visualize_ring_aromaticity(mol):
    # Detect all rings and classify them
    rings = mol.GetRingInfo().AtomRings()
    aromatic_rings = []
    non_aromatic_rings = []

    # Determine whether each ring is aromatic
    for ring in rings:
        is_aromatic = True
        # Check whether all bonds in the ring are aromatic
        bond_ids = []
        for i in range(len(ring)):
            a1 = ring[i]
            a2 = ring[(i + 1) % len(ring)]
            bond = mol.GetBondBetweenAtoms(a1, a2)
            if bond and not bond.GetIsAromatic():
                is_aromatic = False
                break
        if is_aromatic:
            aromatic_rings.append(ring)
        else:
            non_aromatic_rings.append(ring)

    # Prepare visualization parameters
    atom_colors = {}
    bond_colors = {}

    # Aromatic rings rendered in red
    for ring in aromatic_rings:
        for atom in ring:
            atom_colors[atom] = (1, 0, 0)  # RGB red
        for i in range(len(ring)):
            a1 = ring[i]
            a2 = ring[(i + 1) % len(ring)]
            bond = mol.GetBondBetweenAtoms(a1, a2)
            if bond:
                bond_colors[bond.GetIdx()] = (1, 0, 0)

    # Non-aromatic rings rendered in green
    for ring in non_aromatic_rings:
        for atom in ring:
            atom_colors[atom] = (0, 1, 0)  # RGB green
        for i in range(len(ring)):
            a1 = ring[i]
            a2 = ring[(i + 1) % len(ring)]
            bond = mol.GetBondBetweenAtoms(a1, a2)
            if bond:
                bond_colors[bond.GetIdx()] = (0, 1, 0)
    # Add atom labels
    mol = Chem.Mol(mol)
    # Label each atom
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        atom.SetProp('atomNote', str(idx))  # Use atom index as label

    # Generate SVG image
    drawer = rdMolDraw2D.MolDraw2DSVG(600, 400)
    if len(rings)!=0:
        drawer.DrawMolecule(
            mol,
            highlightAtoms=atom_colors.keys(),
            highlightAtomColors=atom_colors,
            highlightBonds=bond_colors.keys(),
            highlightBondColors=bond_colors
        )
        drawer.FinishDrawing()

    # Return SVG payload
    svg = drawer.GetDrawingText().replace('svg:', '')
    return aromatic_rings, non_aromatic_rings, svg
    # return rings, non_aromatic_rings, svg

### Merge qualifying non-aromatic rings
def merge_aromatic_rings(rings):
    """
    Merge aromatic ring tuples when the conditions are satisfied.

    Args:
        rings (list of tuple): Non-aromatic ring tuples.

    Returns:
        list of tuple: Merged ring tuples.
    """
    # Convert tuples to sets for easier operations
    rings = [set(ring) for ring in rings]

    # Track whether further merging is required
    merged = True

    # Keep merging until no more rings can be combined
    while merged:
        merged = False
        new_rings = []

        # Iterate over pairs of rings
        while rings:
            current_ring = rings.pop(0)
            merged_with_existing = False

            # Check whether current_ring can merge with a ring in new_rings
            for i, new_ring in enumerate(new_rings):
                # Merge when both rings contain >=5 atoms and share >=2 atoms
                if len(current_ring) >= 5 and len(new_ring) >= 5 and len(current_ring & new_ring) >= 2:
                    new_rings[i] = current_ring | new_ring  # Merge rings
                    merged_with_existing = True
                    merged = True
                    break

            # Otherwise add current_ring as-is
            if not merged_with_existing:
                new_rings.append(current_ring)

        # Update ring list
        rings = new_rings

    # Convert sets back to sorted tuples
    return [tuple(sorted(ring)) for ring in rings]

def merge_single_h_neighbors(mol, merged_rings):
    """
    Merge single-atom nodes (only connected to hydrogens) into adjacent
    non-aromatic rings.
    Args:
        mol: RDKit molecule
        merged_rings: merged non-aromatic rings, each entry is a set of atom ids
    Returns:
        Updated non-aromatic ring list
    """
    new_rings = []

    for ring in merged_rings:
        extended_ring = set(ring)
        # Track atoms that should be merged
        candidates = set()

        # Iterate over all atoms in the ring
        for atom_idx in ring:
            atom = mol.GetAtomWithIdx(atom_idx)

            # Iterate over atom neighbors
            for neighbor in atom.GetNeighbors():
                neighbor_idx = neighbor.GetIdx()

                # Skip atoms already in the ring
                if neighbor_idx in ring:
                    continue

                # Conditions:
                # 1) All non-ring neighbors are hydrogens
                # 2) The atom itself is not hydrogen
                if neighbor.GetAtomicNum() == 1:
                    continue  # Skip hydrogen atoms

                all_hydrogen = True
                for nbr in neighbor.GetNeighbors():
                    nbr_idx = nbr.GetIdx()
                    # Ignore ring members and self
                    if nbr_idx == atom_idx or nbr_idx in ring:
                        continue
                    # Abort if a non-hydrogen neighbor exists
                    if nbr.GetAtomicNum() != 1:
                        all_hydrogen = False
                        break

                if all_hydrogen:
                    candidates.add(neighbor_idx)

        # Merge qualifying atoms into the ring
        extended_ring.update(candidates)
        new_rings.append(frozenset(extended_ring))

    # Deduplicate results
    return [set(ring) for ring in list({ring for ring in new_rings})]

# Identify functional atoms
def mark_functional_groups(mol, rings):
    """
    Mark functional atoms (hetero atoms, multiple-bond carbons, acetal carbons,
    etc.) while excluding atoms that belong to rings.
    Returns a list of atom indices representing functional groups.
    """
    PATT = {
    'HETEROATOM': '[!#6]',                  # Matches non-carbon atoms
    'DOUBLE_TRIPLE_BOND': '*=,#*',        # Matches double (=) or triple (#) bonds
    # 'ACETAL': '[CX4]'                       # Initial SMARTS pattern for sp3 carbon
    }

    # Compile SMARTS patterns
    PATT = {k: Chem.MolFromSmarts(v) for k, v in PATT.items()}

    marks = []
    # Match in the order of PATT definitions
    for patt in PATT.values():
        for subs in mol.GetSubstructMatches(patt):
            subs = [sub for sub in subs if sub not in marks]
            for sub in subs :
                if sub not in rings:
                    marks.append(sub)

    # Hetero atom matching examples:
    # heteroatom_matches = mol.GetSubstructMatches(PATT['HETEROATOM'])
    # for match in heteroatom_matches:
    #     for atom_idx in match:
    #         if atom_idx not in rings:
    #             functional_atoms[atom_idx] = "Heteroatom"
    #
    # # Match carbons on multiple bonds
    # double_triple_matches = mol.GetSubstructMatches(PATT['DOUBLE_TRIPLE_BOND'])
    # for match in double_triple_matches:
    #     for atom_idx in match:
    #         if atom_idx not in rings:
    #             print("C:", atom_idx)
    #             functional_atoms[atom_idx] = "Multiple Bond Carbon"
    #
    for atom in mol.GetAtoms():
        if atom.GetIdx() in rings:  # Skip ring atoms
            continue
        elif atom.GetTotalNumHs() == 0 and len([n for n in atom.GetNeighbors() if n.GetAtomicNum() not in [6, 1]]) >= 2:
            if atom.GetIdx() not in marks:
                marks.append(atom.GetIdx())
    #
    # # Current idea: process one by one and sort in the end:
    #
    # # Assign priorities
    # """
    # (1) Hetero atoms highest priority
    # (2) Multiple-bond carbons next
    # (3) Acetal carbons afterwards
    # """
    # sorted_functional_atoms = dict(sorted(
    #     functional_atoms.items(),
    #     key=lambda x: (
    #         0 if x[1] == "Heteroatom" else  # Highest priority: hetero atoms
    #         1 if x[1] == "Multiple Bond Carbon" else  # Next: multiple-bond carbons
    #         2 if x[1] == "Acetal Carbon" else  # Then: acetal carbons
    #         3  # Other cases
    #     )
    # ))
    # print("sorted_functional_atoms:",sorted_functional_atoms)
    return marks

# Merge functional atoms with each other and with adjacent non-functional atoms
def merge_functional_groups(mol, marks, rings):
    """
    Merge functional groups and their neighboring carbon atoms into motifs,
    excluding ring atoms.
    Returns:
    - fgs: list of motifs, each a set of atom indices
    - adjacency_matrices: adjacency matrix for each motif
    """
    fgs = []  # Function Groups
    adjacency_matrices = []  # List of adjacency matrices

    node_visited = set()  # Tracks functional + regular atoms
    edge_visited = list()  # Tracks functional-functional merges

    # Initialize: each marked atom is a functional group
    atom2fg = [[] for _ in range(mol.GetNumAtoms())]  # atom2fg[i]: FG indices for atom i

    for atom in marks:  # init: each marked atom is its own FG
        fgs.append({atom})
        atom2fg[atom] = [len(fgs) - 1]
        node_visited.add(atom)

    # Handle functional-functional and functional-nonfunctional merges
    for atom_idx in marks:
        # Iterate over neighbors
        for neighbor in mol.GetAtomWithIdx(atom_idx).GetNeighbors():
            neighbor_idx = neighbor.GetIdx()

            # Skip ring atoms
            if neighbor_idx in rings:
                continue

            # Merge functional neighbors (functional + functional)
            if neighbor_idx in marks:
                if {atom_idx, neighbor_idx} not in edge_visited:
                    assert len(atom2fg[atom_idx]) == 1 and len(atom2fg[neighbor_idx]) == 1
                    # Merge neighbor_idx FG into atom_idx FG
                    fgs[atom2fg[atom_idx][0]].update(fgs[atom2fg[neighbor_idx][0]])     # Include neighbor FG
                    fgs[atom2fg[neighbor_idx][0]] = set()
                    atom2fg[neighbor_idx] = atom2fg[atom_idx]
                    edge_visited.append({atom_idx, neighbor_idx})

            # Merge functional atom with adjacent non-functional atoms
            else:
                if neighbor_idx not in node_visited:
                    fgs[atom2fg[atom_idx][0]].add(neighbor_idx)
                    atom2fg[neighbor_idx].extend(atom2fg[atom_idx])
                    node_visited.add(neighbor_idx)

    # Remove empty motifs
    tmp = []
    for fg in fgs:
        if len(fg) == 0:
            continue
        tmp.append(fg)
    fgs = tmp

    # Build adjacency matrices per motif
    for fg in fgs:
        fg_list = sorted(fg)  # Keep atom indices ordered
        size = len(fg_list)
        adj_matrix = np.zeros((size, size), dtype=int)  # Initialize adjacency matrix

        # Populate adjacency matrix
        for i, atom_idx in enumerate(fg_list):
            atom = mol.GetAtomWithIdx(atom_idx)
            for neighbor in atom.GetNeighbors():
                neighbor_idx = neighbor.GetIdx()
                if neighbor_idx in fg:
                    j = fg_list.index(neighbor_idx)
                    adj_matrix[i, j] = 1
                    adj_matrix[j, i] = 1  # Symmetric fill

        adjacency_matrices.append(adj_matrix)

    return fgs, adjacency_matrices

# Given a functional group and adjacency matrix, check connectivity after removing an atom
def is_connected_after_removal(fg, adjacency_matrix, removed_atom):
    """
    - True: remaining nodes stay connected after removing the atom.
    - False: removing the atom disconnects the nodes.
    """
    # Sort atoms and locate the index of the removed atom
    fg_list = sorted(fg)
    removed_index = fg_list.index(removed_atom)

    # Build a new adjacency matrix without the removed atom
    new_adjacency_matrix = np.delete(adjacency_matrix, removed_index, axis=0)  # Remove row
    new_adjacency_matrix = np.delete(new_adjacency_matrix, removed_index, axis=1)  # Remove column

    # Remaining node count
    remaining_nodes = len(fg_list) - 1

    # No nodes left -> disconnected
    if remaining_nodes == 0:
        return False

    # Use BFS to check connectivity
    visited = [False] * remaining_nodes
    queue = [0]
    visited[0] = True

    while queue:
        current_node = queue.pop(0)
        for neighbor in range(remaining_nodes):
            if new_adjacency_matrix[current_node][neighbor] == 1 and not visited[neighbor]:
                visited[neighbor] = True
                queue.append(neighbor)

    # Connected only if every node is visited
    return all(visited)

# Locate non-ring chains of carbon atoms with only single bonds
def find_non_ring_single_bond_only_carbon_chains_with_adjacency(mol):
    """Identify non-ring carbon chains where every bond is a single bond.

    Args:
        mol (rdkit.Chem.rdchem.Mol): RDKit molecule

    Returns:
        tuple: (chains, chains_adjacency)
            chains: list of carbon chains
            chains_adjacency: adjacency matrix for each chain
    """
    # 1. Collect all ring atoms
    rings = rdmolops.GetSSSR(mol)
    ring_atoms = set()
    for ring in rings:
        ring_atoms.update(ring)

    # 2. Filter carbon atoms not in rings whose bonds are all single
    carbon_atoms = []
    for atom in mol.GetAtoms():
        if (atom.GetAtomicNum() == 6 and
                atom.GetIdx() not in ring_atoms and
                all(bond.GetBondType() == Chem.BondType.SINGLE for bond in atom.GetBonds())):
            carbon_atoms.append(atom.GetIdx())

    # 3. Build adjacency list limited to single-bond carbon pairs
    adj = {i: [] for i in carbon_atoms}
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()

        # Only handle single-bond carbon pairs
        if (bond.GetBondType() == Chem.BondType.SINGLE and
                a1 in carbon_atoms and
                a2 in carbon_atoms):
            adj[a1].append(a2)
            adj[a2].append(a1)

    # 4. Run BFS to find connected components and build adjacency matrices
    visited = set()
    chains = []
    chains_adjacency = []

    for atom in carbon_atoms:
        if atom not in visited:
            queue = deque([atom])
            visited.add(atom)
            current_chain = []
            chain_adj = {}  # Track adjacency within this chain

            while queue:
                current = queue.popleft()
                current_chain.append(current)

                for neighbor in adj[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
                    # Record edges within the chain
                    chain_adj.setdefault(current, []).append(neighbor)
                    chain_adj.setdefault(neighbor, []).append(current)

            # Build adjacency matrix if the chain length >= 2
            if len(current_chain) >= 2:
                # Map atom ids to contiguous indices
                index_map = {idx: i for i, idx in enumerate(sorted(current_chain))}
                n = len(current_chain)
                adjacency_matrix = np.zeros((n, n), dtype=int)

                # Populate adjacency entries
                for node, neighbors in chain_adj.items():
                    if node in index_map:
                        i = index_map[node]
                        for neighbor in neighbors:
                            if neighbor in index_map:
                                j = index_map[neighbor]
                                adjacency_matrix[i, j] = 1
                                adjacency_matrix[j, i] = 1

                chains.append(sorted(current_chain))
                chains_adjacency.append(adjacency_matrix)

    return chains, chains_adjacency

# Adjust functional groups with respect to carbon chains
def reset_fgs_carbon(fgs, fgs_adjacency,carbon_chains, carbon_chains_adjacency, marks):
    for index_c,carbon_chain in enumerate(carbon_chains[:]):
        fg_set = get_unique(fgs)
        # Handle carbon chains of length 2
        if len(carbon_chain) == 2:
            C1 = carbon_chain[0]
            C2 = carbon_chain[1]

            # Skip chains containing functional carbons
            if C1 in marks or C2 in marks:
                carbon_chains.remove(carbon_chain)
                continue

            # Skip if both are already part of motifs
            if C1 in fg_set and C2 in fg_set:
                carbon_chains.remove(carbon_chain)
                continue
            # If only one atom belongs to a motif, remove it
            for i, fg in enumerate(fgs):
                if C1 in fg and C2 not in fg:
                    fgs[i].remove(C1)
                    break
                if C1 not in fg and C2 in fg:
                    fgs[i].remove(C2)
                    break

        # Chains longer than 2 atoms
        if len(carbon_chain) > 2:
            for carbon in carbon_chain[:]:
                # Skip functional carbons
                if carbon in marks:
                    carbon_chain.remove(carbon)
                    continue

                # If the carbon sits in a motif, evaluate removal
                for i, fg in enumerate(fgs):
                    # Remove from motif only if connectivity is preserved AND chain remains connected
                    if carbon in fg and is_connected_after_removal(fg, fgs_adjacency[i], carbon):
                        fgs[i].remove(carbon)
                        break
                    # Otherwise, remove from the chain if its connectivity stays intact
                    elif carbon in fg and not is_connected_after_removal(fg, fgs_adjacency[i], carbon):
                        if is_connected_after_removal(carbon_chain,carbon_chains_adjacency[index_c],carbon):
                            carbon_chain.remove(carbon)
                            break
    return fgs, carbon_chains


def get_unique(iterable):
    iterable_set = set()
    for iter in iterable:
        iterable_set.update(iter)
    return iterable_set

def remove_subsets_ring(rings):
    """
    Remove sets that are strict subsets of other sets in the list.

    Args:
        rings (list of set): collection of sets.

    Returns:
        list of set: sets with subsets removed.
    """
    # Result list
    result = []

    # Iterate over each set
    for current_set in rings:
        # Skip if it's a subset of another set
        if not any(current_set.issubset(other_set) and current_set != other_set for other_set in rings):
            result.append(current_set)

    return result

def mol_get_motif(mol):
    # Identify aromatic and non-aromatic rings
    aromatic_rings, non_aromatic_rings, svg = visualize_ring_aromaticity(mol)
    # print(aromatic_rings)
    # print(non_aromatic_rings)
    # Merge non-aromatic rings
    merged_non_aromatic_rings = merge_aromatic_rings(non_aromatic_rings)
    # print(merged_non_aromatic_rings)
    updated_non_aromatic_rings = merge_single_h_neighbors(mol, merged_non_aromatic_rings)
    rings = updated_non_aromatic_rings + [set(i) for i in aromatic_rings]
    rings = remove_subsets_ring(rings)
    rings_set = set()
    for ring in rings:
        rings_set.update(ring)

    # print("Rings:",rings)
    # 1. Mark functional atoms
    marks = mark_functional_groups(mol, rings_set)
    # print(marks)

    # 2. Merge functional groups and adjacent carbons
    fgs, fgs_adjacency = merge_functional_groups(mol, marks, rings_set)
    # print("Before adjustments:",fgs)


    # 3. Process pure carbon chains
    carbon_chains, carbon_chains_adjacency = find_non_ring_single_bond_only_carbon_chains_with_adjacency(mol)
    # print("Carbon chains:",carbon_chains)

    # 4. Update FGs based on carbon chains
    fgs, carbon_chains = reset_fgs_carbon(fgs, fgs_adjacency,carbon_chains, carbon_chains_adjacency, marks)
    carbon_chains = [set(carbon_chains) for carbon_chains in carbon_chains]
    # motifs = process_carbon_chains(mol, fgs, rings_set)

    # 4. Handle isolated carbon atoms
    motifs_result = rings + fgs + carbon_chains # (non-cycle 2, ring 1, chain 3)
    # print("motifs_result",motifs_result)

    # Type mapping (reference):
    # motifs_type_dict = {}
    # for i,ring in enumerate(rings):
    #     if i==0:
    #         motifs_type_dict[0] = list()
    #     motifs_type_dict[0].append(set(ring))
    #
    # for i,fg in enumerate(fgs):
    #     if i==0:
    #         motifs_type_dict[1] = list()
    #     motifs_type_dict[1].append(set(fg))
    #
    # for i,carbon_chain in enumerate(carbon_chains):
    #     if i==0:
    #         motifs_type_dict[2] = list()
    #     motifs_type_dict[2].append(set(carbon_chain))

    # Return motif types


    # Carbon atoms (methyl = 4)
    i=0
    motifs_list = get_unique(motifs_result)
    for atom in mol.GetAtoms():
        atom_id = atom.GetIdx()
        if atom_id not in motifs_list:
            motifs_result.append(set([atom_id]))



    motifs_type = list()
    for motif in motifs_result:
        if motif in rings:
            motifs_type.append(0)
        elif motif in fgs:
            motifs_type.append(1)
        elif motif in carbon_chains:
            motifs_type.append(2)
        else:
            motifs_type.append(3)

    motifs_result = [list(motif) for motif in motifs_result]

    return motifs_type,motifs_result


def get_motif_smiles(mol, motifs_result):

    motif_smiles_list = []

    for motif_indices in motifs_result:
        # Ensure indices are zero-based (RDKit uses zero-based indices)
        adjusted_indices = [idx - 1 if idx > 0 else idx for idx in motif_indices]  # Adjust if necessary

        try:
            # Generate fragment SMILES
            motif_smiles = Chem.MolFragmentToSmiles(mol, atomsToUse=adjusted_indices, isomericSmiles=True)
            motif_smiles_list.append(motif_smiles)
        except Exception as e:
            print(f"Error generating SMILES for motif {motif_indices}: {e}")
            motif_smiles_list.append(None)

    return motif_smiles_list


if __name__ == "__main__":
    # Load data
    with open('data/ZINC15/zinc15_250k.txt') as f:
        smiles_list = f.read().splitlines()[:1000]
    # datasets = MoleculeNet(root="../dataset/", name="Tox21")
    # dataset = MoleculeNet(root="../dataset/", name="BBBP")
    # smiles = 'Cc1occc1C(=O)Nc2ccccc2'
    # mol = Chem.MolFromSmiles(smiles)
    # motifs_type, motifs_result = mol_get_motif(mol)
    # print(motifs_type)
    # print(motifs_result)
    # motifs_type, motifs_result = mol_get_motif(mol)
    # data =
    # smiles = 'Cc1occc1C(=O)Nc2ccccc2'
    # mol = Chem.MolFromSmiles(smiles)
    # motifs_type, motifs_result = mol_get_motif(mol)
    # svg = visualize_motif(mol, motifs_result,method="save")
    #
    # with open(f'./Image/text.png', 'wb') as file:
    #     file.write(svg)
    #
    # print(motifs_result)

    for i in range(1000):
        mol = Chem.MolFromSmiles(smiles_list[i])
        motifs_type, motifs_result = mol_get_motif(mol)
        motif_smiles = get_motif_smiles(mol, motifs_result)

    # svg = visualize_motif(mol, motifs_result, method="save")
    # with open('./Image/new_motif/temp.png', 'wb') as file:
    #     file.write(svg)
    #     print("temp-save-complete")
    #
    # print(motifs_result,motif_smiles)
    # for i, smiles in enumerate(smiles_list):
    #
    #     mol = Chem.MolFromSmiles(smiles)
    #
    #     motifs_type,motifs_result = mol_get_motif(mol)
    #     svg = visualize_motif(mol, motifs_result,method="save")
    #
    #     # Save SVG locally
    #     with open(f'./Image/new_motif/{smiles}.png', 'wb') as file:
    #         file.write(svg)
    #         print(f"{smiles}-save-complete")

