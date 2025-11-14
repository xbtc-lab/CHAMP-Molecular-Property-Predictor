import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
from umap.umap_ import UMAP
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import rdkit.Chem as Chem
import warnings
import os

def reg_visual_umap(X, y,argse,epoch):
    # Ensure inputs are NumPy arrays
    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()  # Convert tensors to NumPy
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()  # Convert tensors to NumPy
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()  # Convert tensors to NumPy

    warnings.filterwarnings("ignore", category=UserWarning, message="n_jobs value.*")
    # Reduce dimensionality with UMAP
    umap = UMAP(n_components=2, random_state=42)
    X_2d = umap.fit_transform(X)

    # Normalize targets to [-1, 1]
    y_normalized = (y - y.min()) / (y.max() - y.min()) * 2 - 1

    # Plot scatter
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_normalized, cmap='coolwarm', s=5, alpha=0.8)

    # Colorbar for mapping values
    cbar = plt.colorbar(scatter)
    cbar.set_label('Normalized Target Value', rotation=270, labelpad=15)

    # Titles and axis labels
    plt.title(f"Regression Task Visualization (UMAP)-{argse.dataset}:{epoch+1}")
    plt.xlabel("Dimension 1 (UMAP)")
    plt.ylabel("Dimension 2 (UMAP)")

    path = f"./Experiment/reg_Image/{argse.dataset}"
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(f"./Experiment/reg_Image/{argse.dataset}/{epoch}_{argse.use_head_interaction}_{argse.use_gating}_UMAP.png")
    # Display figure
    plt.show()

def reg_visual_pca(X, y,argse,epoch):
    # Ensure inputs are NumPy arrays
    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()  # Convert tensors to NumPy

    warnings.filterwarnings("ignore", category=UserWarning, message="n_jobs value.*")
    # Dimensionality reduction using PCA
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X)

    # Normalize targets to [-1, 1]
    y_normalized = (y - y.min()) / (y.max() - y.min()) * 2 - 1

    # Plot scatter
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_normalized, cmap='coolwarm', s=5, alpha=0.8)

    # Colorbar legend
    cbar = plt.colorbar(scatter)
    cbar.set_label('Normalized Target Value', rotation=270, labelpad=15)

    # Titles and axis labels
    plt.title(f"Regression Task Visualization (PCA)-{argse.dataset}:{epoch+1}")
    plt.xlabel("Dimension 1 (PCA)")
    plt.ylabel("Dimension 2 (PCA)")

    path = f"./Experiment/reg_Image/{argse.dataset}"
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(f"./Experiment/reg_Image/{argse.dataset}/{epoch}_{argse.use_head_interaction}_{argse.use_gating}_PCA.png")
    # Display figure
    plt.show()


def reg_visual_TSNE(X, y,argse,epoch):
    # Ensure inputs are NumPy arrays
    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()  # Convert tensors to NumPy

    warnings.filterwarnings("ignore", category=UserWarning, message="n_jobs value.*")
    # Dimensionality reduction via t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    X_2d = tsne.fit_transform(X)

    # Normalize targets to [-1, 1]
    y_normalized = (y - y.min()) / (y.max() - y.min()) * 2 - 1

    # Plot scatter
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_normalized, cmap='coolwarm', s=5, alpha=0.8)

    # Colorbar legend
    cbar = plt.colorbar(scatter)
    cbar.set_label('Normalized Target Value', rotation=270, labelpad=15)

    # Titles and axis labels
    plt.title(f"Regression Task Visualization (TSNE)-{argse.dataset}:{epoch+1}")
    plt.xlabel("Dimension 1 (TSNE)")
    plt.ylabel("Dimension 2 (TSNE)")

    path = f"./Experiment/reg_Image/{argse.dataset}"
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(f"./Experiment/reg_Image/{argse.dataset}/{epoch}_{argse.use_head_interaction}_{argse.use_gating}_TSNE.png")
    # Display figure
    plt.show()


def task_visual(X, Y, epoch):
    # Ensure inputs are NumPy arrays
    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()
    if isinstance(Y, torch.Tensor):
        Y = Y.cpu().numpy()
        Y = Y.flatten()

    umap = UMAP(n_components=2, random_state=42)
    embedding = umap.fit_transform(X)  # Reduce to 2D
    unique_labels = np.unique(Y)

    # Plot scatter
    plt.figure(figsize=(10, 8))

    # Define color map
    colors = plt.cm.get_cmap('tab10', len(unique_labels))

    for i, label in enumerate(unique_labels):
        # Filter points for current label
        mask = Y == label
        plt.scatter(
            embedding[mask, 0],  # x coordinate
            embedding[mask, 1],  # y coordinate
            label=f'Class {label}',
            color=colors(i),
            s=10,
            alpha=0.7
        )


    # Remove axes spines/ticks
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.xticks([])  # Remove x ticks
    plt.yticks([])  # Remove y ticks


    # Legend and labels
    plt.legend(title='Classes', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(f'UMAP Projection of Data{epoch+1}', fontsize=16)
    plt.xlabel('UMAP Dimension 1', fontsize=12)
    plt.ylabel('UMAP Dimension 2', fontsize=12)
    plt.tight_layout()

    # Display plot
    plt.show()


def find_substructure_indices(molecule_smiles, substructure_smiles_list):
    """
    Args:
        molecule_smiles (str): Molecule SMILES.
        substructure_smiles_list (list): Functional group SMILES strings.
    Returns:
        list: Matching substructure indices.
    """
    # Convert SMILES into RDKit molecule
    molecule = Chem.MolFromSmiles(molecule_smiles)
    if molecule is None:
        raise ValueError("Invalid molecule SMILES")

    # Collect matches
    matched_indices = []

    # Iterate over substructures
    for idx, substructure_smiles in enumerate(substructure_smiles_list):
        # Convert substructure SMILES
        substructure = Chem.MolFromSmiles(substructure_smiles)
        if substructure is None:
            raise ValueError(f"Invalid substructure SMILES at index {idx}: {substructure_smiles}")

        # Check for substructure matches
        if molecule.HasSubstructMatch(substructure):
            matched_indices.append(idx)

    return matched_indices


def plot_embeddings(data, method='tsne'):
    # Filter out samples without labels
    filtered_data = [(label, tensor) for label, tensor in data if label is not None]

    if not filtered_data:
        print("No valid data to plot.")
        return

    labels, tensors = zip(*filtered_data)
    labels = list(labels)
    tensor_matrix = torch.stack(tensors)  # shape: (N, D)

    # Reduce to 2D
    if method == 'pca':
        reduced = PCA(n_components=2).fit_transform(tensor_matrix.cpu().numpy())
    elif method == 'tsne':
        reduced = TSNE(n_components=2, random_state=42).fit_transform(tensor_matrix.cpu().numpy())
    elif method == 'umap':
        reduced = UMAP(n_components=2).fit_transform(tensor_matrix.cpu().numpy())
    else:
        raise ValueError("method must be 'pca' or 'tsne'")

    # Plot scatter
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab10', alpha=0.8)
    plt.colorbar(scatter, ticks=sorted(set(labels)))
    plt.title(f'2D Scatter Plot ({method.upper()})')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    with open("","rb") as file:
        pickle.load(file)
