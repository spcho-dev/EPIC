import pandas as pd
import numpy as np
import torch
from torch_geometric.data import HeteroData
from pathlib import Path

def load_data(cancer_type: str, data_dir: Path = Path('./Data')):
    """
    Loads multi-omics data into a PyTorch Geometric HeteroData object.

    This function performs the following steps:
    1. Loads ground truth driver genes for the specific cancer type.
    2. Loads mutation (binary) and gene expression (continuous) matrices.
    3. Loads the PPI network.
    4. Constructs a heterogeneous graph with:
       - Nodes: 'gene', 'patient'
       - Edges:
         - ('gene', 'ppi', 'gene'): Weighted by interaction confidence.
         - ('patient', 'mutates', 'gene'): Weighted by gene expression levels.
       - Labels: Assigns 1.0 if the mutated gene is a known driver, else 0.0.

    Args:
        cancer_type (str): The cancer cohort code (e.g., 'BRCA', 'LUAD').
        data_dir (Path): Path to the data directory.

    Returns:
        tuple: (HeteroData object, mutation DataFrame, gene_map, patient_map, gene_map_inv)
    """
    print(f"--- 1. Loading data for {cancer_type} ---")
    
    # --- Define file paths ---
    exp_file = data_dir / cancer_type / 'HiSeqV2_common_samples_genes_sorted.tsv'
    mut_file = data_dir / cancer_type / 'mc3_gene_level_common_samples_genes_sorted.tsv'
    ppi_file = data_dir / 'STRING_ppi_edgelist.tsv'
    
    # --- Load Ground Truth Labels ---
    label_file = data_dir / f'pos-{cancer_type}-genename.txt'
    if not label_file.exists():
        raise FileNotFoundError(f"Label file not found: {label_file}")

    driver_gene_list = pd.read_table(label_file, header=None, names=['name'])['name'].tolist()
    driver_gene_set = set(driver_gene_list)
    print(f"Loaded {len(driver_gene_set)} known driver genes from {label_file.name}.")

    # --- Load Raw Data ---
    exp_df = pd.read_table(exp_file, index_col=0)
    mut_df = pd.read_table(mut_file, index_col=0)
    ppi_df = pd.read_table(ppi_file)

    # --- Consistency Check & Mapping ---
    # Ensure gene and patient indices match between mutation and expression data
    genes = mut_df.index.tolist()
    patients = mut_df.columns.tolist()
    
    assert genes == exp_df.index.tolist(), "Gene indices do not match"
    assert patients == exp_df.columns.tolist(), "Patient columns do not match"

    # Create mappings: Name -> Index
    gene_map = {name: i for i, name in enumerate(genes)}
    patient_map = {name: i for i, name in enumerate(patients)}
    gene_map_inv = {i: name for name, i in gene_map.items()} # Index -> Name mapping for retrieval
    
    num_genes = len(genes)
    num_patients = len(patients)

    print(f"Found {num_genes} genes and {num_patients} patients.")

    # --- Initialize HeteroData Object ---
    data = HeteroData()
    data['gene'].num_nodes = num_genes
    data['patient'].num_nodes = num_patients
    
    # Initialize node features (using indices as placeholders)
    data['gene'].x = torch.arange(num_genes)
    data['patient'].x = torch.arange(num_patients)


    # --- 1. Construct PPI Edges (Gene-Gene) ---
    # Filter PPI interactions to include only genes present in the dataset
    ppi_df = ppi_df[ppi_df['partner1'].isin(gene_map) & ppi_df['partner2'].isin(gene_map)]
    
    src = [gene_map[g] for g in ppi_df['partner1']]
    dst = [gene_map[g] for g in ppi_df['partner2']]
    
    # Create undirected edges (source -> dest and dest -> source)
    edge_index_ppi = torch.tensor([src + dst, dst + src], dtype=torch.long)
    
    # Edge attribute: Interaction confidence (normalized)
    confidence = torch.tensor(ppi_df['confidence'].values / 1000.0, dtype=torch.float)
    edge_attr_ppi = torch.cat([confidence, confidence]).unsqueeze(-1)
    
    data['gene', 'ppi', 'gene'].edge_index = edge_index_ppi
    data['gene', 'ppi', 'gene'].edge_attr = edge_attr_ppi
    print(f"Added {edge_index_ppi.size(1)} PPI edges.")


    # --- 2. Construct Mutation Edges (Patient-Gene) ---
    # Identify non-zero mutations from the matrix
    mut_arr = mut_df.values
    exp_arr = exp_df.values
    
    gene_indices, patient_indices = np.where(mut_arr > 0)
    
    # Create edge indices (Patient -> Gene)
    edge_index_mut = torch.tensor([patient_indices, gene_indices], dtype=torch.long)
    
    # Edge attribute: Gene expression value for the specific patient-gene pair
    edge_attr_mut = torch.tensor(exp_arr[gene_indices, patient_indices], dtype=torch.float).unsqueeze(-1)
    
    # Generate Edge Labels (Ground Truth):
    # Iterate through all mutations and label as 1.0 if the gene is a known driver, else 0.0
    edge_labels = []
    for gene_idx in gene_indices:
        gene_name = gene_map_inv[gene_idx]
        is_driver = 1.0 if gene_name in driver_gene_set else 0.0
        edge_labels.append(is_driver)
        
    edge_label_mut = torch.tensor(edge_labels, dtype=torch.float)

    data['patient', 'mutates', 'gene'].edge_index = edge_index_mut
    data['patient', 'mutates', 'gene'].edge_attr = edge_attr_mut
    data['patient', 'mutates', 'gene'].edge_label = edge_label_mut 


    # --- 3. Construct Reverse Edges (Gene -> Patient) ---
    # Essential for message passing in both directions (Gene <-> Patient)
    data['gene', 'mutated_by', 'patient'].edge_index = torch.tensor([gene_indices, patient_indices], dtype=torch.long)
    data['gene', 'mutated_by', 'patient'].edge_attr = edge_attr_mut
    
    print(f"Added {edge_index_mut.size(1)} mutation edges (and reverse edges).")
    print(f"Positive driver labels: {int(edge_label_mut.sum())} / Total mutations: {len(edge_label_mut)}")
    print("--- Data loading complete. ---")
    
    return data, mut_df, gene_map, patient_map, gene_map_inv

if __name__ == '__main__':
    data, *rest = load_data('BRCA')
    print("\n--- HeteroData Object Summary ---")
    print(data)
    print("\nMetadata:", data.metadata())