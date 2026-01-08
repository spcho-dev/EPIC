import torch
import pandas as pd
import numpy as np
from pathlib import Path
import os
from collections import defaultdict, Counter

from data_loader import load_data
from model import EPIC, LinkPredictor 

# --- Configuration & Hyperparameters ---
CANCER_TYPE = 'BRCA'
DATA_DIR = Path('./Data')
MODEL_SAVE_DIR = Path('./trained_models')
RESULT_SAVE_DIR = Path(f'./outputs/{CANCER_TYPE}/result')

# Model Architecture (Must match training config)
EMBED_DIM = 64
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT_RATE = 0.3

# Ensure output directory exists
os.makedirs(RESULT_SAVE_DIR, exist_ok=True)

# --- 1. Load Data and Pre-trained Models ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the same dataset structure used during training
data, mut_df, gene_map, patient_map, gene_map_inv = load_data(CANCER_TYPE, DATA_DIR)
data = data.to(device)

num_genes = data['gene'].num_nodes
num_patients = data['patient'].num_nodes

# Initialize Model Architecture
gnn = EPIC(
    num_genes, num_patients, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT_RATE
).to(device)

predictor = LinkPredictor(
    hidden_dim=HIDDEN_DIM, 
    dropout_rate=DROPOUT_RATE
).to(device)

# Load Pre-trained Weights
gnn_path = MODEL_SAVE_DIR / f'epic_gnn_{CANCER_TYPE}.pth'
pred_path = MODEL_SAVE_DIR / f'epic_predictor_{CANCER_TYPE}.pth'

if not gnn_path.exists() or not pred_path.exists():
    raise FileNotFoundError("Pre-trained model files not found. Please run train.py first.")

gnn.load_state_dict(torch.load(gnn_path))
predictor.load_state_dict(torch.load(pred_path))

gnn.eval()
predictor.eval()

print("Models loaded successfully.")

# --- 2. Generate Node Embeddings ---
# Perform a single forward pass to get the final learned embeddings for all patients and genes
with torch.no_grad():
    final_embeddings, _ = gnn(data) 

# --- 3. Generate Patient-Specific Rankings ---
print("--- 3. Generating patient-specific rankings ---")

patient_list = mut_df.columns.tolist()
results = defaultdict(list) 

for patient_name in patient_list:
    patient_idx = patient_map.get(patient_name)
    if patient_idx is None:
        continue
        
    # Get all mutated genes for this patient
    mutations_for_patient = mut_df[patient_name].values
    mutated_gene_indices_np = np.where(mutations_for_patient > 0)[0]
    
    if len(mutated_gene_indices_np) == 0:
        results[patient_name] = []
        continue

    # Prepare input tensors for the Link Predictor
    patient_idx_tensor = torch.tensor([patient_idx] * len(mutated_gene_indices_np), device=device)
    gene_idx_tensor = torch.tensor(mutated_gene_indices_np, device=device)
    
    pred_edge_index = torch.stack([patient_idx_tensor, gene_idx_tensor])

    # Predict scores for each mutation event
    with torch.no_grad():
        # Predictor returns (logits, dist_driver, dist_passenger)
        # We use logits (dist_passenger - dist_driver) as the priority score
        pred_output_tuple = predictor(final_embeddings, pred_edge_index)
        scores = pred_output_tuple[0].squeeze(-1)

    # Sort genes by score (High score = high likelihood of being a driver)
    sorted_indices = torch.argsort(scores, descending=True)
    
    # Map indices back to gene names
    sorted_gene_names = [
        gene_map_inv[mutated_gene_indices_np[i]] for i in sorted_indices.cpu().numpy()
    ]
    
    results[patient_name] = sorted_gene_names

print("Ranking generation complete.")

# --- 4. Aggregate Rankings (Population Level) ---
print("--- 4. Aggregating rankings (Condorcet method) ---")

# We use a Borda/Condorcet-like voting method to find consensus drivers across the cohort.
# Genes appearing in the top 50% of an individual's ranking receive a vote.
gene_votes = Counter()

for patient_name, sorted_genes in results.items():
    if not sorted_genes:
        continue 
    
    num_mutated = len(sorted_genes)
    cutoff_index = int(np.ceil(num_mutated * 0.5)) 
    ballot = sorted_genes[:cutoff_index]
    gene_votes.update(ballot)

# Save Aggregated Priority List
priority_list = gene_votes.most_common()
priority_df = pd.DataFrame(priority_list, columns=['Gene', 'Frequency'])
priority_output_file = RESULT_SAVE_DIR / 'priority.txt'
priority_df.to_csv(priority_output_file, sep='\t', index=False)

print(f"\n--- Priority file (Condorcet) saved to: ---")
print(priority_output_file)
print(priority_df.head(5))

# --- 5. Save Final Individual Rankings ---
# Save the full ranked list for each patient into a single file.
# Rows are gene ranks (1st, 2nd...), Columns are Patient IDs.
max_len = max(len(genes) for genes in results.values()) if results else 0
df_dict = {}

for patient_name, sorted_genes in results.items():
    # Pad lists with empty strings to match DataFrame dimensions
    padded_genes = sorted_genes + [''] * (max_len - len(sorted_genes))
    df_dict[patient_name] = padded_genes

final_df = pd.DataFrame(df_dict)
output_file = RESULT_SAVE_DIR / 'final_ranking.txt'
final_df.to_csv(output_file, sep='\t', index=False)

print(f"\n--- 5. Final ranking file saved to: ---")
print(output_file)
print(final_df.head(5))