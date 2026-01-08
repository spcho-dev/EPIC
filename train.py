import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import os
import numpy as np

from data_loader import load_data
from model import EPIC, LinkPredictor 

# --- Hyperparameters & Configuration ---
CANCER_TYPE = 'BRCA'
DATA_DIR = Path('./Data')
MODEL_SAVE_DIR = Path('./trained_models')

# Model Architecture
EMBED_DIM = 64
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT_RATE = 0.3

# Training Settings
LEARNING_RATE = 1e-4
EPOCHS = 1000   # 500 epochs for LUSC, 1000 epochs setting for other carcinomas by default
WEIGHT_DECAY = 1e-5

# Ensure save directory exists
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)


# --- Custom Loss Functions ---

class FocalLoss(nn.Module):
    """
    Implements Focal Loss to address class imbalance between driver and passenger mutations.
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce_with_logits = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, pred_logits, true_labels):
        bce_loss = self.bce_with_logits(pred_logits, true_labels)
        prob = torch.sigmoid(pred_logits)
        
        # Calculate focal weights
        p_t = prob * true_labels + (1 - prob) * (1 - true_labels)
        alpha_t = self.alpha * true_labels + (1 - self.alpha) * (1 - true_labels)
        focal_weight = alpha_t * torch.pow((1 - p_t), self.gamma)
        
        return (focal_weight * bce_loss).mean()


class DynamicLossWeighter(nn.Module):
    """
    Implements uncertainty-based dynamic loss weighting (Multi-task learning).
    Learns to balance Classification Loss, Variance Loss, and Diversity Loss automatically.
    
    Formula: L_total = sum( 0.5 * exp(-log_var_i) * L_i + 0.5 * log_var_i )
    """
    def __init__(self, num_losses=3):
        super(DynamicLossWeighter, self).__init__()
        # Parameters representing log(sigma^2) for each loss component
        self.log_vars = nn.Parameter(torch.zeros(num_losses))

    def forward(self, loss_bce, loss_flow, loss_div):
        losses = [loss_bce, loss_flow, loss_div]
        total_loss = 0.0
        
        for i, loss_val in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            weighted_loss = 0.5 * precision * loss_val
            regularization = 0.5 * self.log_vars[i]
            total_loss += (weighted_loss + regularization)
            
        return total_loss

    def get_weights(self):
        """Returns the current learned weights for monitoring."""
        return 0.5 * torch.exp(-self.log_vars.detach())


def compute_iff_loss(flow_dicts):
    """
    Computes Information Flow constraints to prevent over-smoothing.
    1. Variance Loss: Ensures information update magnitude is consistent.
    2. Diversity Loss: Prevents information flow from collapsing into a single direction (mean field).
    """
    total_flow_loss = 0.0
    total_div_loss = 0.0
    
    if not flow_dicts: 
        return 0.0, 0.0
        
    for flow in flow_dicts:
        all_flow_vectors = torch.cat([flow['gene'], flow['patient']], dim=0)
        
        # Variance constraint
        flow_norms = torch.norm(all_flow_vectors, p=2, dim=-1)
        total_flow_loss += torch.var(flow_norms)
        
        # Diversity constraint (Cosine similarity to mean direction)
        flow_directions = F.normalize(all_flow_vectors, p=2, dim=-1)
        mean_direction = torch.mean(flow_directions, dim=0, keepdim=True)
        cosine_sim = F.cosine_similarity(flow_directions, mean_direction)
        total_div_loss += torch.mean(cosine_sim.pow(2))
        
    return total_flow_loss / len(flow_dicts), total_div_loss / len(flow_dicts)


# --- Main Training Flow ---

if __name__ == "__main__":
    # 1. Load Data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    data, *_ = load_data(CANCER_TYPE, DATA_DIR)
    data = data.to(device)
    
    print(f"Training on all {data['patient', 'mutates', 'gene'].edge_index.size(1)} mutation edges.")
    num_genes = data['gene'].num_nodes
    num_patients = data['patient'].num_nodes

    # 2. Initialize Models
    gnn = EPIC(
        num_genes=num_genes,
        num_patients=num_patients,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout_rate=DROPOUT_RATE
    ).to(device)

    predictor = LinkPredictor(
        hidden_dim=HIDDEN_DIM,
        dropout_rate=DROPOUT_RATE
    ).to(device)

    loss_weighter = DynamicLossWeighter(num_losses=3).to(device)

    # 3. Setup Optimizer (Includes model params and loss weights)
    parameters = list(gnn.parameters()) + list(predictor.parameters()) + list(loss_weighter.parameters())
    optimizer = optim.Adam(parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # 4. Setup Main Loss
    loss_fn_bce = FocalLoss(alpha=0.75, gamma=2.0) 
    print("Using Focal Loss (alpha=0.75, gamma=2.0) for the classification task.")

    # 5. Training Loop
    print("--- Starting Training (EPIC Framework) ---")
    
    true_labels = data['patient', 'mutates', 'gene'].edge_label
    pred_edge_index = data['patient', 'mutates', 'gene'].edge_index

    for epoch in range(1, EPOCHS + 1):
        gnn.train()
        predictor.train()
        loss_weighter.train()
        
        optimizer.zero_grad()
        
        # A. GNN Encoder Forward Pass
        # Returns node embeddings and information flow vectors
        x_dict, flow_dicts = gnn(data)
        
        # B. Link Predictor Forward Pass
        # Returns logits based on prototype distances
        pred_logits, _, _ = predictor(x_dict, pred_edge_index)
        pred_logits = pred_logits.squeeze(-1)

        # C. Compute Losses
        # C-1. Classification Loss (Focal)
        loss_bce = loss_fn_bce(pred_logits, true_labels)
        
        # C-2. Information Constraints (Variance & Diversity)
        loss_flow, loss_div = compute_iff_loss(flow_dicts)
        
        # C-3. Weighted Total Loss (Dynamic Uncertainty Weighting)
        loss = loss_weighter(loss_bce, loss_flow, loss_div)
        
        # D. Backpropagation
        loss.backward()
        optimizer.step()
        
        # E. Logging
        if epoch % 10 == 0 or epoch == 1:
            current_weights = loss_weighter.get_weights()
            w_bce, w_flow, w_div = current_weights[0], current_weights[1], current_weights[2]
            
            print(f"Epoch {epoch}/{EPOCHS} | Total Loss: {loss.item():.4f} "
                  f"(Focal: {loss_bce.item():.4f}, Flow: {loss_flow.item():.4f}, Div: {loss_div.item():.4f})")
            print(f"          Weights (BCE: {w_bce:.3f}, Flow: {w_flow:.3f}, Div: {w_div:.3f})")

    print("--- Training complete. ---")

    # 6. Save Models
    torch.save(gnn.state_dict(), MODEL_SAVE_DIR / f'epic_gnn_{CANCER_TYPE}.pth')
    torch.save(predictor.state_dict(), MODEL_SAVE_DIR / f'epic_predictor_{CANCER_TYPE}.pth')

    print(f"Final models saved to {MODEL_SAVE_DIR}")