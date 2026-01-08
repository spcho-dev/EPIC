import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATv2Conv

class EPIC(nn.Module):
    """
    EPIC Graph Encoder: Information Constrained Graph Learning.
    
    This module implements a Heterogeneous Graph Neural Network (GNN) using GATv2Conv layers.
    It incorporates an 'Information Flow' mechanism (residual connections) to update node embeddings.
    
    Key Features:
    - Heterogeneous Message Passing: Handles 'gene-ppi-gene' and 'patient-mutates-gene' edges separately.
    - Information Flow (IFF): Explicitly returns the residual update vectors (flow_dicts) to be used
      for calculating Variance and Diversity constraints in the loss function.
    """
    def __init__(self, num_genes, num_patients, embed_dim, hidden_dim, num_layers=2, dropout_rate=0.3):
        super().__init__()
        
        # 1. Node Embedding Layers (Learnable Initial Features)
        self.gene_emb = nn.Embedding(num_genes, embed_dim)
        self.patient_emb = nn.Embedding(num_patients, embed_dim)
        
        # 2. Edge Feature Encoders (Linearly project scalar attributes to hidden dim)
        self.ppi_edge_encoder = nn.Linear(1, hidden_dim)  # PPI confidence score
        self.mut_edge_encoder = nn.Linear(1, hidden_dim)  # Gene expression value

        # 3. Input Projection (Embed Dim -> Hidden Dim)
        self.gene_in_proj = nn.Linear(embed_dim, hidden_dim)
        self.patient_in_proj = nn.Linear(embed_dim, hidden_dim)
        
        # 4. GNN Layers (HeteroConv with GATv2)
        self.convs = nn.ModuleList()
        self.dropout = nn.Dropout(dropout_rate)
        
        for _ in range(num_layers):
            self.convs.append(HeteroConv({
                ('gene', 'ppi', 'gene'): GATv2Conv(
                    (hidden_dim, hidden_dim), hidden_dim, edge_dim=hidden_dim, add_self_loops=False
                ),
                ('patient', 'mutates', 'gene'): GATv2Conv(
                    (hidden_dim, hidden_dim), hidden_dim, edge_dim=hidden_dim, add_self_loops=False
                ),
                ('gene', 'mutated_by', 'patient'): GATv2Conv(
                    (hidden_dim, hidden_dim), hidden_dim, edge_dim=hidden_dim, add_self_loops=False
                ),
            }, aggr='sum'))

    def forward(self, data):
        """
        Forward pass of the GNN encoder.

        Returns:
            x_dict (dict): Final node embeddings for 'gene' and 'patient'.
            flow_dicts (list): List of residual update vectors (information flows) for each layer,
                               used for computing Variance and Diversity losses.
        """
        # A. Initialize Node Features
        x_dict = {
            'gene': self.gene_emb(data['gene'].x),
            'patient': self.patient_emb(data['patient'].x)
        }
        
        # B. Encode Edge Attributes
        edge_attr_dict = {
            ('gene', 'ppi', 'gene'): self.ppi_edge_encoder(data['gene', 'ppi', 'gene'].edge_attr),
            ('patient', 'mutates', 'gene'): self.mut_edge_encoder(data['patient', 'mutates', 'gene'].edge_attr),
            ('gene', 'mutated_by', 'patient'): self.mut_edge_encoder(data['gene', 'mutated_by', 'patient'].edge_attr),
        }
        
        # C. Input Projection
        x_dict = {
            'gene': F.relu(self.gene_in_proj(x_dict['gene'])),
            'patient': F.relu(self.patient_in_proj(x_dict['patient']))
        }
        
        # D. Message Passing Layers with Information Flow Tracking
        flow_dicts = [] 
        for conv in self.convs:
            # 1. Compute Message (Flow)
            flow_dict = conv(x_dict, data.edge_index_dict, edge_attr_dict)
            flow_dict = {key: self.dropout(F.relu(x)) for key, x in flow_dict.items()}
            
            # 2. Update Node State (Residual Connection: h_new = h_old + flow)
            x_dict = {
                'gene': x_dict['gene'] + flow_dict['gene'],
                'patient': x_dict['patient'] + flow_dict['patient']
            }
            
            # 3. Track Flow for Loss Calculation
            flow_dicts.append(flow_dict)
            
        return x_dict, flow_dicts


class LinkPredictor(nn.Module):
    """
    Event Prototyping Head (Metric Learning).
    
    This module implements the "Event Prototyping" mechanism:
    1. Constructs 'Event Embeddings' by fusing Patient and Gene representations.
    2. Learns two latent prototypes: 'Driver Prototype' and 'Passenger Prototype'.
    3. Calculates priority scores based on the relative distance to these prototypes.
    """
    def __init__(self, hidden_dim, dropout_rate=0.3):
        super().__init__()
        
        # 1. Fusion Layer (Event Embedding Construction)
        # Concatenates Patient and Gene embeddings and projects to latent event space.
        self.fuser_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim) # Final Event Embedding z_uv
        )
        
        # 2. Learnable Prototypes
        # Initialize random vectors representing the ideal 'Driver' and 'Passenger' centers.
        self.driver_prototype = nn.Parameter(torch.randn(1, hidden_dim))
        self.passenger_prototype = nn.Parameter(torch.randn(1, hidden_dim))

    def forward(self, x_dict, edge_index):
        """
        Computes the priority score (logits) for each patient-gene mutation edge.

        Args:
            x_dict (dict): Dictionary of node embeddings from EPIC encoder.
            edge_index (Tensor): Indices of mutation edges to evaluate.

        Returns:
            pred_logits (Tensor): Score indicating likelihood of being a driver.
                                  Higher score => Closer to Driver Prototype.
            dist_driver (Tensor): Squared Euclidean distance to Driver Prototype.
            dist_passenger (Tensor): Squared Euclidean distance to Passenger Prototype.
        """
        # A. Extract Node Embeddings for given edges
        patient_emb = x_dict['patient'][edge_index[0]]
        gene_emb = x_dict['gene'][edge_index[1]]
        
        # B. Generate Event Embeddings (N, hidden_dim)
        combined_emb = torch.cat([patient_emb, gene_emb], dim=-1)
        event_embedding = self.fuser_mlp(combined_emb)
        
        # C. Compute Prototype Distances (Metric Learning)
        # Calculate squared Euclidean distance between each event and the prototypes.
        # Shape: (N, 1)
        dist_driver = torch.cdist(event_embedding.unsqueeze(1), self.driver_prototype.unsqueeze(0)).pow(2).squeeze(-1)
        dist_passenger = torch.cdist(event_embedding.unsqueeze(1), self.passenger_prototype.unsqueeze(0)).pow(2).squeeze(-1)

        # D. Compute Logits
        # Logic: We want Driver events to have SMALL dist_driver and LARGE dist_passenger.
        # Score = dist_passenger - dist_driver
        # High Score => Close to Driver, Far from Passenger.
        pred_logits = dist_passenger - dist_driver
        
        return pred_logits, dist_driver, dist_passenger