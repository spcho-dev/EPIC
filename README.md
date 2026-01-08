## EPIC: Event Prototyping via Information Constrained Graph Learning for personalized cancer driver gene prediction

**EPIC** is a novel deep learning framework for personalized cancer driver gene prediction. Unlike traditional node-centric approaches, EPIC redefines driver prediction as a metric learning task in an "event embedding space." It leverages a heterogeneous Graph Neural Network with an Information Constrained learning strategy to preserve patient-specific genomic contexts and prevent over-smoothing.

> **Paper**: *EPIC: Event Prototyping via Information Constrained graph learning for personalized cancer driver gene prediction* > **Authors**: Sang-Pil Cho and Young-Rae Cho


### Key Features

* **Event Prototyping**: Models mutation events by fusing patient and gene embeddings, scoring them based on distances to learnable "Driver" and "Passenger" prototypes.
* **Information Constrained Graph Learning**: Introduces **Variance** and **Diversity** constraints on information flows to prevent feature collapse in deep GNNs.
* **Heterogeneous Graph Integration**: Seamlessly integrates multi-omics data (Somatic Mutations, Gene Expression) and biological networks (PPI).
* **Personalized Prediction**: Prioritizes rare, patient-specific driver mutations that are often overlooked by population-based methods.


### Repository Structure

```bash
EPIC/
├── Data/                        # Input data directory (TCGA mutations, expression, PPI)
│   ├── BRCA/                    # Cancer-specific data
│   ├── HNSC/
│   ├── LUAD/
│   ├── PRAD/
│   └── STRING_ppi_edgelist.tsv
├── outputs/                     # Directory for prediction results and plots
├── trained_models/              # Directory for saving trained model weights
├── data_loader.py               # Data preprocessing and HeteroData construction
├── model.py                     # EPIC model architecture (GNN Encoder + LinkPredictor)
├── train.py                     # Main training script
├── predict.py                   # Inference script for generating patient-specific rankings
├── evaluation_cohort.py         # Cohort-level performance evaluation
├── evaluation_individual.py     # Individual-level (Personalized) performance evaluation
└── README.md
```


### Model Architecture
The framework consists of two main components implemented in `model.py`:

1. **EPIC Encoder (`EPIC` class)**
   - A **Heterogeneous Graph Neural Network** (GNN) backbone based on `GATv2Conv`.
   - Aggregates multi-omics context from the **PPI network** (Gene-Gene) and **Mutation bipartite graph** (Patient-Gene).
   - Implements an **Information Flow** mechanism that tracks residual feature updates (flow vectors) to enforce **Variance** and **Diversity** constraints, ensuring representation robustness against over-smoothing.

2. **LinkPredictor (`LinkPredictor` class)**
   - Implements the **Event Prototyping** strategy.
   - **Event Embedding**: Fuses the learned Patient and Gene embeddings into a unified latent vector representing the specific mutation event.
   - **Metric Learning**: Learns two global prototypes—**Driver Prototype** ($\mathbf{p}_{driver}$) and **Passenger Prototype** ($\mathbf{p}_{passenger}$).
   - **Scoring**: Calculates the priority score based on the relative Euclidean distance:
     $$Score = \|\mathbf{z} - \mathbf{p}_{passenger}\|^2 - \|\mathbf{z} - \mathbf{p}_{driver}\|^2$$



### Usage

#### 1. Data Preparation
Place your raw data files in the `Data/` directory. The project expects the following structure for each cancer type (e.g., `BRCA`):
* `pos-BRCA-genename.txt`: Ground truth driver genes.
* `HiSeqV2_common_samples_genes_sorted.tsv`: Gene expression matrix.
* `mc3_gene_level_common_samples_genes_sorted.tsv`: Somatic mutation matrix.
* `STRING_ppi_edgelist.tsv`: Global PPI network file.

#### 2. Training
Train the EPIC model. This script initializes the heterogeneous graph, applies the Information Constrained GNN encoder, and optimizes the Event Prototyping objective.

```bash
python train.py
```

#### 3. Prediction
Generate personalized driver gene rankings for each patient using the trained model.

```bash
python predict.py
```

#### 4. Evaluation
Evaluate the model's performance using Two-Track metrics (Cohort-level & Individual-level).

**Cohort-level Evaluation** (Population-wide oncogenic signal detection):

```bash
python evaluation_cohort.py
```

**Individual-level Evaluation** (Personalized clinical utility):

```bash
python evaluation_individual.py
```

* **Outputs**: CSV files containing Precision, Recall, and F1-scores, along with visualization plots in `outputs/{cancer_type}/result/`.

