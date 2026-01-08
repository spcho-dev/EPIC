## EPIC: Event Prototyping via Information Constrained Graph Learning for personalized cancer driver gene prediction

**EPIC** is a novel deep learning framework for personalized cancer driver gene prediction. Unlike traditional node-centric approaches, EPIC redefines driver prediction as a metric learning task in an "event embedding space." It leverages a heterogeneous Graph Neural Network with an Information Constrained learning strategy to preserve patient-specific genomic contexts and prevent over-smoothing.

> **Paper**: *EPIC: Event Prototyping via Information Constrained graph learning for personalized cancer driver gene prediction* > **Authors**: Sang-Pil Cho and Young-Rae Cho


### Key Features

* **Heterogeneous Graph Integration**: Seamlessly integrates multi-omics data (Somatic Mutations, Gene Expression) and biological networks (PPI).
* **Event Prototyping**: Models mutation events by fusing patient and gene embeddings, scoring them based on distances to learnable "Driver" and "Passenger" prototypes.
* **Information Constrained Graph Learning**: Introduces **Variance** and **Diversity** constraints on information flows to prevent feature collapse in deep GNNs.
* **Personalized Prediction**: Prioritizes rare, patient-specific driver mutations that are often overlooked by population-based methods.


### Repository Structure

```bash
EPIC/
├── Data/                        # Input data directory (TCGA mutations, expression, PPI)
│   ├── xena_org_multiomics/     # Original raw data from UCSC Xena Browser
│   │   ├── BRCA/                # Original HiSeqV2 and mc3_gene_level files
│   │   ├── HNSC/
│   │   ├── LUAD/
│   │   └── PRAD/
│   ├── BRCA/                    # Preprocessed Cancer-specific data
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
   - **Metric Learning**: Learns two global prototypes—**Driver Prototype** and **Passenger Prototype**.
   - **Scoring**: Calculates the priority score based on the relative Euclidean distance. The score is defined as the difference between the distance to the Passenger Prototype and the distance to the Driver Prototype. A higher score indicates that the event is closer to the Driver Prototype and further from the Passenger Prototype.


### Usage

### 1. Data Preparation
The raw multi-omics data were obtained from the **[UCSC Xena Browser](https://xenabrowser.net/datapages/)**. The original files (Gene Expression: `HiSeqV2` and Somatic Mutation: `mc3_gene_level`) are stored in the `Data/xena_org_multiomics/` directory.

To ensure data consistency across modalities, we performed a rigorous preprocessing step. We intersected the samples and genes across the mutation and expression datasets, retaining only those entities present in both. Consequently, the final preprocessed dataset comprises a unified set of **18,616 genes** across all cancer types. The preprocessed, ready-to-use data are located in the respective cancer type folders (e.g., `Data/BRCA/`).

* **Original Source**: `Data/xena_org_multiomics/`
* **Model Input**: `Data/{Cancer_Type}/` (e.g., `Data/BRCA/HiSeqV2_common_samples_genes_sorted.tsv`)
* **Global Network**: `Data/STRING_ppi_edgelist.tsv`


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

