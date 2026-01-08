import pandas as pd
import matplotlib.pyplot as plt
import os

def load_known_driver_genes(file_path):
    """
    Loads the set of known cancer driver genes from a text file.
    Assumes one gene name per line.
    """
    with open(file_path, 'r') as f:
        known_genes = set(gene.strip() for gene in f.readlines() if gene.strip())
    return known_genes

def load_priority_genes(file_path):
    """
    Loads the prioritized list of genes predicted by the model.
    The input file is expected to be a TSV/CSV with a 'Gene' column.
    """
    # Load using pandas to automatically handle headers and separators
    df = pd.read_csv(file_path, sep='\t')
    
    # Return the ranked list of genes
    return df['Gene'].tolist()

def calculate_metrics(priority_genes, known_driver_genes, top_n):
    """
    Calculates Precision, Recall, and F1-Score at various Top-N thresholds.

    Args:
        priority_genes (list): List of genes ranked by the model.
        known_driver_genes (set): Set of ground truth driver genes.
        top_n (int): The maximum rank (N) to evaluate up to.

    Returns:
        list: A list of [N, Precision, Recall, F1_Score] for each cutoff N.
    """
    results = []
    # Ensure we don't exceed the number of predicted genes
    max_n = min(top_n, len(priority_genes))
    
    for N in range(1, max_n + 1):
        # Select the top N predicted genes
        top_n_genes = set(priority_genes[:N])
        
        # Calculate True Positives (TP)
        tp = len(top_n_genes & known_driver_genes)
        
        # Calculate Metrics
        precision = tp / len(top_n_genes) if len(top_n_genes) > 0 else 0.0
        recall = tp / len(known_driver_genes) if len(known_driver_genes) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        results.append([N, precision, recall, f1])
    
    return results

def save_results(results, output_file):
    """Saves the calculated metrics to a CSV file."""
    df = pd.DataFrame(results, columns=["Top_N", "Precision", "Recall", "F1_Score"])
    df.to_csv(output_file, index=False)

def plot_metrics(results, cancer_name, save_path=None):
    """
    Plots the Precision, Recall, and F1-Score curves.
    If 'save_path' is provided, saves the plot to an image file.
    """
    df = pd.DataFrame(results, columns=["Top_N", "Precision", "Recall", "F1_Score"])
    
    plt.figure(figsize=(10, 6))
    plt.plot(df["Top_N"], df["Precision"], label="Precision", marker='o', linestyle='-')
    plt.plot(df["Top_N"], df["Recall"], label="Recall", marker='s', linestyle='-')
    plt.plot(df["Top_N"], df["F1_Score"], label="F1-Score", marker='^', linestyle='-')
    
    plt.xlabel("Top N Genes")
    plt.ylabel("Score")
    plt.title(f"Cohort-Level Performance Metrics: {cancer_name}")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save the plot image if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
    
    plt.tight_layout()
    plt.show()

# --- Main Execution ---

if __name__ == "__main__":
    # Configuration
    cancer_name = "BRCA"
    top_n = 150  # Evaluation Cutoff: BRCA 150, HNSC 100, LUAD 150, PRAD 40

    # Define File Paths
    # Inputs
    priority_genes_file = f"./outputs/{cancer_name}/result/priority.txt"
    known_genes_file = f"./Data/pos-{cancer_name}-genename.txt"
    
    # Outputs
    output_dir = f"./outputs/{cancer_name}/result"
    os.makedirs(output_dir, exist_ok=True)
    
    output_csv_file = os.path.join(output_dir, f"cohort_precision_recall_f1_scores_{cancer_name}.csv")
    output_plot_file = os.path.join(output_dir, f"cohort_precision_recall_f1_scores_{cancer_name}.png")

    print(f"--- evaluating Cohort-Level Performance for {cancer_name} ---")

    # 1. Load Data
    if not os.path.exists(known_genes_file) or not os.path.exists(priority_genes_file):
        raise FileNotFoundError(f"Input files not found. Check paths: {known_genes_file}, {priority_genes_file}")

    known_driver_genes = load_known_driver_genes(known_genes_file)
    priority_genes = load_priority_genes(priority_genes_file)
    
    print(f"Loaded {len(known_driver_genes)} known driver genes.")
    print(f"Loaded {len(priority_genes)} priority ranked genes.")

    # 2. Calculate Metrics
    results = calculate_metrics(priority_genes, known_driver_genes, top_n)

    # 3. Save Results (CSV)
    save_results(results, output_csv_file)
    print(f"Metrics saved to {output_csv_file}")

    # 4. Plot & Save Image
    plot_metrics(results, cancer_name, save_path=output_plot_file)