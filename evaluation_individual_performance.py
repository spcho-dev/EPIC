import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Configuration ---
# List of cancer cohorts to evaluate (must match directory names)
CANCER_LIST = ['BRCA', 'HNSC', 'LUAD', 'PRAD'] 

# Input file path patterns
BASE_PATH = "./outputs"
PREDICTION_FILE_TEMPLATE = "{cancer}/result/final_ranking.txt"
KNOWN_GENES_PATH = "./Data"
KNOWN_GENES_TEMPLATE = "pos-{cancer}-genename.txt"

# Output file path patterns
OUTPUT_FILE_TEMPLATE = "individual_avg_metrics_{cancer}.csv"
PLOT_FILE_TEMPLATE = "individual_avg_metrics_{cancer}.png"

MAX_TOP_N = 5  # Evaluation range (Top 1 to Top 5)

# --- 2. Helper Functions ---

def load_known_driver_genes(file_path):
    """
    Loads the set of known cancer driver genes (Ground Truth).
    Returns a set of gene names.
    """
    try:
        with open(file_path, 'r') as f:
            # Strip whitespace and newlines
            known_genes = set(gene.strip() for gene in f.readlines() if gene.strip())
        return known_genes
    except FileNotFoundError:
        print(f"  [Error] Ground truth file not found: {file_path}")
        return None

def calculate_individual_metrics(cancer_name, cdg_set, max_n):
    """
    Calculates individual-level performance metrics for a specific cancer type.
    
    This function implements the 'Personalized Evaluation' logic:
    1. Loads patient-specific gene rankings.
    2. For each patient, calculates Precision, Recall, and F1 at Top-N.
       - Recall is normalized by the number of driver genes ACTUALLY present in that patient's mutations.
    3. Averages the metrics across all patients in the cohort.

    Args:
        cancer_name (str): Cancer cohort name (e.g., 'BRCA').
        cdg_set (set): Set of ground truth driver genes.
        max_n (int): Maximum rank to evaluate (e.g., 5).

    Returns:
        list: Averaged metrics [N, Precision, Recall, F1, Patient_Count] for each N.
    """
    
    # 1. Load patient-specific predictions
    pred_file = os.path.join(BASE_PATH, PREDICTION_FILE_TEMPLATE.format(cancer=cancer_name))
    try:
        # Load whitespace-separated values
        pred_df = pd.read_csv(pred_file, sep=r'\s+')
    except FileNotFoundError:
        print(f"  [Error] Prediction file not found: {pred_file}")
        return None
    except pd.errors.EmptyDataError:
        print(f"  [Error] Prediction file is empty: {pred_file}")
        return None

    patient_ids = pred_df.columns
    print(f"  > {cancer_name}: Loaded predictions for {len(patient_ids)} patients.")

    # Dictionary to store metrics for all patients at each Top-N level
    # Structure: {1: {'precisions': [], 'recalls': [], ...}, 2: {...}}
    results_by_n = {n: {'precisions': [], 'recalls': [], 'f1s': []} for n in range(1, max_n + 1)}

    # 2. Iterate through each patient (Evaluate individually)
    for patient_id in patient_ids:
        # Get patient's full ranked list (remove NaNs)
        M_k_list = pred_df[patient_id].dropna().tolist()
        M_k_set = set(M_k_list)
        
        if not M_k_list:
            continue 

        # Define Personal Ground Truth (R_k)
        # R_k = (Patient's Mutations) INTERSECTION (All Known Drivers)
        R_k_set = M_k_set & cdg_set
        num_personal_drivers = len(R_k_set) 

        # 3. Calculate metrics for each Top-N cutoff
        for n in range(1, max_n + 1):
            if n > len(M_k_list):
                break 
            
            # Top-N predicted genes for this patient
            C_k_n_set = set(M_k_list[:n])
            
            # True Positives: Predicted genes that are known drivers
            TP = len(C_k_n_set & cdg_set)

            # Precision@N
            precision = TP / n
            
            # Recall@N (Personalized): TP / (Number of drivers in this patient)
            recall = TP / num_personal_drivers if num_personal_drivers > 0 else 0.0
            
            # F1-Score@N
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            # Store results
            results_by_n[n]['precisions'].append(precision)
            results_by_n[n]['recalls'].append(recall)
            results_by_n[n]['f1s'].append(f1)

    # 4. Average metrics across the cohort
    averaged_results = []
    for n in range(1, max_n + 1):
        p_list = results_by_n[n]['precisions']
        r_list = results_by_n[n]['recalls']
        f1_list = results_by_n[n]['f1s']
        
        if not p_list:
            break 
            
        avg_p = np.mean(p_list)
        avg_r = np.mean(r_list)
        avg_f1 = np.mean(f1_list)
        
        averaged_results.append([n, avg_p, avg_r, avg_f1, len(p_list)])
        
    return averaged_results

# --- 3. Main Execution ---
if __name__ == "__main__":
    print(f"Starting Individual-Level Performance Evaluation...")
    
    for cancer in CANCER_LIST:
        print(f"\n--- Processing {cancer} ---")
        OUTPUT_DIR = f"./outputs/{cancer}/result"
        
        # 1. Load Ground Truth
        known_genes_file = os.path.join(KNOWN_GENES_PATH, KNOWN_GENES_TEMPLATE.format(cancer=cancer))
        cdg_set = load_known_driver_genes(known_genes_file)
        
        if cdg_set is None or not cdg_set:
            print(f"  > Failed to load ground truth for {cancer}. Skipping.")
            continue
        
        print(f"  > {cancer}: Loaded {len(cdg_set)} known driver genes.")
        
        # 2. Calculate Personalized Metrics
        avg_metrics_data = calculate_individual_metrics(cancer, cdg_set, MAX_TOP_N)
        
        if avg_metrics_data is None:
            print(f"  > Error calculating metrics for {cancer}. Skipping.")
            continue

        # 3. Save Results to CSV
        output_df = pd.DataFrame(
            avg_metrics_data, 
            columns=["Top_N", "Precision", "Recall", "F1_Score", "Patient_Count"]
        )
        
        csv_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE_TEMPLATE.format(cancer=cancer))
        output_df.to_csv(csv_path, index=False, float_format='%.6f')
        
        print(f"  > [Success] Saved metrics CSV: {csv_path}")

        # 4. Plot Results
        plt.figure(figsize=(10, 6))

        x = output_df["Top_N"]

        plt.plot(x, output_df["Precision"], marker='o', label='Precision', linewidth=2)
        plt.plot(x, output_df["Recall"], marker='s', label='Recall', linewidth=2)
        plt.plot(x, output_df["F1_Score"], marker='^', label='F1-Score', linewidth=2)

        plt.title(f'Individual-Level Performance Metrics: {cancer}', fontsize=15, fontweight='bold')
        plt.xlabel('Top N', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.ylim(0, 1.05)
        plt.xticks(x)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)

        plot_path = os.path.join(OUTPUT_DIR, PLOT_FILE_TEMPLATE.format(cancer=cancer))
        plt.savefig(plot_path, dpi=300)
        print(f"  > [Success] Saved performance plot: {plot_path}")

        plt.tight_layout()
        plt.show()

    print("\n--- Complete all computational processing ---")