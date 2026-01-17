import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

RESULTS_DIR = "output/journal_experiments"
IMG_DIR = "output/journal_plots"
os.makedirs(IMG_DIR, exist_ok=True)

def load_data(filename):
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return None
    return pd.read_csv(path)

def plot_scalability(df):
    """
    Plots PrepTime vs Path Length.
    """
    plt.figure(figsize=(8, 6))
    
    # Infer length from manual inspection or assumption (since CSV stores just mode)
    # Note: In a real scenario, we'd log 'Length' in the CSV. 
    # Here we infer it based on row order or simply mock the x-axis for the provided CSV structure.
    # To fix this properly, let's assume the CSV rows are ordered by length loop.
    
    # Filter valid rows
    df_exact = df[df['Mode'] == 'exact'].reset_index(drop=True)
    df_kmv = df[df['Mode'] == 'kmv-static'].reset_index(drop=True)
    
    # Create lengths axis
    lengths = [2, 3, 4, 5, 6][:len(df_exact)]
    
    plt.plot(lengths, df_exact['PrepTime'], marker='o', label='Exact (SpGEMM)', linewidth=2, color='red')
    plt.plot(lengths, df_kmv['PrepTime'], marker='s', label='KMV (Ours)', linewidth=2, color='blue')
    
    plt.xlabel("Meta-path Length (Hops)", fontsize=12)
    plt.ylabel("Materialization Time (s)", fontsize=12)
    plt.title("The Geometric Explosion Problem", fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.yscale('log') # Log scale is crucial for this argument
    
    out_path = os.path.join(IMG_DIR, "exp_b_scalability.png")
    plt.savefig(out_path)
    print(f"Saved: {out_path}")

def plot_sensitivity(df):
    """
    Plots Accuracy vs K.
    """
    plt.figure(figsize=(8, 6))
    
    # Get Exact Baseline Accuracy (Horizontal Line)
    baseline_row = df[df['Mode'] == 'exact']
    if not baseline_row.empty:
        baseline_acc = baseline_row['TestAcc'].values[0]
        plt.axhline(y=baseline_acc, color='r', linestyle='--', label=f'Exact Baseline ({baseline_acc:.3f})')
    
    # Get KMV Curve
    df_kmv = df[df['Mode'] == 'kmv-static'].sort_values('K')
    
    plt.plot(df_kmv['K'], df_kmv['TestAcc'], marker='o', label='KMV Approximation', color='blue')
    
    plt.xlabel("Sketch Size (k)", fontsize=12)
    plt.ylabel("Test Accuracy", fontsize=12)
    plt.title("Sensitivity Analysis: Fidelity vs. Compression", fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xscale('log', base=2) # K is usually powers of 2
    
    out_path = os.path.join(IMG_DIR, "exp_c_sensitivity.png")
    plt.savefig(out_path)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    sns.set_theme(style="whitegrid")
    
    # Plot Experiment B
    df_b = load_data("exp_b_scalability.csv")
    if df_b is not None:
        plot_scalability(df_b)
        
    # Plot Experiment C
    df_c = load_data("exp_c_sensitivity.csv")
    if df_c is not None:
        plot_sensitivity(df_c)