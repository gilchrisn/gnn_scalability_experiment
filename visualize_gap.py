"""
Visualization script for GNN Fidelity Gap analysis.
Plots Cosine Fidelity against Sketch Size (k).
"""
import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_fidelity_plot(csv_path: str, output_image: str):
    """
    Generates a high-resolution plot showing the convergence of KMV sketches.
    Adheres to formal research plotting standards.
    """
    if not os.path.exists(csv_path):
        print(f"[Error] CSV not found: {csv_path}")
        return

    # Data ingestion
    df = pd.read_csv(csv_path)
    df = df.sort_values('K')

    # Figure setup
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    # Fidelity Curve: Exact vs KMV
    ax.plot(df['K'], df['Fidelity_Cosine'], 
            marker='o', markersize=8, linewidth=2.5, 
            color='#2c3e50', label=r'Cosine Fidelity $\rho$')

    # Logarithmic scaling for the K-space (Base 2)
    ax.set_xscale('log', base=2)
    ax.set_xticks(df['K'])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    # Labels using LaTeX notation
    ax.set_xlabel(r'Sketch Size $k$ (Log Scale)', fontsize=14)
    ax.set_ylabel(r'Representation Fidelity $\rho$', fontsize=14)
    ax.set_title('GNN Fidelity Convergence: Exact vs. KMV Materialization', fontsize=16)
    
    # Visualizing the "Confidence Zone"
    ax.axhline(y=0.99, color='r', linestyle='--', alpha=0.3, label='99% Fidelity Threshold')
    ax.fill_between(df['K'], df['Fidelity_Cosine'], 1.0, color='#34495e', alpha=0.05)

    # Annotation of the saturation point
    last_val = df['Fidelity_Cosine'].iloc[-1]
    
    # Use raw strings and ensure $ encapsulates the symbols correctly
    ax.plot(df['K'], df['Fidelity_Cosine'], 
            marker='o', markersize=8, linewidth=2.5, 
            color='#2c3e50', label=r'Cosine Fidelity $\rho$')

    # Fixed Annotation Line
    ax.annotate(r'Saturation $\rho \approx ' + f'{last_val:.4f}' + r'$', 
                xy=(df['K'].iloc[-1], last_val), 
                xytext=(df['K'].iloc[-1]/4, last_val-0.02),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))

    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend(loc='lower right')
    plt.tight_layout()
    
    plt.savefig(output_image, dpi=300)
    print(f"[Success] Publication-quality plot saved to: {output_image}")

if __name__ == "__main__":
    target_csv = "output/results/gap_analysis_cpp.csv"
    target_img = "output/results/fidelity_convergence.png"
    generate_fidelity_plot(target_csv, target_img)