import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
RESULTS_DIR = "output/results"
PLOTS_DIR = "output/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# Clean, publication-ready styling
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

def load_and_clean_data(pattern):
    """
    Rigorously parses CSVs, fixing the K/Rule column shift and 
    coercing string flags ("VOID_OOM") into mathematically actionable NaNs.
    """
    files = glob.glob(os.path.join(RESULTS_DIR, pattern))
    if not files:
        return pd.DataFrame()
    
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    
    # 1. Realign shifted K and Rule columns
    if 'K' in df.columns:
        mask = df['K'].astype(str).str.contains('[a-zA-Z]', regex=True, na=False)
        if mask.any():
            temp = df.loc[mask, 'K']
            df.loc[mask, 'K'] = df.loc[mask, 'Rule']
            df.loc[mask, 'Rule'] = temp
            
        df['K'] = pd.to_numeric(df['K'], errors='coerce')

    # 2. Force numeric types for metrics
    metrics = ['Time', 'Accuracy', 'Fidelity', 'Speedup', 
               'Dirichlet_Exact', 'Dirichlet_KMV', 'MAD_Exact', 'MAD_KMV']
    for col in metrics:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    return df

def shorthand_rule(rule_str):
    """Compresses 'author_to_paper,paper_to_author' into 'A->P->A' for readability."""
    if pd.isna(rule_str): return "Unknown"
    tokens = str(rule_str).replace('rev_', '').replace('inverse_', '').split(',')
    nodes = []
    for i, t in enumerate(tokens):
        if '_to_' in t:
            src, dst = t.split('_to_')
            if i == 0: nodes.append(src[0].upper())
            nodes.append(dst[0].upper())
        else:
            nodes.append(t[:2].upper())
    return "->".join(nodes) if nodes else str(rule_str)

def plot_absolute_robustness(df, model_arch, metric='Time', y_label='Execution Time (s)', log_y=True):
    df_model = df[df['ModelArch'] == model_arch].copy()
    if df_model.empty: return

    datasets = sorted(df_model['Dataset'].unique())[:4]
    
    # 1. MASSIVE VERTICAL CANVAS (16x22)
    fig, axes = plt.subplots(2, 2, figsize=(16, 22)) 
    axes = axes.flatten()
    colors = sns.color_palette("tab10")

    for i, ds in enumerate(datasets):
        ax = axes[i]
        ds_df = df_model[df_model['Dataset'] == ds]
        rules = ds_df['Rule'].unique()
        
        y_max = 0
        oom_messages = []

        for j, rule in enumerate(rules):
            color = colors[j % len(colors)]
            rule_df = ds_df[ds_df['Rule'] == rule]
            rule_short = shorthand_rule(rule)
            
            exact_row = rule_df[rule_df['Method'] == 'Exact']
            kmv_df = rule_df[rule_df['Method'] == 'KMV'].sort_values('K')
            
            if not kmv_df.empty:
                ax.plot(kmv_df['K'], kmv_df[metric], marker='o', color=color, 
                        linewidth=2.5, label=f"KMV: {rule_short}")
                y_max = max(y_max, kmv_df[metric].max(skipna=True))
                
            has_exact = not exact_row.empty and not pd.isna(exact_row[metric].iloc[0])
            if has_exact:
                exact_val = exact_row[metric].iloc[0]
                ax.axhline(exact_val, linestyle='--', color=color, alpha=0.8, 
                           label=f"Exact: {rule_short}")
                y_max = max(y_max, exact_val)
            else:
                oom_messages.append(f"❌ Exact OOM: {rule_short}")

        ax.set_xlabel("Sketch Size (K)", fontsize=13)
        ax.set_ylabel(y_label, fontsize=13)
        ax.set_xscale('log', base=2)
        if log_y and y_max > 0:
            ax.set_yscale('log')
            
        ax.set_title(f"Dataset: {ds}", fontsize=18, fontweight='bold', pad=25)

        # 2. LEGEND (Bottom Left)
        ax.legend(fontsize=10, loc='upper left', bbox_to_anchor=(0.0, -0.15), 
                  ncol=1, frameon=False)

        # 3. OOM BOX (Bottom Right of the same axis)
        if oom_messages:
            oom_text = "System Failures:\n" + "\n".join(oom_messages)
            # Alignment: top-right of the text block is at (1.0, -0.15) relative to axes
            ax.text(1.0, -0.15, oom_text, transform=ax.transAxes, 
                    fontsize=10, color='darkred', fontweight='bold', 
                    ha='right', va='top', 
                    bbox=dict(facecolor='#ffe6e6', alpha=0.7, edgecolor='red', boxstyle='round,pad=0.5'))

    fig.suptitle(f"Metric: {metric} | Model Architecture: {model_arch}", fontsize=24, fontweight='bold', y=0.97)
    
    # 4. AGGRESSIVE SPACING
    # hspace=0.8 provides a huge gap for the side-by-side metadata to live in
    fig.subplots_adjust(top=0.92, bottom=0.12, hspace=0.8, wspace=0.3)
    
    out_file = os.path.join(PLOTS_DIR, f"robustness_{metric.lower()}_{model_arch}.png")
    plt.savefig(out_file, dpi=300, bbox_inches='tight') 
    plt.close()
    print(f"[Plot] Saved {out_file}")

def plot_depth_bias(df, model_arch, metric_exact, metric_kmv, title, filename):
    df_model = df[df['ModelArch'] == model_arch].copy()
    if df_model.empty: return

    datasets = sorted(df_model['Dataset'].unique())[:4]
    
    # Matching the 16x22 scale
    fig, axes = plt.subplots(2, 2, figsize=(16, 22))
    axes = axes.flatten()
    colors = sns.color_palette("Set1")

    for i, ds in enumerate(datasets):
        ax = axes[i]
        ds_df = df_model[df_model['Dataset'] == ds]
        rules = ds_df['Rule'].unique()
        
        oom_messages = []
        for j, rule in enumerate(rules):
            color = colors[j % len(colors)]
            rule_df = ds_df[ds_df['Rule'] == rule].sort_values('Depth')
            rule_short = shorthand_rule(rule)
            
            ax.plot(rule_df['Depth'], rule_df[metric_kmv], marker='s', color=color, 
                    linewidth=2, label=f"KMV: {rule_short}")
            
            if rule_df[metric_exact].notna().any():
                ax.plot(rule_df['Depth'], rule_df[metric_exact], marker='o', color=color, 
                        linestyle='--', linewidth=2, alpha=0.7, label=f"Exact: {rule_short}")
            
            if rule_df[metric_exact].isna().all():
                 oom_messages.append(f"❌ Exact OOM: {rule_short}")

        ax.set_xlabel("GNN Depth (L)", fontsize=13)
        ax.set_ylabel(title, fontsize=13)
        ax.set_title(f"Dataset: {ds}", fontsize=18, fontweight='bold', pad=25)

        # Legend Left / OOM Right
        ax.legend(fontsize=10, loc='upper left', bbox_to_anchor=(0.0, -0.15), 
                  ncol=1, frameon=False)

        if oom_messages:
            oom_text = "System Failures:\n" + "\n".join(oom_messages)
            ax.text(1.0, -0.15, oom_text, transform=ax.transAxes, 
                    fontsize=10, color='darkred', fontweight='bold', 
                    ha='right', va='top', 
                    bbox=dict(facecolor='#ffe6e6', alpha=0.7, edgecolor='red', boxstyle='round,pad=0.5'))

    fig.suptitle(f"{title} vs. Depth ({model_arch})", fontsize=24, fontweight='bold', y=0.97)
    fig.subplots_adjust(top=0.92, bottom=0.12, hspace=0.8, wspace=0.3)
    
    out_file = os.path.join(PLOTS_DIR, f"{filename}_{model_arch}.png")
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Saved {out_file}")

def main():
    print("--- Generating Axiomatic Evaluation Plots ---")
    
    df_rob = load_and_clean_data("robustness_*.csv")
    df_depth = load_and_clean_data("depth_bias_*.csv")
    
    for model in ['GCN', 'SAGE']:
        print(f"\nProcessing {model} Visualizations...")
        
        # 1. Robustness: Absolute Execution Time
        plot_absolute_robustness(df_rob, model, metric='Time', y_label='Execution Time (Seconds)', log_y=True)
        
        # 2. Robustness: Absolute Accuracy
        plot_absolute_robustness(df_rob, model, metric='Accuracy', y_label='Test Accuracy', log_y=False)
        
        # 3. Depth: Dirichlet Energy
        plot_depth_bias(df_depth, model, 'Dirichlet_Exact', 'Dirichlet_KMV', 
                        'Dirichlet Energy (Oversmoothing)', 'depth_dirichlet')
        
        # 4. Depth: Mean Average Distance
        plot_depth_bias(df_depth, model, 'MAD_Exact', 'MAD_KMV', 
                        'Mean Average Distance (Collapse)', 'depth_mad')

if __name__ == "__main__":
    main()