import pandas as pd
import os
import sys

# Configuration
METAPATH_LENGTHS = [2, 3, 4]
OUTPUT_PREFIX = "benchmark_results"
LATEX_OUTPUT_FILE = "appendix_tables.tex"

def format_relative(val, baseline):
    """Calculates relative ratio and formats string."""
    if baseline <= 0 or val <= 0:
        return "N/A" # Handle failures (marked as -1 in benchmark script)
    
    ratio = val / baseline
    return f"{ratio:.3f}"

def generate_latex_for_length(length):
    filename = f"{OUTPUT_PREFIX}_length_{length}.csv"
    
    if not os.path.exists(filename):
        print(f"% File {filename} not found. Skipping.")
        return ""

    df = pd.read_csv(filename)
    
    # Filter out total failures if any
    # (Though we keep them to show N/A if needed)
    
    # Group by Dataset to find baselines
    datasets = df['Dataset'].unique()
    
    latex_lines = []
    latex_lines.append(r"\begin{table}[h!]")
    latex_lines.append(r"\centering")
    latex_lines.append(r"\caption{Relative Performance for Metapath Length " + str(length) + r" (Normalized to Exact Method)}")
    latex_lines.append(r"\label{tab:results_len_" + str(length) + r"}")
    latex_lines.append(r"\resizebox{\textwidth}{!}{%")
    latex_lines.append(r"\begin{tabular}{l l c c c}")
    latex_lines.append(r"\toprule")
    latex_lines.append(r"\textbf{Dataset} & \textbf{Method} & \textbf{K} & \textbf{Rel. Time} & \textbf{Rel. Edges} \\")
    latex_lines.append(r"\midrule")

    for dataset in datasets:
        ds_data = df[df['Dataset'] == dataset]
        
        # FIX: Escape underscores outside the f-string
        safe_dataset_name = dataset.replace('_', r'\_')
        
        # 1. Get Baseline (Exact)
        exact_row = ds_data[ds_data['Method'] == 'Exact']
        
        if exact_row.empty:
            base_time = -1
            base_edges = -1
            baseline_status = "Missing"
        else:
            base_time = exact_row.iloc[0]['Total_Mean']
            base_edges = exact_row.iloc[0]['Edges_Mean']
            
            # Check if Exact failed (OOM/Crash marked as -1)
            if base_time == -1:
                baseline_status = "Failed"
            else:
                baseline_status = "OK"

        # 2. Add Baseline Row
        if baseline_status == "OK":
            latex_lines.append(f"{safe_dataset_name} & Exact & - & 1.000 & 1.000 \\\\")
        elif baseline_status == "Failed":
             latex_lines.append(f"{safe_dataset_name} & Exact (OOM) & - & - & - \\\\")

        # 3. Add KMV Rows
        kmv_rows = ds_data[ds_data['Method'].str.startswith('KMV')]
        
        for _, row in kmv_rows.iterrows():
            k_val = row['K']
            
            # Handle KMV Failures
            if row['Total_Mean'] == -1:
                latex_lines.append(f" & KMV & {k_val} & OOM & OOM \\\\")
                continue

            # Calculate Relatives
            if baseline_status == "OK":
                rel_time = format_relative(row['Total_Mean'], base_time)
                rel_edges = format_relative(row['Edges_Mean'], base_edges)
            else:
                rel_time = "N/A"
                rel_edges = "N/A"
            
            latex_lines.append(f" & KMV & {k_val} & {rel_time} & {rel_edges} \\\\")
        
        # Add spacing between datasets
        latex_lines.append(r"\midrule")

    # Remove last midrule and add bottomrule
    if latex_lines[-1] == r"\midrule":
        latex_lines.pop()
        
    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}}")
    latex_lines.append(r"\end{table}")
    latex_lines.append("") # Empty line for spacing
    
    return "\n".join(latex_lines)

def main():
    print(f"Generating LaTeX tables from {OUTPUT_PREFIX}_length_X.csv files...")
    
    all_latex = []
    
    for length in METAPATH_LENGTHS:
        print(f"Processing Length {length}...")
        table_code = generate_latex_for_length(length)
        if table_code:
            all_latex.append(table_code)
            
    full_document = "\n".join(all_latex)
    
    # Print to console
    print("\n" + "="*40)
    print("COPY BELOW TO OVERLEAF")
    print("="*40 + "\n")
    print(full_document)
    print("\n" + "="*40)
    
    # Save to file
    with open(LATEX_OUTPUT_FILE, "w") as f:
        f.write(full_document)
    print(f"Also saved to {LATEX_OUTPUT_FILE}")

if __name__ == "__main__":
    main()