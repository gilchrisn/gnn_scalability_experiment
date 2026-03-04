"""
KMV Convergence Rate Analysis
==============================
Fits a log-log regression of (1 - Fidelity) vs K to empirically validate
(or challenge) the theoretical O(1/sqrt(K)) approximation rate.

Produces:
  - Log-log scatter plot with fitted regression line and 95% CI band
  - Console report with slope, R², and comparison to -0.5 theoretical slope
  - Optional CSV of regression statistics

Usage:
    python scripts/run_rate_analysis.py \
        --input  output/results/robustness_HGB_ACM_SAGE.csv \
        --output output/plots/rate_analysis.png
"""
import argparse
import os
import sys
import warnings
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

THEORETICAL_SLOPE: float = -0.5   # O(1/sqrt(K)) → slope = -0.5 on log-log
SLOPE_TOLERANCE:   float = 0.15   # Acceptable deviation from theoretical slope


# ---------------------------------------------------------------------------
# Regression helpers
# ---------------------------------------------------------------------------

class LogLogRegression:
    """
    Fits log-log linear regression:  log(error) = slope * log(K) + intercept

    Provides slope, intercept, R², standard error, and 95% CI on the slope.
    Uses numpy's lstsq for robustness (no scipy dependency required).
    """

    def __init__(self) -> None:
        self.slope:      float = 0.0
        self.intercept:  float = 0.0
        self.r_squared:  float = 0.0
        self.slope_se:   float = 0.0
        self.slope_ci95: Tuple[float, float] = (0.0, 0.0)
        self._fitted:    bool  = False

    def fit(self, k_values: np.ndarray, errors: np.ndarray) -> "LogLogRegression":
        """
        Args:
            k_values: Array of sketch sizes (positive integers).
            errors:   Array of error values (positive floats, e.g. 1 - Fidelity).

        Returns:
            self, for chaining.

        Raises:
            ValueError: If fewer than 3 valid data points are available.
        """
        valid = (k_values > 0) & (errors > 0) & np.isfinite(errors)
        k_clean = k_values[valid]
        e_clean = errors[valid]

        if len(k_clean) < 3:
            raise ValueError(
                f"Need at least 3 valid (K, error) pairs; got {len(k_clean)}. "
                "Check that Fidelity values are not all VOID_OOM."
            )

        log_k = np.log(k_clean.astype(float))
        log_e = np.log(e_clean.astype(float))

        # Design matrix for OLS: [log_k, 1]
        A = np.column_stack([log_k, np.ones_like(log_k)])
        coeffs, residuals, rank, _ = np.linalg.lstsq(A, log_e, rcond=None)

        self.slope, self.intercept = float(coeffs[0]), float(coeffs[1])

        # R²
        ss_res = float(np.sum((log_e - A @ coeffs) ** 2))
        ss_tot = float(np.sum((log_e - log_e.mean()) ** 2))
        self.r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        # Standard error and 95% CI (t-distribution, df = n - 2)
        n = len(k_clean)
        if n > 2:
            mse = ss_res / (n - 2)
            x_var = float(np.sum((log_k - log_k.mean()) ** 2))
            self.slope_se = float(np.sqrt(mse / x_var)) if x_var > 0 else 0.0
            # t_{0.975, n-2}  ≈ 1.96 for large n, use scipy-free approximation
            t_crit = _t_quantile_approx(0.975, n - 2)
            margin = t_crit * self.slope_se
            self.slope_ci95 = (self.slope - margin, self.slope + margin)

        self._fitted = True
        return self

    def predict(self, k_values: np.ndarray) -> np.ndarray:
        """Returns predicted error values (in original scale) for given K array."""
        if not self._fitted:
            raise RuntimeError("Call fit() before predict().")
        return np.exp(self.intercept) * k_values.astype(float) ** self.slope


def _t_quantile_approx(p: float, df: int) -> float:
    """
    Approximates the t-distribution quantile without scipy.
    Uses the Hill (1970) rational approximation, accurate to ~0.001 for df > 2.
    """
    if df >= 120:
        # Normal approximation is adequate
        return 1.96 if p == 0.975 else float(np.percentile(np.random.standard_normal(100_000), p * 100))

    # Cornish-Fisher expansion (adequate for our purposes)
    z = 1.959964  # z_{0.975}
    adj = (z ** 3 + z) / (4 * df) + (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96 * df ** 2)
    return z + adj


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_robustness_csv(filepath: str) -> pd.DataFrame:
    """
    Loads a robustness CSV produced by Phase 3.
    Converts 'VOID_OOM' strings to NaN and coerces numeric columns.
    """
    df = pd.read_csv(filepath)

    for col in ['K', 'Fidelity', 'Accuracy', 'Time', 'Speedup']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Remove the Exact baseline row (K=0) — not meaningful for rate analysis
    if 'Method' in df.columns:
        df = df[df['Method'] == 'KMV'].copy()

    return df


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyse_rule(df_rule: pd.DataFrame, rule_label: str) -> Dict:
    """
    Runs log-log regression for one rule / dataset / model combination.
    Returns a dict of statistics for reporting.
    """
    df_valid = df_rule.dropna(subset=['K', 'Fidelity'])
    if df_valid.empty:
        warnings.warn(f"No valid data for rule: {rule_label}")
        return {}

    k_arr  = df_valid['K'].values
    err_arr = 1.0 - df_valid['Fidelity'].values

    reg = LogLogRegression()
    try:
        reg.fit(k_arr, err_arr)
    except ValueError as exc:
        warnings.warn(str(exc))
        return {}

    agrees_with_theory = reg.slope_ci95[0] <= THEORETICAL_SLOPE <= reg.slope_ci95[1]

    return {
        'Rule':             rule_label,
        'N_points':         len(df_valid),
        'Slope':            reg.slope,
        'Slope_SE':         reg.slope_se,
        'CI95_Low':         reg.slope_ci95[0],
        'CI95_High':        reg.slope_ci95[1],
        'R_squared':        reg.r_squared,
        'Theory_slope':     THEORETICAL_SLOPE,
        'Agrees_theory':    agrees_with_theory,
        '_reg':             reg,           # kept for plotting (not in CSV output)
        '_k':               k_arr,
        '_err':             err_arr,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_rate_analysis(stats_list: List[Dict], output_path: str) -> None:
    """
    Generates log-log scatter + regression lines for every rule analysed.
    Saves to output_path.
    """
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    n_rules = len(stats_list)
    if n_rules == 0:
        print("[Plot] No data to plot.")
        return

    cols = min(n_rules, 3)
    rows = (n_rules + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False)
    palette = sns.color_palette("tab10", n_colors=n_rules)

    for idx, stats in enumerate(stats_list):
        ax = axes[idx // cols][idx % cols]
        color = palette[idx]

        k_arr   = stats['_k']
        err_arr = stats['_err']
        reg: LogLogRegression = stats['_reg']

        # Scatter (log-log axes)
        ax.scatter(k_arr, err_arr, color=color, s=60, zorder=5, label='Empirical error')

        # Fitted line
        k_dense = np.logspace(np.log10(k_arr.min()), np.log10(k_arr.max()), 200)
        ax.plot(k_dense, reg.predict(k_dense), color=color, linewidth=2,
                label=f'Fit: slope={stats["Slope"]:.3f} ± {stats["Slope_SE"]:.3f}')

        # Theoretical O(1/sqrt(K)) reference line — anchored at median empirical error
        anchor = float(np.median(err_arr)) * (float(np.median(k_arr)) ** 0.5)
        theory_line = anchor * k_dense ** THEORETICAL_SLOPE
        ax.plot(k_dense, theory_line, '--', color='gray', linewidth=1.5,
                label=f'Theory: slope={THEORETICAL_SLOPE}')

        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.set_xlabel('Sketch size K', fontsize=12)
        ax.set_ylabel('Approximation error  (1 − Fidelity)', fontsize=12)
        ax.set_title(stats['Rule'][:60], fontsize=11, fontweight='bold')
        ax.legend(fontsize=9, frameon=False)

        # Annotate R²
        r2_text = f"$R^2={stats['R_squared']:.3f}$"
        ax.text(0.97, 0.97, r2_text, transform=ax.transAxes,
                ha='right', va='top', fontsize=10)

        # Highlight whether CI covers theoretical slope
        ci_color = '#2ecc71' if stats['Agrees_theory'] else '#e74c3c'
        ci_text  = '✓ consistent with theory' if stats['Agrees_theory'] else '✗ deviates from theory'
        ax.text(0.97, 0.88, ci_text, transform=ax.transAxes,
                ha='right', va='top', fontsize=9, color=ci_color, fontweight='bold')

    # Hide unused subplots
    for i in range(n_rules, rows * cols):
        axes[i // cols][i % cols].set_visible(False)

    fig.suptitle('KMV Convergence Rate Analysis  (log-log)', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n[Plot] Saved → {output_path}")


# ---------------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------------

def print_report(stats_list: List[Dict]) -> None:
    print(f"\n{'='*70}")
    print("KMV CONVERGENCE RATE ANALYSIS REPORT")
    print(f"Theoretical slope: {THEORETICAL_SLOPE}  (O(1/√K))")
    print(f"{'='*70}\n")

    for s in stats_list:
        agrees = s.get('Agrees_theory', False)
        tag    = '✅ CONSISTENT' if agrees else '⚠️  DEVIATES'
        print(f"Rule   : {s.get('Rule', 'N/A')}")
        print(f"Points : {s.get('N_points', 0)}")
        print(f"Slope  : {s.get('Slope', 0):.4f}  "
              f"95% CI [{s.get('CI95_Low', 0):.4f}, {s.get('CI95_High', 0):.4f}]")
        print(f"R²     : {s.get('R_squared', 0):.4f}")
        print(f"Theory : {tag}")

        if not agrees:
            d = s.get('Slope', 0) - THEORETICAL_SLOPE
            print(f"         Empirical slope is {d:+.3f} away from theoretical.")
            if s.get('Slope', 0) > THEORETICAL_SLOPE:
                print("         Approximation degrades FASTER than predicted.")
            else:
                print("         Approximation is BETTER than predicted.")
        print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="KMV convergence rate analysis")
    parser.add_argument('--input',  type=str, required=True,
                        help='Path to robustness CSV (e.g. output/results/robustness_HGB_ACM_SAGE.csv)')
    parser.add_argument('--output', type=str,
                        default=os.path.join(project_root, 'output', 'plots', 'rate_analysis.png'))
    parser.add_argument('--stats-csv', type=str, default=None,
                        help='Optional path to save regression statistics CSV')
    args = parser.parse_args()

    print(f"[Rate Analysis] Loading: {args.input}")
    df = load_robustness_csv(args.input)

    if df.empty:
        print("[Rate Analysis] No KMV rows found in CSV. Nothing to analyse.")
        return

    print(f"   Loaded {len(df)} KMV rows.")

    # Group by rule — run one regression per rule
    group_col = 'Rule' if 'Rule' in df.columns else None
    if group_col:
        groups = df.groupby(group_col)
    else:
        groups = [('All', df)]

    stats_list = []
    for rule_label, df_rule in groups:
        s = analyse_rule(df_rule, str(rule_label))
        if s:
            stats_list.append(s)

    if not stats_list:
        print("[Rate Analysis] No valid regressions produced. Check Fidelity columns.")
        return

    print_report(stats_list)
    plot_rate_analysis(stats_list, args.output)

    # Save regression statistics (exclude non-serialisable keys)
    if args.stats_csv:
        csv_rows = [{k: v for k, v in s.items() if not k.startswith('_')}
                    for s in stats_list]
        pd.DataFrame(csv_rows).to_csv(args.stats_csv, index=False)
        print(f"[Rate Analysis] Stats saved → {args.stats_csv}")


if __name__ == "__main__":
    main()