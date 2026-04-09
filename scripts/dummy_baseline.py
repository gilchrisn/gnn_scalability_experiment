"""
Dummy classifier baselines for multi-label datasets (e.g. HGB_IMDB).

Strategies
----------
most_frequent  For each label, always predict the majority value in the
               training set (typically all-zeros for sparse genres).
prior          For each label, predict 1 with probability = training
               frequency (stochastic; averaged over --trials runs).
mlp            Feature-only MLP: learns from node features with NO graph
               topology. Isolates whether SAGE gains anything from H.

Usage
-----
    python scripts/dummy_baseline.py HGB_IMDB
    python scripts/dummy_baseline.py HGB_IMDB --strategies most_frequent prior mlp
    python scripts/dummy_baseline.py HGB_IMDB --trials 20
"""
from __future__ import annotations

import argparse
import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import f1_score

current_dir  = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.config import config
from src.data import DatasetFactory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load(dataset: str):
    cfg = config.get_dataset_config(dataset)
    g, info = DatasetFactory.get_data(cfg.source, cfg.dataset_name, cfg.target_node)
    labels     = info["labels"]          # [N, C] float32 multi-hot
    train_mask = info["masks"]["train"]
    test_mask  = info["masks"]["test"]
    features   = info["features"]        # [N, D]
    num_classes = info["num_classes"]

    assert labels.dim() == 2, (
        f"Expected multi-label [N, C], got shape {labels.shape}. "
        "Run on a single-label dataset? Adjust this script."
    )

    # Restrict to labeled rows (at least one active genre)
    labeled = labels.sum(dim=1) > 0
    tr = train_mask & labeled
    te = test_mask  & labeled

    print(f"Dataset : {dataset}")
    print(f"Nodes   : {labels.size(0)} total, {labeled.sum()} labeled")
    print(f"Train   : {tr.sum()} | Test: {te.sum()} | Classes: {num_classes}")

    # Per-class positive rate in training set
    freq = labels[tr].mean(dim=0)
    print(f"Train label freq (per class): {freq.tolist()}\n")

    return labels, features, tr, te, num_classes, freq


def _macro_f1(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    return f1_score(
        preds.long(), targets.long(),
        task="multilabel", num_labels=num_classes, average="macro",
    ).item()


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

def most_frequent(labels, freq, tr, te, num_classes, **_):
    """Always predict the majority binary value per label."""
    majority = (freq >= 0.5).long()           # shape [C]
    preds = majority.unsqueeze(0).expand(te.sum(), -1)  # [N_test, C]
    f1 = _macro_f1(preds, labels[te].long(), num_classes)
    print(f"[most_frequent]  macro-F1 = {f1:.4f}")
    return f1


def prior(labels, freq, tr, te, num_classes, trials=10, **_):
    """Sample predictions from training-set label frequencies."""
    results = []
    targets = labels[te].long()
    for _ in range(trials):
        preds = torch.bernoulli(freq.unsqueeze(0).expand(te.sum(), -1))
        results.append(_macro_f1(preds.long(), targets, num_classes))
    mean_f1 = sum(results) / len(results)
    print(f"[prior]          macro-F1 = {mean_f1:.4f}  (avg over {trials} trials)")
    return mean_f1


def mlp(labels, features, tr, te, num_classes, epochs=200, hidden=64, **_):
    """Feature-only MLP — no graph edges at all."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    labeled = labels.sum(dim=1) > 0
    x   = features.to(device)
    y   = labels.float().to(device)
    tr_ = tr.to(device)
    te_ = te.to(device)

    model = nn.Sequential(
        nn.Linear(features.size(1), hidden),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(hidden, num_classes),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    best_state, best_f1 = None, 0.0
    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        out  = model(x)
        loss = F.binary_cross_entropy_with_logits(out[tr_], y[tr_])
        loss.backward()
        opt.step()

        if ep % 20 == 0 or ep == epochs:
            model.eval()
            with torch.no_grad():
                out_e = model(x)
                preds = (out_e[te_] > 0).long()
                f1 = _macro_f1(preds.cpu(), y[te_].long().cpu(), num_classes)
            if f1 > best_f1:
                best_f1 = f1
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            if ep % 40 == 0:
                print(f"  [MLP ep {ep:3d}]  loss={loss.item():.4f}  test-F1={f1:.4f}")

    print(f"[mlp]            macro-F1 = {best_f1:.4f}  (best over training)")
    return best_f1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

STRATEGY_FNS = {
    "most_frequent": most_frequent,
    "prior":         prior,
    "mlp":           mlp,
}

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", help="e.g. HGB_IMDB")
    parser.add_argument("--strategies", nargs="+",
                        default=["most_frequent", "prior", "mlp"],
                        choices=list(STRATEGY_FNS),
                        help="Which baselines to run (default: all)")
    parser.add_argument("--trials", type=int, default=10,
                        help="Repetitions for the 'prior' strategy (default 10)")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Training epochs for the MLP baseline (default 200)")
    parser.add_argument("--hidden", type=int, default=64,
                        help="Hidden dim for the MLP (default 64, matches SAGE)")
    parser.add_argument("--master-csv", type=str, default=None,
                        help="Append rows to master_results.csv using the exp3 schema")
    args = parser.parse_args()

    labels, features, tr, te, num_classes, freq = _load(args.dataset)

    print("=" * 50)
    results = {}
    for name in args.strategies:
        results[name] = STRATEGY_FNS[name](
            labels=labels, features=features, freq=freq,
            tr=tr, te=te, num_classes=num_classes,
            trials=args.trials, epochs=args.epochs, hidden=args.hidden,
        )

    print("=" * 50)
    print("Summary")
    for name, f1 in results.items():
        print(f"  {name:<16} {f1:.4f}")

    if args.master_csv:
        import csv as _csv
        _MASTER_FIELDS = [
            "Dataset", "MetaPath", "L", "Method", "k_value", "Density_Matched_w",
            "Materialization_Time", "Inference_Time",
            "Mat_RAM_MB", "Inf_RAM_MB",
            "Edge_Count", "Graph_Density",
            "CKA_L1", "CKA_L2", "CKA_L3", "CKA_L4",
            "Pred_Similarity", "Macro_F1", "Dirichlet_Energy", "exact_status",
        ]
        path = args.master_csv
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        is_new = not os.path.exists(path) or os.path.getsize(path) == 0
        with open(path, "a", newline="", encoding="utf-8") as fh:
            w = _csv.DictWriter(fh, fieldnames=_MASTER_FIELDS)
            if is_new:
                w.writeheader()
            for name, f1 in results.items():
                row = {f: "" for f in _MASTER_FIELDS}
                row["Dataset"]   = args.dataset
                row["Method"]    = name
                row["Macro_F1"]  = round(f1, 6)
                w.writerow(row)
        print(f"\nResults written → {path}")


if __name__ == "__main__":
    main()
