"""aggregate_kgrw_bench.py — Summarize kgrw_bench.csv across datasets.

Reads results/<DS>/kgrw_bench.csv for HGB_{DBLP,ACM,IMDB}, aggregates
mean±std across 5 seeds for each (L, method, k, w'), and emits:
  - per-dataset L=2 full w-range table (MPRW w=1..128 + KGRW k,w' grid)
  - crossover analysis: MPRW marginal cost curve
  - final comparison at matched density

Output: stdout + results/kgrw_bench_summary.md
"""
from __future__ import annotations

import csv
import io
import statistics
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASETS = ["HGB_DBLP", "HGB_ACM", "HGB_IMDB"]
L_TARGET = 2


def _load(ds: str) -> list[dict]:
    p = ROOT / "results" / ds / "kgrw_bench.csv"
    rows = []
    with open(p, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if str(row.get("L", "")) != str(L_TARGET):
                continue
            rows.append(row)
    return rows


def _f(s):
    try: return float(s) if s else float("nan")
    except ValueError: return float("nan")


def _norm(x: str) -> str:
    if not x: return ""
    try: return str(int(float(x)))
    except ValueError: return x


def _agg(rows: list[dict], method: str, k: str, w: str) -> dict | None:
    matches = [r for r in rows if r["Method"] == method
               and _norm(r["k"]) == _norm(k)
               and _norm(r["w_prime"]) == _norm(w)]
    if not matches: return None
    edges = [_f(r["Edge_Count"]) for r in matches]
    times = [_f(r["Mat_Time_s"]) for r in matches]
    f1s   = [_f(r["Macro_F1"]) for r in matches if r["Macro_F1"]]
    ckas  = [_f(r["CKA"]) for r in matches if r["CKA"]]
    pas   = [_f(r["Pred_Agreement"]) for r in matches if r["Pred_Agreement"]]
    def ms(v):
        if not v: return float("nan"), float("nan")
        return statistics.mean(v), (statistics.stdev(v) if len(v) > 1 else 0.0)
    return {
        "n": len(matches),
        "edges": ms(edges), "time": ms(times),
        "f1": ms(f1s), "cka": ms(ckas), "pa": ms(pas),
    }


def main() -> None:
    lines: list[str] = []
    def p(s=""):
        print(s); lines.append(s)

    p("# KGRW vs MPRW wide-w-sweep summary (L=2, 5 seeds)")
    p(f"Generated from results/<DS>/kgrw_bench.csv on {Path().cwd().name}")
    p()

    for ds in DATASETS:
        rows = _load(ds)
        p(f"## {ds}  (L={L_TARGET}, {len({r['Seed'] for r in rows})} seeds)")
        p()
        p("| Method | k | w' | Edges | Mat Time(s) | F1 | CKA | PA |")
        p("|---|---|---|---|---|---|---|---|")

        # MPRW row per w
        w_list = sorted({int(float(r["w_prime"])) for r in rows if r["Method"] == "MPRW" and r["w_prime"]})
        for w in w_list:
            a = _agg(rows, "MPRW", "", str(w))
            if not a: continue
            p(f"| MPRW |  | {w} | {a['edges'][0]:,.0f} | {a['time'][0]:.4f}±{a['time'][1]:.4f} | "
              f"{a['f1'][0]:.4f}±{a['f1'][1]:.4f} | {a['cka'][0]:.4f}±{a['cka'][1]:.4f} | "
              f"{a['pa'][0]:.4f}±{a['pa'][1]:.4f} |")

        # KGRW row per (k,w')
        k_list = sorted({int(float(r["k"])) for r in rows if r["Method"] == "KGRW" and r["k"]})
        for k in k_list:
            wp_list = sorted({int(float(r["w_prime"])) for r in rows if r["Method"] == "KGRW" and r["w_prime"] and int(float(r["k"])) == k})
            for wp in wp_list:
                a = _agg(rows, "KGRW", str(k), str(wp))
                if not a: continue
                p(f"| KGRW | {k} | {wp} | {a['edges'][0]:,.0f} | {a['time'][0]:.4f}±{a['time'][1]:.4f} | "
                  f"{a['f1'][0]:.4f}±{a['f1'][1]:.4f} | {a['cka'][0]:.4f}±{a['cka'][1]:.4f} | "
                  f"{a['pa'][0]:.4f}±{a['pa'][1]:.4f} |")
        p()

        # Crossover analysis: MPRW marginal cost curve (edges/ms) and saturation
        p(f"### {ds} — MPRW marginal behavior")
        prev_e = 0.0
        p("| w | Mean edges | dedges | Time(ms) | dedges/ms |")
        p("|---|---|---|---|---|")
        for w in w_list:
            a = _agg(rows, "MPRW", "", str(w))
            if not a: continue
            e, t = a["edges"][0], a["time"][0] * 1000
            de = e - prev_e
            ratio = de / t if t > 0 else float("nan")
            p(f"| {w} | {e:,.0f} | {de:+,.0f} | {t:.1f} | {ratio:.0f} |")
            prev_e = e
        p()

        # Match KGRW to MPRW at similar edge count
        p(f"### {ds} — matched-density comparison")
        p("For each MPRW w, find KGRW (k,w') with closest edge count. Compare quality.")
        p("| MPRW w | MPRW edges | MPRW CKA | Match KGRW | KGRW edges | KGRW CKA | dCKA (KGRW-MPRW) |")
        p("|---|---|---|---|---|---|---|")
        for w in w_list:
            a_m = _agg(rows, "MPRW", "", str(w))
            if not a_m: continue
            # Find closest KGRW config
            best_k, best_wp, best_diff, best_a = None, None, float("inf"), None
            for k in k_list:
                wp_list2 = sorted({int(float(r["w_prime"])) for r in rows if r["Method"] == "KGRW" and int(float(r["k"])) == k})
                for wp in wp_list2:
                    a_k = _agg(rows, "KGRW", str(k), str(wp))
                    if not a_k: continue
                    diff = abs(a_k["edges"][0] - a_m["edges"][0])
                    if diff < best_diff:
                        best_diff, best_k, best_wp, best_a = diff, k, wp, a_k
            if best_a is None: continue
            d_cka = best_a["cka"][0] - a_m["cka"][0]
            p(f"| {w} | {a_m['edges'][0]:,.0f} | {a_m['cka'][0]:.4f} | k={best_k} w'={best_wp} | "
              f"{best_a['edges'][0]:,.0f} | {best_a['cka'][0]:.4f} | {d_cka:+.4f} |")
        p()

    # Write markdown summary
    out = ROOT / "results" / "kgrw_bench_summary.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
