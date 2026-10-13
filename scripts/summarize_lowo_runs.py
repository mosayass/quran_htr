"""
scripts/summarize_lowo_runs.py
Reads reports/lowo_runs.jsonl and produces the complete Task 2.25 summary table
with mean +/- std across seeds, forgetting metrics, and the learning curve.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="reports/lowo_runs.jsonl")
    args = ap.parse_args()

    p = Path(args.runs) if Path(args.runs).is_absolute() else ROOT / args.runs
    if not p.exists():
        print(f"Error: {p} not found")
        return

    runs = []
    for line in open(p, encoding="utf-8"):
        if line.strip():
            try:
                runs.append(json.loads(line))
            except Exception:
                pass

    print(f"Loaded {len(runs)} runs from {p}\n")

    # Group runs by (fold, mix)
    grouped = defaultdict(list)
    lc_grouped = defaultdict(lambda: defaultdict(list))

    for r in runs:
        fold = r["fold"]
        mix = r["mix"]
        if fold == "CLC":
            lc_size = r.get("lc_size")
            lc_grouped[mix][lc_size].append(r)
        else:
            grouped[(fold, mix)].append(r)

    print("=========================================================================================")
    print("TASK 2.25: MULTI-SEED LOWO FINE-TUNING SUMMARY (Folds A, B, C across seeds)")
    print("=========================================================================================")
    print(f"{'Fold':5s} | {'Mix':12s} | {'Seeds':5s} | {'Pre CER':8s} | {'Overall CER (mean +/- std)':27s} | {'Synth Forgetting':18s} | {'KHATT Forgetting':18s}")
    print("-" * 115)

    for (fold, mix), rlist in sorted(grouped.items()):
        seeds = sorted(list({r["seed"] for r in rlist}))
        cers = [r["result"]["cer_overall"] for r in rlist]
        pre_cers = [r["step0_cer"] for r in rlist]
        mean_pre = np.mean(pre_cers)
        mean_cer = np.mean(cers)
        std_cer = np.std(cers) if len(cers) > 1 else 0.0

        synth_f = [r["forgetting"]["synth_test_cer"] for r in rlist if r.get("forgetting", {}).get("synth_test_cer") is not None]
        khatt_f = [r["forgetting"]["khatt_val_cer"] for r in rlist if r.get("forgetting", {}).get("khatt_val_cer") is not None]

        synth_str = f"{np.mean(synth_f):.2f}% +/- {np.std(synth_f):.2f}" if len(synth_f) > 1 else (f"{synth_f[0]:.2f}%" if synth_f else "N/A")
        khatt_str = f"{np.mean(khatt_f):.2f}% +/- {np.std(khatt_f):.2f}" if len(khatt_f) > 1 else (f"{khatt_f[0]:.2f}%" if khatt_f else "N/A")

        print(f"{fold:5s} | {mix:12s} | n={len(rlist):<3d} | {mean_pre:6.2f}%  | {mean_cer:6.2f}% +/- {std_cer:5.2f}%        | {synth_str:18s} | {khatt_str:18s}")

    print("\n=========================================================================================")
    print("TASK 2.25: FOLD C PERSONALIZATION LEARNING CURVE (w02)")
    print("=========================================================================================")
    print(f"{'Mix':12s} | {'N=10 CER':12s} | {'N=20 CER':12s} | {'N=40 CER':12s} | {'N=75 CER':12s} | Base CER (35.48%)")
    print("-" * 85)

    for mix, sdict in sorted(lc_grouped.items()):
        vals = []
        for n in [10, 20, 40, 75]:
            r_n = sdict.get(n, [])
            if r_n:
                c_n = [r["result"]["cer_overall"] for r in r_n]
                m_c = np.mean(c_n)
                s_c = np.std(c_n) if len(c_n) > 1 else 0.0
                vals.append(f"{m_c:.2f}%" if len(c_n) == 1 else f"{m_c:.1f}+/-{s_c:.1f}%")
            else:
                vals.append("N/A")
        print(f"{mix:12s} | {vals[0]:12s} | {vals[1]:12s} | {vals[2]:12s} | {vals[3]:12s} |")


if __name__ == "__main__":
    main()
