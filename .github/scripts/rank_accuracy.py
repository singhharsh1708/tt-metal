#!/usr/bin/env python3
"""Rank SFPU accuracy-sweep output by error, to surface numerical outliers.

The LLK accuracy harness already compares every sampled point against a torch
golden and records signed_error / rel_error / signed_ulp_error per element --
but it only sanity-asserts (finite, non-zero) and never gates on magnitude.
This reads what it wrote and ranks it, so the tail is visible.

Usage:  rank_accuracy.py <dir with {op}.parquet|.csv> [-n 40] [--md out.md]
"""
import argparse, glob, os, sys
import numpy as np
import pandas as pd

GROUP = ["op", "input_format", "output_format", "approx_mode", "fast_mode", "dest_acc"]

def load(d):
    files = sorted(glob.glob(os.path.join(d, "*.parquet"))) or \
            sorted(glob.glob(os.path.join(d, "*.csv")))
    if not files:
        sys.exit(f"no .parquet or .csv under {d}")
    frames = []
    for f in files:
        frames.append(pd.read_parquet(f) if f.endswith(".parquet") else pd.read_csv(f))
    return pd.concat(frames, ignore_index=True)

def summarise(df):
    rows = []
    for key, g in df.groupby(GROUP, dropna=False):
        ulp = pd.to_numeric(g.get("signed_ulp_error"), errors="coerce").to_numpy(dtype=float)
        rel = pd.to_numeric(g.get("rel_error"), errors="coerce").to_numpy(dtype=float)
        a_ulp = np.abs(ulp[np.isfinite(ulp)])
        a_rel = np.abs(rel[np.isfinite(rel)])
        fh = g.get("is_finite_hw"); fg = g.get("is_finite_golden")
        # A finiteness disagreement is a hard defect: one side is NaN/inf, the other is not.
        mismatch = int((fh.astype(bool) != fg.astype(bool)).sum()) if fh is not None and fg is not None else 0
        rows.append({
            **dict(zip(GROUP, key)),
            "n": len(g),
            "max_abs_ulp": float(a_ulp.max()) if a_ulp.size else np.nan,
            "p999_abs_ulp": float(np.percentile(a_ulp, 99.9)) if a_ulp.size else np.nan,
            "max_abs_rel": float(a_rel.max()) if a_rel.size else np.nan,
            "finite_mismatch": mismatch,
            # ULP is undefined for bfp/int outputs; count how much of the sweep it covered.
            "ulp_defined_frac": round(float(a_ulp.size) / max(len(g), 1), 3),
        })
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("-n", type=int, default=40)
    ap.add_argument("--md")
    a = ap.parse_args()

    s = summarise(load(a.dir))

    hard = s[s.finite_mismatch > 0].sort_values("finite_mismatch", ascending=False)
    worst = s.sort_values("max_abs_ulp", ascending=False, na_position="last").head(a.n)

    out = []
    out.append(f"Variants swept: {len(s)}  |  rows: {int(s.n.sum())}\n")
    if len(hard):
        out.append(f"\n## Finiteness disagreements ({len(hard)} variants)\n")
        out.append("HW is finite where the torch golden is not, or vice versa. Each is a hard defect.\n")
        out.append(hard.head(a.n).to_markdown(index=False))
    else:
        out.append("\n## Finiteness disagreements\n\nNone.\n")
    out.append(f"\n## Worst {a.n} variants by max |ULP| error\n")
    out.append(worst.to_markdown(index=False))
    text = "\n".join(out)
    print(text)
    if a.md:
        open(a.md, "w").write(text)

if __name__ == "__main__":
    main()
