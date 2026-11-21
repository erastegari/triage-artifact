#!/usr/bin/env python3
"""Preprocessing step: filter the large raw OULAD VLE interaction log to the two modules
used in this study (BBB, DDD), producing data/vle_bbb_ddd.csv.

Input : <DATA_DIR>/studentVle.csv   (from the official OULAD download)
Output: <DATA_DIR>/vle_bbb_ddd.csv  (used by repro.py)

Run this once after downloading OULAD and before any analysis script.
"""
import os, pandas as pd
from config import DATA_DIR

MODULES = ["BBB", "DDD"]
src = os.path.join(DATA_DIR, "studentVle.csv")
dst = os.path.join(DATA_DIR, "vle_bbb_ddd.csv")

if not os.path.exists(src):
    raise SystemExit(
        f"Could not find {src}.\n"
        "Download OULAD (https://analyse.kmi.open.ac.uk/open_dataset) and place its CSV files "
        f"in {DATA_DIR} (or set OULAD_DATA)."
    )

print(f"Reading {src} (this file is large; reading in chunks)...")
chunks = []
for ch in pd.read_csv(src, chunksize=1_000_000):
    chunks.append(ch[ch["code_module"].isin(MODULES)])
out = pd.concat(chunks, ignore_index=True)
out.to_csv(dst, index=False)
print(f"Wrote {dst}: {len(out):,} rows for modules {MODULES}.")
