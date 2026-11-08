"""Guess a manifest.csv from a LiverHccSeg folder.  ALWAYS open the CSV and fix it by hand:
the dataset's real file naming may differ from the guesses below.

python scripts/make_manifest.py --data_root /data/LiverHccSeg --out manifest.csv
"""
import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

PHASES = {"pre": "pre", "art": "art", "arterial": "art", "pv": "pv", "portal": "pv", "del": "del", "delayed": "del"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out", default="manifest.csv")
    args = ap.parse_args()
    root = Path(args.data_root)
    pats = defaultdict(dict)
    for p in sorted(root.rglob("*")):
        if not p.name.lower().endswith((".nii", ".nii.gz")):
            continue
        name = p.name.lower()
        pid = p.parent.name
        tokens = set(re.split(r"[^a-z0-9]+", name))
        rel = str(p.relative_to(root))
        if tokens & {"seg", "mask", "label", "segmentation", "tumor", "tumour", "hcc"}:
            pats[pid].setdefault("mask", rel)
        else:
            for tok, ph in PHASES.items():
                if tok in tokens:
                    pats[pid].setdefault(ph, rel)
    cols = ["patient_id", "pre", "art", "pv", "del", "mask"]
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for pid, d in pats.items():
            w.writerow([pid] + [d.get(c, "") for c in cols[1:]])
    print(f"wrote {len(pats)} patients to {args.out} - please inspect it before training")


if __name__ == "__main__":
    main()
