#!/usr/bin/env bash
# Honest ablation study: same patient-wise split + seed for every variant.
# usage: bash scripts/run_ablations.sh /path/to/LiverHccSeg manifest.csv [fold]
set -e
ROOT=$1; MAN=$2; FOLD=${3:-0}
COMMON="--data_root $ROOT --manifest $MAN --fold $FOLD"
python train.py $COMMON --out runs/full
python train.py $COMMON --out runs/no_pfwf        --no_pfwf
python train.py $COMMON --out runs/no_cres2net    --encoder plain
python train.py $COMMON --out runs/no_transformer --no_transformer --no_gsa --no_direction
python train.py $COMMON --out runs/no_gsa         --no_gsa
python train.py $COMMON --out runs/no_direction   --no_direction
python train.py $COMMON --out runs/unet_skip      --skip unet
python train.py $COMMON --out runs/baseline_unet  --no_pfwf --encoder plain --no_transformer --no_gsa --no_direction --skip unet
