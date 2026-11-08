# Liver tumour segmentation: PFWF -> CRes2Net -> MDA-SGT

Re-implementation of *"An advanced transformer framework for liver tumor segmentation using MRI images"*
(Sivanagaraju et al., Biomed. Signal Process. Control 107 (2025) 107808) on the LiverHccSeg MRI dataset,
written as a base for a B.Tech final-year project.

## STATUS - read this first
* `liverseg/preprocessing.py` (PFWF) and `liverseg/metrics.py` were run and pass `tests/test_numpy.py`.
* The PyTorch code (models, training, data loading) was written WITHOUT being executed (no torch / GPU / dataset
  in the authoring environment). Run `python tests/smoke_test.py` first; it exercises every model variant.
  If something breaks, fix it or paste the traceback to Claude.
* The data loader has never seen the real LiverHccSeg files. Verify folder layout, phase names and label values.

## Setup
```bash
pip install -r requirements.txt
python tests/test_numpy.py
python tests/smoke_test.py
```
Google Colab (free GPU): upload the zip, `!unzip`, `!pip install nibabel`, then run the commands below
(torch is preinstalled).

## Data
Download LiverHccSeg (Gross et al., Zenodo, 2023; derived from TCGA-LIHC). Then:
```bash
python scripts/make_manifest.py --data_root /path/LiverHccSeg --out manifest.csv
```
Open `manifest.csv` and fix it. Columns: `patient_id, pre, art, pv, del, mask`. Check which integer in the
mask is tumour vs liver and set `--tumor_labels` (default `2` is a GUESS).

## Train / evaluate
```bash
python train.py --data_root /path/LiverHccSeg --manifest manifest.csv --phases pv --out runs/full
python evaluate.py --run_dir runs/full --viz 5
bash scripts/run_ablations.sh /path/LiverHccSeg manifest.csv 0     # all ablations, same split
```
Use `--phases art,pv` for multi-phase input (2 channels). On CPU-only or small GPUs lower
`--img_size 128 --base 16 --embed_dim 192 --depth 2 --batch_size 8`.
Splits are PATIENT-wise k-fold (`--n_folds 5 --fold k`); report the mean over all folds in your thesis.

## Paper -> code map
| Paper | File |
|---|---|
| PFWF pre-processing (Table 2) | `liverseg/preprocessing.py` |
| CRes2Net: Res2Net block (Eq. 2) + SE block | `liverseg/models/blocks.py` |
| Patch embed + pos. embed + transformer layers (Eq. 3-7), TSA | `liverseg/models/attention.py` |
| GSA (Eq. 10-11), TSA/GSA fusion (Eq. 12) | `liverseg/models/attention.py` |
| Direction guidance (Eq. 8) | `liverseg/models/attention.py` |
| Multi-scale (dense) skip connections (Eq. 13-15), full model | `liverseg/models/mda_sgt.py` |
| Dice, IoU, VOE, specificity, HD95, ASSD | `liverseg/metrics.py` |

## Interpretation choices (the paper is under-specified)
* PFWF: Table 2 is ambiguous - see the docstring for how it was interpreted.
* Transformer: the paper gives no depth/heads; defaults are 4 layers, 8 heads, dim 384 (paper mentions 768).
  Applied at the 1/16 bottleneck with patch size 1.
* TSA/GSA fusion: `x + lam1*TSA + lam2*GSA`, lambdas start at 0 (as in the paper) with an identity path.
  If training is slow to use attention, try `--lambda_init 0.1`.
* Direction field is learned without explicit supervision. Loss is BCE+Dice (paper's "sparse categorical entropy"
  does not fit a binary mask). Tumour = binary foreground, 2D slices at training time, 3D metrics at test time.

## Problems in the paper to be aware of (do not copy its claims blindly)
* Reports Dice 97.28% but HD95 = 82 mm and VOE = 23.4% - inconsistent (IoU 94.7% implies VOE ~5%).
* Ablation table: Dice and DPC columns are identical; MAE/RMSE in text and conclusion differ.
* 560/140 split of 700 items looks slice-level => likely patient leakage, inflating scores.
* Contains text from another field ("spoofing cues"), and reference [5] is unrelated.
Expect a clean patient-wise result to be well below 97% Dice. That is normal.

## Ideas for your "new contribution"
1. Leakage-free evaluation: patient-wise 5-fold CV, compare with the paper's claims (the ablation runner does this).
2. Multi-phase input (`--phases pre,art,pv,del`) vs single phase.
3. 2.5D input (neighbouring slices as channels) or a 3D variant.
4. Better loss (Focal/Tversky/boundary loss), test-time augmentation (`--tta`), or post-processing (largest components).
5. Lightweight model (smaller embed_dim/depth) with a parameter/FLOP-vs-Dice comparison.
6. Uncertainty maps (MC-dropout) or cross-dataset testing (e.g. ATLAS MRI dataset).
