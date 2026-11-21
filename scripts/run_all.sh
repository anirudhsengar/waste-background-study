#!/usr/bin/env bash
# Full pipeline from the raw data, in order. Run scripts/download_data.sh first.
# The two judgement steps (near-duplicate verdicts, mask visual check) were made by an AI model (Claude) on
# label-hidden contact sheets and are recorded in configs/, so every stage runs unattended.
# Set PYTHON to choose the interpreter (default: python). The shipped neural predictions make run_gpu.sh skip
# every run; to retrain them too: rm results/preds/{cnn,resnet18_ft,mobilenet_v3,swin_t}_*.npz results/rq4.json
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
$PY -m src.s01_inventory              # counts, 224-frame cache
$PY -m src.s02_duplicates candidates  # hash + embedding candidates, review sheets
# judgement step (recorded, AI rater): data/work/dup_review/*.png -> configs/duplicate_review.csv
$PY -m src.s02_duplicates groups      # duplicate groups, label-conflict drops, agreement with Prayitno & Putra
$PY -m src.s03_split                  # frozen group-aware split (+ official split for s10)
$PY -m src.s04_masks u2net            # U2-Net saliency (slow on CPU)
$PY -m src.s04_masks evaluate         # reference IoU, Otsu baseline, eligibility rules
$PY -m src.s04_masks sheets           # blind contact sheets of eligible test masks
# judgement step (recorded, AI rater): visual check -> configs/mask_visual_check.csv
$PY -m src.s04_masks finalize         # eligibility.csv
$PY -m src.s05_variants               # 4-class counterfactuals and probes (U2-Net masks; supplementary)
$PY -m src.s05_variants6              # 6-class counterfactuals from human polygons, corner probe (primary)
$PY -m src.s07_svm                    # majority baseline, colour SVM, SIFT SVM (all regimes and probes)
PYTHON=$PY bash scripts/run_gpu.sh    # neural suite and duplicate-leakage check (CUDA GPU recommended)
$PY -m src.s09_analysis               # performance, reliance, probes, planned tests -> summary.json
$PY -m src.s12_contrasts              # regime contrasts, seed variability, sensitivity, sorting errors
$PY -m src.s11_figures                # paper Figures 1-3
