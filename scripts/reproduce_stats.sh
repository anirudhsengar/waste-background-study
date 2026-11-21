#!/usr/bin/env bash
# CPU only, no datasets needed: regenerate every statistics table and Figures 2-3 from the shipped per-image
# predictions in results/preds/, then compare the regenerated tables with the reference versions shipped in
# results/ (copied aside before they are overwritten, so this also works without a git checkout).
#   PYTHON=.venv/bin/python bash scripts/reproduce_stats.sh   # interpreter (default: python on PATH)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
REF=$(mktemp -d)
trap 'rm -rf "$REF"' EXIT
cp results/*.csv results/*.json "$REF"/
$PY -m src.s09_analysis
$PY -m src.s12_contrasts
$PY -m src.s11_figures fig2 fig3
$PY scripts/compare_results.py --ref-dir "$REF"
