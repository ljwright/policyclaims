#!/usr/bin/env bash
# Classify the human-reviewed samples in table/ with Jev 1.13, benchmark speed,
# and build the accuracy report. Costs about $0.15 in OpenRouter credit.
# Usage: bash run_jev_validation.sh            (Python pipeline)
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
OUT=concordance/jev_outputs
W=${JEV_WORKERS:-8}

# 1. Development set: 400 abstracts stratified by study design (DeepSeek labels only).
$PY code/3b_run_jev_classification.py table/manual_review_by_design_50_each.csv --output $OUT/jev_design400.csv --workers $W

# 2. Gold-standard workbook (204 abstracts with human review), three repeated runs for test-retest.
for i in 1 2 3; do
  $PY code/3b_run_jev_classification.py table/gold_standard_30_march.xlsx --sheet in --output $OUT/jev_gold_standard_run$i.csv --workers $W
done

# 3. Blinded stratified sample of 400 abstracts (reviewers DB and EC).
$PY code/3b_run_jev_classification.py table/supp_stratified_sample_400_blinded_db.xlsx --output $OUT/jev_stratified400.csv --workers $W

# 4. Speed benchmark at several concurrency levels (100 abstracts each).
$PY code/11_jev_speed_benchmark.py --n 100 --workers 1 4 8 16

# 5. Accuracy report (also picks up the full-corpus run if data/json_files/filtered/all_abstracts_JEV.csv exists).
$PY code/12_jev_accuracy.py
