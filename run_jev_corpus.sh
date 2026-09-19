#!/usr/bin/env bash
# Full-corpus Jev run, after code/1_fetch_abstracts.py has populated data/json_files/.
# Steps: filter records -> classify with Jev -> build the analysis dataset from the
# Jev labels -> accuracy report (Jev vs DeepSeek and human review, Table 1 replication).
# Costs about $3.30 in OpenRouter credit (~45,800 requests); resumable if interrupted.
# Usage: bash run_jev_corpus.sh [workers] [budget_usd]
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
W=${1:-16}
BUDGET=${2:-8}

$PY code/2_filter_records.py --dir data/json_files --log data/json_files/excluded_records.csv
$PY code/3b_run_jev_classification.py data/json_files/filtered/all_abstracts.json --workers "$W" --budget-usd "$BUDGET"
$PY code/4_build_analysis_dataset.py --label-col jev_policy_claim --labels-csv data/json_files/filtered/all_abstracts_JEV.csv
$PY code/12_jev_accuracy.py
