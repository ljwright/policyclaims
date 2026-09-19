#!/usr/bin/env bash
# R version of run_jev_corpus.sh. Because the Python and R scripts share input and
# output formats, R/03 resumes from data/json_files/filtered/all_abstracts_JEV.csv
# if the Python run already produced it (no new API spend); otherwise it classifies
# from scratch (~$3.30). Outputs get the _R / _jev suffixes.
# Usage: bash run_jev_corpus_R.sh [workers] [budget_usd]
set -euo pipefail
cd "$(dirname "$0")"
W=${1:-16}
BUDGET=${2:-8}

# The R filter writes its own combined file so it can be checked against the Python one;
# classification then runs on all_abstracts.json (shared) so a Python run is reused.
Rscript R/02_filter_records.R --dir data/json_files --log data/json_files/excluded_records_R.csv --combined-name all_abstracts_R.json
python3 - <<'PY'
import json
a = [r["scopus_id"] for r in json.load(open("data/json_files/filtered/all_abstracts.json"))]
b = [r["scopus_id"] for r in json.load(open("data/json_files/filtered/all_abstracts_R.json"))]
print(f"filter check: python={len(a)} R={len(b)} identical order={a == b}")
PY
Rscript R/03_run_jev_classification.R data/json_files/filtered/all_abstracts.json --workers "$W" --budget-usd "$BUDGET"
Rscript R/04_build_analysis_dataset.R --label-col jev_policy_claim --labels-csv data/json_files/filtered/all_abstracts_JEV.csv \
        --analysis-csv data/analysis/analysis_dataset_jev_R.csv --minimal-export data/analysis/policy_claims_minimal_jev_R.csv
Rscript R/12_jev_accuracy.R --suffix _R
