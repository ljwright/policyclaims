#!/usr/bin/env bash
# R version of run_jev_validation.sh: classify the human-reviewed samples with
# Jev 1.13 using the R pipeline, benchmark speed, and build the accuracy report.
# Outputs go to concordance/jev_outputs_R/ and table/*_R.* so they can be compared
# with the Python outputs. Costs about $0.15 in OpenRouter credit.
set -euo pipefail
cd "$(dirname "$0")"
OUT=concordance/jev_outputs_R
W=${JEV_WORKERS:-8}

Rscript R/03_run_jev_classification.R table/manual_review_by_design_50_each.csv --output $OUT/jev_design400.csv --workers $W
for i in 1 2 3; do
  Rscript R/03_run_jev_classification.R table/gold_standard_30_march.xlsx --sheet in --output $OUT/jev_gold_standard_run$i.csv --workers $W
done
Rscript R/03_run_jev_classification.R table/supp_stratified_sample_400_blinded_db.xlsx --output $OUT/jev_stratified400.csv --workers $W
Rscript R/11_jev_speed_benchmark.R --n 100 --workers 1,4,8,16
Rscript R/12_jev_accuracy.R --outputs-dir $OUT --suffix _R
