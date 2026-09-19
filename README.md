
## Overview

The paper analyzes 45,807 abstracts from ten Epidemiology and Public Health journals and classifies whether the abstract contains a policy claim. 
The objective is descriptive. The study quantifies trends in the prevalence of policy claims by time, country, journal, field, and study design, with classification performed using a large language model plus human validation.

Full paper in American Journal of Epidemiology: https://academic.oup.com/aje/advance-article/doi/10.1093/aje/kwag196/8761610

---

## Methodology

### Corpus construction

- Journals: Ten established epidemiology and public health journals that publish original empirical research. The journal list extended prior manual evaluations and was finalized after discussion among the authors.
- Time window: 1990 to 2024, spanning periods before and during the rise of the policy impact agenda.
- Source and fields: Abstracts and metadata were retrieved through the Scopus API. Retrieved fields included publication year, keywords, citation counts, and corresponding author country.
- Inclusion criteria: Records classified as research articles. Additional filtering removed non-empirical content such as systematic reviews and commentaries.

### Classification of policy claims

- Definition: A policy claim is a concluding abstract statement that calls for policy attention or action, ranging from explicit recommendations to broader implications for policy.
- Model: DeepSeek V3.1 was run at low temperature to improve determinism. Prompts were designed to identify both explicit and implicit policy recommendations.
- Aim: The classification was used to map policy claims at scale for descriptive purposes. The study does not assess the validity of individual claims.

### Analytic outputs

- Primary measures: Prevalence of policy claims by year, country, journal, keywords/topics, and study design.
- Deliverables: Summary tables and figures for the manuscript and supplementary materials.

### Use of AI coding assistants
Multiple LLM coding assistants were used to support code drafting, refactoring, debugging, and code review/checking. These included Codex (GPT-5.5) and Claude Code (Claude Sonnet 4.6 and Claude Opus 4.7). The authors reviewed and retained responsibility for the analysis code, outputs, and interpretation.

## Data availability
Due to licensing restrictions, the full set of Scopus abstracts cannot be shared; not all publishers enable free sharing of abstracts, see https://i4oa.org.  

The shareable derived dataset is provided in `derived_data/`. It contains publicly available bibliographic metadata (DOI, title, journal, publication year, keywords, and corresponding author country) together with large language model classifications, but excludes full abstracts. The private `data/` directory contains licensed Scopus source files and intermediate analysis files with abstracts, and is not intended for redistribution. Researchers with Scopus access can reproduce the complete corpus using the included identifiers and code.

## Cost and time to process
The cost and time to process such a large number of abstracts are dependent on the LLM compute / API costs; for the Deepseek API, for example, the analysis incurred ~$3 and ~10 hours of processing time. Since Deepseek is open-weight, this or other open-weight models can be run on local hardware with sufficiently high RAM. 

---

## Jev 1.13 replication (Python and R)

This branch re-runs the policy-claim classification with **Jev 1.13** (TypeSafe AI, via the
[OpenRouter Decisions API](https://openrouter.ai/typesafe/jev-1.13)) to test how fast and how accurate
it is compared with the DeepSeek V3.1 labels used in the paper, and with the human reviews stored in
`table/`. Everything below runs from the repository root. Nothing in the original pipeline was changed
except that `code/4_build_analysis_dataset.py` gained a `--label-col` option.

### How Jev differs from a chat model

Jev is a "System One" decision model: it does not generate text, so there is no prompt and no JSON to
parse. A request sends a `state` (the abstract) plus typed *questions* and gets back calibrated
probabilities. It cannot be used through `/chat/completions`; OpenRouter routes it via
`POST https://openrouter.ai/api/alpha/decisions`. Pricing is per input token only ($0.042 per million;
about $0.00007 per abstract with the two questions below).

The questions live in one file, **`code/jev_questions.json`**, loaded by both `code/jev_client.py` and
`R/jev_client.R` (each verifies the file's hash, so the two implementations cannot drift apart):

* `policy_claim` (Noul, yes/no probability) - the primary label: `P(policy claim) >= 0.5`.
* `policy_claim_choice` (yes/no Choice) - the same judgement as a choice, with a confidence value.

The wording mirrors the DeepSeek prompt in `code/3_run_llm_classification.py` (definition, inclusion and
exclusion rules, the same five worked examples). It was checked once on the 400-abstract study-design
sample (DeepSeek labels only) and then frozen before the human-reviewed samples were scored; the 0.5
threshold was fixed in advance.

### Setup

```bash
cp .env.example .env        # add OPENROUTER_API_KEY (and SCOPUS_API_KEY for step 1)
pip install pandas numpy scipy scikit-learn requests python-dotenv tqdm openpyxl matplotlib
Rscript -e 'install.packages(c("tidyverse","httr2","jsonlite","readxl","here","digest"))'
```

The repository's `.env` takes precedence over an `OPENROUTER_API_KEY` in the environment or `~/.Renviron`.

### Run

Validation on the human-reviewed samples (~1,400 requests, about $0.15, ~3 minutes):

```bash
bash run_jev_validation.sh      # Python -> concordance/jev_outputs/, table/jev_*.csv|md
bash run_jev_validation_R.sh    # R      -> concordance/jev_outputs_R/, table/jev_*_R.csv|md
```

Full corpus (needs the Scopus abstracts, which cannot be redistributed; ~45,800 requests, about $3.30,
under an hour at 16 concurrent requests):

```bash
python code/1_fetch_abstracts.py                                  # or: Rscript R/01_fetch_abstracts.R
python code/2_filter_records.py --dir data/json_files              # or: Rscript R/02_filter_records.R
python code/3b_run_jev_classification.py data/json_files/filtered/all_abstracts.json --workers 16 --budget-usd 8
                                                                   # or: Rscript R/03_run_jev_classification.R data/json_files/filtered/all_abstracts.json --workers 16
python code/4_build_analysis_dataset.py --label-col jev_policy_claim   # writes data/analysis/*_jev.csv
python code/12_jev_accuracy.py                                     # adds the corpus comparison and Table 1 replication
```

Runs are resumable (already-scored rows are skipped) and stop at `--budget-usd`.

**If the Scopus COMPLETE view is refused** (HTTP 401 "not authorized to access the requested view": the key has no
institutional entitlement, e.g. off the university network and no `INST_TOKEN`), abstracts for the paper's
analytic sample can instead be fetched from PubMed, free and without a key. All ten journals are in MEDLINE;
records are matched to `derived_data/policy_claims_minimal.csv` by DOI, then by normalised title:

```bash
python code/1b_fetch_abstracts_pubmed.py        # or: Rscript R/01b_fetch_abstracts_pubmed.R
python code/3b_run_jev_classification.py data/json_files/filtered/all_abstracts_pubmed.json --workers 16 --budget-usd 8
python code/12_jev_accuracy.py --corpus data/json_files/filtered/all_abstracts_pubmed_JEV.csv
```

PubMed abstracts are the same publisher-supplied text as Scopus's but can differ in section labels and trailing
copyright notices, so this is a close rather than exact re-run of the paper's input. Match statistics are written
to `data/json_files/pubmed/match_stats.csv`.

### Results on the validation samples (19 September 2026)

Full tables: `table/jev_accuracy_report.md` (Python) and `table/jev_accuracy_report_R.md` (R, identical
point estimates). Cohen's kappa with bootstrap 95% CIs; the human references are those used in the paper.

| Comparison | n | DeepSeek V3.1 | Jev 1.13 (Noul) |
|---|---|---|---|
| Agreement with adjudicated gold standard (kappa) | 204 | 0.80 (0.70-0.89) | 0.79 (0.69-0.89) |
| ... sensitivity / specificity | 204 | 0.96 / 0.92 | 0.89 / 0.94 |
| Agreement with blinded stratified review, adjudicated (kappa) | 400 | 0.66 (0.57-0.73) | 0.77 (0.69-0.83) |
| ... sensitivity / specificity | 400 | 0.66 / 0.95 | 0.81 / 0.94 |
| Human-human agreement on the same 400 (kappa) | 400 | 0.87 | 0.87 |
| Test-retest across 3 runs (kappa) | 204 | 0.90-0.98 | 1.00 (max change in P = 0.06) |
| Agreement Jev vs DeepSeek (kappa) | 204 / 400 | - | 0.82 / 0.84 |
| Throughput (abstracts per second) | | ~1.3 (5 workers) | 20 (8 workers), 26 (16 workers) |
| Latency per request (p50) | | - | ~0.3 s |
| Full corpus (46,279 abstracts): wall time / cost | | ~10 h / ~$3 | 14.7 min / $3.27 (16 workers, 52 abstracts/s) |

Jev's probabilities are well calibrated against the human labels (AUC 0.98 on the gold standard, 0.96 on
the blinded 400) and its policy-claim rate by period tracks the manual rate more closely than DeepSeek's
(`table/jev_claim_rate_by_period_400.csv`).

### Results on the full corpus (Scopus re-download of 19 September 2026)

The abstracts were re-fetched from Scopus (51,061 raw records, 46,279 after the paper's filters; the paper's
2025 download gave 50,533 and 45,807) and classified in one run. 45,671 abstracts matched the published derived
dataset by DOI and carry both labels (`derived_data/policy_claims_jev.csv` holds the Jev labels, probabilities
and the DeepSeek label for every record, without abstracts).

| | DeepSeek V3.1 (paper) | Jev 1.13 |
|---|---|---|
| Policy-claim rate, all years | 25.6% | 31.1% |
| By period: 1990-99 / 2000-09 / 2010-19 / 2020-24 | 17.7 / 22.8 / 28.4 / 35.8% | 23.0 / 27.6 / 34.2 / 42.0% |
| Lowest and highest journal | Epidemiology 3.7%, Lancet Public Health 62.4% | Epidemiology 4.8%, Lancet Public Health 76.9% |
| Agreement with the other model | 92.6% (kappa 0.82, 95% CI 0.81-0.83) | same |
| Jev sensitivity / specificity taking DeepSeek as reference | | 0.96 / 0.91 |

Jev labels more abstracts as making a policy claim, but the paper's findings hold under either model: the rise
over time, the ordering of journals and of countries, and the higher rates in qualitative and cross-sectional
studies than in experimental and case-control studies (`table/jev_table1_replication.csv`,
`table/jev_corpus_by_*.csv`, `figures/jev_vs_deepseek_trend.png`). Both models undercount claims relative to
the blinded human reviewers, Jev less so (see the previous table).

### Files added

| File | Purpose |
|---|---|
| `code/jev_questions.json` | The questions asked of Jev (single source of truth, hashed) |
| `code/jev_client.py`, `R/jev_client.R` | API client: request/response, retries, budget cap, latency capture |
| `code/3b_run_jev_classification.py`, `R/03_run_jev_classification.R` | Classify any table with an `abstract` column; resumable; writes `*_timing.json` |
| `code/11_jev_speed_benchmark.py`, `R/11_jev_speed_benchmark.R` | Throughput/latency/cost at several concurrency levels, extrapolated to the corpus |
| `code/12_jev_accuracy.py`, `R/12_jev_accuracy.R` | Accuracy vs DeepSeek and human reviewers; test-retest; corpus comparison; Table 1 replication |
| `R/01_fetch_abstracts.R`, `R/02_filter_records.R`, `R/04_build_analysis_dataset.R` | tidyverse ports of steps 1, 2 and 4 |
| `code/1b_fetch_abstracts_pubmed.py`, `R/01b_fetch_abstracts_pubmed.R` | PubMed fallback for the abstracts (matches the derived dataset by DOI/title) |
| `run_jev_validation.sh`, `run_jev_validation_R.sh` | Drivers for the validation runs |
| `concordance/jev_outputs*/` | Jev outputs for the validation samples (no abstracts) |
| `table/jev_*`, `figures/jev_*` | Results |
| `derived_data/policy_claims_jev.csv` | Jev labels and probabilities for the full corpus (no abstracts) |

---

## File Structure

```
├── code                  # data processing, LLM classification, validation, and analysis scripts (incl. Jev *_jev* scripts)
├── R                     # tidyverse port of the pipeline for the Jev replication
├── concordance           # repeated LLM run outputs for concordance analyses
├── data                  # private source/intermediate files; not shared because abstracts are licensed
│   ├── analysis          # full analytic datasets with abstracts
│   └── json_files        # Scopus JSON exports and filtered/LLM-labelled abstract files
├── derived_data          # public/shareable derived metadata and LLM classifications
├── figures               # main and supplementary figures
└── table                 # validation files and exported main/supplementary tables
           
```

## Analysis Workflow

The analysis follows the sequence laid out in the `code/` directory:

File provenance: Scopus JSON exports in `data/json_files/` are filtered by `code/2_filter_records.py` into `data/json_files/filtered/all_abstracts.json`; `code/3_run_llm_classification.py` adds policy-claim labels in `data/json_files/filtered/all_abstracts_LLM.csv`; `code/4_build_analysis_dataset.py` merges those labels and writes `data/analysis/analysis_dataset.csv`; `code/5_add_study_design_and_topics.py` adds design/topic variables in `data/analysis/analysis_dataset_enriched_v2.csv`, which is used for the main analyses. Earlier study-design classifier variants are archived in `code/study_design_supplemental/` for provenance and sensitivity checks.

1. **Download metadata**  
   Query Scopus for each journal over 1990-2024 and save abstracts and metadata fields including publication year, keywords, citation counts, and corresponding author country.

2. **Clean corpus**  
  Restrict the dataset to research articles and remove non-empirical items, systematic reviews, and commentaries to produce an analysis-ready corpus.

3. **Classify policy claims**  
   Run DeepSeek V3.1 at low temperature on each abstract using the study prompt and generate a binary indicator for the presence of a policy claim.
   (Jev 1.13 replication: `code/3b_run_jev_classification.py` / `R/03_run_jev_classification.R`, see the section above.)

4. **Human validation**  
   Draw samples for blinded human review and compute agreement metrics against model outputs to assess reliability of the automated classification.

5. **Primary analyses**  
   Estimate prevalence by year, country, journal, field, and study design. Generate time series, country rankings, and journal contrasts.

6. **Keyword analyses**  
   Describe variation in claim rates across keywords and examine changes over time by topic.  

7. **Reporting**  
   Export figures and tables for the manuscript and supplementary materials.
   
---

# Authors and acknowledgments
David Bann<sup>1</sup>  \
Mengyao Wang<sup>2</sup>


### Author Affiliations:

1. Centre for Longitudinal Studies, University College London, UK
2. Department of Biostatistics, Yale University, US 
