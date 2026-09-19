#!/usr/bin/env Rscript
# 04_build_analysis_dataset.R - combine the raw Scopus records with LLM labels into
# the analysis dataset and the public minimal export. R port of
# code/4_build_analysis_dataset.py, with a --label-col option so the labels can
# come from Jev (jev_policy_claim) instead of DeepSeek (llm_policy_claim).
#
# Steps: load raw data/json_files/*.json (or filtered/all_abstracts.json as a
# fallback), apply the filters from 02_filter_records.R, keep 1990-2024, merge
# labels by scopus_id then DOI, standardise columns, write
#   data/analysis/analysis_dataset[_jev].csv   (full, with abstracts; not shared)
#   data/analysis/policy_claims_minimal[_jev].csv (public-safe, no abstracts)
#
# Usage: Rscript R/04_build_analysis_dataset.R --labels-csv data/json_files/filtered/all_abstracts_JEV.csv --label-col jev_policy_claim

suppressPackageStartupMessages({ library(dplyr); library(purrr); library(stringr); library(tibble); library(jsonlite); library(readr) })
ROOT <- here::here()
resolve_path <- function(p) if (startsWith(p, "/")) p else file.path(ROOT, p)   # options may be absolute or repo-relative
source(file.path(ROOT, "R", "02_filter_records.R"))

args <- commandArgs(trailingOnly = TRUE)
opt <- function(flag, default) { i <- match(flag, args); if (is.na(i)) default else args[[i + 1]] }
label_col <- opt("--label-col", "llm_policy_claim")
tag <- if (label_col == "llm_policy_claim") "" else "_jev"
raw_dir <- resolve_path(opt("--raw-json-dir", "data/json_files"))
labels_csv <- resolve_path(opt("--labels-csv", if (tag == "") "data/json_files/filtered/all_abstracts_LLM.csv" else "data/json_files/filtered/all_abstracts_JEV.csv"))
analysis_csv <- resolve_path(opt("--analysis-csv", sprintf("data/analysis/analysis_dataset%s.csv", tag)))
minimal_csv <- resolve_path(opt("--minimal-export", sprintf("data/analysis/policy_claims_minimal%s.csv", tag)))

files <- list.files(raw_dir, "\\.json$", full.names = TRUE)
records <- if (length(files)) list_flatten(map(files, load_json_records)) else {   # purrr::list_flatten (jsonlite masks flatten)
  fb <- file.path(raw_dir, "filtered", "all_abstracts.json"); if (!file.exists(fb)) stop("No JSON files under ", raw_dir, " and no fallback at ", fb)
  cat("No raw JSON files; loading combined file:", fb, "\n"); load_json_records(fb)
}
cat(sprintf("Raw records: %d\n", length(records)))

flt <- filter_records(records)
kept <- records_to_tibble(flt$kept) |> mutate(publication_year = suppressWarnings(as.integer(publication_year))) |> filter(between(publication_year, 1990, 2024))
cat(sprintf("Kept after content filters and year window: %d\nDropped during filtering: %d\nExclusion reasons: %s\n", nrow(kept), length(flt$dropped), paste(sprintf("%s=%d", names(flt$counts), unlist(flt$counts)), collapse = ", ")))

lab <- read_csv(labels_csv, show_col_types = FALSE, guess_max = 100000) |> select(scopus_id, doi, label = all_of(label_col)) |> mutate(label = as.logical(label))
lab_sid <- lab |> filter(!is.na(scopus_id)) |> distinct(scopus_id, .keep_all = TRUE) |> select(scopus_id, label)
lab_doi <- lab |> filter(!is.na(doi), doi != "") |> distinct(doi, .keep_all = TRUE) |> select(doi, label_doi = label)
merged <- kept |> mutate(scopus_id = as.character(scopus_id), doi = as.character(doi)) |>
  left_join(lab_sid, by = "scopus_id") |> left_join(lab_doi, by = "doi") |> mutate(llm_policy_claim = coalesce(label, label_doi)) |> select(-label, -label_doi)
cat(sprintf("Missing LLM label after merge: %d\n", sum(is.na(merged$llm_policy_claim))))

analysis <- merged |> filter(!is.na(llm_policy_claim)) |>
  mutate(
    doi = str_to_lower(str_trim(as.character(doi))) |> str_remove("^https?://(dx\\.)?doi\\.org/") |> na_if("") |> na_if("nan") |> na_if("none"),
    title = str_trim(as.character(title)), journal = str_to_title(str_trim(as.character(journal))),
    keywords = na_if(str_trim(as.character(keywords)), ""), abstract = as.character(abstract),
    article_type = str_to_lower(str_trim(as.character(article_type))),
    corresponding_author_country = str_to_upper(coalesce(na_if(na_if(str_trim(as.character(corresponding_author_country)), ""), "nan"), "UNKNOWN")),
    cited_by_count = suppressWarnings(as.integer(cited_by_count)), llm_policy_claim = as.logical(llm_policy_claim),
    claim = as.integer(llm_policy_claim), doi_norm = doi, abstract_word_count = str_count(coalesce(abstract, ""), "\\S+")
  ) |>
  select(scopus_id, doi, title, journal, publication_year, keywords, abstract, article_type, corresponding_author_country, cited_by_count, llm_policy_claim, claim, doi_norm, abstract_word_count)
cat(sprintf("Analytic N: %d\nYears: %d-%d\n", nrow(analysis), min(analysis$publication_year), max(analysis$publication_year)))

dir.create(dirname(analysis_csv), showWarnings = FALSE, recursive = TRUE)
write_csv(analysis, analysis_csv, na = "")
minimal <- analysis |> select(doi, title, journal, publication_year, keywords, corresponding_author_country, llm_policy_claim) |>
  filter(!is.na(doi), doi != "", !is.na(title)) |> distinct(doi, .keep_all = TRUE)
write_csv(minimal, minimal_csv, na = "")
cat(sprintf("Wrote analysis dataset: %s\nWrote minimal export: %s (%d rows)\n", analysis_csv, minimal_csv, nrow(minimal)))
