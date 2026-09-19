#!/usr/bin/env Rscript
# 03_run_jev_classification.R - classify abstracts for policy claims with Jev 1.13
# via OpenRouter. R (tidyverse) port of code/3b_run_jev_classification.py.
#
# Input:  a table with an `abstract` column: JSON (list of records), CSV or XLSX.
# Output: <input>_JEV.csv (or --output) with the input columns (minus the abstract
#         unless --keep-abstract) plus jev_* columns, and a *_timing.json summary.
#         Runs are resumable: rows already scored in the output are skipped, and
#         progress is checkpointed after every chunk of requests.
#
# Usage:
#   Rscript R/03_run_jev_classification.R data/json_files/filtered/all_abstracts.json
#   Rscript R/03_run_jev_classification.R table/gold_standard_30_march.xlsx --sheet in \
#           --output concordance/jev_outputs/jev_gold_standard_run1_R.csv
# Options: --sheet NAME  --output PATH  --workers N (8)  --budget-usd X (8)
#          --sample PCT  --limit N  --seed N (42)  --keep-abstract  --no-resume
#
# Requires OPENROUTER_API_KEY in .env.

suppressPackageStartupMessages({
  library(dplyr); library(readr); library(purrr); library(stringr); library(tibble); library(jsonlite)
})
source(file.path(here::here(), "R", "jev_client.R"))

MIN_ABSTRACT_CHARS <- 30   # same rule as the Python scripts

# --- minimal argument parsing (no extra dependencies) -----------------------
parse_args <- function(args) {
  opts <- list(sheet = NULL, output = NULL, workers = 8, budget_usd = 8, sample = NULL,
               limit = NULL, seed = 42, keep_abstract = FALSE, no_resume = FALSE, input = NULL)
  i <- 1
  while (i <= length(args)) {
    a <- args[[i]]
    take <- function() { i <<- i + 1; args[[i]] }
    switch(a,
      "--sheet" = opts$sheet <- take(),
      "--output" = opts$output <- take(),
      "--workers" = opts$workers <- as.integer(take()),
      "--budget-usd" = opts$budget_usd <- as.numeric(take()),
      "--sample" = opts$sample <- as.numeric(take()),
      "--limit" = opts$limit <- as.integer(take()),
      "--seed" = opts$seed <- as.integer(take()),
      "--keep-abstract" = opts$keep_abstract <- TRUE,
      "--no-resume" = opts$no_resume <- TRUE,
      { if (str_starts(a, "--")) stop("Unknown option: ", a); opts$input <- a }
    )
    i <- i + 1
  }
  if (is.null(opts$input)) stop("Usage: Rscript R/03_run_jev_classification.R <input> [options]")
  opts
}

read_table <- function(path, sheet = NULL) {
  ext <- tolower(tools::file_ext(path))
  if (ext == "json") {
    d <- fromJSON(path)
    if (is.list(d) && !is.data.frame(d) && !is.null(d$records)) d <- d$records
    as_tibble(d)
  } else if (ext == "csv") {
    read_csv(path, show_col_types = FALSE, guess_max = 100000)
  } else if (ext %in% c("xlsx", "xls")) {
    readxl::read_excel(path, sheet = sheet %||% 1)
  } else stop("Unsupported input type: ", path)
}

jev_cols <- names(jev_empty_result())

# Coerce the jev_* columns of a data frame to the canonical types of
# jev_empty_result(), so rows_update() never faces an incompatible cast
# (e.g. an all-NA column read back from CSV as logical).
coerce_jev_types <- function(df) {
  template <- jev_empty_result()
  for (col in jev_cols) {
    if (!col %in% names(df)) df[[col]] <- template[[col]][NA_integer_]
    tmpl <- template[[col]]
    df[[col]] <- if (is.logical(tmpl)) as.logical(df[[col]]) else if (is.integer(tmpl)) as.integer(df[[col]])
                 else if (is.numeric(tmpl)) as.numeric(df[[col]]) else as.character(df[[col]])
  }
  df
}

save_output <- function(df, out_csv, keep_abstract) {
  out <- if (keep_abstract) df else select(df, -any_of("abstract"))
  # list columns (e.g. keywords from JSON) are flattened for CSV
  out <- mutate(out, across(where(is.list), ~ map_chr(.x, \(v) paste(unlist(v), collapse = "; "))))
  dir.create(dirname(out_csv), showWarnings = FALSE, recursive = TRUE)
  write_csv(out, out_csv, na = "")
}

main <- function() {
  opts <- parse_args(commandArgs(trailingOnly = TRUE))
  api_key <- jev_api_key()
  in_path <- opts$input
  out_csv <- opts$output %||% {
    suffix <- if (!is.null(opts$sample)) sprintf("_sample%g_seed%d", opts$sample, opts$seed) else ""
    file.path(dirname(in_path), paste0(tools::file_path_sans_ext(basename(in_path)), suffix, "_JEV.csv"))
  }
  timing_json <- str_replace(out_csv, "\\.csv$", "_timing.json")

  cat(strrep("=", 60), "\nJev 1.13 policy-claim classification (OpenRouter Decisions API) - R\n",
      "Input:  ", in_path, "\nOutput: ", out_csv, "\nQuestions hash: ", JEV_QUESTIONS_HASH, "\n", strrep("=", 60), "\n", sep = "")

  df <- read_table(in_path, opts$sheet) |>
    mutate(source_row = row_number() - 1L, .before = 1) |>          # 0-based, same as Python
    mutate(abstract = str_trim(coalesce(as.character(abstract), "")))
  n0 <- nrow(df)
  df <- filter(df, nchar(abstract) >= MIN_ABSTRACT_CHARS)
  cat(sprintf("Dropped %d rows with missing/short abstracts (< %d chars).\n", n0 - nrow(df), MIN_ABSTRACT_CHARS))
  if (!is.null(opts$sample)) {
    set.seed(opts$seed)
    df <- slice_sample(df, prop = opts$sample / 100) |> arrange(source_row)
    cat(sprintf("Sampled %d rows (%g%%).\n", nrow(df), opts$sample))
  }
  if (!is.null(opts$limit)) df <- slice_head(df, n = opts$limit)

  # resume from an existing output
  if (file.exists(out_csv) && !opts$no_resume) {
    prev <- read_csv(out_csv, show_col_types = FALSE, guess_max = 100000) |>
      filter(!is.na(jev_p_yes)) |> select(source_row, all_of(jev_cols))
    df <- left_join(select(df, -any_of(jev_cols)), prev, by = "source_row") |> coerce_jev_types()
    cat(sprintf("Resuming: %d rows already classified in %s.\n", sum(!is.na(df$jev_p_yes)), basename(out_csv)))
  } else {
    df <- select(df, -any_of(jev_cols)) |> coerce_jev_types()   # adds empty, correctly typed jev_* columns
  }

  todo <- filter(df, is.na(jev_p_yes))
  est <- jev_estimate_cost_usd(todo$abstract)
  usage <- jev_key_usage(api_key)
  cat(sprintf("To classify: %d abstracts. Estimated cost ~$%.3f. Key usage so far $%.3f of limit $%s (remaining $%s).\n",
              nrow(todo), est, usage$usage_usd, usage$limit_usd, usage$limit_remaining_usd))
  if (est > opts$budget_usd) stop(sprintf("Estimated cost $%.2f exceeds --budget-usd %.2f. Raise the cap or use --sample/--limit.", est, opts$budget_usd))
  if (nrow(todo) == 0) { cat("Nothing to do.\n"); return(invisible()) }

  started <- format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z", tz = "UTC")
  t0 <- Sys.time()
  # checkpoint after every chunk: merge the chunk's results into df and rewrite the CSV
  on_chunk <- function(idx, rows) {
    rows$source_row <- todo$source_row[idx]
    df <<- rows_update(df, rows, by = "source_row", unmatched = "ignore")
    save_output(df, out_csv, opts$keep_abstract)
  }
  res <- jev_classify_many(todo$abstract, api_key = api_key, workers = opts$workers,
                           budget_usd = opts$budget_usd, on_chunk = on_chunk, progress = TRUE)
  wall <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  save_output(df, out_csv, opts$keep_abstract)

  ok <- filter(df, !is.na(jev_p_yes))
  n_new <- sum(res$ok)
  timing <- list(
    model = JEV_MODEL, model_version_reported = names(sort(table(ok$jev_model), decreasing = TRUE))[1],
    questions_hash = JEV_QUESTIONS_HASH, implementation = "R",
    started_utc = started, finished_utc = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z", tz = "UTC"),
    workers = opts$workers, n_rows_in_output = nrow(df), n_classified_total = nrow(ok),
    n_classified_this_run = n_new, n_failed = sum(!is.na(df$jev_error)),
    wall_seconds_this_run = round(wall, 2),
    throughput_abstracts_per_second = if (n_new > 0) round(n_new / wall, 3) else NULL,
    latency_ms_p50 = unname(quantile(ok$jev_latency_ms, 0.5, na.rm = TRUE)),
    latency_ms_p90 = unname(quantile(ok$jev_latency_ms, 0.9, na.rm = TRUE)),
    latency_ms_p99 = unname(quantile(ok$jev_latency_ms, 0.99, na.rm = TRUE)),
    latency_ms_mean = mean(ok$jev_latency_ms, na.rm = TRUE),
    input_tokens_total = sum(ok$jev_input_tokens, na.rm = TRUE), input_tokens_mean = mean(ok$jev_input_tokens, na.rm = TRUE),
    cost_usd_total = sum(ok$jev_cost_usd, na.rm = TRUE), cost_usd_mean = mean(ok$jev_cost_usd, na.rm = TRUE),
    share_policy_claim_noul = mean(ok$jev_policy_claim, na.rm = TRUE),
    share_policy_claim_choice = mean(ok$jev_choice == "yes", na.rm = TRUE)
  )
  write_json(timing, timing_json, auto_unbox = TRUE, pretty = TRUE, digits = NA)

  cat(sprintf("\nClassified %d/%d rows (%d failed) in %.1fs (%.2f abstracts/s with %d workers).\n",
              nrow(ok), nrow(df), timing$n_failed, wall, n_new / wall, opts$workers))
  cat(sprintf("Spent this run: $%.4f. Policy-claim share (Noul): %.3f\n", sum(res$jev_cost_usd, na.rm = TRUE), timing$share_policy_claim_noul))
  cat("Wrote ", out_csv, " and ", timing_json, "\n", sep = "")
}

main()
