#!/usr/bin/env Rscript
# 11_jev_speed_benchmark.R - how fast (and how cheap) is Jev 1.13 via OpenRouter
# at several concurrency levels? R port of code/11_jev_speed_benchmark.py.
#
# The same --n abstracts are classified once per --workers value (no caching),
# recording wall time, throughput, client-observed latency percentiles, input
# tokens and cost, then extrapolated to the 45,807-abstract corpus and compared
# with the DeepSeek run in the README (~10 h, ~$3, 5 workers).
#
# Usage: Rscript R/11_jev_speed_benchmark.R [--n 100] [--workers 1,4,8,16] [--seed 42]
#        [--input table/manual_review_by_design_50_each.csv] [--budget-usd 0.5] [--suffix _R]
# Output: table/jev_speed_benchmark<suffix>.csv and .md

suppressPackageStartupMessages({ library(dplyr); library(readr); library(purrr); library(stringr); library(tibble); library(jsonlite) })
source(file.path(here::here(), "R", "jev_client.R"))
ROOT <- here::here()

N_CORPUS <- 45807
DEEPSEEK_REFERENCE <- list(model = "deepseek-chat (DeepSeek V3.1)", workers = 5, hours_for_corpus = 10, cost_usd_for_corpus = 3)

args <- commandArgs(trailingOnly = TRUE)
opt <- function(flag, default) { i <- match(flag, args); if (is.na(i)) default else args[[i + 1]] }
n <- as.integer(opt("--n", 100)); workers <- as.integer(str_split_1(opt("--workers", "1,4,8,16"), ","))
seed <- as.integer(opt("--seed", 42)); budget <- as.numeric(opt("--budget-usd", 0.5)); suffix <- opt("--suffix", "_R")
input <- opt("--input", file.path(ROOT, "table", "manual_review_by_design_50_each.csv"))

df <- read_csv(input, show_col_types = FALSE) |> filter(nchar(str_trim(abstract)) >= 30)
set.seed(seed)
abstracts <- slice_sample(df, n = min(n, nrow(df)))$abstract
cat(sprintf("Benchmarking %d abstracts at workers=%s (model %s)\n", length(abstracts), paste(workers, collapse = ","), JEV_MODEL))

bench <- function(w) {
  t0 <- Sys.time()
  res <- jev_classify_many(abstracts, workers = w, budget_usd = budget, chunk_size = length(abstracts), progress = FALSE)
  wall <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  ok <- filter(res, ok)
  thr <- nrow(ok) / wall
  tibble(
    model = JEV_MODEL, workers = w, n_abstracts = nrow(ok), n_failed = nrow(res) - nrow(ok),
    wall_seconds = round(wall, 2), throughput_per_second = round(thr, 3),
    latency_ms_p50 = round(quantile(ok$jev_latency_ms, 0.5), 1), latency_ms_p90 = round(quantile(ok$jev_latency_ms, 0.9), 1),
    latency_ms_p99 = round(quantile(ok$jev_latency_ms, 0.99), 1), latency_ms_mean = round(mean(ok$jev_latency_ms), 1),
    input_tokens_mean = round(mean(ok$jev_input_tokens), 1),
    cost_usd_total = round(sum(ok$jev_cost_usd), 5), cost_usd_per_abstract = round(sum(ok$jev_cost_usd) / nrow(ok), 7),
    est_hours_for_corpus = round(N_CORPUS / thr / 3600, 2), est_cost_usd_for_corpus = round(sum(ok$jev_cost_usd) / nrow(ok) * N_CORPUS, 2)
  )
}

res <- map(workers, function(w) {
  cat(sprintf("  workers=%d ...", w)); r <- bench(w)
  cat(sprintf(" %s abs/s, p50 %s ms, $%s\n", r$throughput_per_second, r$latency_ms_p50, r$cost_usd_total)); Sys.sleep(2); r
}) |> list_rbind() |> mutate(run_utc = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z", tz = "UTC"))

md_table <- function(d) {
  cells <- d |> mutate(across(everything(), ~ ifelse(is.na(.x), "", as.character(.x))))
  c(paste0("| ", paste(names(d), collapse = " | "), " |"), paste0("|", strrep("---|", ncol(d))),
    pmap_chr(cells, function(...) paste0("| ", paste(c(...), collapse = " | "), " |")))
}

csv_path <- file.path(ROOT, "table", paste0("jev_speed_benchmark", suffix, ".csv"))
md_path <- file.path(ROOT, "table", paste0("jev_speed_benchmark", suffix, ".md"))
write_csv(res, csv_path)
best <- slice_max(res, throughput_per_second, n = 1)
ds_thr <- N_CORPUS / (DEEPSEEK_REFERENCE$hours_for_corpus * 3600)
lines <- c(
  "# Jev 1.13 speed benchmark (R implementation)", "",
  sprintf("Run %s from this machine via OpenRouter; %d abstracts per setting (seed %d, source `%s`).",
          res$run_utc[[1]], length(abstracts), seed, str_remove(input, paste0("^", ROOT, "/"))),
  sprintf("Latency is the client-observed round trip (curl total time) and includes network time. Cost is as reported by OpenRouter (input tokens only, $%.3f/M).", PRICE_USD_PER_INPUT_TOKEN * 1e6), "",
  md_table(select(res, -run_utc)), "",
  "## Extrapolation to the full corpus (n = 45,807)", "",
  "| model | workers | throughput (abstracts/s) | est. hours | est. cost (USD) |", "|---|---|---|---|---|",
  sprintf("| %s (README) | %d | %.2f | %.1f | %.2f |", DEEPSEEK_REFERENCE$model, DEEPSEEK_REFERENCE$workers, ds_thr, DEEPSEEK_REFERENCE$hours_for_corpus, DEEPSEEK_REFERENCE$cost_usd_for_corpus),
  sprintf("| %s (best setting here) | %d | %.2f | %.2f | %.2f |", JEV_MODEL, best$workers, best$throughput_per_second, best$est_hours_for_corpus, best$est_cost_usd_for_corpus), "",
  sprintf("Speed-up over the DeepSeek run: about %.0fx at %d workers. TypeSafe's published limit is 1,200 requests/minute (20/s), so throughput saturates around there.",
          best$throughput_per_second / ds_thr, best$workers)
)
writeLines(lines, md_path)
cat("Wrote", csv_path, "and", md_path, "\n"); cat(lines, sep = "\n")
