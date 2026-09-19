#!/usr/bin/env Rscript
# 12_jev_accuracy.R - accuracy of Jev 1.13 policy-claim labels against DeepSeek and
# human reviewers. R (tidyverse) port of code/12_jev_accuracy.py; see that file's
# header for the comparisons. Reads the Jev outputs produced by either
# code/3b_run_jev_classification.py or R/03_run_jev_classification.R.
#
# Usage: Rscript R/12_jev_accuracy.R [--outputs-dir concordance/jev_outputs] [--suffix _R]
#        [--corpus data/json_files/filtered/all_abstracts_JEV.csv]
# Outputs (in table/ and figures/, with <suffix>): jev_accuracy_summary, jev_accuracy_report.md,
#   jev_test_retest, jev_mismatches_gold, jev_claim_rate_by_period_400, jev_corpus_by_*, jev_table1_replication

suppressPackageStartupMessages({
  library(dplyr); library(readr); library(purrr); library(stringr); library(tibble); library(tidyr); library(jsonlite); library(ggplot2)
})
source(file.path(here::here(), "R", "jev_client.R"))
ROOT <- here::here(); TABLE <- file.path(ROOT, "table"); FIG <- file.path(ROOT, "figures")
resolve_path <- function(p) if (startsWith(p, "/")) p else file.path(ROOT, p)   # options may be absolute or repo-relative

args <- commandArgs(trailingOnly = TRUE)
opt <- function(flag, default) { i <- match(flag, args); if (is.na(i)) default else args[[i + 1]] }
OUT_DIR <- resolve_path(opt("--outputs-dir", "concordance/jev_outputs"))
SUFFIX <- opt("--suffix", "_R")
CORPUS_JEV <- resolve_path(opt("--corpus", "data/json_files/filtered/all_abstracts_JEV.csv"))
DERIVED <- file.path(ROOT, "derived_data", "policy_claims_minimal.csv")
PERIODS <- tibble(period = c("1990-1999", "2000-2009", "2010-2019", "2020-2024"), lo = c(1990, 2000, 2010, 2020), hi = c(1999, 2009, 2019, 2024))
DEEPSEEK_RETEST_REFERENCE <- c("run1 vs run2" = 0.905, "run1 vs run3" = 0.930, "run2 vs run3" = 0.978)  # 8_llm_validation.ipynb
out_path <- function(name, ext = "csv") file.path(TABLE, paste0(name, SUFFIX, ".", ext))

# ---------------------------------------------------------------- helpers ----
norm_bool <- function(x) {
  s <- str_to_lower(str_trim(as.character(x)))
  case_when(s %in% c("1", "1.0", "true", "t", "yes", "y") ~ TRUE, s %in% c("0", "0.0", "false", "f", "no", "n") ~ FALSE, TRUE ~ NA)
}
norm_doi <- function(x) {
  s <- str_to_lower(str_trim(as.character(x))) |> str_remove("^https?://(dx\\.)?doi\\.org/")
  if_else(is.na(s) | s == "" | s == "nan", NA_character_, s)
}
wilson_ci <- function(k, n, z = qnorm(0.975)) {
  if (n == 0) return(c(NA, NA)); p <- k / n; denom <- 1 + z^2 / n
  centre <- (p + z^2 / (2 * n)) / denom; half <- z * sqrt((p * (1 - p) + z^2 / (4 * n)) / n) / denom
  c(centre - half, centre + half)
}
chi2_2xk_p <- function(successes, totals) chisq.test(rbind(successes, totals - successes), correct = FALSE)$p.value
cohen_kappa <- function(a, b) {   # binary Cohen's kappa; equals sklearn.metrics.cohen_kappa_score
  a <- as.logical(a); b <- as.logical(b); po <- mean(a == b)
  pe <- mean(a) * mean(b) + (1 - mean(a)) * (1 - mean(b)); if (pe == 1) NA_real_ else (po - pe) / (1 - pe)
}
auc_mw <- function(y, p) {         # Mann-Whitney AUC with average ranks for ties (= sklearn roc_auc_score)
  y <- as.logical(y); n1 <- sum(y); n0 <- sum(!y); if (n1 == 0 || n0 == 0) return(NA_real_)
  r <- rank(p); (sum(r[y]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}
fmt_p <- function(p) if (p < 0.001) "<0.001" else sprintf("%.3f", p)

# Agreement statistics treating `ref` as the reference standard (bootstrap CIs, 1,000 resamples, seed 42).
metrics <- function(ref, pred, label, n_boot = 1000, seed = 42) {
  m <- tibble(ref = as.logical(ref), pred = as.logical(pred)) |> drop_na()
  n <- nrow(m); if (n == 0) return(tibble(comparison = label, n = 0L))
  r <- m$ref; p <- m$pred
  tp <- sum(r & p); tn <- sum(!r & !p); fp <- sum(!r & p); fn <- sum(r & !p)
  set.seed(seed)
  bs <- map(seq_len(n_boot), function(i) { s <- sample.int(n, n, replace = TRUE); c(mean(r[s] == p[s]), cohen_kappa(r[s], p[s])) })
  ag_bs <- map_dbl(bs, 1); k_bs <- map_dbl(bs, 2)
  div <- function(a, b) if (b > 0) a / b else NA_real_
  tibble(
    comparison = label, n = n,
    agreement_pct = round(100 * mean(r == p), 1),
    agreement_ci = sprintf("(%.1f, %.1f)", 100 * quantile(ag_bs, 0.025, na.rm = TRUE), 100 * quantile(ag_bs, 0.975, na.rm = TRUE)),
    kappa = round(cohen_kappa(r, p), 3),
    kappa_ci = sprintf("(%.3f, %.3f)", quantile(k_bs, 0.025, na.rm = TRUE), quantile(k_bs, 0.975, na.rm = TRUE)),
    sensitivity = round(div(tp, tp + fn), 3), specificity = round(div(tn, tn + fp), 3),
    ppv = round(div(tp, tp + fp), 3), npv = round(div(tn, tn + fn), 3),
    prevalence_ref_pct = round(100 * mean(r), 1), prevalence_pred_pct = round(100 * mean(p), 1),
    tp = tp, fp = fp, fn = fn, tn = tn
  )
}
prob_metrics <- function(ref, p) {
  m <- tibble(ref = as.logical(ref), p = as.numeric(p)) |> drop_na()
  list(auc = round(auc_mw(m$ref, m$p), 3), brier = round(mean((m$p - m$ref)^2), 3), n = nrow(m))
}
calibration_table <- function(ref, p, breaks = c(0, 0.1, 0.3, 0.5, 0.7, 0.9, 1)) {
  tibble(ref = as.logical(ref), p = as.numeric(p)) |> drop_na() |>
    mutate(p_yes_bin = cut(p, breaks, include.lowest = TRUE)) |>
    group_by(p_yes_bin) |> summarise(n = n(), observed_rate = round(100 * mean(ref), 1), mean_p_yes_pct = round(100 * mean(p), 1), .groups = "drop")
}
md_table <- function(d) {
  if (is.null(d) || nrow(d) == 0) return("_(no data)_")
  fmt <- function(v) if (is.numeric(v)) ifelse(is.na(v), "", ifelse(abs(v - round(v)) < 1e-9, format(round(v)), formatC(v, digits = 3, format = "f"))) else ifelse(is.na(v), "", as.character(v))
  cells <- d |> mutate(across(everything(), fmt))
  paste(c(paste0("| ", paste(names(d), collapse = " | "), " |"), paste0("|", strrep("---|", ncol(d))),
          pmap_chr(cells, function(...) paste0("| ", paste(c(...), collapse = " | "), " |"))), collapse = "\n")
}
load_jev <- function(path) {
  if (!file.exists(path)) { message("[skip] ", basename(path), " not found"); return(NULL) }
  read_csv(path, show_col_types = FALSE, guess_max = 100000) |> filter(!is.na(jev_p_yes)) |>
    mutate(jev_noul = jev_p_yes >= NOUL_THRESHOLD, jev_choice_yes = str_to_lower(as.character(jev_choice)) == "yes")
}
drop_counts <- function(d) select(d, -any_of(c("tp", "fp", "fn", "tn")))

summary_rows <- list(); report <- character()
add_summary <- function(rows) summary_rows[[length(summary_rows) + 1]] <<- rows
add_report <- function(...) report <<- c(report, ...)

# ----------------------------------------------------------- 1. dev set ----
d <- load_jev(file.path(OUT_DIR, "jev_design400.csv"))
if (!is.null(d)) {
  ref <- norm_bool(d$llm_policy_claim)
  rows <- bind_rows(metrics(ref, d$jev_noul, "dev400: Jev noul vs DeepSeek"), metrics(ref, d$jev_choice_yes, "dev400: Jev choice vs DeepSeek"))
  add_summary(rows)
  add_report("## 1. Development set: design-stratified sample (n=400), reference = DeepSeek", "",
             "Used only to check the question wording; no human labels here.", "", md_table(drop_counts(rows)), "",
             "Calibration of Jev's P(policy claim) against the DeepSeek label:", "", md_table(calibration_table(ref, d$jev_p_yes)), "")
}

# ------------------------------------------------------ 2. gold standard ----
wb <- readxl::read_excel(file.path(TABLE, "gold_standard_30_march.xlsx"), sheet = "in", .name_repair = "unique_quiet") |> mutate(source_row = row_number() - 1L, .before = 1)
d <- load_jev(file.path(OUT_DIR, "jev_gold_standard_run1.csv"))
if (!is.null(d)) {
  g <- inner_join(wb, select(d, source_row, jev_p_yes, jev_noul, jev_choice_yes, jev_choice_confidence), by = "source_row")
  refs <- c("agreed_gold_standard" = "adjudicated gold standard", "agreed_gold_standard_with_exclusions" = "gold standard excl. non-empirical/truncated",
            "DB review" = "reviewer DB (author 1)", "EC re-review" = "reviewer EC (author 5)", "MW review" = "reviewer MW (author 2)")
  ds <- norm_bool(g$llm_policy_claim)
  rows <- imap(refs, function(nice, col) {
    r <- norm_bool(g[[col]])
    bind_rows(metrics(r, ds, paste("gold: DeepSeek vs", nice)), metrics(r, g$jev_noul, paste("gold: Jev noul vs", nice)), metrics(r, g$jev_choice_yes, paste("gold: Jev choice vs", nice)))
  }) |> list_rbind() |> bind_rows(metrics(ds, g$jev_noul, "gold: Jev noul vs DeepSeek"), metrics(ds, g$jev_choice_yes, "gold: Jev choice vs DeepSeek"))
  add_summary(rows)
  gold <- norm_bool(g$agreed_gold_standard); pm <- prob_metrics(gold, g$jev_p_yes)
  add_report("## 2. Gold standard workbook (n=204 abstracts with human review)", "",
             "The paper reports DeepSeek vs the adjudicated gold standard: kappa 0.80, sensitivity 76.8%, specificity 98.6% (n=204).", "",
             md_table(drop_counts(rows)), "",
             sprintf("Jev P(policy claim) vs the adjudicated gold standard: AUC = %s, Brier score = %s (n=%d).", pm$auc, pm$brier, pm$n), "",
             "Calibration against the adjudicated gold standard:", "", md_table(calibration_table(gold, g$jev_p_yes)), "")
  g |> mutate(gold = gold, deepseek = ds) |> filter(!is.na(gold), gold != jev_noul) |>
    mutate(abstract_end = str_sub(str_trim(as.character(abstract)), -400)) |>
    select(scopus_id, doi, title, gold, deepseek, jev_noul, jev_p_yes, jev_choice_yes, abstract_end) |>
    write_csv(out_path("jev_mismatches_gold"))
  pl <- ggplot(mutate(g, gold = if_else(gold, "gold: policy claim", "gold: no policy claim")), aes(jev_p_yes, fill = gold)) +
    geom_histogram(bins = 20, alpha = 0.7, position = "identity") + geom_vline(xintercept = NOUL_THRESHOLD, linetype = 2) +
    labs(x = "Jev P(policy claim)", y = "abstracts", fill = NULL, title = "Jev probability by adjudicated human label (n=204)") + theme_minimal(base_size = 10) + theme(legend.position = "top")
  dir.create(FIG, showWarnings = FALSE); ggsave(file.path(FIG, paste0("jev_p_yes_by_gold_label", SUFFIX, ".png")), pl, width = 6, height = 3.6, dpi = 150)
}

# ---------------------------------------------------- 3. stratified 400 ----
d <- load_jev(file.path(OUT_DIR, "jev_stratified400.csv"))
if (!is.null(d)) {
  s <- readxl::read_excel(file.path(TABLE, "supp_stratified_sample_400_blinded_db.xlsx")) |>
    left_join(readxl::read_excel(file.path(TABLE, "supp_stratified_sample_400_internal.xlsx")) |> select(review_id, publication_year, llm_policy_claim), by = "review_id") |>
    inner_join(select(d, review_id, jev_p_yes, jev_noul, jev_choice_yes), by = "review_id") |>
    mutate(across(c(DB_policy_claim, EC_policy_claim, db_double_checked, llm_policy_claim), norm_bool),
           # adjudication rule from 8_llm_validation.ipynb: DB where DB and EC agree, otherwise DB's double-check
           manual_final = if_else(DB_policy_claim == EC_policy_claim, DB_policy_claim, db_double_checked))
  refs <- c("manual_final" = "adjudicated manual label", "DB_policy_claim" = "reviewer DB (blinded)", "EC_policy_claim" = "reviewer EC (blinded)")
  rows <- imap(refs, function(nice, col) bind_rows(
    metrics(s[[col]], s$llm_policy_claim, paste("strat400: DeepSeek vs", nice)), metrics(s[[col]], s$jev_noul, paste("strat400: Jev noul vs", nice)),
    metrics(s[[col]], s$jev_choice_yes, paste("strat400: Jev choice vs", nice)))) |> list_rbind() |>
    bind_rows(metrics(s$DB_policy_claim, s$EC_policy_claim, "strat400: reviewer EC vs reviewer DB (human-human)"), metrics(s$llm_policy_claim, s$jev_noul, "strat400: Jev noul vs DeepSeek"))
  add_summary(rows)
  pm <- prob_metrics(s$manual_final, s$jev_p_yes)
  per <- pmap(PERIODS, function(period, lo, hi) {
    sub <- filter(s, publication_year >= lo, publication_year <= hi); n <- nrow(sub)
    row <- tibble(period = period, n = n)
    for (key in c("manual", "deepseek", "jev")) {
      col <- c(manual = "manual_final", deepseek = "llm_policy_claim", jev = "jev_noul")[[key]]
      k <- sum(sub[[col]], na.rm = TRUE); ci <- wilson_ci(k, n)
      row[[paste0(key, "_k")]] <- k; row[[paste0(key, "_rate_pct")]] <- round(100 * k / n, 1); row[[paste0(key, "_95ci")]] <- sprintf("(%.1f, %.1f)", 100 * ci[1], 100 * ci[2])
    }
    row
  }) |> list_rbind()
  pvals <- map_chr(c("manual", "deepseek", "jev"), ~ sprintf("%s p=%s", .x, fmt_p(chi2_2xk_p(per[[paste0(.x, "_k")]], per$n))))
  per <- select(per, -ends_with("_k")); write_csv(per, out_path("jev_claim_rate_by_period_400"))
  add_report("## 3. Blinded stratified sample (n=400, 1990-2024), reviewers DB and EC", "",
             "Adjudicated label = DB where DB and EC agree, otherwise DB's double-check (rule from 8_llm_validation.ipynb).", "", md_table(drop_counts(rows)), "",
             sprintf("Jev P(policy claim) vs the adjudicated manual label: AUC = %s, Brier = %s.", pm$auc, pm$brier), "",
             paste0("Policy-claim rate by period (Supplementary Table 8 style; Wilson 95% CIs; chi-square p across periods: ", paste(pvals, collapse = ", "), "):"), "", md_table(per), "")
}

# -------------------------------------------------------- 4. test-retest ----
runs <- list.files(OUT_DIR, "^jev_gold_standard_run\\d+\\.csv$", full.names = TRUE)
if (length(runs) >= 2) {
  frames <- set_names(map(runs, ~ load_jev(.x) |> select(source_row, jev_p_yes, jev_noul)), str_remove_all(basename(runs), "jev_gold_standard_|\\.csv"))
  rt <- combn(names(frames), 2, simplify = FALSE) |> map(function(pr) {
    m <- inner_join(frames[[pr[1]]], frames[[pr[2]]], by = "source_row", suffix = c("_a", "_b"))
    tibble(comparison = paste(pr[1], "vs", pr[2]), n = nrow(m), kappa = round(cohen_kappa(m$jev_noul_a, m$jev_noul_b), 3),
           agreement_pct = round(100 * mean(m$jev_noul_a == m$jev_noul_b), 1),
           mean_abs_diff_p_yes = round(mean(abs(m$jev_p_yes_a - m$jev_p_yes_b)), 4), max_abs_diff_p_yes = round(max(abs(m$jev_p_yes_a - m$jev_p_yes_b)), 3),
           deepseek_kappa_reference = unname(DEEPSEEK_RETEST_REFERENCE[paste(pr[1], "vs", pr[2])]))
  }) |> list_rbind()
  write_csv(rt, out_path("jev_test_retest"))
  add_report("## 4. Test-retest reliability (repeated Jev runs on the gold-standard abstracts)", "",
             "DeepSeek reference (paper): kappa 0.90-0.98 across three runs of 200 abstracts at temperature 0.1.", "", md_table(rt), "")
} else message("[skip] fewer than two gold-standard runs for test-retest")

# ------------------------------------------------------------- 5. corpus ----
if (!file.exists(CORPUS_JEV)) {
  add_report("## 5. Full corpus", "", sprintf("_Not yet run: `%s` not found. Fetch the abstracts (01_fetch_abstracts.R, 02_filter_records.R) and run 03_run_jev_classification.R._", str_remove(CORPUS_JEV, paste0("^", ROOT, "/"))), "")
} else {
  jv <- load_jev(CORPUS_JEV) |> mutate(doi_n = norm_doi(doi), tkey = str_c(str_to_lower(str_trim(title)), "|", str_to_lower(str_trim(journal)), "|", publication_year))
  dv <- read_csv(DERIVED, show_col_types = FALSE) |> mutate(doi_n = norm_doi(doi), tkey = str_c(str_to_lower(str_trim(title)), "|", str_to_lower(str_trim(journal)), "|", publication_year))
  m <- jv |> left_join(dv |> filter(!is.na(doi_n)) |> distinct(doi_n, .keep_all = TRUE) |> select(doi_n, llm_policy_claim, design_combined), by = "doi_n") |>
    left_join(dv |> distinct(tkey, .keep_all = TRUE) |> select(tkey, llm_policy_claim_t = llm_policy_claim, design_combined_t = design_combined), by = "tkey") |>
    mutate(llm_policy_claim = coalesce(llm_policy_claim, llm_policy_claim_t), design_combined = coalesce(design_combined, design_combined_t), deepseek = norm_bool(llm_policy_claim)) |>
    select(-llm_policy_claim_t, -design_combined_t)
  # Shareable derived file (metadata + Jev results, no abstracts); the Python run writes derived_data/policy_claims_jev.csv
  m |> mutate(deepseek_policy_claim = deepseek) |>
    select(any_of(c("scopus_id", "doi", "title", "journal", "publication_year", "keywords", "corresponding_author_country", "design_combined",
                    "jev_policy_claim", "jev_p_yes", "jev_choice", "jev_choice_p_yes", "jev_choice_confidence", "jev_model", "jev_questions_hash", "deepseek_policy_claim"))) |>
    write_csv(file.path(ROOT, "derived_data", paste0("policy_claims_jev", SUFFIX, ".csv")), na = "")
  matched <- m |> filter(!is.na(deepseek)) |>
    mutate(agree = deepseek == jev_noul, period = cut(as.integer(publication_year), c(1989, 1999, 2009, 2019, 2024), labels = PERIODS$period),
           country = str_to_upper(str_trim(as.character(corresponding_author_country))))
  rows <- bind_rows(metrics(matched$deepseek, matched$jev_noul, "corpus: Jev noul vs DeepSeek", n_boot = 200), metrics(matched$deepseek, matched$jev_choice_yes, "corpus: Jev choice vs DeepSeek", n_boot = 200))
  add_summary(rows)
  rates <- function(df, by) df |> group_by(.data[[by]]) |>
    summarise(n = n(), deepseek_rate_pct = round(100 * mean(deepseek), 1), jev_rate_pct = round(100 * mean(jev_noul), 1), agreement_pct = round(100 * mean(agree), 1),
              kappa = round(if (n_distinct(deepseek) > 1) cohen_kappa(deepseek, jev_noul) else NA_real_, 3), .groups = "drop")
  by_year <- rates(matched, "publication_year"); write_csv(by_year, out_path("jev_corpus_by_year"))
  by_journal <- rates(matched, "journal") |> arrange(deepseek_rate_pct); write_csv(by_journal, out_path("jev_corpus_by_journal"))
  top15 <- count(matched, country, sort = TRUE) |> slice_head(n = 15) |> pull(country)
  by_country <- rates(filter(matched, country %in% top15), "country") |> arrange(deepseek_rate_pct); write_csv(by_country, out_path("jev_corpus_by_country"))
  by_design <- rates(matched, "design_combined"); write_csv(by_design, out_path("jev_corpus_by_design"))
  t1_block <- function(df, by, name) {
    groups <- if (is.null(by)) list("All abstracts" = df) else split(df, df[[by]])
    imap(groups, function(sub, lvl) {
      row <- tibble(row = if (is.null(by)) lvl else paste0(name, ": ", lvl))
      for (per in c(PERIODS$period, "All years")) {
        ss <- if (per == "All years") sub else filter(sub, period == per)
        row[[paste("DeepSeek", per)]] <- if (nrow(ss)) round(100 * mean(ss$deepseek), 1) else NA; row[[paste("Jev", per)]] <- if (nrow(ss)) round(100 * mean(ss$jev_noul), 1) else NA
      }
      row
    }) |> list_rbind()
  }
  t1 <- bind_rows(t1_block(matched, NULL, ""), t1_block(matched, "journal", "Journal"), t1_block(filter(matched, country %in% top15), "country", "Country"))
  write_csv(t1, out_path("jev_table1_replication"))
  pl <- by_year |> pivot_longer(c(deepseek_rate_pct, jev_rate_pct), names_to = "model", values_to = "rate") |>
    mutate(model = recode(model, deepseek_rate_pct = "DeepSeek V3.1 (paper)", jev_rate_pct = "Jev 1.13")) |>
    ggplot(aes(publication_year, rate, colour = model)) + geom_line() + geom_point(size = 1) +
    labs(x = "publication year", y = "% abstracts with a policy claim", colour = NULL, title = "Policy-claim rate by year: DeepSeek vs Jev") + theme_minimal(base_size = 10) + theme(legend.position = "top")
  ggsave(file.path(FIG, paste0("jev_vs_deepseek_trend", SUFFIX, ".png")), pl, width = 7, height = 3.8, dpi = 150)
  add_report("## 5. Full corpus: Jev vs DeepSeek", "",
             sprintf("Matched %s of %s Jev-classified abstracts to the derived dataset by DOI (fallback title+journal+year); %s unmatched.", format(nrow(matched), big.mark = ","), format(nrow(m), big.mark = ","), format(sum(is.na(m$deepseek)), big.mark = ",")), "",
             md_table(drop_counts(rows)), "", "### By period (Table 1 style, % with policy claim)", "", md_table(filter(t1, str_starts(row, "All|Journal"))), "",
             "### By journal", "", md_table(by_journal), "", "### Top-15 countries (first author)", "", md_table(by_country), "", "### By study design", "", md_table(by_design), "",
             sprintf("![trend](../figures/jev_vs_deepseek_trend%s.png)", SUFFIX), "")
}

# -------------------------------------------------------------- 6. speed ----
timing_files <- c(list.files(OUT_DIR, "_timing\\.json$", full.names = TRUE), list.files(file.path(ROOT, "data"), "_timing\\.json$", full.names = TRUE, recursive = TRUE))
add_report("## 6. Speed and cost", "", "DeepSeek reference (README): ~10 hours and ~$3 for 45,807 abstracts (5 concurrent requests), i.e. ~1.3 abstracts/s.", "")
if (length(timing_files)) {
  tim <- map(timing_files, function(p) { t <- fromJSON(p); tibble(run = str_remove(basename(p), "_timing\\.json$"), n = t$n_classified_this_run, workers = t$workers, wall_s = t$wall_seconds_this_run,
    abstracts_per_s = t$throughput_abstracts_per_second %||% NA, latency_p50_ms = round(t$latency_ms_p50 %||% NA), latency_p90_ms = round(t$latency_ms_p90 %||% NA),
    cost_usd = round(t$cost_usd_total %||% 0, 4), tokens_per_abstract = round(t$input_tokens_mean %||% 0)) }) |> list_rbind()
  add_report("Jev runs recorded by the classification script:", "", md_table(tim), "")
}
bench <- file.path(TABLE, paste0("jev_speed_benchmark", SUFFIX, ".md")); if (!file.exists(bench)) bench <- file.path(TABLE, "jev_speed_benchmark.md")
if (file.exists(bench)) add_report("Concurrency benchmark:", "", readLines(bench)[-(1:2)], "")

# -------------------------------------------------------------- write ----
header <- c("# Jev 1.13 vs DeepSeek V3.1 and human review: policy-claim classification accuracy (R implementation)", "",
            sprintf("Generated by `R/12_jev_accuracy.R`. Jev question set `%s` (hash `%s`), label rule: P(policy claim) >= %s (`noul`); `choice` = yes/no Choice question.", JEV_QUESTIONS_VERSION, JEV_QUESTIONS_HASH, NOUL_THRESHOLD),
            "Kappa CIs are 1,000-resample bootstraps (seed 42), matching 7_concordance.py.", "")
write_csv(list_rbind(summary_rows), out_path("jev_accuracy_summary"))
writeLines(c(header, report), out_path("jev_accuracy_report", "md"))
cat(sprintf("Wrote %s and %s (%d comparisons)\n", out_path("jev_accuracy_report", "md"), out_path("jev_accuracy_summary"), nrow(list_rbind(summary_rows))))
