#!/usr/bin/env Rscript
# 02_filter_records.R - keep only original empirical research articles.
# R port of code/2_filter_records.py (same regexes, same order of checks).
#
# Drops records that are missing an abstract; commentaries, replies, letters,
# editorials or corrections (document-type field, title or abstract regex); or
# reviews of any kind (systematic, scoping, umbrella, narrative, rapid,
# meta-analysis, or "review" in the title/keywords).
#
# Usage: Rscript R/02_filter_records.R [--dir data/json_files] [--log data/json_files/excluded_records.csv]
# Writes data/json_files/filtered/<file>.filtered.json per input and the combined
# data/json_files/filtered/all_abstracts.json. The functions are also sourced by
# 04_build_analysis_dataset.R.

suppressPackageStartupMessages({ library(dplyr); library(purrr); library(stringr); library(tibble); library(jsonlite); library(readr) })
ROOT <- here::here()
resolve_path <- function(p) if (startsWith(p, "/")) p else file.path(ROOT, p)   # options may be absolute or repo-relative

COMMENTARY_TITLE_RE <- regex("\\b(comment(ary)?|authors?\\s*reply|response|letter)\\b[:\\s-]?", ignore_case = TRUE)
COMMENTARY_ABS_RE   <- regex("\\b(this\\s+commentary|authors?\\s+reply|in\\s+response\\s+to)\\b", ignore_case = TRUE)
REVIEW_TITLE_RE     <- regex("(\\b(systematic|scoping|umbrella|narrative|rapid)\\s+review(s)?\\b)|\\bmeta-?analysis(es)?\\b|\\b(review|reviews)\\b", ignore_case = TRUE)
DOCTYPE_EXCLUDE <- c("comment", "commentary", "reply", "letter", "editorial", "author reply", "authors reply", "authors’ reply",
                     "correction", "erratum", "retraction", "news", "perspective",
                     "review", "systematic review", "scoping review", "umbrella review", "narrative review", "rapid review", "meta-analysis", "meta analysis")

# NFKC-normalise and collapse whitespace; lists (e.g. keywords) are joined with spaces, as in the Python version.
norm_text <- function(x) {
  if (is.null(x)) return("")
  if (is.list(x)) x <- paste(unlist(x), collapse = " ")
  if (is.na(x[[1]])) return("")
  stringi::stri_trans_nfkc(as.character(x)) |> str_replace_all("\\s+", " ") |> str_trim()
}

# Returns the exclusion reason ("" = keep), one record at a time.
exclusion_reason <- function(rec) {
  title <- norm_text(rec$title); abstract <- norm_text(rec$abstract); keywords <- norm_text(rec$keywords)
  if (!nzchar(abstract)) return("empty_abs")
  # NOTE: mirrors the Python script, which looks for `document_type`/`doctype`/`type`;
  # the fetch scripts store the Scopus subtype as `article_type`, so this check only
  # fires for inputs that carry one of those three field names.
  doc_type <- str_to_lower(norm_text(rec$document_type %||% rec$doctype %||% rec$type %||% ""))
  if (any(str_detect(doc_type, fixed(DOCTYPE_EXCLUDE)))) return("doc_type")
  if (str_detect(title, REVIEW_TITLE_RE) || str_detect(keywords, REVIEW_TITLE_RE)) return("review_term")
  if (str_detect(title, COMMENTARY_TITLE_RE)) return("title_commentary")
  if (str_detect(abstract, COMMENTARY_ABS_RE)) return("abs_commentary")
  ""
}

load_json_records <- function(path) {
  d <- fromJSON(path, simplifyVector = FALSE)
  if (!is.null(names(d)) && !is.null(d$records)) d$records else d
}

# Split a list of records into kept / dropped (with reason) and a count table.
filter_records <- function(records) {
  reasons <- map_chr(records, exclusion_reason)
  list(kept = records[reasons == ""], dropped = records[reasons != ""], dropped_reasons = reasons[reasons != ""],
       counts = as.list(table(reasons[reasons != ""])))
}

records_to_tibble <- function(records) {
  map(records, function(r) { r$keywords <- paste(unlist(r$keywords), collapse = "; "); as_tibble(map(r, ~ if (is.null(.x)) NA else .x)) }) |> list_rbind()
}

main <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  opt <- function(flag, default) { i <- match(flag, args); if (is.na(i)) default else args[[i + 1]] }
  in_dir <- resolve_path(opt("--dir", "data/json_files"))
  log_csv <- opt("--log", NULL)
  out_dir <- file.path(in_dir, "filtered"); dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  files <- list.files(in_dir, "\\.json$", full.names = TRUE)
  cat(sprintf("Found %d .json files in %s\n", length(files), in_dir))
  combined <- list()
  for (f in files) {
    res <- filter_records(load_json_records(f))
    out <- file.path(out_dir, str_replace(basename(f), "\\.json$", ".filtered.json"))
    write_json(res$kept, out, auto_unbox = TRUE, pretty = TRUE, null = "null")
    if (!is.null(log_csv) && length(res$dropped)) {
      lg <- records_to_tibble(res$dropped) |> mutate(`_excl_reason` = res$dropped_reasons) |> select(any_of(c("scopus_id", "title", "document_type", "abstract", "_excl_reason")))
      write_csv(lg, resolve_path(log_csv), append = file.exists(resolve_path(log_csv)))
    }
    summary <- if (length(res$counts)) paste(sprintf("%s=%d", names(res$counts), unlist(res$counts)), collapse = ", ") else "none"
    cat(sprintf("[filter] %s: kept=%d dropped=%d (%s) -> %s\n", basename(f), length(res$kept), length(res$dropped), summary, basename(out)))
    combined <- c(combined, res$kept)
  }
  write_json(combined, file.path(out_dir, "all_abstracts.json"), auto_unbox = TRUE, pretty = TRUE, null = "null")
  cat(sprintf("[combine] Wrote %d records to %s\n", length(combined), file.path(out_dir, "all_abstracts.json")))
}

if (sys.nframe() == 0) main()
