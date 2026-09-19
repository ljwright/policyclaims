#!/usr/bin/env Rscript
# 01_fetch_abstracts.R - fetch article metadata and abstracts from the Scopus API
# for the ten journals over 1990-2024, one JSON file per journal in data/json_files/.
# R (tidyverse/httr2) port of code/1_fetch_abstracts.py; the output files have the
# same fields, so either implementation can feed 02_filter_records.R or
# code/2_filter_records.py.
#
# Queries are issued year by year (Scopus caps cursor pagination at 5,000 results)
# and deduplicated by scopus_id. Re-running is safe: years already complete are
# skipped unless --refresh is given. Abstracts (view=COMPLETE) require an API key
# with institutional entitlement: on campus, or with an INST_TOKEN.
#
# Requires SCOPUS_API_KEY (and optionally INST_TOKEN) in .env.
# Usage: Rscript R/01_fetch_abstracts.R [--refresh]

suppressPackageStartupMessages({ library(httr2); library(jsonlite); library(dplyr); library(purrr); library(stringr); library(tibble) })
ROOT <- here::here()
JSON_DIR <- file.path(ROOT, "data", "json_files"); dir.create(JSON_DIR, showWarnings = FALSE, recursive = TRUE)
SCOPUS_API_URL <- "https://api.elsevier.com/content/search/scopus"
PAUSE_SECONDS <- 0.2
YEARS <- 1990:2024

read_dotenv <- function(name) {
  v <- Sys.getenv(name); env <- file.path(ROOT, ".env")
  if (file.exists(env)) { ln <- str_subset(readLines(env, warn = FALSE), paste0("^", name, "=")); if (length(ln)) v <- str_trim(str_remove(ln[[1]], paste0("^", name, "="))) }
  v
}
SCOPUS_API_KEY <- read_dotenv("SCOPUS_API_KEY"); INST_TOKEN <- read_dotenv("INST_TOKEN")
if (!nzchar(SCOPUS_API_KEY)) stop("SCOPUS_API_KEY not found in .env")
cat("SCOPUS_API_KEY assigned: TRUE; INST_TOKEN:", nzchar(INST_TOKEN), "\n")

# Scopus source id -> ISSN and short name (same list as the Python script)
JOURNALS <- tribble(
  ~source_id,    ~issn,       ~abbrev,
  "S27024",      "0300-5771", "int_j_epidemiol",
  "S20224",      "0749-3797", "am_j_prev_med",
  "S20040",      "0091-7435", "prev_med",
  "S15470582",   "1044-3983", "epidemiology",
  "S156988948",  "0143-005X", "j_epidemiol_comm_health",
  "S2764808104", "2468-2667", "lancet_public_health",
  "S168049282",  "0090-0036", "am_j_public_health",
  "S4210220588", "1101-1262", "eur_j_public_health",
  "S170967050",  "0002-9262", "am_j_epidemiol",
  "S48690275",   "0393-2990", "eur_j_epidemiol"
)

scopus_request <- function(query, ...) {
  req <- request(SCOPUS_API_URL) |>
    req_headers(`X-ELS-APIKey` = SCOPUS_API_KEY, Accept = "application/json") |>
    req_url_query(query = query, ...) |>
    req_retry(max_tries = 4, is_transient = ~ resp_status(.x) %in% c(429, 500, 502, 503, 504), backoff = ~ 2^.x) |>
    req_error(is_error = function(resp) FALSE)
  if (nzchar(INST_TOKEN)) req <- req_headers(req, `X-ELS-Insttoken` = INST_TOKEN)
  req
}

fetch_total_results <- function(issn, year) {
  resp <- scopus_request(sprintf("ISSN(%s) AND PUBYEAR = %d AND DOCTYPE(ar)", issn, year), count = 1, view = "STANDARD") |> req_perform()
  if (resp_status(resp) != 200) { message(sprintf("[fetch_total_results] HTTP %d for ISSN=%s year=%d", resp_status(resp), issn, year)); return(0L) }
  as.integer(resp_body_json(resp)$`search-results`$`opensearch:totalResults`)
}

process_scopus_work <- function(e) {
  cover <- e$`prism:coverDate` %||% "Unknown"
  affs <- e$affiliation
  keywords <- e$authkeywords %||% ""
  list(
    scopus_id = e$`dc:identifier` %||% "Unknown",
    doi = e$`prism:doi` %||% "",
    title = e$`dc:title` %||% "",
    journal = e$`prism:publicationName` %||% "",
    publication_year = if (nzchar(cover)) str_split_1(cover, "-")[[1]] else "Unknown",
    keywords = if (nzchar(keywords)) as.list(str_split_1(keywords, fixed(" | "))) else list(),
    abstract = e$`dc:description` %||% "",
    article_type = str_to_lower(e$subtypeDescription %||% ""),
    # Historical field name retained for downstream compatibility: the value is the first listed affiliation's country.
    corresponding_author_country = if (length(affs)) (affs[[1]]$`affiliation-country` %||% "Unknown") else "Unknown",
    cited_by_count = e$`citedby-count` %||% 0
  )
}

fetch_scopus_works_for_year <- function(issn, year) {
  records <- list(); cursor <- "*"
  repeat {
    resp <- scopus_request(sprintf("ISSN(%s) AND PUBYEAR = %d AND DOCTYPE(ar)", issn, year), cursor = cursor, view = "COMPLETE") |> req_perform()
    if (resp_status(resp) != 200) { message(sprintf("[fetch_scopus_works_for_year] HTTP %d (ISSN=%s, year=%d): %s", resp_status(resp), issn, year, str_sub(resp_body_string(resp), 1, 200))); break }
    sr <- resp_body_json(resp)$`search-results`
    entries <- sr$entry %||% list()
    if (!length(entries) || !is.null(entries[[1]]$error)) break
    records <- c(records, map(entries, process_scopus_work))
    nxt <- sr$cursor$`@next`
    if (is.null(nxt) || length(entries) < 25) break
    cursor <- nxt
    Sys.sleep(PAUSE_SECONDS)
  }
  records
}

journal_file <- function(source_id, abbrev) file.path(JSON_DIR, sprintf("%s_%s.json", abbrev, source_id))

load_existing <- function(path) if (file.exists(path)) fromJSON(path, simplifyVector = FALSE) else list()

save_journal <- function(path, records) { write_json(records, path, auto_unbox = TRUE, pretty = TRUE, null = "null"); cat(sprintf("[save_journal_data] Saved %d articles to %s\n", length(records), path)) }

fetch_or_load_journal <- function(source_id, issn, abbrev, refresh = FALSE) {
  path <- journal_file(source_id, abbrev)
  records <- if (refresh) list() else load_existing(path)
  ids <- map_chr(records, ~ .x$scopus_id %||% .x$doi %||% "")
  for (year in YEARS) {
    have <- sum(map_chr(records, ~ as.character(.x$publication_year %||% "")) == as.character(year))
    total <- fetch_total_results(issn, year)
    if (total <= have) { cat(sprintf("[%s %d] Already have %d/%d. Skipping.\n", abbrev, year, have, total)); next }
    cat(sprintf("[%s %d] Found %d existing. Need %d total. Fetching...\n", abbrev, year, have, total))
    new <- fetch_scopus_works_for_year(issn, year)
    new <- new[!map_chr(new, "scopus_id") %in% ids]
    records <- c(records, new); ids <- c(ids, map_chr(new, "scopus_id"))
    cat(sprintf("[%s %d] +%d new articles fetched. Total = %d\n", abbrev, year, length(new), length(records)))
    save_journal(path, records)   # incremental save
  }
  invisible(records)
}

refresh <- "--refresh" %in% commandArgs(trailingOnly = TRUE)
pwalk(JOURNALS, function(source_id, issn, abbrev) {
  cat(strrep("=", 50), "\n", sprintf("Processing: %s (%s)", source_id, abbrev), "\n", strrep("=", 50), "\n", sep = "")
  fetch_or_load_journal(source_id, issn, abbrev, refresh = refresh)
})
cat("All done.\n")
