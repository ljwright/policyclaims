#!/usr/bin/env Rscript
# 01b_fetch_abstracts_pubmed.R - fill in abstracts for the paper's analytic sample
# from PubMed when the Scopus COMPLETE view is not available. R (tidyverse/httr2/
# xml2) port of code/1b_fetch_abstracts_pubmed.py.
#
# derived_data/policy_claims_minimal.csv holds the metadata for every analysed
# record but not the abstracts. All ten journals are in MEDLINE, so the abstracts
# are fetched with the free NCBI E-utilities (esearch by ISSN and year, efetch in
# batches) and matched to the derived rows by DOI, then by normalised title.
#
# Output: data/json_files/filtered/all_abstracts_pubmed_R.json, same fields as the
# Scopus pipeline (scopus_id blank; pmid, abstract_source, match_method added).
# Raw PubMed records are cached per journal in data/json_files/pubmed_R/.
# Optional NCBI_API_KEY in .env raises the rate limit from 3 to 10 requests/s.
# Usage: Rscript R/01b_fetch_abstracts_pubmed.R [--journals am_j_epidemiol,prev_med] [--refresh]

suppressPackageStartupMessages({ library(httr2); library(xml2); library(jsonlite); library(dplyr); library(purrr); library(stringr); library(readr); library(tibble) })
ROOT <- here::here()
DERIVED <- file.path(ROOT, "derived_data", "policy_claims_minimal.csv")
CACHE_DIR <- file.path(ROOT, "data", "json_files", "pubmed_R")
OUT_JSON <- file.path(ROOT, "data", "json_files", "filtered", "all_abstracts_pubmed_R.json")
EUTILS <- "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
BATCH <- 200

read_dotenv <- function(name) {
  v <- Sys.getenv(name); env <- file.path(ROOT, ".env")
  if (file.exists(env)) { ln <- str_subset(readLines(env, warn = FALSE), paste0("^", name, "=")); if (length(ln)) v <- str_trim(str_remove(ln[[1]], paste0("^", name, "="))) }
  v
}
NCBI_API_KEY <- read_dotenv("NCBI_API_KEY")
PAUSE <- if (nzchar(NCBI_API_KEY)) 0.11 else 0.35   # 10/s with a key, 3/s without

# ISSN -> short name -> journal name as it appears in the derived dataset
JOURNALS <- tribble(
  ~issn,       ~abbrev,                   ~journal,
  "0300-5771", "int_j_epidemiol",         "International Journal Of Epidemiology",
  "0749-3797", "am_j_prev_med",           "American Journal Of Preventive Medicine",
  "0091-7435", "prev_med",                "Preventive Medicine",
  "1044-3983", "epidemiology",            "Epidemiology",
  "0143-005X", "j_epidemiol_comm_health", "Journal Of Epidemiology And Community Health",
  "2468-2667", "lancet_public_health",    "The Lancet Public Health",
  "0090-0036", "am_j_public_health",      "American Journal Of Public Health",
  "1101-1262", "eur_j_public_health",     "European Journal Of Public Health",
  "0002-9262", "am_j_epidemiol",          "American Journal Of Epidemiology",
  "0393-2990", "eur_j_epidemiol",         "European Journal Of Epidemiology"
)

eutils <- function(endpoint, params, post = FALSE) {
  Sys.sleep(PAUSE)
  params <- c(list(db = "pubmed", tool = "policyclaims"), params, if (nzchar(NCBI_API_KEY)) list(api_key = NCBI_API_KEY))
  req <- request(paste0(EUTILS, endpoint)) |> req_timeout(120) |>
    req_retry(max_tries = 5, is_transient = ~ resp_status(.x) %in% c(429, 500, 502, 503, 504), backoff = ~ 2^.x, retry_on_failure = TRUE)
  req <- if (post) req_body_form(req, !!!params) else req_url_query(req, !!!params)
  req_perform(req) |> resp_body_xml()
}

esearch_history <- function(term) {
  x <- eutils("esearch.fcgi", list(term = term, retmax = 0, usehistory = "y"))
  list(count = as.integer(xml_text(xml_find_first(x, "//Count"))), webenv = xml_text(xml_find_first(x, "//WebEnv")), qk = xml_text(xml_find_first(x, "//QueryKey")))
}

parse_article <- function(a) {
  parts <- map_chr(xml_find_all(a, ".//Article/Abstract/AbstractText"), function(n) {
    label <- xml_attr(n, "Label"); text <- str_trim(xml_text(n))
    if (!is.na(label) && nzchar(str_trim(label)) && toupper(str_trim(label)) != "UNLABELLED") paste0(str_to_sentence(str_trim(label)), ": ", text) else text
  })
  dois <- c(xml_text(xml_find_all(a, ".//PubmedData/ArticleIdList/ArticleId[@IdType='doi']")), xml_text(xml_find_all(a, ".//Article/ELocationID[@EIdType='doi']")))
  year <- xml_text(xml_find_first(a, ".//Article/Journal/JournalIssue/PubDate/Year"))
  if (is.na(year)) year <- str_sub(xml_text(xml_find_first(a, ".//Article/Journal/JournalIssue/PubDate/MedlineDate")), 1, 4)
  list(pmid = xml_text(xml_find_first(a, ".//MedlineCitation/PMID")),
       doi = if (length(dois)) str_to_lower(str_trim(dois[[1]])) else "",
       title = str_trim(xml_text(xml_find_first(a, ".//Article/ArticleTitle"))),
       year = year, abstract = str_trim(paste(parts, collapse = " ")),
       publication_types = as.list(xml_text(xml_find_all(a, ".//PublicationTypeList/PublicationType"))))
}

fetch_journal_year <- function(issn, year) {
  h <- esearch_history(sprintf("%s[is] AND %d[dp]", issn, year))
  if (h$count == 0) return(list())
  starts <- seq(0, h$count - 1, by = BATCH)
  map(starts, function(start) {
    x <- eutils("efetch.fcgi", list(WebEnv = h$webenv, query_key = h$qk, retstart = start, retmax = BATCH, retmode = "xml"), post = TRUE)
    map(xml_find_all(x, "//PubmedArticle"), parse_article)
  }) |> list_flatten()
}

norm_title <- function(s) stringi::stri_trans_general(coalesce(as.character(s), ""), "Any-Latin; Latin-ASCII") |> str_to_lower() |> str_remove_all("[^a-z0-9]")
norm_doi <- function(s) coalesce(str_to_lower(str_trim(as.character(s))), "") |> str_remove("^https?://(dx\\.)?doi\\.org/")

main <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  opt <- function(flag, default) { i <- match(flag, args); if (is.na(i)) default else args[[i + 1]] }
  only <- opt("--journals", NULL); refresh <- "--refresh" %in% args
  dir.create(CACHE_DIR, showWarnings = FALSE, recursive = TRUE); dir.create(dirname(OUT_JSON), showWarnings = FALSE, recursive = TRUE)
  derived <- read_csv(DERIVED, show_col_types = FALSE) |> mutate(doi_n = norm_doi(doi), title_n = norm_title(title))
  out <- list(); unmatched <- list(); stats <- list(); t0 <- Sys.time()
  for (j in seq_len(nrow(JOURNALS))) {
    issn <- JOURNALS$issn[[j]]; abbrev <- JOURNALS$abbrev[[j]]; journal_name <- JOURNALS$journal[[j]]
    if (!is.null(only) && !abbrev %in% str_split_1(only, ",")) next
    target <- filter(derived, journal == journal_name)
    if (nrow(target) == 0) next
    cache <- file.path(CACHE_DIR, paste0(abbrev, ".json"))
    pub <- if (file.exists(cache) && !refresh) fromJSON(cache, simplifyVector = FALSE) else list()
    # PubMed's publication date can differ from Scopus's cover year by one year, so fetch one extra year each side
    years <- seq(min(target$publication_year) - 1, max(target$publication_year) + 1)
    for (year in years) {
      if (!is.null(pub[[as.character(year)]])) next
      pub[[as.character(year)]] <- fetch_journal_year(issn, year)
      write_json(pub, cache, auto_unbox = TRUE, null = "null")
      cat(sprintf("[%s %d] PubMed records: %d (derived rows: %d)\n", abbrev, year, length(pub[[as.character(year)]]), sum(target$publication_year == year)))
    }
    recs <- list_flatten(unname(pub[as.character(years)])) |> keep(~ nzchar(.x$abstract %||% ""))
    by_doi <- recs |> keep(~ nzchar(.x$doi)) |> (\(r) set_names(r, map_chr(r, "doi")))()
    by_title <- set_names(recs, map_chr(recs, ~ norm_title(.x$title))); by_title <- by_title[!duplicated(names(by_title))]
    n_doi <- 0L; n_title <- 0L
    for (i in seq_len(nrow(target))) {
      row <- target[i, ]
      rec <- if (nzchar(row$doi_n)) by_doi[[row$doi_n]] else NULL; method <- if (!is.null(rec)) "doi" else NULL
      if (is.null(rec) && nzchar(row$title_n)) { rec <- by_title[[row$title_n]]; method <- if (!is.null(rec)) "title" else NULL }
      if (is.null(rec)) { unmatched[[length(unmatched) + 1]] <- tibble(journal = journal_name, publication_year = row$publication_year, doi = row$doi, title = row$title); next }
      if (method == "doi") n_doi <- n_doi + 1L else n_title <- n_title + 1L
      out[[length(out) + 1]] <- list(
        scopus_id = "", pmid = rec$pmid, doi = if (!is.na(row$doi)) row$doi else rec$doi, title = row$title, journal = journal_name,
        publication_year = as.character(row$publication_year),
        keywords = if (!is.na(row$keywords)) as.list(str_trim(str_split_1(row$keywords, ";"))) else list(),
        abstract = rec$abstract, article_type = "article",
        corresponding_author_country = if (!is.na(row$corresponding_author_country)) row$corresponding_author_country else "Unknown",
        cited_by_count = NULL, abstract_source = "pubmed", match_method = method, llm_policy_claim = as.logical(row$llm_policy_claim))
    }
    stats[[length(stats) + 1]] <- tibble(journal = journal_name, derived_rows = nrow(target), matched_doi = n_doi, matched_title = n_title,
                                         unmatched = nrow(target) - n_doi - n_title, match_pct = round(100 * (n_doi + n_title) / nrow(target), 1))
    cat(sprintf("[%s] matched %d/%d (%s%%): doi=%d, title=%d\n", abbrev, n_doi + n_title, nrow(target), stats[[length(stats)]]$match_pct, n_doi, n_title))
  }
  write_json(out, OUT_JSON, auto_unbox = TRUE, null = "null", pretty = TRUE)
  write_csv(bind_rows(unmatched), file.path(CACHE_DIR, "unmatched.csv"))
  st <- bind_rows(stats); write_csv(st, file.path(CACHE_DIR, "match_stats.csv")); print(st)
  cat(sprintf("\nWrote %d records with abstracts to %s (%d unmatched) in %.0fs\n", length(out), OUT_JSON, length(unmatched), as.numeric(difftime(Sys.time(), t0, units = "secs"))))
}

main()
