# jev_client.R - shared helpers for classifying abstracts with TypeSafe's Jev 1.13
# through the OpenRouter Decisions API. R (tidyverse) port of code/jev_client.py.
#
# Jev is a "System One" decision model: it does not generate text. You send a
# `state` (here: the abstract) plus typed `questions` and it returns a typed
# answer per question with calibrated probabilities:
#
#   POST https://openrouter.ai/api/alpha/decisions
#   {"model": "typesafe/jev-1.13", "state": {...}, "questions": {...}}
#
# It cannot be called through /chat/completions. Pricing is per input token
# only ($0.042 per million tokens as of 2026-09-19); output tokens are free.
#
# The questions are NOT defined here: both the Python and R pipelines load them
# from code/jev_questions.json so the two implementations ask exactly the same
# thing. This file provides:
#
#   jev_api_key()          read OPENROUTER_API_KEY from the environment or .env
#   jev_key_usage()        spend and limit for the key, as OpenRouter sees it
#   jev_build_request()    one httr2 request for one abstract (with retries)
#   jev_parse_response()   response -> one-row tibble of jev_* columns
#   jev_classify_many()    concurrent classification with a hard budget cap,
#                          checkpointing after every chunk
#
# Usage: source("R/jev_client.R") from anywhere inside the repository.

suppressPackageStartupMessages({
  library(httr2)
  library(jsonlite)
  library(dplyr)
  library(purrr)
  library(tibble)
  library(stringr)
})

JEV_MODEL <- "typesafe/jev-1.13"
OPENROUTER_DECISIONS_URL <- "https://openrouter.ai/api/alpha/decisions"
OPENROUTER_KEY_URL <- "https://openrouter.ai/api/v1/auth/key"
PRICE_USD_PER_INPUT_TOKEN <- 0.042 / 1e6   # OpenRouter list price, 2026-09-19
PRICE_USD_PER_OUTPUT_TOKEN <- 0

# ---------------------------------------------------------------------------
# Paths and the shared question specification
# ---------------------------------------------------------------------------
jev_repo_root <- function() here::here()   # `here` locates the repository root (.git)

jev_questions_path <- function() file.path(jev_repo_root(), "code", "jev_questions.json")

# Canonical JSON identical to Python's json.dumps(x, sort_keys=True, ensure_ascii=FALSE),
# so the hash stored in jev_questions.json can be re-checked from R.
jev_canonical_json <- function(x) {
  if (is.list(x) && !is.null(names(x))) {
    keys <- sort(names(x))
    inner <- map_chr(keys, function(k) paste0(toJSON(k, auto_unbox = TRUE), ": ", jev_canonical_json(x[[k]])))
    paste0("{", paste(inner, collapse = ", "), "}")
  } else if (is.list(x)) {
    paste0("[", paste(map_chr(x, jev_canonical_json), collapse = ", "), "]")
  } else if (is.null(x)) {
    "null"
  } else {
    as.character(toJSON(x, auto_unbox = TRUE, digits = NA))
  }
}

jev_questions_hash <- function(questions) {
  substr(digest::digest(jev_canonical_json(questions), algo = "sha256", serialize = FALSE), 1, 12)
}

jev_load_spec <- function(path = jev_questions_path()) {
  spec <- fromJSON(path, simplifyVector = FALSE)   # keep nested lists as-is
  actual <- jev_questions_hash(spec$questions)
  if (!identical(actual, spec$questions_hash)) {
    stop(sprintf("%s: questions_hash %s does not match the questions (%s). If you edited the questions on purpose, bump `version` and update questions_hash.",
                 basename(path), spec$questions_hash, actual))
  }
  spec
}

JEV_SPEC <- jev_load_spec()
JEV_QUESTIONS <- JEV_SPEC$questions
JEV_QUESTIONS_VERSION <- JEV_SPEC$version
JEV_QUESTIONS_HASH <- JEV_SPEC$questions_hash
# Label rule for the primary (Noul) question: P(policy claim) >= threshold.
NOUL_THRESHOLD <- JEV_SPEC$noul_threshold

# ---------------------------------------------------------------------------
# Credentials and account usage
# ---------------------------------------------------------------------------
jev_api_key <- function() {
  # The repository's .env takes precedence over the environment, so the key you
  # put in the project is the one used (same rule as code/jev_client.py). This
  # avoids silently picking up a stale OPENROUTER_API_KEY from ~/.Renviron.
  key <- ""
  env_file <- file.path(jev_repo_root(), ".env")
  if (file.exists(env_file)) {
    line <- readLines(env_file, warn = FALSE) |> str_subset("^OPENROUTER_API_KEY=")
    if (length(line)) key <- str_trim(str_remove(line[[1]], "^OPENROUTER_API_KEY="))
  }
  if (!nzchar(key)) key <- Sys.getenv("OPENROUTER_API_KEY")
  if (!nzchar(key)) stop("OPENROUTER_API_KEY not found. Add it to .env (see .env.example) or export it.")
  key
}

jev_key_usage <- function(api_key = jev_api_key()) {
  d <- request(OPENROUTER_KEY_URL) |>
    req_auth_bearer_token(api_key) |>
    req_perform() |>
    resp_body_json() |>
    pluck("data")
  tibble(
    usage_usd = as.numeric(d$usage %||% 0),
    limit_usd = as.numeric(d$limit %||% NA),
    limit_remaining_usd = as.numeric(d$limit_remaining %||% NA),
    expires_at = d$expires_at %||% NA_character_
  )
}

# ---------------------------------------------------------------------------
# One request / one response
# ---------------------------------------------------------------------------
jev_build_request <- function(abstract, api_key, timeout = 60, max_tries = 4) {
  body <- list(
    model = JEV_MODEL,
    state = list(abstract = str_trim(abstract %||% "")),
    questions = JEV_QUESTIONS
  )
  request(OPENROUTER_DECISIONS_URL) |>
    req_auth_bearer_token(api_key) |>
    req_headers(`HTTP-Referer` = "https://github.com/dbann/policyclaims",
                `X-Title` = "policyclaims-jev-replication") |>
    req_body_json(body, auto_unbox = TRUE) |>
    req_timeout(timeout) |>
    # Retry rate limits / server errors with exponential backoff (honours Retry-After).
    req_retry(max_tries = max_tries,
              is_transient = function(resp) resp_status(resp) %in% c(429, 500, 502, 503, 504, 529),
              backoff = function(i) min(1.5^i, 60),
              retry_on_failure = TRUE) |>
    # Do not turn HTTP errors into R errors: jev_parse_response() records them.
    req_error(is_error = function(resp) FALSE)
}

jev_empty_result <- function(error = NA_character_) {
  tibble(
    ok = FALSE,
    jev_p_yes = NA_real_, jev_policy_claim = NA, jev_choice = NA_character_,
    jev_choice_p_yes = NA_real_, jev_choice_confidence = NA_real_,
    jev_input_tokens = NA_integer_, jev_output_tokens = NA_integer_, jev_cost_usd = NA_real_,
    jev_latency_ms = NA_real_, jev_attempts = NA_integer_, jev_model = NA_character_,
    jev_request_id = NA_character_, jev_error = error,
    jev_questions_hash = JEV_QUESTIONS_HASH,
    jev_timestamp = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z", tz = "UTC")
  )
}

jev_parse_response <- function(resp) {
  if (inherits(resp, "error") || inherits(resp, "condition")) {
    return(jev_empty_result(paste("network:", conditionMessage(resp))))
  }
  status <- resp_status(resp)
  if (status != 200) {
    return(jev_empty_result(sprintf("HTTP %d: %s", status, str_sub(resp_body_string(resp), 1, 300))))
  }
  p <- resp_body_json(resp)
  noul <- p$answers$policy_claim
  choice <- p$answers$policy_claim_choice
  usage <- p$usage %||% list()
  if (is.null(noul$noul)) return(jev_empty_result("no noul answer in response"))
  p_yes <- as.numeric(noul$noul)
  cost <- usage$cost
  if (is.null(cost) && !is.null(usage$input_tokens)) {
    cost <- usage$input_tokens * PRICE_USD_PER_INPUT_TOKEN + (usage$output_tokens %||% 0) * PRICE_USD_PER_OUTPUT_TOKEN
  }
  jev_empty_result() |>
    mutate(
      ok = TRUE,
      jev_p_yes = p_yes,
      jev_policy_claim = p_yes >= NOUL_THRESHOLD,
      jev_choice = choice$choice %||% NA_character_,
      jev_choice_p_yes = as.numeric(choice$probabilities$yes %||% NA),
      jev_choice_confidence = as.numeric(choice$confidence %||% NA),
      jev_input_tokens = as.integer(usage$input_tokens %||% NA),
      jev_output_tokens = as.integer(usage$output_tokens %||% NA),
      jev_cost_usd = as.numeric(cost %||% NA),
      jev_latency_ms = unname(resp$timing[["total"]]) * 1000,   # curl's total time, last attempt
      jev_attempts = NA_integer_,
      jev_model = p$model %||% NA_character_,
      jev_request_id = p$id %||% NA_character_,
      jev_error = NA_character_
    )
}

jev_classify_one <- function(abstract, api_key = jev_api_key()) {
  resp <- tryCatch(req_perform(jev_build_request(abstract, api_key)), error = identity)
  jev_parse_response(resp)
}

# ---------------------------------------------------------------------------
# Concurrent classification with a hard spending cap
# ---------------------------------------------------------------------------
jev_estimate_cost_usd <- function(abstracts, overhead_tokens = 1300) {
  # ~4 characters per token for the abstract plus the (measured) cost of the two questions
  sum(nchar(abstracts %||% "") / 4 + overhead_tokens) * PRICE_USD_PER_INPUT_TOKEN
}

#' Classify many abstracts concurrently.
#'
#' @param abstracts character vector
#' @param workers   number of concurrent requests (TypeSafe allows 1,200/min)
#' @param budget_usd hard stop once this much has been spent in this call
#' @param chunk_size requests per batch; the budget is checked and `on_chunk`
#'   called (e.g. to checkpoint) after every batch
#' @param on_chunk optional function(indices, results_tibble)
#' @return tibble with one row per abstract, in input order, columns jev_*
jev_classify_many <- function(abstracts, api_key = jev_api_key(), workers = 8, budget_usd = 8,
                              chunk_size = 200, on_chunk = NULL, progress = TRUE) {
  n <- length(abstracts)
  out <- vector("list", n)
  spent <- 0
  chunks <- split(seq_len(n), ceiling(seq_len(n) / chunk_size))
  for (idx in chunks) {
    reqs <- map(abstracts[idx], jev_build_request, api_key = api_key)
    resps <- req_perform_parallel(reqs, max_active = workers, on_error = "continue", progress = progress)
    rows <- map(resps, jev_parse_response)
    out[idx] <- rows
    spent <- spent + sum(map_dbl(rows, ~ .x$jev_cost_usd %||% NA_real_), na.rm = TRUE)
    if (!is.null(on_chunk)) on_chunk(idx, bind_rows(rows))
    if (spent > budget_usd) {
      warning(sprintf("Budget cap of $%.2f reached (spent $%.4f); stopping.", budget_usd, spent), call. = FALSE)
      break
    }
  }
  filled <- !map_lgl(out, is.null)
  out[!filled] <- list(jev_empty_result("not attempted (budget cap)"))
  bind_rows(out) |> mutate(.row = seq_len(n), .before = 1)
}
