#!/usr/bin/env python3
"""
jev_client.py - shared helpers for classifying abstracts with TypeSafe's Jev 1.13
through the OpenRouter Decisions API.

Jev is a "System One" decision model: it does not generate text. You send a
`state` (here: the abstract) plus typed `questions`, and it returns a typed
answer per question with calibrated probabilities. OpenRouter exposes it at

    POST https://openrouter.ai/api/alpha/decisions
    {"model": "typesafe/jev-1.13", "state": ..., "questions": {...}}

(it cannot be used through /chat/completions). Pricing is per *input* token
only ($0.042 per million tokens as of 2026-09-19); output tokens are free.

Everything that defines the classification lives in this module so it can be
audited in one place:

  * QUESTIONS            - the two questions asked about every abstract (a Noul =
                           yes/no probability, and a yes/no Choice), loaded from
                           code/jev_questions.json (shared with the R port)
  * NOUL_THRESHOLD       - how the Noul probability is turned into a label
  * classify_abstract()  - one API call with retries
  * classify_many()      - concurrent classification with a hard budget cap

The question wording mirrors the DeepSeek prompt used in the paper
(code/3_run_llm_classification.py): same definition, same inclusion/exclusion
rules and the same five worked examples. The R port lives in R/jev_client.R.

Requires OPENROUTER_API_KEY in .env (or the environment).
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=True)  # the repo's .env wins over a stale env var

# --------------------------------------------------------------------------
# API constants
# --------------------------------------------------------------------------
OPENROUTER_DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
OPENROUTER_KEY_URL = "https://openrouter.ai/api/v1/auth/key"
JEV_MODEL = "typesafe/jev-1.13"
PRICE_USD_PER_INPUT_TOKEN = 0.042 / 1_000_000  # OpenRouter list price, 2026-09-19
PRICE_USD_PER_OUTPUT_TOKEN = 0.0

# --------------------------------------------------------------------------
# Question definitions: loaded from code/jev_questions.json, the single source
# of truth shared with the R port (R/jev_client.R).
# --------------------------------------------------------------------------
QUESTIONS_PATH = Path(__file__).resolve().parent / "jev_questions.json"


def _hash_questions(questions: dict[str, Any]) -> str:
    payload = json.dumps(questions, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def load_question_spec(path: Path = QUESTIONS_PATH) -> dict[str, Any]:
    spec = json.loads(path.read_text(encoding="utf-8"))
    actual = _hash_questions(spec["questions"])
    if actual != spec["questions_hash"]:
        raise SystemExit(
            f"{path.name}: questions_hash {spec['questions_hash']} does not match the questions ({actual}). "
            "If you edited the questions on purpose, bump `version` and set questions_hash to the new value."
        )
    return spec


_SPEC = load_question_spec()
QUESTIONS: dict[str, dict[str, Any]] = _SPEC["questions"]
QUESTIONS_VERSION: str = _SPEC["version"]
# Label rule for the primary (Noul) question: P(yes) >= threshold -> policy claim.
# 0.5 is the natural cut-point for a calibrated probability and the value used in
# TypeSafe's own docs; it was fixed before any validation data was scored.
NOUL_THRESHOLD: float = float(_SPEC["noul_threshold"])


def questions_hash() -> str:
    """Short fingerprint of the question set, stored with every result."""
    return _hash_questions(QUESTIONS)


def build_state(abstract: str) -> dict[str, str]:
    return {"abstract": (abstract or "").strip()}


def build_request(abstract: str) -> dict[str, Any]:
    return {"model": JEV_MODEL, "state": build_state(abstract), "questions": QUESTIONS}


# --------------------------------------------------------------------------
# Credentials and account usage
# --------------------------------------------------------------------------
class BudgetExceeded(RuntimeError):
    pass


def get_api_key() -> str:
    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise SystemExit(
            "OPENROUTER_API_KEY not found. Add it to .env (see .env.example) or export it."
        )
    return key


def get_key_usage(api_key: str, timeout: float = 20.0) -> dict[str, Any]:
    """Return OpenRouter's view of this key: usage (USD), limit, limit_remaining."""
    r = requests.get(OPENROUTER_KEY_URL, headers={"Authorization": f"Bearer {api_key}"}, timeout=timeout)
    r.raise_for_status()
    d = r.json()["data"]
    return {
        "usage_usd": float(d.get("usage") or 0.0),
        "limit_usd": d.get("limit"),
        "limit_remaining_usd": d.get("limit_remaining"),
        "expires_at": d.get("expires_at"),
    }


# --------------------------------------------------------------------------
# One classification call
# --------------------------------------------------------------------------
@dataclass
class JevResult:
    ok: bool
    jev_p_yes: float | None = None          # Noul: P(policy claim)
    jev_policy_claim: bool | None = None    # jev_p_yes >= NOUL_THRESHOLD
    jev_choice: str | None = None           # Choice: "yes" / "no"
    jev_choice_p_yes: float | None = None   # Choice: probabilities["yes"]
    jev_choice_confidence: float | None = None
    jev_input_tokens: int | None = None
    jev_output_tokens: int | None = None
    jev_cost_usd: float | None = None       # as reported by OpenRouter
    jev_latency_ms: float | None = None     # client-observed round trip (successful attempt)
    jev_attempts: int = 0
    jev_model: str | None = None            # versioned model id reported by the API
    jev_request_id: str | None = None
    jev_error: str | None = None
    jev_questions_hash: str = field(default_factory=questions_hash)
    jev_timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


_thread_local = threading.local()


def _session() -> requests.Session:
    s = getattr(_thread_local, "session", None)
    if s is None:
        s = requests.Session()
        _thread_local.session = s
    return s


def parse_answers(payload: dict[str, Any]) -> JevResult:
    answers = payload.get("answers", {})
    noul = answers.get("policy_claim", {})
    choice = answers.get("policy_claim_choice", {})
    usage = payload.get("usage", {}) or {}
    p_yes = noul.get("noul")
    if p_yes is None:
        return JevResult(ok=False, jev_error=f"no noul answer in response: {json.dumps(payload)[:300]}")
    p_yes = float(p_yes)
    probs = choice.get("probabilities") or {}
    in_tok = usage.get("input_tokens")
    out_tok = usage.get("output_tokens")
    cost = usage.get("cost")
    if cost is None and in_tok is not None:
        cost = in_tok * PRICE_USD_PER_INPUT_TOKEN + (out_tok or 0) * PRICE_USD_PER_OUTPUT_TOKEN
    return JevResult(
        ok=True,
        jev_p_yes=p_yes,
        jev_policy_claim=bool(p_yes >= NOUL_THRESHOLD),
        jev_choice=choice.get("choice"),
        jev_choice_p_yes=float(probs["yes"]) if "yes" in probs else None,
        jev_choice_confidence=float(choice["confidence"]) if choice.get("confidence") is not None else None,
        jev_input_tokens=in_tok,
        jev_output_tokens=out_tok,
        jev_cost_usd=float(cost) if cost is not None else None,
        jev_model=payload.get("model"),
        jev_request_id=payload.get("id"),
    )


def classify_abstract(
    abstract: str,
    *,
    api_key: str,
    timeout: float = 60.0,
    max_retries: int = 4,
    backoff_base: float = 1.5,
) -> JevResult:
    """Classify one abstract. Retries on 429/5xx/network errors with backoff."""
    body = build_request(abstract)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/dbann/policyclaims",
        "X-Title": "policyclaims-jev-replication",
    }
    last_error = "unknown"
    for attempt in range(1, max_retries + 1):
        t0 = time.perf_counter()
        try:
            resp = _session().post(OPENROUTER_DECISIONS_URL, headers=headers, json=body, timeout=timeout)
            latency_ms = (time.perf_counter() - t0) * 1000.0
        except requests.RequestException as exc:
            last_error = f"network: {exc}"
            time.sleep(backoff_base ** attempt)
            continue

        if resp.status_code == 200:
            try:
                result = parse_answers(resp.json())
            except ValueError as exc:
                last_error = f"bad json: {exc}"
                time.sleep(backoff_base ** attempt)
                continue
            result.jev_latency_ms = latency_ms
            result.jev_attempts = attempt
            return result

        if resp.status_code in (402, 403):
            # Out of credit / key over limit: never retry, surface immediately.
            raise BudgetExceeded(f"HTTP {resp.status_code}: {resp.text[:300]}")

        if resp.status_code in (429, 500, 502, 503, 504, 529):
            retry_after = resp.headers.get("Retry-After")
            wait = float(retry_after) if retry_after and retry_after.replace(".", "", 1).isdigit() else backoff_base ** attempt
            last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
            time.sleep(min(wait, 60.0))
            continue

        # Any other 4xx is a request problem; do not retry.
        return JevResult(ok=False, jev_attempts=attempt, jev_error=f"HTTP {resp.status_code}: {resp.text[:300]}")

    return JevResult(ok=False, jev_attempts=max_retries, jev_error=last_error)


# --------------------------------------------------------------------------
# Concurrent classification with a hard spending cap
# --------------------------------------------------------------------------
class BudgetGuard:
    """Tracks cumulative spend from API responses and stops the run at a cap.

    Also re-reads the key's usage from OpenRouter every `check_every` calls so a
    cap holds even if several runs share the same key.
    """

    def __init__(self, api_key: str, cap_usd: float, check_every: int = 500):
        self.api_key = api_key
        self.cap_usd = cap_usd
        self.check_every = check_every
        self.lock = threading.Lock()
        self.spent_this_run = 0.0
        self.calls = 0
        self.key_usage_at_start = get_key_usage(api_key)["usage_usd"]
        self.stop = threading.Event()

    def add(self, cost: float | None) -> None:
        with self.lock:
            self.calls += 1
            self.spent_this_run += float(cost or 0.0)
            over = self.spent_this_run > self.cap_usd
            do_check = self.calls % self.check_every == 0
        if do_check:
            try:
                usage = get_key_usage(self.api_key)
                remaining = usage.get("limit_remaining_usd")
                if remaining is not None and float(remaining) <= 0.05:
                    over = True
            except requests.RequestException:
                pass
        if over:
            self.stop.set()

    def check(self) -> None:
        if self.stop.is_set():
            raise BudgetExceeded(
                f"Budget cap of ${self.cap_usd:.2f} reached (spent ${self.spent_this_run:.4f} this run)."
            )


def estimate_cost_usd(abstracts: Iterable[str], overhead_tokens: int = 1300) -> float:
    """Rough pre-run estimate: ~4 characters per token for the abstract plus the
    fixed cost of the two questions (measured at ~1,300 tokens)."""
    total_tokens = sum(len(a or "") / 4.0 + overhead_tokens for a in abstracts)
    return total_tokens * PRICE_USD_PER_INPUT_TOKEN


def classify_many(
    items: list[tuple[Any, str]],
    *,
    api_key: str,
    workers: int = 8,
    budget_usd: float = 8.0,
    on_result: Callable[[Any, JevResult], None] | None = None,
    progress: bool = True,
) -> dict[Any, JevResult]:
    """Classify (key, abstract) pairs concurrently. Returns {key: JevResult}.

    `on_result` is called from the main thread after each completion (use it to
    checkpoint). Raises BudgetExceeded once `budget_usd` has been spent in this
    run; results obtained so far are still delivered through `on_result`.
    """
    import concurrent.futures

    try:
        from tqdm.auto import tqdm
    except ImportError:  # pragma: no cover
        tqdm = None

    guard = BudgetGuard(api_key, budget_usd)
    results: dict[Any, JevResult] = {}

    def work(key: Any, abstract: str) -> tuple[Any, JevResult]:
        guard.check()
        res = classify_abstract(abstract, api_key=api_key)
        guard.add(res.jev_cost_usd)
        return key, res

    bar = tqdm(total=len(items), desc="Jev classification", unit="abs") if (progress and tqdm) else None
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(work, k, a) for k, a in items]
        try:
            for fut in concurrent.futures.as_completed(futures):
                key, res = fut.result()
                results[key] = res
                if on_result:
                    on_result(key, res)
                if bar:
                    bar.update(1)
                    bar.set_postfix_str(f"spent=${guard.spent_this_run:.4f}")
        except BudgetExceeded:
            for f in futures:
                f.cancel()
            raise
        finally:
            if bar:
                bar.close()
    return results
