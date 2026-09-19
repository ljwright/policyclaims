#!/usr/bin/env python3
"""
Classifies each abstract as containing a policy claim using TypeSafe's Jev 1.13
via the OpenRouter Decisions API. This is the Jev counterpart of
3_run_llm_classification.py (DeepSeek); the question wording is defined in
jev_client.py.

Input:  any table with an `abstract` column - JSON (list of records), CSV, or
        XLSX (use --sheet). E.g. data/json_files/filtered/all_abstracts.json or
        table/gold_standard_30_march.xlsx.
Output: a CSV next to the input (or --output) with the input columns (minus the
        abstract, unless --keep-abstract) plus one jev_* column per result field,
        and a *_timing.json summary of wall time, throughput, latency
        percentiles, tokens and cost.

Runs are resumable: rows already scored in the output are skipped. Progress is
checkpointed every --save-every rows. A hard --budget-usd cap stops the run.

Examples
  python code/3b_run_jev_classification.py data/json_files/filtered/all_abstracts.json
  python code/3b_run_jev_classification.py table/gold_standard_30_march.xlsx --sheet in \
      --output concordance/jev_outputs/jev_gold_standard_run1.csv
  python code/3b_run_jev_classification.py data/json_files/filtered/all_abstracts.json --sample 1

Requires OPENROUTER_API_KEY in .env.

Run after: 2_filter_records.py
Run before: 4_build_analysis_dataset.py (pass --labels-csv <output> --label-col jev_policy_claim)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jev_client as jev  # noqa: E402

MIN_ABSTRACT_CHARS = 30  # same rule as 3_run_llm_classification.py
JEV_COLUMNS = list(jev.JevResult(ok=True).as_dict().keys())


def read_table(path: Path, sheet: str | None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "records" in data:
            data = data["records"]
        return pd.DataFrame.from_records(data)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path, sheet_name=sheet or 0)
    raise SystemExit(f"Unsupported input type: {path}")


def prepare(df: pd.DataFrame, abstract_col: str) -> pd.DataFrame:
    df = df.copy()
    df.insert(0, "source_row", np.arange(len(df)))  # stable key for resuming
    if abstract_col not in df.columns:
        raise SystemExit(f"Column '{abstract_col}' not found. Columns: {list(df.columns)}")
    n0 = len(df)
    df[abstract_col] = df[abstract_col].astype("string").fillna("").str.strip()
    df = df[df[abstract_col].str.len() >= MIN_ABSTRACT_CHARS].copy()
    print(f"Dropped {n0 - len(df)} rows with missing/short abstracts (< {MIN_ABSTRACT_CHARS} chars).")
    return df.reset_index(drop=True)


def save(df: pd.DataFrame, out_csv: Path, abstract_col: str, keep_abstract: bool) -> None:
    out = df if keep_abstract else df.drop(columns=[abstract_col], errors="ignore")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_csv.with_suffix(out_csv.suffix + ".tmp")
    out.to_csv(tmp, index=False)
    tmp.replace(out_csv)


def summarize_timing(df: pd.DataFrame, *, wall_s: float, n_new: int, workers: int, started: str) -> dict:
    ok = df[df["jev_p_yes"].notna()]
    lat = ok["jev_latency_ms"].dropna().astype(float)
    tokens_in = ok["jev_input_tokens"].dropna().astype(float)
    cost = ok["jev_cost_usd"].dropna().astype(float)
    pct = lambda q: float(np.percentile(lat, q)) if len(lat) else None
    return {
        "model": jev.JEV_MODEL,
        "model_version_reported": ok["jev_model"].dropna().mode().iat[0] if ok["jev_model"].notna().any() else None,
        "questions_hash": jev.questions_hash(),
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "workers": workers,
        "n_rows_in_output": int(len(df)),
        "n_classified_total": int(len(ok)),
        "n_classified_this_run": int(n_new),
        "n_failed": int(df["jev_error"].notna().sum()) if "jev_error" in df else 0,
        "wall_seconds_this_run": round(wall_s, 2),
        "throughput_abstracts_per_second": round(n_new / wall_s, 3) if wall_s > 0 and n_new else None,
        "latency_ms_p50": pct(50), "latency_ms_p90": pct(90), "latency_ms_p99": pct(99),
        "latency_ms_mean": float(lat.mean()) if len(lat) else None,
        "input_tokens_total": int(tokens_in.sum()), "input_tokens_mean": float(tokens_in.mean()) if len(tokens_in) else None,
        "cost_usd_total": float(cost.sum()), "cost_usd_mean": float(cost.mean()) if len(cost) else None,
        "share_policy_claim_noul": float(ok["jev_policy_claim"].astype(bool).mean()) if len(ok) else None,
        "share_policy_claim_choice": float((ok["jev_choice"] == "yes").mean()) if len(ok) else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_file", type=Path)
    ap.add_argument("--sheet", default=None, help="Sheet name for XLSX input (default: first sheet)")
    ap.add_argument("--abstract-col", default="abstract")
    ap.add_argument("--output", type=Path, default=None, help="Output CSV (default: <input>_JEV.csv next to input)")
    ap.add_argument("--run-name", default=None, help="Tag appended to the default output name, e.g. run2")
    ap.add_argument("-s", "--sample", type=float, default=None, metavar="PCT", help="Random PCT%% subsample")
    ap.add_argument("--limit", type=int, default=None, help="Only the first N rows (after sampling)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=8, help="Concurrent requests (TypeSafe allows 1,200 req/min)")
    ap.add_argument("--budget-usd", type=float, default=8.0, help="Hard stop once this much has been spent in this run")
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--keep-abstract", action="store_true", help="Keep the abstract text in the output CSV")
    ap.add_argument("--no-resume", action="store_true", help="Ignore an existing output file and start over")
    args = ap.parse_args()

    api_key = jev.get_api_key()
    in_path = args.input_file
    if args.output:
        out_csv = args.output
    else:
        tag = f"_{args.run_name}" if args.run_name else ""
        suffix = f"_sample{args.sample:g}_seed{args.seed}" if args.sample else ""
        out_csv = in_path.with_name(f"{in_path.stem}{suffix}_JEV{tag}.csv")
    timing_json = out_csv.with_name(out_csv.stem + "_timing.json")

    print("=" * 60)
    print("Jev 1.13 policy-claim classification (OpenRouter Decisions API)")
    print(f"Input:  {in_path}\nOutput: {out_csv}\nQuestions hash: {jev.questions_hash()}")
    print("=" * 60)

    df = prepare(read_table(in_path, args.sheet), args.abstract_col)
    if args.sample:
        df = df.sample(frac=args.sample / 100.0, random_state=args.seed).sort_values("source_row").reset_index(drop=True)
        print(f"Sampled {len(df)} rows ({args.sample}%).")
    if args.limit:
        df = df.head(args.limit).copy()

    for col in JEV_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    if out_csv.exists() and not args.no_resume:
        prev = pd.read_csv(out_csv)
        prev = prev[prev["jev_p_yes"].notna()]
        if len(prev):
            merged = df.drop(columns=JEV_COLUMNS).merge(prev[["source_row"] + JEV_COLUMNS], on="source_row", how="left")
            df = merged
            print(f"Resuming: {int(df['jev_p_yes'].notna().sum())} rows already classified in {out_csv.name}.")

    todo = df[df["jev_p_yes"].isna()]
    items = list(zip(todo["source_row"].tolist(), todo[args.abstract_col].tolist()))
    est = jev.estimate_cost_usd(todo[args.abstract_col].tolist())
    usage = jev.get_key_usage(api_key)
    print(f"To classify: {len(items)} abstracts. Estimated cost ~${est:.3f}. "
          f"Key usage so far ${usage['usage_usd']:.3f} of limit ${usage['limit_usd']} (remaining ${usage['limit_remaining_usd']}).")
    if est > args.budget_usd:
        raise SystemExit(f"Estimated cost ${est:.2f} exceeds --budget-usd {args.budget_usd:.2f}. Raise the cap or use --sample/--limit.")
    if not items:
        print("Nothing to do.")
        return

    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    t0 = time.perf_counter()
    done = {"n": 0}
    df = df.set_index("source_row", drop=False)

    def on_result(key, res: jev.JevResult) -> None:
        for k, v in res.as_dict().items():
            df.at[key, k] = v
        done["n"] += 1
        if done["n"] % args.save_every == 0:
            save(df.reset_index(drop=True), out_csv, args.abstract_col, args.keep_abstract)

    try:
        jev.classify_many(items, api_key=api_key, workers=args.workers, budget_usd=args.budget_usd, on_result=on_result)
    except jev.BudgetExceeded as exc:
        print(f"\nSTOPPED: {exc}")
    finally:
        wall = time.perf_counter() - t0
        df = df.reset_index(drop=True)
        save(df, out_csv, args.abstract_col, args.keep_abstract)
        timing = summarize_timing(df, wall_s=wall, n_new=done["n"], workers=args.workers, started=started)
        timing_json.write_text(json.dumps(timing, indent=2))

    n_ok = int(df["jev_p_yes"].notna().sum())
    n_fail = int(df["jev_error"].notna().sum())
    print(f"\nClassified {n_ok}/{len(df)} rows ({n_fail} failed) in {wall:.1f}s "
          f"({done['n']/wall:.2f} abstracts/s with {args.workers} workers).")
    print(f"Spent this run: ${df['jev_cost_usd'].astype(float).sum():.4f}. "
          f"Policy-claim share (Noul): {timing['share_policy_claim_noul']}")
    print(f"Wrote {out_csv} and {timing_json}")


if __name__ == "__main__":
    main()
