#!/usr/bin/env python3
"""
Measures how fast Jev 1.13 (via OpenRouter) classifies abstracts, and what it
costs, at several concurrency levels; extrapolates to the full corpus and
compares with the DeepSeek run reported in the README (~10 hours and ~$3 for
45,807 abstracts with 5 concurrent workers).

For each --workers value the same --n abstracts are classified once (no
caching), recording wall time, throughput, client-observed latency
percentiles, input tokens and cost as reported by OpenRouter.

Latency includes the network round trip from this machine to OpenRouter and on
to TypeSafe, so absolute numbers depend on where the script is run.

Output: table/jev_speed_benchmark.csv and table/jev_speed_benchmark.md

Example
  python code/11_jev_speed_benchmark.py --n 100 --workers 1 4 8 16
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

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "table" / "manual_review_by_design_50_each.csv"
N_CORPUS = 45_807  # analytic sample in the paper
DEEPSEEK_REFERENCE = {  # from README.md "Cost and time to process"
    "model": "deepseek-chat (DeepSeek V3.1)",
    "workers": 5,
    "hours_for_corpus": 10.0,
    "cost_usd_for_corpus": 3.0,
}


def bench(abstracts: list[str], workers: int, api_key: str, budget_usd: float) -> dict:
    items = list(enumerate(abstracts))
    t0 = time.perf_counter()
    results = jev.classify_many(items, api_key=api_key, workers=workers, budget_usd=budget_usd, progress=False)
    wall = time.perf_counter() - t0
    ok = [r for r in results.values() if r.ok]
    lat = np.array([r.jev_latency_ms for r in ok], dtype=float)
    tok = np.array([r.jev_input_tokens or 0 for r in ok], dtype=float)
    cost = float(sum(r.jev_cost_usd or 0.0 for r in ok))
    n = len(ok)
    thr = n / wall if wall > 0 else float("nan")
    return {
        "workers": workers,
        "n_abstracts": n,
        "n_failed": len(results) - n,
        "wall_seconds": round(wall, 2),
        "throughput_per_second": round(thr, 3),
        "latency_ms_p50": round(float(np.percentile(lat, 50)), 1) if n else None,
        "latency_ms_p90": round(float(np.percentile(lat, 90)), 1) if n else None,
        "latency_ms_p99": round(float(np.percentile(lat, 99)), 1) if n else None,
        "latency_ms_mean": round(float(lat.mean()), 1) if n else None,
        "input_tokens_mean": round(float(tok.mean()), 1) if n else None,
        "cost_usd_total": round(cost, 5),
        "cost_usd_per_abstract": round(cost / n, 7) if n else None,
        "est_hours_for_corpus": round(N_CORPUS / thr / 3600.0, 2) if thr else None,
        "est_cost_usd_for_corpus": round(cost / n * N_CORPUS, 2) if n else None,
    }


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in r.tolist()) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Table with an 'abstract' column")
    ap.add_argument("--sheet", default=None)
    ap.add_argument("--n", type=int, default=100, help="Abstracts per concurrency level")
    ap.add_argument("--workers", type=int, nargs="+", default=[1, 4, 8, 16])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--budget-usd", type=float, default=0.5)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "table")
    args = ap.parse_args()

    api_key = jev.get_api_key()
    if args.input.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(args.input, sheet_name=args.sheet or 0)
    elif args.input.suffix.lower() == ".json":
        df = pd.DataFrame.from_records(json.loads(args.input.read_text(encoding="utf-8")))
    else:
        df = pd.read_csv(args.input)
    df = df[df["abstract"].astype(str).str.strip().str.len() >= 30]
    abstracts = df.sample(n=min(args.n, len(df)), random_state=args.seed)["abstract"].astype(str).tolist()
    print(f"Benchmarking {len(abstracts)} abstracts at workers={args.workers} (model {jev.JEV_MODEL})")

    rows = []
    for w in args.workers:
        print(f"  workers={w} ...", end="", flush=True)
        r = bench(abstracts, w, api_key, args.budget_usd)
        rows.append(r)
        print(f" {r['throughput_per_second']} abs/s, p50 {r['latency_ms_p50']} ms, ${r['cost_usd_total']}")
        time.sleep(2)

    res = pd.DataFrame(rows)
    res.insert(0, "model", jev.JEV_MODEL)
    res["run_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "jev_speed_benchmark.csv"
    md_path = args.out_dir / "jev_speed_benchmark.md"
    res.to_csv(csv_path, index=False)

    best = res.loc[res["throughput_per_second"].idxmax()]
    ds = DEEPSEEK_REFERENCE
    ds_thr = N_CORPUS / (ds["hours_for_corpus"] * 3600)
    lines = [
        "# Jev 1.13 speed benchmark",
        "",
        f"Run {res['run_utc'].iat[0]} from this machine via OpenRouter; {len(abstracts)} abstracts per setting "
        f"(seed {args.seed}, source `{args.input.relative_to(ROOT) if args.input.is_relative_to(ROOT) else args.input}`).",
        "Latency is the client-observed round trip and includes network time. Cost is as reported by OpenRouter "
        f"(input tokens only, ${jev.PRICE_USD_PER_INPUT_TOKEN*1e6:.3f}/M).",
        "",
        md_table(res.drop(columns=["run_utc"])),
        "",
        "## Extrapolation to the full corpus (n = 45,807)",
        "",
        "| model | workers | throughput (abstracts/s) | est. hours | est. cost (USD) |",
        "|---|---|---|---|---|",
        f"| {ds['model']} (README) | {ds['workers']} | {ds_thr:.2f} | {ds['hours_for_corpus']:.1f} | {ds['cost_usd_for_corpus']:.2f} |",
        f"| {jev.JEV_MODEL} (best setting here) | {int(best['workers'])} | {best['throughput_per_second']:.2f} | "
        f"{best['est_hours_for_corpus']:.2f} | {best['est_cost_usd_for_corpus']:.2f} |",
        "",
        f"Speed-up over the DeepSeek run: about {best['throughput_per_second']/ds_thr:.0f}x at {int(best['workers'])} workers. "
        "TypeSafe's published limit is 1,200 requests/minute (20/s), so throughput saturates around there.",
    ]
    md_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {csv_path} and {md_path}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
