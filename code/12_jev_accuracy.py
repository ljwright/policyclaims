#!/usr/bin/env python3
"""
Evaluates the accuracy of Jev 1.13 policy-claim labels against (a) the DeepSeek
labels used in the paper and (b) the human reviews stored in table/.

Comparisons (each reports n, % agreement, Cohen's kappa with bootstrap 95% CI,
sensitivity, specificity, PPV, NPV, and both prevalences):

  1. Development set   table/manual_review_by_design_50_each.csv (n=400, DeepSeek
                       labels only). Used to sanity-check the question wording.
  2. Gold standard     table/gold_standard_30_march.xlsx, sheet `in` (n=204 with
                       abstracts): adjudicated gold standard, gold standard with
                       exclusions, and the individual reviewers (DB, EC, MW).
                       DeepSeek is scored against the same references.
  3. Stratified 400    table/supp_stratified_sample_400_blinded_db.xlsx (blinded
                       DB/EC reviews, adjudicated as in 8_llm_validation.ipynb)
                       plus claim rates by period (Supplementary Table 8 style).
  4. Test-retest       repeated Jev runs on the gold-standard abstracts.
  5. Full corpus       if data/json_files/filtered/all_abstracts_JEV.csv exists:
                       Jev vs DeepSeek (derived_data/policy_claims_minimal.csv)
                       matched on DOI, overall and by year / journal / country,
                       plus a Table 1 style replication.
  6. Speed             any *_timing.json and table/jev_speed_benchmark.csv.

Jev labels: `noul` = P(policy claim) >= 0.5 (primary); `choice` = yes/no Choice.

Inputs are the outputs of 3b_run_jev_classification.py (see run_jev_validation.sh).
Outputs: table/jev_accuracy_summary.csv, table/jev_accuracy_report.md,
         table/jev_test_retest.csv, table/jev_mismatches_gold.csv,
         table/jev_claim_rate_by_period_400.csv, table/jev_corpus_by_*.csv,
         table/jev_table1_replication.csv, figures/jev_*.png

Run after: 3b_run_jev_classification.py (validation runs and, optionally, the corpus)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jev_client as jev  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "table"
FIG = ROOT / "figures"
JEV_OUT = ROOT / "concordance" / "jev_outputs"
DERIVED = ROOT / "derived_data" / "policy_claims_minimal.csv"
CORPUS_JEV = ROOT / "data" / "json_files" / "filtered" / "all_abstracts_JEV.csv"
PERIODS = {"1990-1999": (1990, 1999), "2000-2009": (2000, 2009), "2010-2019": (2010, 2019), "2020-2024": (2020, 2024)}
DEEPSEEK_RETEST_REFERENCE = {"run1 vs run2": 0.905, "run1 vs run3": 0.930, "run2 vs run3": 0.978}  # 8_llm_validation.ipynb


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def norm_bool(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    s = str(x).strip().lower()
    if s in {"1", "1.0", "true", "t", "yes", "y"}:
        return True
    if s in {"0", "0.0", "false", "f", "no", "n"}:
        return False
    return np.nan


def norm_doi(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    s = s.replace("https://doi.org/", "").replace("http://doi.org/", "").replace("http://dx.doi.org/", "")
    return s if s and s != "nan" else np.nan


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return centre - half, centre + half


def chi2_2xk_p(successes, totals) -> float:
    from scipy.stats import chi2
    s = np.asarray(successes, float); t = np.asarray(totals, float); f = t - s
    es = t * s.sum() / t.sum(); ef = t * f.sum() / t.sum()
    stat = (((s - es) ** 2) / es).sum() + (((f - ef) ** 2) / ef).sum()
    return float(chi2.sf(stat, len(t) - 1))


def metrics(ref, pred, *, label: str, n_boot: int = 1000, seed: int = 42) -> dict:
    """Agreement statistics treating `ref` as the reference standard."""
    m = pd.DataFrame({"ref": ref, "pred": pred}).dropna()
    r = m["ref"].astype(bool).to_numpy(); p = m["pred"].astype(bool).to_numpy()
    n = len(m)
    if n == 0:
        return {"comparison": label, "n": 0}
    tp = int((r & p).sum()); tn = int((~r & ~p).sum()); fp = int((~r & p).sum()); fn = int((r & ~p).sum())
    agree = (r == p).mean()
    kappa = cohen_kappa_score(r, p) if (r.any() and (~r).any()) or (p.any() and (~p).any()) else np.nan
    rng = np.random.default_rng(seed)
    idx = np.arange(n); ag_bs, k_bs = [], []
    for _ in range(n_boot):
        s = rng.choice(idx, n, replace=True)
        ag_bs.append((r[s] == p[s]).mean())
        k_bs.append(cohen_kappa_score(r[s], p[s]) if len(set(r[s])) > 1 or len(set(p[s])) > 1 else np.nan)
    ci = lambda v: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5)))
    a_ci, k_ci = ci(ag_bs), ci(k_bs)
    div = lambda a, b: a / b if b else np.nan
    return {
        "comparison": label, "n": n,
        "agreement_pct": round(100 * agree, 1), "agreement_ci": f"({100*a_ci[0]:.1f}, {100*a_ci[1]:.1f})",
        "kappa": round(float(kappa), 3), "kappa_ci": f"({k_ci[0]:.3f}, {k_ci[1]:.3f})",
        "sensitivity": round(div(tp, tp + fn), 3), "specificity": round(div(tn, tn + fp), 3),
        "ppv": round(div(tp, tp + fp), 3), "npv": round(div(tn, tn + fn), 3),
        "prevalence_ref_pct": round(100 * r.mean(), 1), "prevalence_pred_pct": round(100 * p.mean(), 1),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def prob_metrics(ref, p) -> dict:
    m = pd.DataFrame({"ref": ref, "p": p}).dropna()
    r = m["ref"].astype(bool).to_numpy(); q = m["p"].astype(float).to_numpy()
    if len(m) == 0 or r.all() or (~r).all():
        return {"auc": np.nan, "brier": np.nan, "n": len(m)}
    return {"auc": round(float(roc_auc_score(r, q)), 3), "brier": round(float(np.mean((q - r) ** 2)), 3), "n": len(m)}


def calibration_table(ref, p, bins=(0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0)) -> pd.DataFrame:
    m = pd.DataFrame({"ref": ref, "p": p}).dropna()
    m["bin"] = pd.cut(m["p"].astype(float), bins=list(bins), include_lowest=True)
    g = m.groupby("bin", observed=True)["ref"].agg(n="size", observed_rate=lambda s: round(100 * s.astype(bool).mean(), 1))
    g["mean_p_yes_pct"] = m.groupby("bin", observed=True)["p"].mean().round(3) * 100
    return g.reset_index().rename(columns={"bin": "p_yes_bin"})


def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    if df is None or len(df) == 0:
        return "_(no data)_"
    cols = list(df.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        cells = []
        for v in row.tolist():
            if isinstance(v, float) and not pd.isna(v):
                cells.append(floatfmt.format(v) if abs(v - round(v)) > 1e-9 else str(int(round(v))))
            else:
                cells.append("" if (v is None or (isinstance(v, float) and pd.isna(v))) else str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def load_jev(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"[skip] {path.relative_to(ROOT)} not found")
        return None
    d = pd.read_csv(path)
    d = d[d["jev_p_yes"].notna()].copy()
    d["jev_noul"] = d["jev_p_yes"].astype(float) >= jev.NOUL_THRESHOLD
    d["jev_choice_yes"] = d["jev_choice"].astype(str).str.lower().eq("yes")
    return d


# ----------------------------------------------------------------------------
# sections
# ----------------------------------------------------------------------------
def section_dev(summary: list, report: list) -> None:
    d = load_jev(JEV_OUT / "jev_design400.csv")
    if d is None:
        return
    ref = d["llm_policy_claim"].map(norm_bool)
    rows = [metrics(ref, d["jev_noul"], label="dev400: Jev noul vs DeepSeek"),
            metrics(ref, d["jev_choice_yes"], label="dev400: Jev choice vs DeepSeek")]
    summary.extend(rows)
    report += ["## 1. Development set: design-stratified sample (n=400), reference = DeepSeek", "",
               "Used only to check the question wording; no human labels here.", "",
               md_table(pd.DataFrame(rows).drop(columns=["tp", "fp", "fn", "tn"])), "",
               "Calibration of Jev's P(policy claim) against the DeepSeek label:", "",
               md_table(calibration_table(ref, d["jev_p_yes"])), ""]


def section_gold(summary: list, report: list) -> pd.DataFrame | None:
    wb = pd.read_excel(TABLE / "gold_standard_30_march.xlsx", sheet_name="in")
    wb.insert(0, "source_row", np.arange(len(wb)))
    d = load_jev(JEV_OUT / "jev_gold_standard_run1.csv")
    if d is None:
        return None
    g = wb.merge(d[["source_row", "jev_p_yes", "jev_noul", "jev_choice_yes", "jev_choice_confidence"]], on="source_row", how="inner")
    refs = [("agreed_gold_standard", "adjudicated gold standard"),
            ("agreed_gold_standard_with_exclusions", "gold standard excl. non-empirical/truncated"),
            ("DB review", "reviewer DB (author 1)"), ("EC re-review", "reviewer EC (author 5)"), ("MW review", "reviewer MW (author 2)")]
    ds = g["llm_policy_claim"].map(norm_bool)
    rows = []
    for col, nice in refs:
        r = g[col].map(norm_bool)
        rows.append(metrics(r, ds, label=f"gold: DeepSeek vs {nice}"))
        rows.append(metrics(r, g["jev_noul"], label=f"gold: Jev noul vs {nice}"))
        rows.append(metrics(r, g["jev_choice_yes"], label=f"gold: Jev choice vs {nice}"))
    rows.append(metrics(ds, g["jev_noul"], label="gold: Jev noul vs DeepSeek"))
    rows.append(metrics(ds, g["jev_choice_yes"], label="gold: Jev choice vs DeepSeek"))
    summary.extend(rows)

    gold = g["agreed_gold_standard"].map(norm_bool)
    pm = prob_metrics(gold, g["jev_p_yes"])
    report += ["## 2. Gold standard workbook (n=204 abstracts with human review)", "",
               "The paper reports DeepSeek vs the adjudicated gold standard: kappa 0.80, sensitivity 76.8%, specificity 98.6% (n=204).", "",
               md_table(pd.DataFrame(rows).drop(columns=["tp", "fp", "fn", "tn"])), "",
               f"Jev P(policy claim) vs the adjudicated gold standard: AUC = {pm['auc']}, Brier score = {pm['brier']} (n={pm['n']}).", "",
               "Calibration against the adjudicated gold standard:", "", md_table(calibration_table(gold, g["jev_p_yes"])), ""]

    mism = g[(gold.notna()) & (gold != g["jev_noul"])].copy()
    mism["gold"] = gold[mism.index]
    mism["deepseek"] = ds[mism.index]
    mism["abstract_end"] = mism["abstract"].astype(str).str.strip().str[-400:]
    mism[["scopus_id", "doi", "title", "gold", "deepseek", "jev_noul", "jev_p_yes", "jev_choice_yes", "abstract_end"]].to_csv(TABLE / "jev_mismatches_gold.csv", index=False)

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 3.6))
        for val, colr, lab in [(False, "#4C72B0", "gold: no policy claim"), (True, "#DD8452", "gold: policy claim")]:
            ax.hist(g.loc[gold == val, "jev_p_yes"].astype(float), bins=20, range=(0, 1), alpha=0.7, color=colr, label=lab)
        ax.axvline(jev.NOUL_THRESHOLD, color="k", ls="--", lw=1); ax.set_xlabel("Jev P(policy claim)"); ax.set_ylabel("abstracts"); ax.legend(frameon=False)
        ax.set_title("Jev probability by adjudicated human label (n=204)", fontsize=10); fig.tight_layout()
        FIG.mkdir(exist_ok=True); fig.savefig(FIG / "jev_p_yes_by_gold_label.png", dpi=150); plt.close(fig)
    except Exception as exc:  # pragma: no cover
        print(f"[warn] figure skipped: {exc}")
    return g


def section_strat400(summary: list, report: list) -> None:
    blinded = pd.read_excel(TABLE / "supp_stratified_sample_400_blinded_db.xlsx")
    internal = pd.read_excel(TABLE / "supp_stratified_sample_400_internal.xlsx")[["review_id", "publication_year", "llm_policy_claim"]]
    d = load_jev(JEV_OUT / "jev_stratified400.csv")
    if d is None:
        return
    s = blinded.merge(internal, on="review_id", how="left").merge(
        d[["review_id", "jev_p_yes", "jev_noul", "jev_choice_yes"]], on="review_id", how="inner")
    for c in ["DB_policy_claim", "EC_policy_claim", "db_double_checked", "llm_policy_claim"]:
        s[c] = s[c].map(norm_bool)
    # adjudication rule from 8_llm_validation.ipynb: DB label where DB and EC agree, else DB's double-check
    s["manual_final"] = np.where(s["DB_policy_claim"] == s["EC_policy_claim"], s["DB_policy_claim"], s["db_double_checked"])
    s["manual_final"] = s["manual_final"].map(norm_bool)
    rows = []
    for col, nice in [("manual_final", "adjudicated manual label"), ("DB_policy_claim", "reviewer DB (blinded)"), ("EC_policy_claim", "reviewer EC (blinded)")]:
        rows.append(metrics(s[col], s["llm_policy_claim"], label=f"strat400: DeepSeek vs {nice}"))
        rows.append(metrics(s[col], s["jev_noul"], label=f"strat400: Jev noul vs {nice}"))
        rows.append(metrics(s[col], s["jev_choice_yes"], label=f"strat400: Jev choice vs {nice}"))
    rows.append(metrics(s["DB_policy_claim"], s["EC_policy_claim"], label="strat400: reviewer EC vs reviewer DB (human-human)"))
    rows.append(metrics(s["llm_policy_claim"], s["jev_noul"], label="strat400: Jev noul vs DeepSeek"))
    summary.extend(rows)
    pm = prob_metrics(s["manual_final"], s["jev_p_yes"])

    per_rows, succ = [], {"manual": [], "deepseek": [], "jev": []}; tot = []
    for label, (a, b) in PERIODS.items():
        sub = s[(s["publication_year"] >= a) & (s["publication_year"] <= b)]
        n = len(sub); tot.append(n); row = {"period": label, "n": n}
        for key, col in [("manual", "manual_final"), ("deepseek", "llm_policy_claim"), ("jev", "jev_noul")]:
            k = int(sub[col].astype(bool).sum()); lo, hi = wilson_ci(k, n); succ[key].append(k)
            row[f"{key}_rate_pct"] = round(100 * k / n, 1) if n else np.nan
            row[f"{key}_95ci"] = f"({100*lo:.1f}, {100*hi:.1f})"
        per_rows.append(row)
    per = pd.DataFrame(per_rows)
    pvals = {k: chi2_2xk_p(v, tot) for k, v in succ.items()}
    per.to_csv(TABLE / "jev_claim_rate_by_period_400.csv", index=False)
    report += ["## 3. Blinded stratified sample (n=400, 1990-2024), reviewers DB and EC", "",
               "Adjudicated label = DB where DB and EC agree, otherwise DB's double-check (rule from 8_llm_validation.ipynb).", "",
               md_table(pd.DataFrame(rows).drop(columns=["tp", "fp", "fn", "tn"])), "",
               f"Jev P(policy claim) vs the adjudicated manual label: AUC = {pm['auc']}, Brier = {pm['brier']}.", "",
               "Policy-claim rate by period (Supplementary Table 8 style; Wilson 95% CIs; chi-square p across periods: "
               + ", ".join(f"{k} p={'<0.001' if v < 0.001 else f'{v:.3f}'}" for k, v in pvals.items()) + "):", "",
               md_table(per), ""]


def section_retest(report: list) -> None:
    runs = sorted(JEV_OUT.glob("jev_gold_standard_run*.csv"))
    runs = [r for r in runs if "timing" not in r.name]
    if len(runs) < 2:
        print("[skip] fewer than two gold-standard runs for test-retest")
        return
    frames = {}
    for r in runs:
        d = load_jev(r)
        frames[r.stem.replace("jev_gold_standard_", "")] = d.set_index("source_row")[["jev_p_yes", "jev_noul"]]
    rows = []
    for a, b in combinations(frames, 2):
        m = frames[a].join(frames[b], lsuffix="_a", rsuffix="_b", how="inner")
        rows.append({"comparison": f"{a} vs {b}", "n": len(m),
                     "kappa": round(cohen_kappa_score(m["jev_noul_a"], m["jev_noul_b"]), 3),
                     "agreement_pct": round(100 * (m["jev_noul_a"] == m["jev_noul_b"]).mean(), 1),
                     "mean_abs_diff_p_yes": round(float((m["jev_p_yes_a"] - m["jev_p_yes_b"]).abs().mean()), 4),
                     "max_abs_diff_p_yes": round(float((m["jev_p_yes_a"] - m["jev_p_yes_b"]).abs().max()), 3),
                     "deepseek_kappa_reference": DEEPSEEK_RETEST_REFERENCE.get(f"{a} vs {b}", "")})
    rt = pd.DataFrame(rows); rt.to_csv(TABLE / "jev_test_retest.csv", index=False)
    report += ["## 4. Test-retest reliability (repeated Jev runs on the gold-standard abstracts)", "",
               "DeepSeek reference (paper): kappa 0.90-0.98 across three runs of 200 abstracts at temperature 0.1.", "",
               md_table(rt), ""]


def section_corpus(summary: list, report: list) -> None:
    if not CORPUS_JEV.exists():
        report += ["## 5. Full corpus", "", f"_Not yet run: `{CORPUS_JEV.relative_to(ROOT)}` not found. "
                   "Fetch the abstracts (1_fetch_abstracts.py, 2_filter_records.py) and run 3b_run_jev_classification.py._", ""]
        return
    jv = load_jev(CORPUS_JEV)
    dv = pd.read_csv(DERIVED)
    jv["doi_n"] = jv["doi"].map(norm_doi); dv["doi_n"] = dv["doi"].map(norm_doi)
    key = lambda df: (df["title"].astype(str).str.strip().str.lower() + "|" + df["journal"].astype(str).str.strip().str.lower() + "|" + df["publication_year"].astype(str))
    jv["tkey"] = key(jv); dv["tkey"] = key(dv)
    dv_doi = dv.dropna(subset=["doi_n"]).drop_duplicates("doi_n")
    m = jv.merge(dv_doi[["doi_n", "llm_policy_claim", "design_combined"]], on="doi_n", how="left")
    miss = m["llm_policy_claim"].isna()
    if miss.any():
        dv_t = dv.drop_duplicates("tkey")[["tkey", "llm_policy_claim", "design_combined"]].rename(columns=lambda c: c if c == "tkey" else c + "_t")
        m = m.merge(dv_t, on="tkey", how="left")
        m.loc[miss, "llm_policy_claim"] = m.loc[miss, "llm_policy_claim_t"]
        m.loc[miss, "design_combined"] = m.loc[miss, "design_combined_t"]
        m = m.drop(columns=["llm_policy_claim_t", "design_combined_t"])
    m["deepseek"] = m["llm_policy_claim"].map(norm_bool)
    matched = m[m["deepseek"].notna()].copy()
    n_unmatched = int(m["deepseek"].isna().sum())
    rows = [metrics(matched["deepseek"], matched["jev_noul"], label="corpus: Jev noul vs DeepSeek", n_boot=200),
            metrics(matched["deepseek"], matched["jev_choice_yes"], label="corpus: Jev choice vs DeepSeek", n_boot=200)]
    summary.extend(rows)

    def rates(df, by):
        g = df.groupby(by).agg(n=("deepseek", "size"), deepseek_rate_pct=("deepseek", lambda s: 100 * s.astype(bool).mean()),
                               jev_rate_pct=("jev_noul", lambda s: 100 * s.mean()), agreement_pct=("agree", "mean"),
                               kappa=("agree", lambda s: np.nan)).reset_index()
        for i, r in g.iterrows():
            sub = df[df[by] == r[by]]
            g.loc[i, "kappa"] = cohen_kappa_score(sub["deepseek"].astype(bool), sub["jev_noul"]) if sub["deepseek"].nunique() > 1 else np.nan
        return g.round({"deepseek_rate_pct": 1, "jev_rate_pct": 1, "agreement_pct": 1, "kappa": 3})

    matched["agree"] = 100 * (matched["deepseek"].astype(bool) == matched["jev_noul"])
    matched["period"] = pd.cut(matched["publication_year"].astype(int), [1989, 1999, 2009, 2019, 2024], labels=list(PERIODS))
    matched["country"] = matched["corresponding_author_country"].astype(str).str.upper().str.strip()
    by_year = rates(matched, "publication_year"); by_year.to_csv(TABLE / "jev_corpus_by_year.csv", index=False)
    by_journal = rates(matched, "journal").sort_values("deepseek_rate_pct"); by_journal.to_csv(TABLE / "jev_corpus_by_journal.csv", index=False)
    top15 = matched["country"].value_counts().head(15).index
    by_country = rates(matched[matched["country"].isin(top15)], "country").sort_values("deepseek_rate_pct"); by_country.to_csv(TABLE / "jev_corpus_by_country.csv", index=False)
    by_design = rates(matched, "design_combined"); by_design.to_csv(TABLE / "jev_corpus_by_design.csv", index=False)

    # Table 1 replication: % policy claims by period for DeepSeek and Jev
    def t1_block(df, by, name):
        out = []
        for lvl, sub in ([("All abstracts", df)] if by is None else df.groupby(by)):
            row = {"row": f"{name}: {lvl}" if by else lvl}
            for per in list(PERIODS) + ["All years"]:
                ss = sub if per == "All years" else sub[sub["period"] == per]
                row[f"DeepSeek {per}"] = round(100 * ss["deepseek"].astype(bool).mean(), 1) if len(ss) else np.nan
                row[f"Jev {per}"] = round(100 * ss["jev_noul"].mean(), 1) if len(ss) else np.nan
            out.append(row)
        return out
    t1 = pd.DataFrame(t1_block(matched, None, "") + t1_block(matched, "journal", "Journal") + t1_block(matched[matched["country"].isin(top15)], "country", "Country"))
    t1.to_csv(TABLE / "jev_table1_replication.csv", index=False)

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 3.8))
        ax.plot(by_year["publication_year"], by_year["deepseek_rate_pct"], marker="o", ms=3, label="DeepSeek V3.1 (paper)")
        ax.plot(by_year["publication_year"], by_year["jev_rate_pct"], marker="s", ms=3, label="Jev 1.13")
        ax.set_ylabel("% abstracts with a policy claim"); ax.set_xlabel("publication year"); ax.legend(frameon=False)
        ax.set_title("Policy-claim rate by year: DeepSeek vs Jev", fontsize=10); fig.tight_layout()
        FIG.mkdir(exist_ok=True); fig.savefig(FIG / "jev_vs_deepseek_trend.png", dpi=150); plt.close(fig)
    except Exception as exc:  # pragma: no cover
        print(f"[warn] figure skipped: {exc}")

    report += ["## 5. Full corpus: Jev vs DeepSeek", "",
               f"Matched {len(matched):,} of {len(m):,} Jev-classified abstracts to the derived dataset by DOI (fallback title+journal+year); {n_unmatched:,} unmatched.", "",
               md_table(pd.DataFrame(rows).drop(columns=["tp", "fp", "fn", "tn"])), "",
               "### By period (Table 1 style, % with policy claim)", "", md_table(t1[t1["row"].str.startswith("All") | t1["row"].str.startswith("Journal")]), "",
               "### By journal", "", md_table(by_journal), "", "### Top-15 countries (first author)", "", md_table(by_country), "",
               "### By study design", "", md_table(by_design), "",
               "![trend](../figures/jev_vs_deepseek_trend.png)", ""]


def section_speed(report: list) -> None:
    rows = []
    for p in sorted(list(JEV_OUT.glob("*_timing.json")) + list((ROOT / "data").rglob("*_timing.json"))):
        t = json.loads(p.read_text())
        rows.append({"run": p.name.replace("_timing.json", ""), "n": t.get("n_classified_this_run"), "workers": t.get("workers"),
                     "wall_s": t.get("wall_seconds_this_run"), "abstracts_per_s": t.get("throughput_abstracts_per_second"),
                     "latency_p50_ms": round(t["latency_ms_p50"], 0) if t.get("latency_ms_p50") else None,
                     "latency_p90_ms": round(t["latency_ms_p90"], 0) if t.get("latency_ms_p90") else None,
                     "cost_usd": round(t.get("cost_usd_total") or 0, 4), "tokens_per_abstract": round(t.get("input_tokens_mean") or 0)})
    report += ["## 6. Speed and cost", "", "DeepSeek reference (README): ~10 hours and ~$3 for 45,807 abstracts (5 concurrent requests), i.e. ~1.3 abstracts/s.", ""]
    if rows:
        report += ["Jev runs recorded by 3b_run_jev_classification.py:", "", md_table(pd.DataFrame(rows)), ""]
    bench = TABLE / "jev_speed_benchmark.md"
    if bench.exists():
        report += ["Concurrency benchmark (11_jev_speed_benchmark.py):", ""] + bench.read_text().splitlines()[2:] + [""]


def main() -> None:
    global CORPUS_JEV
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=TABLE / "jev_accuracy_report.md")
    ap.add_argument("--corpus", type=Path, default=CORPUS_JEV, help="Jev output for the full corpus (default: %(default)s)")
    args = ap.parse_args()
    CORPUS_JEV = args.corpus
    TABLE.mkdir(exist_ok=True)
    summary: list[dict] = []
    report = ["# Jev 1.13 vs DeepSeek V3.1 and human review: policy-claim classification accuracy", "",
              f"Generated by `code/12_jev_accuracy.py`. Jev question set `{jev.QUESTIONS_VERSION}` (hash `{jev.questions_hash()}`), "
              f"label rule: P(policy claim) >= {jev.NOUL_THRESHOLD} (`noul`); `choice` = yes/no Choice question.",
              "Kappa CIs are 1,000-resample bootstraps (seed 42), matching 7_concordance.py.", ""]
    section_dev(summary, report)
    section_gold(summary, report)
    section_strat400(summary, report)
    section_retest(report)
    section_corpus(summary, report)
    section_speed(report)
    pd.DataFrame(summary).to_csv(TABLE / "jev_accuracy_summary.csv", index=False)
    args.out.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote {args.out} and table/jev_accuracy_summary.csv ({len(summary)} comparisons)")


if __name__ == "__main__":
    main()
