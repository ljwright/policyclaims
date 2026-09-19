#!/usr/bin/env python3
"""
Alternative to 1_fetch_abstracts.py when the Scopus COMPLETE view (abstracts) is
not available: fills in abstracts for the paper's analytic sample from PubMed.

The public derived dataset (derived_data/policy_claims_minimal.csv) already holds
the metadata for every analysed record (DOI, title, journal, year, keywords,
country, DeepSeek label) but not the abstracts, which Scopus does not allow us to
redistribute. All ten journals are indexed in MEDLINE, so their abstracts can be
fetched from PubMed with the free NCBI E-utilities and matched back to the
derived records by DOI, then by normalised title.

Output: data/json_files/filtered/all_abstracts_pubmed.json - one record per
matched derived row with the same fields as the Scopus pipeline (scopus_id is
blank; pmid, abstract_source and match_method are added), ready for
3b_run_jev_classification.py. Unmatched rows are listed in
data/json_files/pubmed/unmatched.csv. Raw PubMed records are cached per journal
in data/json_files/pubmed/ so re-runs are fast.

Note: PubMed abstracts are the publisher-supplied text, like Scopus's, but may
differ in section labels and trailing copyright notices.

Optional: NCBI_API_KEY in .env raises the rate limit from 3 to 10 requests/s.
Usage: python code/1b_fetch_abstracts_pubmed.py [--journals am_j_epidemiol prev_med] [--years 1990-2024]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=True)
DERIVED = ROOT / "derived_data" / "policy_claims_minimal.csv"
CACHE_DIR = ROOT / "data" / "json_files" / "pubmed"
OUT_JSON = ROOT / "data" / "json_files" / "filtered" / "all_abstracts_pubmed.json"
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
NCBI_API_KEY = os.getenv("NCBI_API_KEY", "").strip()
PAUSE = 0.11 if NCBI_API_KEY else 0.35  # 10/s with a key, 3/s without
BATCH = 200

# ISSN -> short name -> journal name as it appears in the derived dataset
JOURNALS = {
    "0300-5771": ("int_j_epidemiol", "International Journal Of Epidemiology"),
    "0749-3797": ("am_j_prev_med", "American Journal Of Preventive Medicine"),
    "0091-7435": ("prev_med", "Preventive Medicine"),
    "1044-3983": ("epidemiology", "Epidemiology"),
    "0143-005X": ("j_epidemiol_comm_health", "Journal Of Epidemiology And Community Health"),
    "2468-2667": ("lancet_public_health", "The Lancet Public Health"),
    "0090-0036": ("am_j_public_health", "American Journal Of Public Health"),
    "1101-1262": ("eur_j_public_health", "European Journal Of Public Health"),
    "0002-9262": ("am_j_epidemiol", "American Journal Of Epidemiology"),
    "0393-2990": ("eur_j_epidemiol", "European Journal Of Epidemiology"),
}


def eutils(endpoint: str, params: dict, *, post: bool = False, retries: int = 5) -> bytes:
    params = {"db": "pubmed", "tool": "policyclaims", **params}
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY
    for attempt in range(1, retries + 1):
        time.sleep(PAUSE)
        try:
            r = requests.post(EUTILS + endpoint, data=params, timeout=120) if post else requests.get(EUTILS + endpoint, params=params, timeout=120)
            if r.status_code == 200:
                return r.content
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
        except requests.RequestException:
            time.sleep(2 ** attempt)
    raise RuntimeError(f"E-utilities {endpoint} failed after {retries} attempts: {params.get('term', '')}")


def esearch_history(term: str) -> tuple[int, str, str]:
    root = ET.fromstring(eutils("esearch.fcgi", {"term": term, "retmax": 0, "usehistory": "y"}))
    return int(root.findtext("Count") or 0), root.findtext("WebEnv") or "", root.findtext("QueryKey") or ""


def parse_article(art: ET.Element) -> dict:
    title_el = art.find(".//Article/ArticleTitle")
    title = "".join(title_el.itertext()).strip() if title_el is not None else ""
    parts = []
    for a in art.findall(".//Article/Abstract/AbstractText"):
        text = "".join(a.itertext()).strip()
        label = (a.get("Label") or "").strip()
        parts.append(f"{label.capitalize()}: {text}" if label and label.upper() != "UNLABELLED" else text)
    dois = [e.text for e in art.findall(".//PubmedData/ArticleIdList/ArticleId") if e.get("IdType") == "doi" and e.text]
    dois += [e.text for e in art.findall(".//Article/ELocationID") if e.get("EIdType") == "doi" and e.text]
    return {
        "pmid": art.findtext(".//MedlineCitation/PMID"),
        "doi": (dois[0] if dois else "").strip().lower(),
        "title": title,
        "year": art.findtext(".//Article/Journal/JournalIssue/PubDate/Year") or (art.findtext(".//Article/Journal/JournalIssue/PubDate/MedlineDate") or "")[:4],
        "abstract": " ".join(parts).strip(),
        "publication_types": [p.text for p in art.findall(".//PublicationTypeList/PublicationType") if p.text],
    }


def fetch_journal_year(issn: str, year: int) -> list[dict]:
    count, webenv, qk = esearch_history(f"{issn}[is] AND {year}[dp]")
    records = []
    for start in range(0, count, BATCH):
        xml = eutils("efetch.fcgi", {"WebEnv": webenv, "query_key": qk, "retstart": start, "retmax": BATCH, "retmode": "xml"}, post=True)
        root = ET.fromstring(xml)
        records.extend(parse_article(a) for a in root.findall(".//PubmedArticle"))
    return records


def norm_title(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def norm_doi(s) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    return re.sub(r"^https?://(dx\.)?doi\.org/", "", str(s).strip().lower())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--journals", nargs="*", default=None, help="Short names to process (default: all ten)")
    ap.add_argument("--years", default="1990-2024")
    ap.add_argument("--refresh", action="store_true", help="Ignore cached PubMed records")
    args = ap.parse_args()
    y0, y1 = (int(x) for x in args.years.split("-"))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    derived = pd.read_csv(DERIVED)
    derived["doi_n"] = derived["doi"].map(norm_doi)
    derived["title_n"] = derived["title"].map(norm_title)
    out, unmatched, stats = [], [], []
    t0 = time.time()
    for issn, (abbrev, journal_name) in JOURNALS.items():
        if args.journals and abbrev not in args.journals:
            continue
        target = derived[(derived["journal"] == journal_name) & derived["publication_year"].between(y0, y1)]
        if target.empty:
            continue
        cache = CACHE_DIR / f"{abbrev}.json"
        pub: dict[int, list[dict]] = {}
        if cache.exists() and not args.refresh:
            pub = {int(k): v for k, v in json.loads(cache.read_text()).items()}
        # PubMed's publication date ([dp]) can differ from Scopus's cover year by one
        # year (online-first articles), so fetch one extra year on each side.
        years = list(range(int(target["publication_year"].min()) - 1, int(target["publication_year"].max()) + 2))
        for year in years:
            if int(year) in pub:
                continue
            pub[int(year)] = fetch_journal_year(issn, int(year))
            cache.write_text(json.dumps({str(k): v for k, v in pub.items()}))
            print(f"[{abbrev} {year}] PubMed records: {len(pub[int(year)])} (derived rows: {int((target['publication_year'] == year).sum())})", flush=True)
        allrecs = [r for y in years for r in pub.get(int(y), []) if r["abstract"]]
        by_doi = {r["doi"]: r for r in allrecs if r["doi"]}
        by_title = {}
        for r in allrecs:
            by_title.setdefault(norm_title(r["title"]), r)
        n_doi = n_title = 0
        for row in target.itertuples():
            rec = by_doi.get(row.doi_n) if row.doi_n else None
            method = "doi" if rec else None
            if rec is None and row.title_n:
                rec = by_title.get(row.title_n)
                method = "title" if rec else None
            if rec is None:
                unmatched.append({"journal": journal_name, "publication_year": row.publication_year, "doi": row.doi, "title": row.title})
                continue
            n_doi += method == "doi"; n_title += method == "title"
            out.append({
                "scopus_id": "", "pmid": rec["pmid"], "doi": row.doi if isinstance(row.doi, str) else rec["doi"],
                "title": row.title, "journal": journal_name, "publication_year": str(row.publication_year),
                "keywords": [k.strip() for k in str(row.keywords).split(";")] if isinstance(row.keywords, str) else [],
                "abstract": rec["abstract"], "article_type": "article",
                "corresponding_author_country": row.corresponding_author_country if isinstance(row.corresponding_author_country, str) else "Unknown",
                "cited_by_count": None, "abstract_source": "pubmed", "match_method": method,
                "llm_policy_claim": bool(row.llm_policy_claim),
            })
        stats.append({"journal": journal_name, "derived_rows": len(target), "matched_doi": n_doi, "matched_title": n_title,
                      "unmatched": len(target) - n_doi - n_title, "match_pct": round(100 * (n_doi + n_title) / len(target), 1)})
        print(f"[{abbrev}] matched {n_doi + n_title}/{len(target)} ({stats[-1]['match_pct']}%): doi={n_doi}, title={n_title}", flush=True)

    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    pd.DataFrame(unmatched).to_csv(CACHE_DIR / "unmatched.csv", index=False)
    st = pd.DataFrame(stats); st.to_csv(CACHE_DIR / "match_stats.csv", index=False)
    print(st.to_string(index=False))
    print(f"\nWrote {len(out)} records with abstracts to {OUT_JSON} ({len(unmatched)} unmatched listed in {CACHE_DIR / 'unmatched.csv'}) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
