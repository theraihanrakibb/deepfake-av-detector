#!/usr/bin/env python3
"""Verify a LaTeX thebibliography file against Crossref and OpenAlex."""

import argparse
import csv
import difflib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

USER_AGENT = "NWPU-Thesis-Reference-Verifier/1.0"


def fetch_json(url):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.load(response)


def normalize(text):
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def similarity(a, b):
    return difflib.SequenceMatcher(None, normalize(a), normalize(b)).ratio()


def parse_entries(text):
    pattern = re.compile(r"\\bibitem\{([^}]+)\}\s*(.+?)(?=\n\\bibitem|\n\\end\{thebibliography\})", re.S)
    for key, body in pattern.findall(text):
        body = " ".join(body.split())
        type_match = re.search(r"\[[A-Z]+(?:/[A-Z]+)*\]", body)
        prefix = body[:type_match.start()] if type_match else body
        title = prefix.rsplit(". ", 1)[-1].strip(" .")
        doi_match = re.search(r"doi\.org/([^}\s]+)", body, re.I)
        url_match = re.search(r"\\url\{([^}]+)\}", body)
        yield {
            "key": key,
            "title": title,
            "doi": doi_match.group(1).rstrip(".") if doi_match else "",
            "url": url_match.group(1) if url_match else "",
            "entry": body,
        }


def crossref_doi(doi):
    data = fetch_json("https://api.crossref.org/works/" + urllib.parse.quote(doi, safe="/"))
    item = data["message"]
    return item.get("title", [""])[0], item.get("DOI", doi), "Crossref DOI"


def crossref_title(title):
    query = urllib.parse.quote(title)
    data = fetch_json(f"https://api.crossref.org/works?rows=3&query.bibliographic={query}")
    items = data["message"].get("items", [])
    if not items:
        return "", "", "Crossref title"
    best = max(items, key=lambda x: similarity(title, x.get("title", [""])[0]))
    return best.get("title", [""])[0], best.get("DOI", ""), "Crossref title"


def openalex_title(title):
    query = urllib.parse.quote(title)
    data = fetch_json(f"https://api.openalex.org/works?search={query}&per-page=3")
    items = data.get("results", [])
    if not items:
        return "", "", "OpenAlex title"
    best = max(items, key=lambda x: similarity(title, x.get("title", "")))
    return best.get("title", ""), best.get("doi", "") or "", "OpenAlex title"


def verify(entry):
    checks = []
    if entry["doi"]:
        try:
            checks.append(crossref_doi(entry["doi"]))
        except (urllib.error.URLError, urllib.error.HTTPError, KeyError, ValueError):
            checks.append(("", entry["doi"], "DOI lookup failed"))
    try:
        time.sleep(0.15)
        checks.append(crossref_title(entry["title"]))
    except (urllib.error.URLError, urllib.error.HTTPError, KeyError, ValueError):
        checks.append(("", "", "Crossref title lookup failed"))
    best = max(checks, key=lambda x: similarity(entry["title"], x[0]))
    score = similarity(entry["title"], best[0])
    if score < 0.60:
        try:
            time.sleep(0.15)
            alt = openalex_title(entry["title"])
            alt_score = similarity(entry["title"], alt[0])
            if alt_score > score:
                best, score = alt, alt_score
        except (urllib.error.URLError, urllib.error.HTTPError, KeyError, ValueError):
            best = (best[0], best[1], best[2] + "; OpenAlex failed")
    if score >= 0.82:
        status = "VERIFIED"
    elif score >= 0.60:
        status = "REVIEW"
    else:
        status = "UNVERIFIED"
    return {
        "key": entry["key"],
        "cited_title": entry["title"],
        "doi": entry["doi"],
        "url": entry["url"],
        "indexed_title": best[0],
        "indexed_doi": best[1],
        "source": best[2],
        "similarity": round(score, 3),
        "status": status,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("references", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    text = args.references.read_text(encoding="utf-8")
    entries = list(parse_entries(text))
    rows = [verify(entry) for entry in entries]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        if row["status"] != "VERIFIED":
            print(f"{row['key']}: {row['status']} ({row['similarity']}) {row['cited_title']}")
    print(f"entries={len(rows)} verified={sum(r['status']=='VERIFIED' for r in rows)} "
          f"review={sum(r['status']=='REVIEW' for r in rows)} "
          f"unverified={sum(r['status']=='UNVERIFIED' for r in rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
