"""
fetch_corpus.py - download plain-text Wikipedia articles about banking into corpus/.

Wikipedia text is licensed CC BY-SA 4.0. Standard library only.

Usage:
    python fetch_corpus.py                       # default banking article list
    python fetch_corpus.py Commerzbank Citigroup # your own titles
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

DEFAULT_TITLES = [
    "JPMorgan_Chase", "Deutsche_Bank", "Goldman_Sachs", "Morgan_Stanley", "HSBC",
    "Barclays", "BNP_Paribas", "UBS", "Credit_Suisse", "Lehman_Brothers",
    "Federal_Reserve", "European_Central_Bank", "Basel_III",
    "Fractional-reserve_banking", "Investment_banking", "2008_financial_crisis",
]
API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "rag-from-scratch-tutorial/0.1 (educational script)"}


def fetch_plaintext(title: str, attempts: int = 5) -> str:
    params = {"action": "query", "prop": "extracts", "explaintext": 1,
              "redirects": 1, "format": "json", "titles": title}
    request = urllib.request.Request(f"{API}?{urllib.parse.urlencode(params)}", headers=HEADERS)
    delay = 5.0
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                pages = json.load(response)["query"]["pages"]
            return next(iter(pages.values())).get("extract", "")
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt == attempts - 1:
                raise
            print(f"  rate limited, waiting {delay:.0f}s ...")
            time.sleep(delay)
            delay *= 2
    return ""


def main() -> None:
    titles = sys.argv[1:] or DEFAULT_TITLES
    out = Path("corpus")
    out.mkdir(exist_ok=True)
    for title in titles:
        target = out / f"{title}.txt"
        if target.exists() and target.stat().st_size > 0:
            print(f"{title}: already downloaded")
            continue
        text = fetch_plaintext(title)
        if not text:
            print(f"{title}: NOT FOUND (check the exact Wikipedia title)")
            continue
        target.write_text(text, encoding="utf-8")
        print(f"{title}: {len(text):,} chars")
        time.sleep(1.0)  # be polite to the Wikipedia API


if __name__ == "__main__":
    main()
