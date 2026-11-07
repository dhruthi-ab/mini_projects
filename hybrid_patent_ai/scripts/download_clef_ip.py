"""Download CLEF-IP 2011 from the TU Wien Research Data repository.

CLEF-IP is licence-gated (CLEF-IP licence agreement) and very large (corpus is
many GB). This script:
  1. searches researchdata.tuwien.at (InvenioRDM API) for CLEF-IP records,
  2. lists the files of each record,
  3. downloads files whose names match --pattern (default: none -> list only).

Usage:
  python scripts/download_clef_ip.py --list
  python scripts/download_clef_ip.py --record <record_id> --pattern qrels --out data/raw
If a file is restricted, the API returns 403: request access on the record page.
"""
import argparse, os, sys, requests

API = "https://researchdata.tuwien.at/api/records"

def search():
    r = requests.get(API, params={"q": "CLEF-IP", "size": 50}, timeout=60)
    r.raise_for_status()
    for h in r.json()["hits"]["hits"]:
        print(f'{h["id"]}\t{h["metadata"]["title"]}\t{h["links"]["self_html"]}')

def files(rec):
    r = requests.get(f"{API}/{rec}/files", timeout=60)
    r.raise_for_status()
    return r.json()["entries"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--record")
    ap.add_argument("--pattern", default=None)
    ap.add_argument("--out", default="data/raw")
    a = ap.parse_args()
    if a.list or not a.record:
        search(); return
    os.makedirs(a.out, exist_ok=True)
    for e in files(a.record):
        size_mb = e.get("size", 0) / 1e6
        print(f'{e["key"]}  {size_mb:.1f} MB')
        if a.pattern and a.pattern.lower() in e["key"].lower():
            dest = os.path.join(a.out, e["key"])
            with requests.get(f"{API}/{a.record}/files/{e['key']}/content", stream=True, timeout=120) as r:
                if r.status_code == 403:
                    sys.exit("403: access restricted - request access / sign the CLEF-IP licence on the record page.")
                r.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            print("  saved", dest)

if __name__ == "__main__":
    main()
