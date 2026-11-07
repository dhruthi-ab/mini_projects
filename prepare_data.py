"""Extract patent IDs, titles, abstracts and claims from a dataset and clean the text.

Supported inputs (auto-detected, or force with --format):
  xml   a folder of CLEF-IP / EPO patent XML files
  jsonl one JSON object per line (HUPD-style: patent_number, title, abstract, claims)
  json  a JSON list of such objects
  csv   a CSV with id / title / abstract / claims-like columns (e.g. data/patents_demo.csv)

Output: <out>.csv with  patent_id, title, abstract, claims, text_raw, text_clean
Usage:
  python prepare_data.py --input data/patents_demo.csv --out data/processed/demo
  python prepare_data.py --input path/to/xml_folder   --out data/processed/clef_ip
"""
import argparse, json, os
import pandas as pd
from src.parse_clef_ip import parse_dir
from src.preprocess import clean_text, tokenize, build_doc_text

ID_KEYS = ["patent_id", "patent_number", "publication_number", "doc_id", "id", "application_number"]
TITLE_KEYS = ["title", "invention_title"]
ABS_KEYS = ["abstract", "abstract_text"]
CLAIM_KEYS = ["claims", "claim", "claims_text", "claim_text", "application_claim_text"]

def pick(row, keys):
    for k in keys:
        if k in row and row[k] is not None and str(row[k]).strip() not in ("", "nan"):
            v = row[k]
            return " ".join(map(str, v)) if isinstance(v, list) else str(v)
    return ""

def load_records(path, fmt):
    if fmt == "auto":
        if os.path.isdir(path): fmt = "xml"
        else: fmt = os.path.splitext(path)[1].lstrip(".").lower()
    if fmt == "xml":
        tmp = "data/processed/_xml_tmp.jsonl"; os.makedirs("data/processed", exist_ok=True)
        print("parsed XML docs:", parse_dir(path, tmp))
        return [json.loads(l) for l in open(tmp, encoding="utf-8")]
    if fmt == "jsonl":
        return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    if fmt == "json":
        return json.load(open(path, encoding="utf-8"))
    if fmt == "csv":
        return pd.read_csv(path).fillna("").to_dict("records")
    raise SystemExit(f"Unsupported format: {fmt}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--format", default="auto", choices=["auto", "xml", "jsonl", "json", "csv"])
    ap.add_argument("--no-claims", action="store_true")
    a = ap.parse_args()
    recs = load_records(a.input, a.format)
    rows = []
    for i, r in enumerate(recs):
        rec = {"patent_id": pick(r, ID_KEYS) or f"doc{i}", "title": pick(r, TITLE_KEYS),
               "abstract": pick(r, ABS_KEYS), "claims": pick(r, CLAIM_KEYS)}
        raw = build_doc_text(rec, not a.no_claims)
        rec["text_raw"] = raw
        rec["text_clean"] = " ".join(tokenize(raw))
        rows.append(rec)
    df = pd.DataFrame(rows).drop_duplicates("patent_id")
    df = df[df["text_clean"].str.len() > 0]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    df.to_csv(a.out + ".csv", index=False)
    print(f"{len(df)} patents -> {a.out}.csv | with claims: {(df.claims != '').sum()}")
    print(df[["patent_id", "title", "text_clean"]].head(3).to_string(max_colwidth=60))

if __name__ == "__main__":
    main()
