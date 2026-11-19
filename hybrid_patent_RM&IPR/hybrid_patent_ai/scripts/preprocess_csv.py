"""Clean a CSV with patent_id,title,abstract[,claims] columns (e.g. data/patents_demo.csv)
and write data/processed/<name>_clean.csv with cleaned text and tokens.
Usage: python scripts/preprocess_csv.py data/patents_demo.csv"""
import os, sys
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.preprocess import clean_text, tokenize

path = sys.argv[1]
df = pd.read_csv(path).fillna("")
for c in ("claims",):
    if c not in df: df[c] = ""
df["full_text"] = df["title"] + " . " + df["abstract"] + " . " + df["claims"]
df["clean_text"] = df["full_text"].map(clean_text)
df["tokens"] = df["full_text"].map(lambda t: " ".join(tokenize(t)))
os.makedirs("data/processed", exist_ok=True)
out = f"data/processed/{os.path.splitext(os.path.basename(path))[0]}_clean.csv"
df[["patent_id", "title", "abstract", "claims", "clean_text", "tokens"]].to_csv(out, index=False)
print(f"{len(df)} records -> {out}")
print(df[["patent_id", "tokens"]].head(3).to_string(index=False))
