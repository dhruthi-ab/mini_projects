# Hybrid AI-Based Patent Prior-Art Retrieval (CLEF-IP 2011)

Pipeline: download -> parse (id, title, abstract, claims) -> clean -> **BM25** -> **Sentence-BERT** ->
**weighted score fusion** + **Reciprocal Rank Fusion** -> **P@K, R@K, MAP, nDCG@K, MRR** -> graphs + analysis.

| File | Purpose |
|---|---|
| `scripts/download_clef_ip.py` | lists / downloads CLEF-IP records from TU Wien Research Data |
| `src/parse_clef_ip.py` | XML -> JSONL (id, title, abstract, claims); qrels loader |
| `src/preprocess.py` | cleaning, tokenising, stopwords, stemming |
| `src/retrieval.py` | BM25, SBERT, min-max fusion, RRF |
| `src/metrics.py` | evaluation metrics |
| `src/plots.py` | comparison graphs |
| `RESEARCH_GAP.md` | literature summary and research gap |
| `prepare_data.py` | extract + clean text from XML / JSONL / CSV datasets |
| `run_experiment.py` | runs everything, writes `results/<name>/` |
| `scripts/make_sample_clef_ip.py` | synthetic CLEF-IP-style data for smoke tests |

## Run on the real dataset
CLEF-IP is licence-gated: sign the CLEF-IP licence / request access on the TU Wien repository, then
```bash
pip install -r requirements.txt
python scripts/download_clef_ip.py --list                       # find the record ids
python scripts/download_clef_ip.py --record <id> --pattern qrels --out data/raw
# extract corpus XML to data/raw/corpus, topics to data/raw/topics
python run_experiment.py --docs-dir data/raw/corpus --topics-dir data/raw/topics \
       --qrels data/raw/qrels.txt --max-docs 100000 --out results/clef_ip_2011
```
Use `--model AI-Growth-Lab/PatentSBERTa` for a patent-specific encoder, `--no-claims` to index title+abstract only.
Check the qrels/topic file layout against your download (the loader accepts TREC `topic 0 doc rel` or `topic doc`); topic XML is parsed like a patent document.
The full corpus (~1.3M docs) needs a GPU for SBERT; `--max-docs` keeps all judged docs plus random distractors.

## Smoke test (synthetic, NOT real results)
```bash
python scripts/make_sample_clef_ip.py
python run_experiment.py --docs-dir data/sample_clef_ip/docs --topics-dir data/sample_clef_ip/topics \
       --qrels data/sample_clef_ip/qrels.txt --semantic-backend lsa --out results/sample
```
`--semantic-backend lsa` is an offline stand-in used only because SBERT could not be installed where this was built.

## Method notes
- Topics are split into validation (tunes fusion weight alpha) and test (reported) to avoid tuning on test.
- Evaluation excludes the query patent itself from its ranking; binary relevance.
- Original demo (`app.py`, `search_engine.py`, `data/patents_demo.csv`) is kept unchanged; it uses synthetic records.
