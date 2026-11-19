"""End-to-end experiment: parse -> preprocess -> BM25 / SBERT / fusion / RRF -> metrics -> plots -> analysis.

Real CLEF-IP:
  python run_experiment.py --docs-dir data/raw/corpus --topics-dir data/raw/topics --qrels data/raw/qrels.txt
Smoke test:
  python scripts/make_sample_clef_ip.py
  python run_experiment.py --docs-dir data/sample_clef_ip/docs --topics-dir data/sample_clef_ip/topics \
         --qrels data/sample_clef_ip/qrels.txt --semantic-backend lsa --out results/sample
"""
import argparse, json, os, random
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from src.parse_clef_ip import parse_dir, load_jsonl, load_qrels
from src.preprocess import build_doc_text
from src.retrieval import BM25, SemanticSearch, weighted_fusion, rrf, rank_docs
from src.metrics import evaluate_query
from src.plots import make_plots

KS = (1, 5, 10, 20, 50, 100)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs-dir", required=True); ap.add_argument("--topics-dir", required=True)
    ap.add_argument("--qrels", required=True); ap.add_argument("--out", default="results")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--semantic-backend", choices=["sbert", "lsa"], default="sbert")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2",
                    help="e.g. AI-Growth-Lab/PatentSBERTa for a patent-specific model")
    ap.add_argument("--no-claims", action="store_true")
    ap.add_argument("--max-docs", type=int, default=None,
                    help="keep all judged-relevant docs + random distractors up to this many docs")
    ap.add_argument("--max-topics", type=int, default=None)
    ap.add_argument("--rrf-k", type=int, default=60); ap.add_argument("--val-frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); os.makedirs(a.processed, exist_ok=True)
    tag = os.path.basename(os.path.normpath(a.out))

    # 1. extract + parse
    cj, tj = f"{a.processed}/{tag}_corpus.jsonl", f"{a.processed}/{tag}_topics.jsonl"
    print("parsed docs:", parse_dir(a.docs_dir, cj), "topics:", parse_dir(a.topics_dir, tj))
    docs, topics, qrels = load_jsonl(cj), load_jsonl(tj), load_qrels(a.qrels)
    topics = [t for t in topics if t["patent_id"] in qrels]
    rng = random.Random(a.seed)
    if a.max_topics: topics = rng.sample(topics, min(a.max_topics, len(topics)))
    if a.max_docs and len(docs) > a.max_docs:
        need = {d for t in topics for d in qrels[t["patent_id"]]}
        keep = [d for d in docs if d["patent_id"] in need]
        rest = [d for d in docs if d["patent_id"] not in need]
        docs = keep + rng.sample(rest, max(0, min(len(rest), a.max_docs - len(keep))))
    ids = [d["patent_id"] for d in docs]; idset = set(ids)
    qrels = {t["patent_id"]: {d for d in qrels[t["patent_id"]] if d in idset and d != t["patent_id"]} for t in topics}
    topics = [t for t in topics if qrels[t["patent_id"]]]
    print(f"{len(docs)} docs, {len(topics)} topics with >=1 relevant doc in corpus")

    # 2. text for indexing / querying
    dtext = [build_doc_text(d, not a.no_claims) for d in docs]
    qtext = [build_doc_text(t, not a.no_claims) for t in topics]

    # 3. scores
    bm = BM25().fit(dtext)
    sem = SemanticSearch(a.model, a.semantic_backend, cache_dir=f"{a.processed}/emb").fit(dtext, ids)
    S_sem = sem.scores_batch(qtext)
    S_lex = np.stack([bm.scores(q) for q in qtext])
    tids = [t["patent_id"] for t in topics]

    def run(fn):
        return {tid: rank_docs(fn(i), ids, exclude=tid, top_n=1000) for i, tid in enumerate(tids)}
    def evalrun(runs, sel):
        return pd.DataFrame([{"topic": t, **evaluate_query(runs[t], qrels[t], KS)} for t in sel]).set_index("topic")

    # 4. split topics: validation (tune alpha) / test (report)
    shuffled = tids[:]; rng.shuffle(shuffled)
    nval = max(1, int(len(shuffled) * a.val_frac)) if len(shuffled) > 1 else 0
    val, test = shuffled[:nval], shuffled[nval:] or shuffled
    alphas = np.round(np.arange(0, 1.01, 0.1), 2)
    sweep = {}
    for al in alphas:
        r = run(lambda i: weighted_fusion(S_lex[i], S_sem[i], al))
        sweep[float(al)] = float(evalrun(r, val)["AP"].mean())
    best_alpha = max(sweep, key=sweep.get)
    print("alpha sweep (val MAP):", {k: round(v, 4) for k, v in sweep.items()}, "-> best", best_alpha)

    runs = {
        "BM25": run(lambda i: S_lex[i]),
        "SBERT" if a.semantic_backend == "sbert" else "LSA (smoke-test)": run(lambda i: S_sem[i]),
        f"Hybrid score fusion (a={best_alpha})": run(lambda i: weighted_fusion(S_lex[i], S_sem[i], best_alpha)),
        f"Hybrid RRF (k={a.rrf_k})": run(lambda i: rrf([S_lex[i], S_sem[i]], a.rrf_k)),
    }
    per_q = {m: evalrun(r, test) for m, r in runs.items()}
    summary = pd.DataFrame({m: df.mean() for m, df in per_q.items()}).T
    summary = summary.rename(columns={"AP": "MAP", "RR": "MRR"})
    summary.to_csv(f"{a.out}/summary_metrics.csv")
    pd.concat({m: df for m, df in per_q.items()}, names=["method", "topic"]).to_csv(f"{a.out}/per_query_metrics.csv")
    pd.Series(sweep, name="val_MAP").to_csv(f"{a.out}/alpha_sweep.csv")
    sample_run = {t: runs[list(runs)[2]][t][:100] for t in test}
    json.dump(sample_run, open(f"{a.out}/hybrid_top100_runs.json", "w"))

    # 5. significance (Wilcoxon on per-query AP vs BM25)
    names = list(runs); sig = []
    for m in names[1:]:
        x, y = per_q[m]["AP"].values, per_q[names[0]]["AP"].values
        try: p = wilcoxon(x, y).pvalue if np.any(x != y) else 1.0
        except ValueError: p = float("nan")
        sig.append((m, p))

    make_plots(summary, pd.Series(sweep), KS, a.out, names)
    write_analysis(a, summary, sig, best_alpha, len(docs), len(test), len(val), names)
    print(summary.round(4).to_string()); print("outputs in", a.out)

def write_analysis(a, S, sig, best_alpha, nd, ntest, nval, names):
    best = S["MAP"].idxmax()
    cols = ["MAP", "MRR", "P@10", "R@10", "R@100", "nDCG@10"]
    tbl = S[cols].round(4).to_markdown() if hasattr(S, "to_markdown") and _has_tab() else S[cols].round(4).to_string()
    warn = ""
    if a.semantic_backend == "lsa":
        warn = ("> **SMOKE TEST ONLY**: synthetic data and an LSA stand-in for Sentence-BERT. "
                "These numbers validate the pipeline, not retrieval quality.\n\n")
    gain = lambda m: (S.loc[m, "MAP"] - S.loc[names[0], "MAP"]) / max(S.loc[names[0], "MAP"], 1e-9) * 100
    lines = [f"# Result analysis\n", warn,
             f"Corpus: {nd} documents; alpha tuned on {nval} validation topics; metrics reported on {ntest} held-out test topics.\n",
             "## Metrics (test topics)\n", "```\n" + tbl + "\n```\n",
             f"## Findings\n",
             f"- Best method by MAP: **{best}** (MAP {S.loc[best, 'MAP']:.4f}).",
             f"- Best fusion weight on validation set: alpha = {best_alpha} (BM25 weight; 1-alpha = semantic).",
             *[f"- {m}: MAP {gain(m):+.1f}% relative to BM25 (Wilcoxon p = {p:.3g} on per-topic AP)." for m, p in sig],
             "\n## How to read this\n",
             "- BM25 rewards exact term overlap; strong when patents share vocabulary, weak under synonymy / drafting-style variation.",
             "- SBERT captures paraphrase but truncates long patents to its max sequence length and can miss rare technical terms.",
             "- Score fusion needs normalisation and a tuned weight; RRF is rank-based, parameter-light and robust to score scales.",
             "- Recall@100 matters most for prior-art search (examiners need all relevant documents in the candidate set).",
             "- p < 0.05 suggests a real difference; with few topics, treat differences cautiously.\n",
             "## Limitations\n",
             "- Binary relevance from citation-based qrels is incomplete: unjudged relevant documents count as misses.",
             "- Subsampling the corpus (--max-docs) inflates all scores versus the full collection.",
             "- Query = full topic patent text; CLEF-IP topics have other formulations (e.g. abstract-only) worth testing."]
    open(f"{a.out}/analysis.md", "w").write("\n".join(lines))

def _has_tab():
    try: import tabulate; return True
    except Exception: return False

if __name__ == "__main__":
    main()
