"""Evaluation Studio for Patent Retrieval Performance and Accuracy Benchmarking.

Computes:
- Precision@K (K=1, 5, 10)
- Recall@K (K=1, 5, 10)
- nDCG@K (K=5, 10)
- Mean Average Precision (MAP)
- Mean Reciprocal Rank (MRR)
- Alpha Sweep Optimization (Finding mathematically optimal lexical vs semantic weight)
"""

import json
import os
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


def compute_average_precision(ranked_ids: List[str], relevant_set: set) -> float:
    """Compute Average Precision (AP) for a ranked list."""
    if not relevant_set:
        return 0.0
    hits = 0
    sum_precisions = 0.0
    for idx, doc_id in enumerate(ranked_ids):
        if doc_id in relevant_set:
            hits += 1
            sum_precisions += hits / (idx + 1)
    return sum_precisions / len(relevant_set)


def compute_reciprocal_rank(ranked_ids: List[str], relevant_set: set) -> float:
    """Compute Reciprocal Rank (RR) of the first relevant document."""
    for idx, doc_id in enumerate(ranked_ids):
        if doc_id in relevant_set:
            return 1.0 / (idx + 1)
    return 0.0


def compute_ndcg_at_k(ranked_ids: List[str], relevant_set: set, k: int = 10) -> float:
    """Compute Normalized Discounted Cumulative Gain (nDCG@K)."""
    k_eval = min(k, len(ranked_ids))
    dcg = 0.0
    for i in range(k_eval):
        rel = 1.0 if ranked_ids[i] in relevant_set else 0.0
        dcg += rel / np.log2(i + 2)

    # Ideal DCG
    ideal_hits = min(k, len(relevant_set))
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return float(dcg / idcg) if idcg > 0 else 0.0


def evaluate_engine_benchmarks(search_engine, benchmark_queries_path: str = "data/benchmark_queries.json") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate retrieval methods across all benchmark test queries."""
    if not os.path.exists(benchmark_queries_path):
        raise FileNotFoundError(f"Benchmark queries file not found at: {benchmark_queries_path}")

    with open(benchmark_queries_path, "r", encoding="utf-8") as f:
        queries = json.load(f)

    modes = [
        ("BM25 (Lexical Only)", "lexical_only", 1.0),
        ("SBERT (Semantic Only)", "semantic_only", 0.0),
        ("Hybrid Fusion (a=0.5)", "hybrid_weighted", 0.5),
        ("Hybrid RRF (k=60)", "hybrid_rrf", 0.5),
        ("Neural Cross-Encoder Re-Ranker", "neural_rerank", 0.5)
    ]

    all_results = []
    ks = [1, 5, 10]

    for mode_name, mode_key, lex_w in modes:
        mode_aps = []
        mode_rrs = []
        mode_p_at_k = {k: [] for k in ks}
        mode_r_at_k = {k: [] for k in ks}
        mode_ndcg_at_k = {k: [] for k in [5, 10]}

        for q in queries:
            query_str = q["query"]
            rel_set = set(q["relevant_ids"])

            # Retrieve top 20
            res_df, _ = search_engine.search(query_str, top_k=20, lexical_weight=lex_w, mode=mode_key)
            retrieved_ids = res_df["patent_id"].tolist() if not res_df.empty else []

            # Metrics
            ap = compute_average_precision(retrieved_ids, rel_set)
            rr = compute_reciprocal_rank(retrieved_ids, rel_set)
            mode_aps.append(ap)
            mode_rrs.append(rr)

            for k in ks:
                top_k_ids = retrieved_ids[:k]
                hits = len(set(top_k_ids).intersection(rel_set))
                mode_p_at_k[k].append(hits / max(1, k))
                mode_r_at_k[k].append(hits / max(1, len(rel_set)))

            for k in [5, 10]:
                mode_ndcg_at_k[k].append(compute_ndcg_at_k(retrieved_ids, rel_set, k=k))

        row = {
            "Method": mode_name,
            "MAP": np.mean(mode_aps),
            "MRR": np.mean(mode_rrs),
            "P@1": np.mean(mode_p_at_k[1]),
            "P@5": np.mean(mode_p_at_k[5]),
            "P@10": np.mean(mode_p_at_k[10]),
            "Recall@5": np.mean(mode_r_at_k[5]),
            "Recall@10": np.mean(mode_r_at_k[10]),
            "nDCG@5": np.mean(mode_ndcg_at_k[5]),
            "nDCG@10": np.mean(mode_ndcg_at_k[10])
        }
        all_results.append(row)

    summary_df = pd.DataFrame(all_results).set_index("Method")

    # Alpha Parameter Sweep Optimization Curve
    alphas = np.round(np.arange(0.0, 1.05, 0.1), 2)
    sweep_data = []

    for al in alphas:
        aps = []
        for q in queries:
            query_str = q["query"]
            rel_set = set(q["relevant_ids"])
            res_df, _ = search_engine.search(query_str, top_k=15, lexical_weight=float(al), mode="hybrid_weighted")
            retrieved_ids = res_df["patent_id"].tolist() if not res_df.empty else []
            aps.append(compute_average_precision(retrieved_ids, rel_set))
        sweep_data.append({"Alpha": float(al), "MAP": float(np.mean(aps))})

    sweep_df = pd.DataFrame(sweep_data)
    return summary_df.round(4), sweep_df
