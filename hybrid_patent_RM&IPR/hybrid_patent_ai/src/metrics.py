"""Ranking metrics (binary relevance). `ranked`: list of doc ids; `rel`: set of relevant ids."""
import math

def precision_at_k(ranked, rel, k):
    return sum(d in rel for d in ranked[:k]) / k

def recall_at_k(ranked, rel, k):
    return sum(d in rel for d in ranked[:k]) / len(rel) if rel else 0.0

def average_precision(ranked, rel):
    hits, s = 0, 0.0
    for i, d in enumerate(ranked, 1):
        if d in rel:
            hits += 1; s += hits / i
    return s / len(rel) if rel else 0.0

def ndcg_at_k(ranked, rel, k):
    dcg = sum(1.0 / math.log2(i + 1) for i, d in enumerate(ranked[:k], 1) if d in rel)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(rel), k) + 1))
    return dcg / idcg if idcg else 0.0

def reciprocal_rank(ranked, rel):
    for i, d in enumerate(ranked, 1):
        if d in rel:
            return 1.0 / i
    return 0.0

def evaluate_query(ranked, rel, ks=(1, 5, 10, 20, 50, 100)):
    m = {"AP": average_precision(ranked, rel), "RR": reciprocal_rank(ranked, rel)}
    for k in ks:
        m[f"P@{k}"] = precision_at_k(ranked, rel, k)
        m[f"R@{k}"] = recall_at_k(ranked, rel, k)
        m[f"nDCG@{k}"] = ndcg_at_k(ranked, rel, k)
    return m
