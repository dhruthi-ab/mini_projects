# Result analysis

> **SMOKE TEST ONLY**: synthetic data and an LSA stand-in for Sentence-BERT. These numbers validate the pipeline, not retrieval quality.


Corpus: 1160 documents; alpha tuned on 30 validation topics; metrics reported on 30 held-out test topics.

## Metrics (test topics)

```
|                             |    MAP |   MRR |   P@10 |   R@10 |   R@100 |   nDCG@10 |
|:----------------------------|-------:|------:|-------:|-------:|--------:|----------:|
| BM25                        | 0.9282 |     1 | 0.5533 | 0.9222 |       1 |    0.9444 |
| LSA (smoke-test)            | 0.9992 |     1 | 0.6    | 1      |       1 |    0.9998 |
| Hybrid score fusion (a=0.0) | 0.9992 |     1 | 0.6    | 1      |       1 |    0.9998 |
| Hybrid RRF (k=60)           | 0.9672 |     1 | 0.5767 | 0.9611 |       1 |    0.972  |
```

## Findings

- Best method by MAP: **LSA (smoke-test)** (MAP 0.9992).
- Best fusion weight on validation set: alpha = 0.0 (BM25 weight; 1-alpha = semantic).
- LSA (smoke-test): MAP +7.7% relative to BM25 (Wilcoxon p = 0.000194 on per-topic AP).
- Hybrid score fusion (a=0.0): MAP +7.7% relative to BM25 (Wilcoxon p = 0.000194 on per-topic AP).
- Hybrid RRF (k=60): MAP +4.2% relative to BM25 (Wilcoxon p = 0.000195 on per-topic AP).

## How to read this

- BM25 rewards exact term overlap; strong when patents share vocabulary, weak under synonymy / drafting-style variation.
- SBERT captures paraphrase but truncates long patents to its max sequence length and can miss rare technical terms.
- Score fusion needs normalisation and a tuned weight; RRF is rank-based, parameter-light and robust to score scales.
- Recall@100 matters most for prior-art search (examiners need all relevant documents in the candidate set).
- p < 0.05 suggests a real difference; with few topics, treat differences cautiously.

## Limitations

- Binary relevance from citation-based qrels is incomplete: unjudged relevant documents count as misses.
- Subsampling the corpus (--max-docs) inflates all scores versus the full collection.
- Query = full topic patent text; CLEF-IP topics have other formulations (e.g. abstract-only) worth testing.