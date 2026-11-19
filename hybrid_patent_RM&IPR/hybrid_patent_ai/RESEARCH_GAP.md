# Existing patent retrieval methods and the research gap

> Only the 2017 survey (Shalaby & Zadrozny) was read in full. The other items were identified from
> search-result titles and must be read and verified before you cite them in a paper.

## 1. Existing methods
| Family | Idea | Reported limits (from the 2017 survey) |
|---|---|---|
| Query reformulation | Pick/weight terms from the topic patent (claims, description, IPC filters) | Long queries; best CLEF-IP 2010 MAP about 0.12 |
| Pseudo-relevance feedback | Expand/reduce the query with top-ranked docs | Parameter-sensitive, unstable; query reduction often beats expansion |
| Thesaurus / semantic expansion | WordNet, query-log, Wikipedia, topic models | Small gains; WordNet raised MAP slightly but lowered PRES and slowed queries |
| Citation / classification (IPC, CPC) | PageRank on citations, class vectors | Incomplete citation graphs; class revisions |
| Interactive search | Faceted / visual refinement | Not automatic; hard to evaluate |
| Dense / neural (later work) | Patent-tuned Sentence-BERT-style encoders, paragraph aggregation (e.g. PARM), embedding benchmarks | Long documents exceed encoder input length; patent vocabulary |

Survey conclusions: automatic performance is "still low (<0.2 MAP)"; 12% of relevant CLEF-IP 2009 documents
share no words with their topic (vocabulary mismatch); CLEF-IP relevance comes from citations, which suits
related-work search rather than patentability search.

## 2. Research gap this project targets
1. **Lexical vs semantic vs hybrid on the same CLEF-IP split.** Older work is mostly lexical + expansion; dense
   and hybrid results are reported on different datasets and settings, so they are hard to compare.
2. **Fusion strategy is rarely compared.** Tuned weighted score fusion (needs normalisation and a validation
   set) vs parameter-light Reciprocal Rank Fusion on long patent queries is a concrete, testable question.
3. **Which patent sections to use.** Title+abstract vs +claims for both the query and the index, given encoder
   length limits (256-512 tokens) versus BM25 using full text.
4. **Vocabulary mismatch.** Whether the semantic component recovers relevant documents that BM25 misses
   (measure: Recall@100 and the share of relevant docs retrieved only by the semantic side).

**Suggested research question:** Does hybrid retrieval (BM25 + Sentence-BERT) with weighted fusion or RRF
improve MAP, nDCG and Recall@K over BM25 alone on CLEF-IP 2011, and which patent sections matter most?

## 3. How this repo answers it
`run_experiment.py` evaluates BM25, SBERT, weighted fusion (alpha tuned on validation topics) and RRF on
held-out topics, with `--no-claims` for the section ablation, Wilcoxon tests, and Recall@100 for the mismatch question.

## Sources
- Shalaby & Zadrozny, [Patent Retrieval: A Literature Review](https://arxiv.org/pdf/1701.00324)
- [PARM: Paragraph Aggregation Retrieval Model](https://arxiv.org/pdf/2201.01614)
- [Benchmarking Patent Embeddings: 22 Models](https://arxiv.org/html/2605.24297)
- [DAPFAM patent retrieval dataset](https://www.researchgate.net/publication/393148782_DAPFAM_A_Domain-Aware_Patent_Retrieval_Dataset_Aggregated_at_the_Family_Level)
- [NLP in the Patent Domain: A Survey](https://arxiv.org/pdf/2403.04105)
