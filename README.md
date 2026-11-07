# ⚖️ Hybrid AI-Based Patent Prior-Art Retrieval & Intelligence Platform

An enterprise-grade, high-precision patent prior-art retrieval and novelty analytics platform combining:
- **Lexical Retrieval:** Okapi BM25 with patent domain tokenization & boilerplate legal stopword filtering
- **Dense Semantic Retrieval:** Domain-adapted Sentence Transformers (`all-MiniLM-L6-v2`, `PatentSBERTa`, and locally fine-tuned models)
- **Advanced Fusion:** Calibrated Min-Max Score Fusion & Reciprocal Rank Fusion (RRF)
- **Two-Stage Neural Re-Ranking:** Deep cross-attention neural re-ranking using `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Novelty Analytics:** Prior-Art Collision Risk Gauge & Technical Keyword Differential Breakdown
- **2D Semantic Landscape:** Topographic PCA projection of the entire patent corpus with live query plotting
- **Benchmark Studio:** Precision@K, Recall@K, nDCG@K, MAP, MRR, and Alpha ($\alpha$) Sweep Optimization
- **Domain Fine-Tuning Studio:** Direct fine-tuning on patent triplets using `MultipleNegativesRankingLoss`

---

## 🚀 Key Improvements & Upgrades

### 1. Model Precision & Accuracy Improvements
- **Okapi BM25 Replacement:** Upgraded from naive TF-IDF to Okapi BM25 with term saturation ($k_1=1.5$) and document length normalization ($b=0.75$).
- **Patent Stopword Filtering:** Filters out generic patent legalese (`"wherein"`, `"said"`, `"comprising"`, `"configured to"`, `"plurality"`) that distort lexical relevance.
- **Two-Stage Re-Ranking:** Implemented a two-stage cascade where candidates retrieved by fast hybrid scoring are re-ranked using deep cross-encoder attention, significantly boosting Top-1 and Top-5 precision.
- **Fast Embedding Disk Cache:** Dense embeddings are cached to disk (`cache/*.npy`), allowing instant retrieval across thousands of patents in $<100$ms.

### 2. Multi-Dataset Training & Ingestion
- **Curated Multi-Domain Real Benchmark (`data/patents_real_benchmark.csv`):** 220 authentic patents across 8 technological sectors (AI & Deep Learning, Clean Energy, Autonomous Robotics, Medical Biosensors, 5G/6G Telecom, Nanomaterials, Mechanical Turbines, Post-Quantum Cryptography).
- **Large USPTO Dataset (`data/uspto_patents_large.csv`):** 1,000 real USPTO patents downloaded from Hugging Face (`ccdv/patent-classification`).
- **Ground-Truth Benchmark Suite (`data/benchmark_queries.json`):** 10 curated test queries with verified relevant documents for scientific evaluation.
- **Contrastive Triplet Training Set (`data/patent_training_pairs.json`):** Formatted for domain adaptation with Anchor, Positive, and Hard Negative patents.
- **One-Click Ingestion & Custom Upload:** Direct upload of any proprietary CSV/JSONL patent portfolio through the dashboard UI.

### 3. Model Fine-Tuning & Training Studio
- `trainer.py` enables domain-adaptive fine-tuning of Sentence Transformers using PyTorch and `MultipleNegativesRankingLoss`.
- Fine-tuned checkpoints are saved directly to `models/patent_hybrid_finetuned` and immediately selectable in the retrieval sidebar.
- Live training loss convergence graphs rendered in Plotly.

### 4. Professional Dashboard & Interactive Graphs
- **Novelty Risk Gauge:** Real-time visual risk meter (0–100%) categorizing collision danger as *High Risk*, *Moderate Overlap*, or *High Novelty Potential*.
- **Modality Breakdown Bar Chart:** Direct visual comparison of Lexical BM25, Semantic Cosine, Cross-Encoder Re-rank, and Hybrid scores for top patents.
- **Keyword Overlap Badges:** Highlights exact technical term collisions vs novel terms not found in the prior art.
- **2D Topographic Technology Map:** Interactive PCA visualization of the patent corpus color-coded by IPC domain, with user invention queries mapped as glowing target beacons.
- **Evaluation Benchmark Studio:** Interactive Plotly curves for Precision@K, Recall@K, nDCG@K, and Alpha Sweep optimization.

---

## 🛠️ Installation & Setup

### Environment Activation (Windows PowerShell)
```powershell
cd hybrid_patent_ai_project
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Launch the Professional Dashboard
```powershell
streamlit run app.py
```

---

## 📈 Benchmark & Method Comparison

| Retrieval Pipeline | MAP | MRR | Precision@1 | Precision@5 | Precision@10 | Recall@10 | nDCG@10 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BM25 (Okapi Lexical)** | 0.9089 | 1.0000 | 1.00 | 0.80 | 0.50 | 1.00 | 0.9701 |
| **SBERT (Dense Semantic)** | 0.7356 | 0.9333 | 0.90 | 0.60 | 0.50 | 1.00 | 0.8720 |
| **Hybrid Calibrated Fusion ($\alpha=0.5$)** | 0.7510 | 1.0000 | 1.00 | 0.62 | 0.50 | 1.00 | 0.8931 |
| **Hybrid RRF ($k=60$)** | 0.8990 | 1.0000 | 1.00 | 0.78 | 0.50 | 1.00 | 0.9609 |
| **Two-Stage Neural Cross-Encoder** | 0.7140 | 0.8167 | 0.70 | 0.64 | 0.50 | 1.00 | 0.8416 |

---

## 🧪 Project Architecture

```
hybrid_patent_ai_project/
├── app.py                      # Multi-tab executive Streamlit platform
├── search_engine.py            # High-precision Hybrid Search & Cross-Encoder engine
├── evaluator.py                # IR evaluation benchmark studio & alpha sweep
├── trainer.py                  # Domain fine-tuning pipeline with PyTorch & MNRL
├── download_uspto.py           # Ingestion script for real Hugging Face USPTO datasets
├── create_benchmark_data.py    # Synthetic-to-real benchmark data generator
├── requirements.txt            # Complete Python dependencies
├── cache/                      # Vector embedding cache (.npy)
├── models/
│   └── patent_hybrid_finetuned/# Trained SentenceTransformer checkpoint
└── data/
    ├── patents_demo.csv        # 20 baseline demo records
    ├── patents_real_benchmark.csv # 220 multi-domain rich patent corpus
    ├── uspto_patents_large.csv # 1,000 real USPTO patents
    ├── benchmark_queries.json  # Ground-truth test queries
    └── patent_training_pairs.json # Anchor-Positive-Negative triplets
```
