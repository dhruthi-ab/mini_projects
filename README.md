# Hybrid AI-Based Patent Prior-Art Retrieval

A beginner-friendly Streamlit demonstration combining:
- Lexical retrieval: TF-IDF + cosine similarity
- Semantic retrieval: Sentence Transformers embeddings
- Hybrid ranking: weighted combination of both scores

## Important dataset note
`data/patents_demo.csv` contains **synthetic demonstration records** written to test the application. They are not real patent documents, do not have real patent IDs, and must not be treated as research ground truth. For a research paper, replace them with a properly licensed public patent dataset and relevance judgments (for example, a suitable CLEF-IP benchmark subset).

## Requirements
- Python 3.10 or newer recommended
- Internet access for the first download of the sentence-embedding model
- A few GB of free disk space is advisable for Python packages and model cache

## Setup (Windows PowerShell)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## Setup (macOS/Linux)
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## How to test
Try a title such as `Smart hydration bottle` and a description such as:
`A connected bottle measures water consumption and reminds the user to drink at intervals.`

## What the scores mean
- Lexical score: TF-IDF cosine similarity
- Semantic score: normalized embedding cosine similarity
- Hybrid score: weighted sum of the two scores

These are ranking scores, not probabilities and not legal opinions about patentability, infringement, or validity. The current semantic score mapping is a simple baseline normalization and does not calibrate scores across models or datasets.

## Research use
This app is a starting implementation, not a validated research result. For experiments:
1. Use a real dataset with relevance judgments.
2. Compare lexical-only, semantic-only, and hybrid retrieval on the same queries.
3. Evaluate Precision@k, Recall@k, MAP, and/or nDCG.
4. Keep a separate validation set for choosing hybrid weights; report final results on held-out test queries.
5. Report limitations and do not claim novelty until the literature review supports it.
