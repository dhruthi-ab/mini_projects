import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


class HybridPatentSearch:
    """Small hybrid search baseline for a CSV patent-like document collection."""

    REQUIRED_COLUMNS = {"patent_id", "title", "abstract"}

    def __init__(self, csv_path, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.patents = pd.read_csv(csv_path).fillna("")
        missing = self.REQUIRED_COLUMNS - set(self.patents.columns)
        if missing:
            raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
        if self.patents.empty:
            raise ValueError("Dataset has no records.")

        self.patents["text"] = (
            self.patents["title"].astype(str).str.strip()
            + ". "
            + self.patents["abstract"].astype(str).str.strip()
        )

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.patents["text"])

        # Downloads model files on first use unless cached locally.
        self.model = SentenceTransformer(model_name)
        self.embeddings = self.model.encode(
            self.patents["text"].tolist(),
            normalize_embeddings=True,
            show_progress_bar=True,
        )

    def search(self, query, top_k=5, lexical_weight=0.5):
        query = str(query).strip()
        if not query:
            return pd.DataFrame()

        if not 0.0 <= lexical_weight <= 1.0:
            raise ValueError("lexical_weight must be between 0 and 1.")

        query_tfidf = self.vectorizer.transform([query])
        lexical_scores = cosine_similarity(query_tfidf, self.tfidf_matrix)[0]

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
        )[0]
        semantic_cosine = self.embeddings @ query_embedding

        # Cosine similarity can be negative. Mapping [-1, 1] to [0, 1]
        # is a simple baseline normalization, not probability calibration.
        semantic_scores = np.clip((semantic_cosine + 1.0) / 2.0, 0.0, 1.0)

        semantic_weight = 1.0 - lexical_weight
        hybrid_scores = (
            lexical_weight * lexical_scores
            + semantic_weight * semantic_scores
        )

        results = self.patents.copy()
        results["lexical_score"] = lexical_scores
        results["semantic_score"] = semantic_scores
        results["hybrid_score"] = hybrid_scores
        results = results.sort_values(
            "hybrid_score", ascending=False
        ).head(max(1, int(top_k)))

        return results.reset_index(drop=True)
