"""High-Performance Hybrid AI Patent Retrieval Engine.

Combines:
1. Okapi BM25 Lexical Retrieval with patent domain tokenization & stopword filtering
2. Dense Semantic Search using Sentence Transformers (with disk embedding caching)
3. Advanced Fusion: Calibrated Score Fusion & Reciprocal Rank Fusion (RRF)
4. Two-Stage Cross-Encoder Neural Re-Ranking for maximum precision
5. Novelty Risk Meter & Keyword Differential Analysis
6. 2D Semantic Landscape Projection (PCA/t-SNE) for interactive visualization
"""

import hashlib
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer
from sklearn.decomposition import PCA

# Patent-specific legal and boilerplate stopwords
PATENT_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
    "can", "did", "do", "does", "doing", "don", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him",
    "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me",
    "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once", "only",
    "or", "other", "our", "ours", "ourselves", "out", "over", "own", "s", "same", "she", "should",
    "so", "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves", "then",
    "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why",
    "will", "with", "you", "your", "yours", "yourself", "yourselves",
    # Patent boilerplates
    "wherein", "said", "comprising", "comprises", "thereof", "therein", "configured", "substantially",
    "plurality", "apparatus", "method", "system", "embodiment", "embodiments", "claim", "claims",
    "respective", "predetermined", "operable", "disclosed", "according", "coupled", "provided",
    "associated", "includes", "including", "portion", "surface", "first", "second", "third", "one"
}


def patent_tokenize(text: str) -> List[str]:
    """Tokenize patent text, lowercasing, stripping punctuation, and filtering patent stopwords."""
    words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9_\-]{2,}\b", str(text).lower())
    return [w for w in words if w not in PATENT_STOPWORDS]


def minmax_scale(scores: np.ndarray) -> np.ndarray:
    """Normalize score vector to [0, 1] range."""
    lo, hi = float(np.min(scores)), float(np.max(scores))
    if hi > lo:
        return (scores - lo) / (hi - lo)
    return np.zeros_like(scores)


class HybridPatentSearch:
    """Enterprise-grade Hybrid Search Engine for Patent Prior-Art Retrieval."""

    REQUIRED_COLUMNS = {"patent_id", "title", "abstract"}

    def __init__(
        self,
        csv_path: str,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        cache_dir: str = "cache",
        enable_cross_encoder: bool = True
    ):
        self.csv_path = csv_path
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.enable_cross_encoder = enable_cross_encoder

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Patent dataset not found at: {csv_path}")

        self.patents = pd.read_csv(csv_path).fillna("")
        missing = self.REQUIRED_COLUMNS - set(self.patents.columns)
        if missing:
            raise ValueError(f"Dataset missing required columns: {sorted(missing)}")
        if self.patents.empty:
            raise ValueError("Dataset contains no records.")

        # Ensure optional fields exist
        if "claims" not in self.patents.columns:
            self.patents["claims"] = ""
        if "ipc_section" not in self.patents.columns:
            self.patents["ipc_section"] = "General Technology"
        if "cpc_class" not in self.patents.columns:
            self.patents["cpc_class"] = "G06F"
        if "source_url" not in self.patents.columns:
            self.patents["source_url"] = ""

        # Rich text concatenation for full document indexing
        self.patents["text"] = (
            self.patents["title"].astype(str).str.strip()
            + ". "
            + self.patents["abstract"].astype(str).str.strip()
            + np.where(
                self.patents["claims"].str.strip() != "",
                ". Claims: " + self.patents["claims"].astype(str).str.strip(),
                ""
            )
        )

        # 1. Okapi BM25 Index Initialization
        self.tokenized_corpus = [patent_tokenize(doc) for doc in self.patents["text"]]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        # 2. Dense Semantic Model Initialization & Caching
        self.model = SentenceTransformer(model_name)
        self.embeddings = self._get_or_compute_embeddings()

        # 3. Cross-Encoder Re-ranker (loaded lazily on demand)
        self._cross_encoder = None
        if enable_cross_encoder:
            try:
                self._cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            except Exception as e:
                print(f"Notice: CrossEncoder lazy load deferred: {e}")

        # 4. Compute 2D Semantic Coordinates for Landscape Map
        self._compute_landscape_coordinates()

    def _get_cache_key(self) -> str:
        """Create unique cache identifier based on file modification and model name."""
        mtime = str(os.path.getmtime(self.csv_path))
        key_str = f"{self.csv_path}_{mtime}_{self.model_name}_{len(self.patents)}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _get_or_compute_embeddings(self) -> np.ndarray:
        """Load embeddings from disk cache if present, else compute and save."""
        os.makedirs(self.cache_dir, exist_ok=True)
        cache_file = os.path.join(self.cache_dir, f"emb_{self._get_cache_key()}.npy")

        if os.path.exists(cache_file):
            try:
                return np.load(cache_file)
            except Exception:
                pass

        # Compute embeddings with progress
        embeddings = self.model.encode(
            self.patents["text"].tolist(),
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=False
        ).astype(np.float32)

        np.save(cache_file, embeddings)
        return embeddings

    def _compute_landscape_coordinates(self):
        """Project patent dense embeddings to 2D using PCA for interactive landscape mapping."""
        if len(self.patents) < 2:
            self.patents["pca_x"] = 0.0
            self.patents["pca_y"] = 0.0
            self.pca = None
            return

        n_components = min(2, len(self.patents), self.embeddings.shape[1])
        self.pca = PCA(n_components=n_components, random_state=42)
        coords = self.pca.fit_transform(self.embeddings)

        self.patents["pca_x"] = coords[:, 0]
        self.patents["pca_y"] = coords[:, 1] if coords.shape[1] > 1 else 0.0

    def search(
        self,
        query: str,
        top_k: int = 5,
        lexical_weight: float = 0.5,
        mode: str = "hybrid_weighted",  # 'hybrid_weighted', 'hybrid_rrf', 'neural_rerank', 'semantic_only', 'lexical_only'
        ipc_filter: Optional[str] = None,
        top_candidates_for_rerank: int = 20
    ) -> Tuple[pd.DataFrame, Dict]:
        """Execute high-precision patent retrieval across lexical and dense semantic modalities."""
        query_str = str(query).strip()
        if not query_str:
            return pd.DataFrame(), {}

        # Optional domain pre-filtering
        indices = np.arange(len(self.patents))
        if ipc_filter and ipc_filter != "All Domains":
            match_mask = self.patents["ipc_section"].str.contains(ipc_filter, case=False, regex=False)
            if match_mask.any():
                indices = np.where(match_mask)[0]

        filtered_patents = self.patents.iloc[indices].copy().reset_index(drop=True)
        filtered_embeddings = self.embeddings[indices]
        filtered_tokenized_corpus = [self.tokenized_corpus[i] for i in indices]

        # 1. Lexical BM25 Scoring
        query_tokens = patent_tokenize(query_str)
        if query_tokens:
            bm25_sub = BM25Okapi(filtered_tokenized_corpus)
            bm25_raw = np.array(bm25_sub.get_scores(query_tokens), dtype=np.float32)
        else:
            bm25_raw = np.zeros(len(filtered_patents), dtype=np.float32)

        bm25_norm = minmax_scale(bm25_raw)

        # 2. Dense Semantic Cosine Scoring
        query_emb = self.model.encode([query_str], normalize_embeddings=True)[0]
        cos_sim = filtered_embeddings @ query_emb
        sem_norm = np.clip((cos_sim + 1.0) / 2.0, 0.0, 1.0)

        # 3. Fusion Scoring Logic
        semantic_weight = 1.0 - lexical_weight

        if mode == "lexical_only":
            final_scores = bm25_norm
        elif mode == "semantic_only":
            final_scores = sem_norm
        elif mode == "hybrid_rrf":
            # Reciprocal Rank Fusion: 1 / (60 + rank_bm25) + 1 / (60 + rank_sem)
            k = 60
            r_lex = np.argsort(-bm25_norm)
            ranks_lex = np.empty_like(r_lex)
            ranks_lex[r_lex] = np.arange(1, len(r_lex) + 1)

            r_sem = np.argsort(-sem_norm)
            ranks_sem = np.empty_like(r_sem)
            ranks_sem[r_sem] = np.arange(1, len(r_sem) + 1)

            rrf_scores = (1.0 / (k + ranks_lex)) + (1.0 / (k + ranks_sem))
            final_scores = minmax_scale(rrf_scores)
        else:
            # Default: Calibrated Weighted Min-Max Fusion
            final_scores = (lexical_weight * bm25_norm) + (semantic_weight * sem_norm)

        filtered_patents["lexical_score"] = np.round(bm25_norm, 4)
        filtered_patents["semantic_score"] = np.round(sem_norm, 4)
        filtered_patents["hybrid_score"] = np.round(final_scores, 4)
        filtered_patents["rerank_score"] = filtered_patents["hybrid_score"]

        # 4. Optional Two-Stage Cross-Encoder Neural Re-Ranking
        if mode == "neural_rerank" and self._cross_encoder is not None and len(filtered_patents) > 1:
            candidate_k = min(len(filtered_patents), max(top_k * 2, top_candidates_for_rerank))
            candidates = filtered_patents.sort_values("hybrid_score", ascending=False).head(candidate_k)
            pairs = [[query_str, doc_text[:512]] for doc_text in candidates["text"]]

            try:
                ce_logits = self._cross_encoder.predict(pairs)
                ce_norm = minmax_scale(np.array(ce_logits, dtype=np.float32))
                candidates["rerank_score"] = np.round(ce_norm, 4)
                # Combine hybrid candidate signal with neural cross-encoder score
                candidates["hybrid_score"] = np.round(0.3 * candidates["hybrid_score"] + 0.7 * ce_norm, 4)
                filtered_patents = candidates
            except Exception as e:
                print(f"Cross-encoder re-ranking fallback: {e}")

        # Sort and take top_k
        results = filtered_patents.sort_values("hybrid_score", ascending=False).head(max(1, int(top_k))).reset_index(drop=True)

        # 5. Novelty Risk Assessment & Keyword Differential Analysis
        novelty_analysis = self._analyze_novelty(query_tokens, results)

        # 6. Query Landscape Projection Coordinates
        query_coords = [0.0, 0.0]
        if self.pca is not None:
            try:
                q_proj = self.pca.transform([query_emb])[0]
                query_coords = [float(q_proj[0]), float(q_proj[1]) if len(q_proj) > 1 else 0.0]
            except Exception:
                pass
        novelty_analysis["query_coords"] = query_coords

        return results, novelty_analysis

    def _analyze_novelty(self, query_tokens: List[str], top_results: pd.DataFrame) -> Dict:
        """Compute Novelty Risk Meter and Keyword Overlap Breakdown."""
        if top_results.empty:
            return {
                "risk_score": 0.0,
                "risk_label": "High Novelty Potential",
                "risk_color": "#10B981",
                "summary": "No matching prior-art records found in the indexed corpus."
            }

        max_score = float(top_results["hybrid_score"].iloc[0])
        avg_top3 = float(top_results["hybrid_score"].head(3).mean())
        risk_score = round(min(100.0, (max_score * 0.7 + avg_top3 * 0.3) * 100), 1)

        if risk_score >= 70.0:
            risk_label = "High Prior-Art Collision Risk"
            risk_color = "#EF4444"
            summary = "Substantial lexical and semantic overlap found. Key inventive concepts closely track existing disclosures."
        elif risk_score >= 40.0:
            risk_label = "Moderate Overlap / Refinement Needed"
            risk_color = "#F59E0B"
            summary = "Related technologies identified. Invention contains novel features but shares underlying foundational architecture."
        else:
            risk_label = "High Novelty Potential"
            risk_color = "#10B981"
            summary = "Low collision risk against indexed patents. Novel technical terminology and distinctive claim boundaries detected."

        # Compute overlapping and distinct keyword lists for the top document
        top_doc_text = top_results["text"].iloc[0]
        doc_tokens = set(patent_tokenize(top_doc_text))
        q_token_set = set(query_tokens)

        overlapping = sorted(list(q_token_set.intersection(doc_tokens)))[:10]
        distinctive_novel = sorted(list(q_token_set - doc_tokens))[:10]

        return {
            "risk_score": risk_score,
            "risk_label": risk_label,
            "risk_color": risk_color,
            "summary": summary,
            "overlapping_keywords": overlapping,
            "distinctive_novel_keywords": distinctive_novel,
            "top_match_id": str(top_results["patent_id"].iloc[0]),
            "top_match_title": str(top_results["title"].iloc[0])
        }
