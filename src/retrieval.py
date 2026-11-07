"""BM25 lexical search, Sentence-BERT semantic search, score fusion and RRF."""
import os
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer
from .preprocess import tokenize

class BM25:
    """Okapi BM25 (Lucene-style idf) on a sparse term-document matrix."""
    def __init__(self, k1=1.5, b=0.75):
        self.k1, self.b = k1, b

    def fit(self, docs):
        self.vec = CountVectorizer(analyzer=tokenize)
        tf = self.vec.fit_transform(docs).tocsr().astype(np.float32)
        N = tf.shape[0]
        df = np.asarray((tf > 0).sum(0)).ravel()
        self.idf = np.log(1 + (N - df + 0.5) / (df + 0.5)).astype(np.float32)
        dl = np.asarray(tf.sum(1)).ravel()
        norm = self.k1 * (1 - self.b + self.b * dl / max(dl.mean(), 1e-9))
        tf = tf.tocoo()
        data = tf.data * (self.k1 + 1) / (tf.data + norm[tf.row])
        self.W = sp.csc_matrix((data * self.idf[tf.col], (tf.row, tf.col)), shape=tf.shape)
        return self

    def scores(self, query):
        q = self.vec.transform([query])
        idx = q.indices
        if len(idx) == 0:
            return np.zeros(self.W.shape[0], dtype=np.float32)
        return np.asarray(self.W[:, idx].sum(1)).ravel()   # binary query terms

class SemanticSearch:
    """Sentence-BERT dense retrieval (cosine). backend='lsa' is an OFFLINE smoke-test
    stand-in only (TF-IDF + SVD) - never report it as Sentence-BERT."""
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", backend="sbert",
                 max_seq_length=256, batch_size=64, cache_dir=None, device=None):
        self.model_name, self.backend = model_name, backend
        self.max_seq_length, self.batch_size = max_seq_length, batch_size
        self.cache_dir, self.device = cache_dir, device

    def fit(self, docs, ids=None):
        if self.backend == "sbert":
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name, device=self.device)
            self.model.max_seq_length = self.max_seq_length
            cache = None
            if self.cache_dir and ids is not None:
                os.makedirs(self.cache_dir, exist_ok=True)
                tag = self.model_name.replace("/", "_")
                cache = os.path.join(self.cache_dir, f"{tag}_{len(ids)}.npy")
            if cache and os.path.exists(cache):
                self.E = np.load(cache)
            else:
                self.E = self.model.encode(docs, batch_size=self.batch_size, normalize_embeddings=True,
                                           show_progress_bar=True).astype(np.float32)
                if cache: np.save(cache, self.E)
        elif self.backend == "lsa":
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.decomposition import TruncatedSVD
            from sklearn.preprocessing import normalize
            self.tfidf = TfidfVectorizer(analyzer=tokenize, sublinear_tf=True, min_df=2)
            X = self.tfidf.fit_transform(docs)
            self.svd = TruncatedSVD(n_components=min(128, X.shape[1] - 1), random_state=0)
            self.E = normalize(self.svd.fit_transform(X)).astype(np.float32)
        else:
            raise ValueError(self.backend)
        return self

    def encode_queries(self, queries):
        if self.backend == "sbert":
            return self.model.encode(queries, batch_size=self.batch_size, normalize_embeddings=True,
                                     show_progress_bar=False).astype(np.float32)
        from sklearn.preprocessing import normalize
        return normalize(self.svd.transform(self.tfidf.transform(queries))).astype(np.float32)

    def scores_batch(self, queries):
        return self.encode_queries(queries) @ self.E.T

def minmax(x):
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)

def weighted_fusion(lex, sem, alpha=0.5):
    """alpha*minmax(BM25) + (1-alpha)*minmax(semantic)."""
    return alpha * minmax(lex) + (1 - alpha) * minmax(sem)

def rrf(score_lists, k=60):
    """Reciprocal Rank Fusion over several score arrays: sum 1/(k+rank)."""
    out = np.zeros(len(score_lists[0]), dtype=np.float64)
    for s in score_lists:
        ranks = np.empty(len(s), dtype=np.int64)
        ranks[np.argsort(-s, kind="stable")] = np.arange(1, len(s) + 1)
        out += 1.0 / (k + ranks)
    return out

def rank_docs(scores, doc_ids, exclude=None, top_n=1000):
    order = np.argsort(-scores, kind="stable")
    res = []
    for i in order:
        if doc_ids[i] == exclude:
            continue
        res.append(doc_ids[i])
        if len(res) >= top_n:
            break
    return res
