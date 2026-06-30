"""
Semantic similarity between the JD and each candidate's profile text.

IMPORTANT - read before assuming this uses sentence-transformers:
This sandbox has no network access and sentence-transformers is not
pre-installed, so this module uses TF-IDF + Truncated SVD (a classic LSA
pipeline, scikit-learn only, no model download, no GPU, no network) as the
semantic-similarity engine. This is a deliberate, documented choice, not an
oversight - see methodology_summary in submission_metadata.yaml.

If you have network access on your own machine, the architecture supports
swapping this for real embeddings (e.g. sentence-transformers/all-MiniLM-L6-v2
or BAAI/bge-small-en) with no change to anything downstream: precompute
candidate embeddings once (offline, pre-computation step, not part of the
5-minute ranking budget), cache them to disk, and load the cache in the
ranking step instead of fitting TF-IDF at ranking time. See
`embed_candidates_offline()` stub at the bottom of this file for where that
would plug in - the swap is a single function, not a redesign.

Why this still catches the JD's "no buzzwords but real experience" case:
TF-IDF + SVD captures co-occurring vocabulary, not just exact keyword
matches - "built a recommendation system," "ranking models," and "discovery
feed" co-occur with the JD's "ranking, retrieval, and matching systems"
vocabulary even without using the words "RAG" or "Pinecone." It is a weaker
signal than a trained embedding model, which is exactly why it's only one of
several weighted components (see ranker.py) rather than the sole driver of
rank.
"""

from __future__ import annotations
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


def _clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\-\+/]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def candidate_text(candidate: dict) -> str:
    """Concatenate the profile fields that actually carry semantic content
    about what the person does / has done. Deliberately excludes the skills
    list itself (skill_matching.py already handles that with trust
    weighting) - mixing it in here would let keyword stuffing leak back
    into the semantic score through a side door."""
    profile = candidate.get("profile", {})
    parts = [
        profile.get("headline", ""),
        profile.get("summary", ""),
        profile.get("current_title", ""),
    ]
    for role in candidate.get("career_history", []):
        parts.append(role.get("title", ""))
        parts.append(role.get("description", ""))
    return _clean(" ".join(p for p in parts if p))


class SemanticSimilarityScorer:
    """Fit once on (JD + all candidate texts in the current batch), then
    score every candidate against the JD vector. Fitting TF-IDF/SVD on the
    batch (rather than a fixed pretrained vocabulary) means the vocabulary
    adapts to whatever's actually in the pool, which matters when scaling
    to the full 100K-candidate file with its own vocabulary distribution.
    """

    def __init__(self, n_components: int = 150, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.vectorizer = TfidfVectorizer(
            max_features=20000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.85,
            stop_words="english",
        )
        self.svd = TruncatedSVD(n_components=n_components, random_state=random_state)
        self._jd_vec = None
        self._fitted = False

    def fit(self, jd_text: str, candidate_texts: list[str]) -> None:
        corpus = [_clean(jd_text)] + candidate_texts
        tfidf = self.vectorizer.fit_transform(corpus)

        # SVD component count can't exceed min(n_samples, n_features) - 1;
        # guard for small batches (e.g. the 50-row dev sample) so this
        # doesn't crash before we ever touch the 100K pool.
        max_components = min(tfidf.shape) - 1
        n_components = min(self.n_components, max_components) if max_components > 0 else 1
        if n_components != self.n_components:
            self.svd = TruncatedSVD(n_components=n_components, random_state=self.random_state)

        reduced = self.svd.fit_transform(tfidf)
        self._jd_vec = reduced[0]
        self._candidate_vecs = reduced[1:]
        self._fitted = True

    def scores(self) -> np.ndarray:
        """Cosine similarity of every candidate vector to the JD vector,
        scaled to [0, 1]. Call fit() first."""
        if not self._fitted:
            raise RuntimeError("call fit() before scores()")
        jd = self._jd_vec
        cand = self._candidate_vecs
        jd_norm = jd / (np.linalg.norm(jd) + 1e-9)
        cand_norm = cand / (np.linalg.norm(cand, axis=1, keepdims=True) + 1e-9)
        cos_sim = cand_norm @ jd_norm
        # cosine similarity from SVD can be negative; rescale [-1,1] -> [0,1]
        return (cos_sim + 1.0) / 2.0


def embed_candidates_offline(candidate_texts: list[str], cache_path: str) -> np.ndarray:
    """STUB - upgrade path, not used by the default pipeline.

    If you have network access on your own dev machine (not required for
    the ranking step itself, which must run offline), you can precompute
    real sentence embeddings once and cache them:

        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(candidate_texts, show_progress_bar=True)
        np.save(cache_path, embeddings)

    Then at ranking time, load the cache (no network, no GPU needed for
    just loading a .npy file) and compute cosine similarity against a
    similarly-precomputed JD embedding. This keeps the actual ranking step
    inside the compute constraints (Section 3 of submission_spec.md) since
    the model call happens during pre-computation, not during ranking.
    """
    raise NotImplementedError(
        "Not wired up in this environment (no network/model access). "
        "See docstring for the swap-in path if you have network locally."
    )