"""TF-IDF based retrieval."""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def tfidf_retrieve(
    question: str,
    facts: list[tuple[str, str]],
    topn: int = 3,
) -> list[tuple[str, str]]:
    """Return the top-n facts most similar to *question* via TF-IDF."""
    if not facts:
        return []
    docs = [text for _, text in facts]
    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        docs_tfidf = vectorizer.fit_transform(docs)
        query_tfidf = vectorizer.transform([question])
        sims = cosine_similarity(query_tfidf, docs_tfidf).flatten()
    except ValueError:
        return facts[:topn]
    ranked = sorted(zip(sims, facts), key=lambda x: -x[0])
    return [f for _, f in ranked[:topn]]
