"""Semantic + structural filtering for KG triplets."""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from ..schema import KGTriplet


def extract_years_from_question(question: str) -> set[str]:
    """Pull 4-digit years out of a question string."""
    return set(re.findall(r'\b((?:19|20)\d{2})\b', question))


def structural_score(triplet: KGTriplet, question: str) -> float:
    """Score a triplet by structural features relative to the question.

    Returns a value in [0, 1] based on period match and keyword overlap.
    """
    score = 0.0
    q_years = extract_years_from_question(question)

    # Period match
    if triplet.period and triplet.period in q_years:
        score += 0.5

    # Keyword overlap between subject and question
    subj_tokens = set(triplet.subject.lower().replace(":", " ").split())
    q_tokens = set(question.lower().split())
    overlap = len(subj_tokens & q_tokens)
    if subj_tokens:
        score += 0.5 * (overlap / len(subj_tokens))

    return min(score, 1.0)


def filter_triplets(
    triplets: list[KGTriplet],
    question: str,
    question_embedding: np.ndarray | None = None,
    triplet_embeddings: np.ndarray | None = None,
    top_k: int = 10,
    semantic_weight: float = 0.5,
) -> list[KGTriplet]:
    """Rank and filter triplets by combined semantic + structural relevance.

    If embeddings are provided, uses cosine similarity weighted with
    structural features. Otherwise falls back to structural scoring only.
    """
    if not triplets:
        return []

    scores: list[float] = []
    for i, triplet in enumerate(triplets):
        struct = structural_score(triplet, question)

        if question_embedding is not None and triplet_embeddings is not None:
            # Cosine similarity
            t_emb = triplet_embeddings[i]
            cos_sim = float(
                np.dot(question_embedding, t_emb)
                / (np.linalg.norm(question_embedding) * np.linalg.norm(t_emb) + 1e-10)
            )
            combined = semantic_weight * cos_sim + (1 - semantic_weight) * struct
        else:
            combined = struct

        scores.append(combined)

    ranked = sorted(
        zip(scores, triplets), key=lambda x: -x[0]
    )
    return [t for _, t in ranked[:top_k]]
