"""Tests for KG triplet extraction and filtering."""

import pytest
import numpy as np

from finqa_chatbot.schema import KGTriplet
from finqa_chatbot.retrieval.kg_filter import (
    extract_years_from_question,
    structural_score,
    filter_triplets,
)


def test_extract_years():
    assert extract_years_from_question("what changed from 2016 to 2017?") == {"2016", "2017"}
    assert extract_years_from_question("no year here") == set()
    assert extract_years_from_question("in 1999 and 2003") == {"1999", "2003"}


def test_structural_score_period_match():
    t = KGTriplet(
        subject="Revenue",
        relation="HAS_VALUE_IN_2017",
        object="100 million",
        period="2017",
    )
    score = structural_score(t, "what was revenue in 2017?")
    assert score > 0.5  # period matches and keyword overlaps


def test_structural_score_no_match():
    t = KGTriplet(
        subject="CostOfGoods",
        relation="HAS_VALUE_IN_2010",
        object="50 million",
        period="2010",
    )
    score = structural_score(t, "what was revenue in 2017?")
    assert score < 0.3  # neither period nor keyword match


def test_filter_triplets_ranking():
    triplets = [
        KGTriplet(subject="Revenue", relation="HAS_VALUE_IN_2017", object="100", period="2017"),
        KGTriplet(subject="Costs", relation="HAS_VALUE_IN_2010", object="50", period="2010"),
        KGTriplet(subject="Revenue", relation="HAS_VALUE_IN_2016", object="90", period="2016"),
    ]
    result = filter_triplets(triplets, "what was the change in revenue from 2016 to 2017?", top_k=2)
    assert len(result) == 2
    # Revenue triplets should rank higher than Costs
    subjects = [t.subject for t in result]
    assert "Revenue" in subjects


def test_filter_triplets_empty():
    assert filter_triplets([], "any question") == []


def test_filter_triplets_with_embeddings():
    triplets = [
        KGTriplet(subject="A", relation="R1", object="1", period="2020"),
        KGTriplet(subject="B", relation="R2", object="2", period="2020"),
    ]
    q_emb = np.array([1.0, 0.0, 0.0])
    t_embs = np.array([[0.9, 0.1, 0.0], [0.1, 0.9, 0.0]])
    result = filter_triplets(
        triplets, "question about 2020",
        question_embedding=q_emb,
        triplet_embeddings=t_embs,
        top_k=1,
    )
    assert len(result) == 1
    assert result[0].subject == "A"  # higher cosine similarity
