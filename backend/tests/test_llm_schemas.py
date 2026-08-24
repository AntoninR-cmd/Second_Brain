from __future__ import annotations

import pytest
from pydantic import ValidationError
from second_brain.llm.schemas import (
    ClusterLabelBatch,
    KnowledgeDraft,
    PassageAnalysis,
    SourceSummary,
)


def test_knowledge_draft_normalizes_and_deduplicates_tags() -> None:
    draft = KnowledgeDraft(
        title="  Préparer le bois avant vernissage  ",
        content=(
            "  Un ponçage progressif prépare la surface du bois avant l'application du vernis.  "
        ),
        tags=["#Bois", " bois ", "FINITION", "#finition"],
        passage_indices=[0, 2],
    )

    assert draft.title == "Préparer le bois avant vernissage"
    assert draft.tags == ["bois", "finition"]
    assert draft.passage_indices == [0, 2]


def test_llm_schemas_reject_extra_fields_and_duplicate_references() -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        SourceSummary(summary="Un résumé suffisamment détaillé pour être accepté ici.", extra=True)

    with pytest.raises(ValidationError, match="uniques"):
        KnowledgeDraft(
            title="Titre autonome",
            content="Une connaissance autonome dont le contenu est suffisamment développé.",
            tags=[],
            passage_indices=[1, 1],
        )


def test_passage_analysis_accepts_no_knowledge_when_source_has_none() -> None:
    analysis = PassageAnalysis(
        passage_index=0,
        summary="Ce passage ne contient qu'une introduction sans fait extractible.",
        knowledge=[],
    )

    assert analysis.knowledge == []


def test_llm_schemas_reject_coerced_types_and_short_normalized_content() -> None:
    with pytest.raises(ValidationError):
        PassageAnalysis.model_validate(
            {
                "passage_index": "0",
                "summary": "Résumé valide.",
                "knowledge": [],
            }
        )

    with pytest.raises(ValidationError):
        KnowledgeDraft.model_validate(
            {
                "title": "Titre valide",
                "content": "a         b",
                "tags": [],
                "passage_indices": [False],
            }
        )

    with pytest.raises(ValidationError, match="caractere #"):
        KnowledgeDraft(
            title="Titre autonome",
            content="Une connaissance autonome suffisamment développée pour être valide.",
            tags=["bo#is"],
            passage_indices=[0],
        )


def test_cluster_labels_allow_the_compact_label_only_format() -> None:
    result = ClusterLabelBatch.model_validate(
        {"labels": [{"cluster_key": "c0001", "label": "Gestion de la fatigue"}]}
    )

    assert result.labels[0].description is None
