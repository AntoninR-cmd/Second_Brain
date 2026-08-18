from __future__ import annotations

from second_brain.llm.prompt_loader import (
    build_passage_analysis_prompt,
    build_source_summary_prompt,
    system_fidelity_prompt,
)
from second_brain.pipeline.chunking import SourceChunk


def test_passage_prompt_contains_bounded_knowledge_and_utf8_text() -> None:
    prompt = build_passage_analysis_prompt(
        source_title="Préparation du bois",
        chunk=SourceChunk(
            index=3,
            text="Poncer progressivement avant d'appliquer le vernis.",
            token_count=12,
            segment_indices=(7, 8),
            start_ms=1200,
            end_ms=4200,
        ),
    )

    assert "Préparation du bois" in prompt
    assert "Indice du passage : 3" in prompt
    assert "zéro à 2 connaissances" in prompt
    assert "7, 8" not in prompt
    assert "{{" not in prompt
    assert "connaissances atomiques" in prompt


def test_summary_and_system_prompts_are_centralized() -> None:
    prompt = build_source_summary_prompt(
        source_title="Source",
        passage_summaries=[(0, "Première idée."), (1, "Deuxième idée.")],
    )

    assert "[PASSAGE 0]" in prompt
    assert "[PASSAGE 1]" in prompt
    assert "fidèle" in prompt
    assert "n'invente" in system_fidelity_prompt()


def test_prompt_values_are_inserted_once_without_rewriting_source_text() -> None:
    source_text = "Texte fidèle avec {{source_title}} et {{passage_index}} littéraux."
    prompt = build_passage_analysis_prompt(
        source_title="{{passage_text}}",
        chunk=SourceChunk(
            index=2,
            text=source_text,
            token_count=12,
        ),
    )

    assert "source « {{passage_text}} »" in prompt
    assert source_text in prompt
