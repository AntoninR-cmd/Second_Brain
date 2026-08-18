from __future__ import annotations

import re
from collections.abc import Sequence
from importlib.resources import files
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from second_brain.pipeline.chunking import SourceChunk

PROMPT_PACKAGE = "second_brain.llm.prompts"
_PLACEHOLDER = re.compile(r"\{\{([a-z_]+)\}\}")


def system_fidelity_prompt() -> str:
    return _load_prompt("system_fidelity.md")


def build_passage_analysis_prompt(
    *,
    source_title: str,
    chunk: SourceChunk,
    max_knowledge: int = 2,
) -> str:
    return _render_prompt(
        "analyze_passage.md",
        source_title=source_title,
        passage_index=str(chunk.index),
        max_knowledge=str(max_knowledge),
        passage_text=chunk.text,
    )


def build_source_summary_prompt(
    *,
    source_title: str,
    passage_summaries: Sequence[tuple[int, str]],
) -> str:
    summaries = "\n\n".join(
        f"[PASSAGE {index}]\n{summary.strip()}" for index, summary in passage_summaries
    )
    return _render_prompt(
        "summarize_source.md",
        source_title=source_title,
        passage_summaries=summaries,
    )


def build_validation_retry_prompt(*, original_prompt: str, validation_error: str) -> str:
    return _render_prompt(
        "retry_invalid_json.md",
        original_prompt=original_prompt,
        validation_error=validation_error[:2000],
    )


def _load_prompt(filename: str) -> str:
    return files(PROMPT_PACKAGE).joinpath(filename).read_text(encoding="utf-8").strip()


def _render_prompt(filename: str, **values: str) -> str:
    prompt = _load_prompt(filename)

    def replacement(match: re.Match[str]) -> str:
        key = match.group(1)
        try:
            return values[key]
        except KeyError as error:
            raise ValueError(f"Valeur de prompt absente : {key}") from error

    return _PLACEHOLDER.sub(replacement, prompt)
