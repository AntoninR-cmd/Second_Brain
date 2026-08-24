from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import replace

from second_brain.graph.types import BrainCluster, BrainMathNode

_WORD_PATTERN = re.compile(r"[^\W\d_]{3,}", re.UNICODE)
_STOP_WORDS = {
    "avec",
    "dans",
    "des",
    "du",
    "elle",
    "entre",
    "est",
    "les",
    "leur",
    "pour",
    "principe",
    "sur",
    "une",
    "the",
    "and",
    "from",
    "that",
    "with",
}


def apply_fallback_labels(
    clusters: Sequence[BrainCluster],
    nodes: Sequence[BrainMathNode],
) -> tuple[BrainCluster, ...]:
    nodes_by_id = {node.id: node for node in nodes}
    ordinals: Counter[int] = Counter()
    labeled: list[BrainCluster] = []
    for cluster in sorted(clusters, key=lambda item: (item.level, str(item.id))):
        if cluster.level == 0:
            labeled.append(replace(cluster, label="Second Brain"))
            continue
        ordinals[cluster.level] += 1
        members = [nodes_by_id[identifier] for identifier in cluster.member_ids]
        label = _tag_label(members) or _title_label(members)
        if not label:
            kind = "Domaine" if cluster.level == 1 else "Theme"
            label = f"{kind} {ordinals[cluster.level]}"
        labeled.append(replace(cluster, label=label[:80]))
    return tuple(labeled)


def _tag_label(nodes: Sequence[BrainMathNode]) -> str | None:
    counts = Counter(
        tag.strip().removeprefix("#").casefold()
        for node in nodes
        for tag in node.tags
        if tag.strip().removeprefix("#")
    )
    if not counts:
        return None
    minimum_frequency = 1 if len(nodes) == 1 else 2
    candidates = [
        tag
        for tag, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        if count >= minimum_frequency
    ][:2]
    if not candidates:
        return None
    return " · ".join(candidate.capitalize() for candidate in candidates)


def _title_label(nodes: Sequence[BrainMathNode]) -> str | None:
    counts = Counter(
        word
        for node in nodes
        for word in (match.casefold() for match in _WORD_PATTERN.findall(node.title))
        if word not in _STOP_WORDS
    )
    words = [word for word, _count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))][
        :3
    ]
    return " ".join(words).capitalize() if words else None
