"""
Get a glimpse of the extra_metadata field for a given pipeline run.
This includes a few example extra_metadata JSON objects and all key paths found within them.
"""

from __future__ import annotations

import json
from typing import Any

from .db import query_df


def _walk_keypaths(obj: Any, prefix: str = "") -> set[str]:
    """
    Recursively walk a JSON-like object to extract all key paths to get all possible fields in extra_metadata.
    :param obj: extra_metadata object
    :param prefix: current key path prefix
    :return: set of key paths
    """
    paths: set[str] = set()

    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else str(k)
            paths.add(p)
            paths |= _walk_keypaths(v, p)
    elif isinstance(obj, list):
        # don’t explode; just mark list and inspect first few items
        paths.add(f"{prefix}[]")
        for item in obj[:3]:
            paths |= _walk_keypaths(item, prefix)

    return paths


def extra_metadata_glimpse(run_id: str, limit: int = 3, max_chars_per_example: int = 1200) -> dict[str, Any]:
    """
    Get a glimpse of the extra_metadata field for a given pipeline run.
    This includes a few example extra_metadata JSON objects and all key paths found within them.

    :param run_id: the pipeline run ID
    :param limit: number of examples to retrieve
    :param max_chars_per_example: maximum characters per example (truncated if longer)
    :return: dict with 'examples', 'keypaths', and 'note'
    """
    df = query_df(
        """
        SELECT pra.extra_metadata
        FROM pipeline_run_articles pra
        WHERE pra.run_id = :run_id
          AND pra.extra_metadata IS NOT NULL LIMIT :limit
        """,
        {"run_id": str(run_id), "limit": int(limit)},
    )

    examples_raw: list[Any] = [x for x in df["extra_metadata"].tolist() if x is not None]

    keypaths: set[str] = set()
    examples: list[str] = []

    for obj in examples_raw:
        keypaths |= _walk_keypaths(obj)

        s = json.dumps(obj, default=str, ensure_ascii=False)
        if len(s) > max_chars_per_example:
            s = s[:max_chars_per_example] + "…"
        examples.append(s)

    return {
        "examples": examples,
        "keypaths": sorted(keypaths),
        "note": "Use only keys shown in keypaths/examples. Do not invent new fields.",
    }
