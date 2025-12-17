"""
Module for managing and retrieving pipeline run information.
"""

from __future__ import annotations

from .db import query_df


def list_pipeline_runs(limit: int = 30) -> list[dict]:
    """
    List recent pipeline runs.

    :param limit: Maximum number of runs to return.
    :return: List of dicts with run_id, question, and created_at.
    """
    df = query_df(
        """
        SELECT run_id, question, created_at
        FROM pipeline_runs
        ORDER BY created_at DESC LIMIT :limit
        """,
        {"limit": limit},
    )
    return df.to_dict(orient="records")


def fetch_pipeline_run(run_id: str) -> dict:
    """
    Fetch detailed information about a specific pipeline run.

    :param run_id: The ID of the pipeline run to fetch.
    :return: Dict with run details including question, final_answer, created_at, nlp_plan, and mocked_tool_outputs.
    """
    df = query_df(
        """
        SELECT run_id, question, final_answer, created_at, nlp_plan, mocked_tool_outputs
        FROM pipeline_runs
        WHERE run_id = :run_id
        """,
        {"run_id": run_id},
    )
    return df.to_dict(orient="records")[0] if not df.empty else {}
