from __future__ import annotations
from .db import query_df


def schema_summary() -> str:
    """
    Get a summary of the database schema for key tables.
    :return: string summarizing the tables and their columns
    """
    df = query_df(
        """
        SELECT table_name, column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name IN ('article_corpus', 'pipeline_run_articles', 'pipeline_runs')
        ORDER BY table_name, ordinal_position
        """
    )

    lines: list[str] = []
    for table in df["table_name"].unique():
        sub = df[df["table_name"] == table]
        cols = ", ".join(f"{r.column_name} {r.data_type}" for r in sub.itertuples(index=False))
        lines.append(f"{table}({cols})")
    return "\n".join(lines)
