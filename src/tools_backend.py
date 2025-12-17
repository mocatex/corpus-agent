"""
This module contains functions to connect to OpenSearch and Postgres.
Fetches articles from Postgres by ID and searches OpenSearch.
"""

from typing import List, Dict, Any, Optional
from opensearchpy import OpenSearch
import psycopg2
import psycopg2.extras
import json

# Connection Settings OpenSearch and Postgres
OS_USERNAME = "admin"
OS_PASSWORD = "VerySecurePassword123!"
PG_DBNAME = "corpus_db"
PG_DBUSER = "corpus"
PG_DBPASS = "corpus"

# OpenSearch Client and settings Port 9200

os_client = OpenSearch(
    hosts=[{"host": "localhost", "port": 9200}],
    http_compress=True,
    use_ssl=True,
    verify_certs=False,
    http_auth=(OS_USERNAME, OS_PASSWORD),
)

pg_conn = psycopg2.connect(
    dbname=PG_DBNAME,
    user=PG_DBUSER,
    password=PG_DBPASS,
    host="localhost",
    port=5432,
)

# -- Functions --

def search_opensearch(query: str, top_k: int = 1000) -> List[Dict[str, Any]]:
    """
    Search OpenSearch and return list of {id, score}.
    Assumes your index has field 'id' used as foreign key in Postgres.
    """
    response = os_client.search(
        index="article-corpus-opensearch",
        body={
            "size": top_k,
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["title", "body"]
                }
            }
        }
    )
    hits = response["hits"]["hits"]
    return [
        {
            "id": hit["_source"]["id"],
            "score": hit["_score"],
        }
        for hit in hits
    ]

def fetch_run_documents_postgres(run_id: str, limit: int | None = None) -> List[Dict[str, Any]]:
    """
    Fetch documents for a given pipeline run by joining pipeline_run_articles -> article_corpus.
    Returns the same fields your pipeline expects, plus rank/os_score
    """
    with pg_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT ac.id,
                   ac.title,
                   ac.body,
                   ac.source_domain,
                   ac.year,
                   pra.rank,
                   pra.os_score,
                   pra.sentiment_score,
                   pra.relevance_score,
                   pra.extra_metadata
            FROM pipeline_run_articles pra
                     JOIN article_corpus ac ON ac.id = pra.article_id
            WHERE pra.run_id = %s
            ORDER BY pra.rank ASC
            """ + (" LIMIT %s" if limit is not None else ""),
            (run_id, limit) if limit is not None else (run_id,),
        )
        rows = cur.fetchall()
    return [dict(r) for r in rows]


def store_run_articles(
    run_id: str,
    question: str,
    search_results: List[Dict[str, Any]],
) -> None:
    """
    Persist retrieved article hits for a single pipeline run into
    the pipeline_run_articles temp table.

    - Deduplicates by article_id (keeps the best os_score)
    - Recomputes rank after dedupe
    - Uses UPSERT so duplicates never crash (idempotent)

    Later NLP tools can UPDATE sentiment_score / relevance_score / extra_metadata.
    """
    if not search_results:
        return

    # Deduplicate hits by article id, keep the highest score
    best_by_id: Dict[int, Dict[str, Any]] = {}
    for hit in search_results:
        try:
            article_id = int(hit["id"])
        except (KeyError, TypeError, ValueError):
            continue
        score = float(hit.get("score", 0.0) or 0.0)

        prev = best_by_id.get(article_id)
        if prev is None or score > float(prev.get("score", 0.0) or 0.0):
            best_by_id[article_id] = {"id": article_id, "score": score}

    deduped = list(best_by_id.values())
    # Sort by score desc to create stable ranks
    deduped.sort(key=lambda h: float(h.get("score", 0.0) or 0.0), reverse=True)

    rows = []
    for rank, hit in enumerate(deduped, start=1):
        rows.append(
            (
                run_id,
                question,
                int(hit["id"]),
                rank,
                float(hit.get("score", 0.0) or 0.0),
            )
        )

    with pg_conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            """
            INSERT INTO pipeline_run_articles
                (run_id, question, article_id, rank, os_score)
            VALUES %s
            ON CONFLICT (run_id, article_id)
            DO UPDATE SET
                question = EXCLUDED.question,
                rank = EXCLUDED.rank,
                os_score = EXCLUDED.os_score
            """,
            rows,
        )

    pg_conn.commit()


# --- New Functions ---

def update_run_articles_nlp_features(run_id: str, documents: List[Dict[str, Any]]) -> None:
    """
    Populate sentiment_score and per-article mocked NLP features into pipeline_run_articles.

    - sentiment_score -> dedicated column
    - all other lightweight NLP features -> extra_metadata (JSONB, merged)

    This is deterministic and cheap (no real NLP executed).
    """
    if not documents:
        return

    topic_labels = ["Economy / Markets", "Technology / Innovation", "Politics / Regulation"]
    emotion_labels = ["neutral", "positive", "negative", "fear", "anger", "joy", "sadness"]

    rows = []
    for d in documents:
        article_id = d.get("id")
        if article_id is None:
            continue

        body = (d.get("body") or "")
        length = len(body)

        # Deterministic mock sentiment (same logic as pipeline metadata)
        sentiment_score = (length % 800) / 400.0 - 1.0

        if sentiment_score < -0.3:
            sentiment_label = "negative"
        elif sentiment_score > 0.3:
            sentiment_label = "positive"
        else:
            sentiment_label = "mixed/neutral"

        topic_label = topic_labels[length % len(topic_labels)]
        emotion_label = emotion_labels[length % len(emotion_labels)]

        extra_metadata = {
            "mock_nlp": {
                "sentiment_label": sentiment_label,
                "topic": topic_label,
                "emotion": emotion_label,
                "body_length": length,
            }
        }

        rows.append(
            (
                float(sentiment_score),
                json.dumps(extra_metadata),
                run_id,
                int(article_id),
            )
        )

    if not rows:
        return

    with pg_conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            """
            UPDATE pipeline_run_articles AS pra
            SET
                sentiment_score = v.sentiment_score,
                extra_metadata = COALESCE(pra.extra_metadata, '{}'::jsonb) || (v.extra_metadata::jsonb)
            FROM (VALUES %s) AS v(sentiment_score, extra_metadata, run_id, article_id)
            WHERE pra.run_id = v.run_id::uuid
              AND pra.article_id = v.article_id
            """,
            rows,
        )

    pg_conn.commit()


def set_run_articles_relevance(run_id: str, selected_article_ids: List[int]) -> None:
    """
    Persist relevance_score for a run.

    - selected articles -> relevance_score = 1.0
    - all others        -> relevance_score = 0.0

    Idempotent and safe to call multiple times.
    """
    with pg_conn.cursor() as cur:
        # Reset all first
        cur.execute(
            "UPDATE pipeline_run_articles SET relevance_score = 0.0 WHERE run_id = %s",
            (run_id,),
        )

        if selected_article_ids:
            cur.execute(
                """
                UPDATE pipeline_run_articles
                SET relevance_score = 1.0
                WHERE run_id = %s AND article_id = ANY(%s)
                """,
                (run_id, selected_article_ids),
            )

    pg_conn.commit()


def append_run_articles_extra_metadata(run_id: str, extra: Dict[str, Any]) -> None:
    if not extra:
        return
    payload = json.dumps(extra)
    with pg_conn.cursor() as cur:
        cur.execute(
            """
            UPDATE pipeline_run_articles
            SET extra_metadata = COALESCE(extra_metadata, '{}'::jsonb) || CAST(%s AS jsonb)
            WHERE run_id = %s
            """,
            (payload, run_id),
        )
    pg_conn.commit()

def store_run_metadata(run_id: str, question: str, nlp_plan: Dict[str, Any], mocked_tool_outputs: Dict[str, Any], final_answer: Optional[str]) -> None:
    """
    Store run-level metadata ONCE per run (nlp_plan + aggregated analytics).
    """
    with pg_conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO pipeline_runs (run_id, question, nlp_plan, mocked_tool_outputs, final_answer)
            VALUES (%s, %s, CAST(%s AS jsonb), CAST(%s AS jsonb), %s)
            ON CONFLICT (run_id)
                DO UPDATE SET question            = EXCLUDED.question,
                              nlp_plan            = EXCLUDED.nlp_plan,
                              mocked_tool_outputs = EXCLUDED.mocked_tool_outputs,
                              final_answer        = COALESCE(EXCLUDED.final_answer, pipeline_runs.final_answer),
            """,
            (
                run_id,
                question,
                json.dumps(nlp_plan or {}),
                json.dumps(mocked_tool_outputs or {}),
                final_answer,
            ),
        )

    pg_conn.commit()


def fetch_run_metadata(run_id: str) -> Dict[str, Any]:
    with pg_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT run_id, question, nlp_plan, mocked_tool_outputs, final_answer, created_at
            FROM pipeline_runs
            WHERE run_id = %s
            """,
            (run_id,),
        )
        row = cur.fetchone()
    return dict(row) if row else {}
