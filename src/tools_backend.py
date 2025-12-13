"""
This module contains functions to connect to OpenSearch and Postgres.
Fetches articles from Postgres by ID and searches OpenSearch.
"""

from typing import List, Dict, Any
from opensearchpy import OpenSearch
import psycopg2
import psycopg2.extras

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
                    "fields": ["title", "text"]
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

    Later NLP tools can UPDATE sentiment_score / relevance_score etc.
    """
    if not search_results:
        return

    rows = []
    for rank, hit in enumerate(search_results, start=1):
        rows.append(
            (
                run_id,
                question,
                int(hit["id"]),
                rank,
                float(hit["score"]),
            )
        )

    with pg_conn.cursor() as cur:
        # make the function idempotent for the same run_id (optional but nice)
        cur.execute(
            "DELETE FROM pipeline_run_articles WHERE run_id = %s",
            (run_id,),
        )
        psycopg2.extras.execute_values(
            cur,
            """
            INSERT INTO pipeline_run_articles
                (run_id, question, article_id, rank, os_score)
            VALUES %s
            """,
            rows,
        )
    pg_conn.commit()



