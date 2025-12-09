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

def fetch_articles_postgres(ids: List[int]) -> List[Dict[str, Any]]:
    """
    Fetch article rows by ID from Postgres.
    """
    if not ids:
        return []

    with pg_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT id, title, body , source_domain, year
            FROM article_corpus
            WHERE id = ANY(%s);
            """,
            (ids,),
        )
        rows = cur.fetchall()
    return [dict(row) for row in rows]


