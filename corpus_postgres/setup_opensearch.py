import psycopg2
import requests
import json
import time
import urllib3

# Supress InsecureRequestWarning since we are only working locally
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

PG_DSN = "dbname=corpus_db user=corpus password=corpus host=localhost port=5432"
OS_URL = "https://localhost:9200/article-corpus-opensearch/_bulk"  # HTTPS now
OS_AUTH = ("admin", "VerySecurePassword123!")          # Your admin login

def get_total_row_count():
    with psycopg2.connect(PG_DSN) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM article_corpus")
            return cur.fetchone()[0]

def stream_rows(batch_size=5_000):
    conn = psycopg2.connect(PG_DSN)
    cur = conn.cursor(name="article_cursor")  # server-side cursor

    cur.execute("SELECT id, year, source_domain, title, body FROM article_corpus")

    while True:
        rows = cur.fetchmany(batch_size)
        if not rows:
            print("No more rows!")
            break
        yield rows

    cur.close()
    conn.close()

def bulk_index():
    total_rows = get_total_row_count()
    print(f"Total rows to index: {total_rows}")

    processed_rows = 0
    batch_num = 0
    log_every_batches = 40
    start = time.time()

    for rows in stream_rows():
        batch_num += 1
        processed_rows += len(rows)
        lines = []
        for (id_, year, source_domain, title, body) in rows:
            # action line
            lines.append(json.dumps({ "index": { "_id": id_ } }))
            # source line
            lines.append(json.dumps({
                "id": id_,
                "year": year,
                "source_domain": source_domain,
                "title": title,
                "body": body,
            }))
        payload = "\n".join(lines) + "\n"

        r = requests.post(
            OS_URL,
            data=payload,
            headers={"Content-Type": "application/x-ndjson"},
            auth=OS_AUTH,
            verify=False  # because OpenSearch uses self-signed certificates
        )
        r.raise_for_status()
        resp = r.json()
        if resp.get("errors"):
            print("Bulk had errors:", resp)

        if batch_num % log_every_batches == 0:
            elapsed = time.time() - start
            rate = processed_rows / elapsed if elapsed > 0 else 0
            remaining = total_rows - processed_rows
            eta_sec = remaining / rate if rate > 0 else 0

            print(
                f"Processed {processed_rows:,}/{total_rows:,} "
                f"({processed_rows / total_rows:.1%}) "
                f"- {rate:,.0f} docs/s, ETA ~ {eta_sec / 60:.1f} min"
            )

    print("Done bulk indexing.")

if __name__ == "__main__":
    bulk_index()