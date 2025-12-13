import duckdb
import psycopg
from pathlib import Path
import time

# Adjust to where your sampled parquet files actually are
SUBSET_DIR = Path(__file__).parent.parent / "dataset" / "CC-News-sample"
PG_CONNINFO = "host=localhost port=5432 dbname=corpus_db user=corpus password=corpus"

BATCH_ROWS = 20_000


def import_file(con: duckdb.DuckDBPyConnection,
                cur,
                parquet_path: Path,
                file_total_rows: int,
                global_total_rows: int,
                progress: dict,
                log_every_batches: int = 5
                ):
    print(f"\n=== Importing {parquet_path.name} ===")
    print(f"  Total rows in file: {file_total_rows:,}")

    offset = 0
    imported_in_file = 0
    batch_idx = 0

    while offset < file_total_rows:
        print(f"  -> Reading rows {offset} .. {offset + BATCH_ROWS} ...", end="", flush=True)
        df = con.execute(f"""
            SELECT
                CAST(year AS SMALLINT) AS year,
                source_domain,
                title,
                text AS body
            FROM read_parquet('{parquet_path}')
            LIMIT {BATCH_ROWS} OFFSET {offset}
        """).df()

        if df.empty:
            print(" no rows (done)")
            break

        csv_data = df.to_csv(index=False, header=False)

        with cur.copy(
                "COPY article_corpus(year, source_domain, title, body) "
                "FROM STDIN WITH (FORMAT csv)"
        ) as copy:
            copy.write(csv_data)

        batch_rows = len(df)
        imported_in_file += batch_rows
        progress["processed"] += batch_rows
        offset += BATCH_ROWS
        batch_idx += 1

        # Logging per-batch
        if batch_idx % log_every_batches == 0:
            elapsed = time.time() - progress["start"]
            rate = progress["processed"] / elapsed if elapsed > 0 else 0
            remaining = global_total_rows - progress["processed"]
            eta_sec = remaining / rate if rate > 0 else 0

            print(
                f"  [File {parquet_path.name}] "
                f"{imported_in_file:,}/{file_total_rows:,} in file "
                f"| Global {progress['processed']:,}/{global_total_rows:,} "
                f"({progress['processed'] / global_total_rows:.1%}) "
                f"| {rate:,.0f} rows/s, ETA ~ {eta_sec / 60:.1f} min"
            )

    print(f"=== Done {parquet_path.name}: {imported_in_file} rows imported ===")


def main():
    con = duckdb.connect()

    files = sorted(SUBSET_DIR.glob("sample_*.parquet"))
    if not files:
        print(f"No sampled parquet files found in {SUBSET_DIR}")
        return

    print("Files to import:")
    for f in files:
        print(" -", f)

    file_counts = {}
    global_total_rows = 0
    print("\nCounting rows per file (DuckDB)...")
    for f in files:
        n = con.execute(f"SELECT COUNT(*) AS n FROM read_parquet('{f}')").fetchone()[0]
        file_counts[f] = n
        global_total_rows += n
        print(f"  {f.name}: {n:,} rows")

    print(f"\nTotal number of rows to import: {global_total_rows:,}\n")

    progress = {
        "processed": 0,
        "start": time.time(),
    }

    with psycopg.connect(PG_CONNINFO) as pgcon:
        with pgcon.cursor() as cur:
            for f in files:
                import_file(con=con,
                            cur=cur,
                            parquet_path=f,
                            file_total_rows=file_counts[f],
                            global_total_rows=global_total_rows,
                            progress=progress,
                            log_every_batches=5)
            pgcon.commit()

    print("\nAll subset files imported.")


if __name__ == "__main__":
    main()
