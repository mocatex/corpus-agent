import duckdb
from pathlib import Path

# ========= CONFIG =========
DATA_DIR = Path(__file__).parent.parent / "dataset" / "CC-News"
PARQUET_GLOB = str(DATA_DIR / "*.parquet")

OUTPUT_DIR = Path(__file__).parent.parent / "dataset" / "CC-News-sample"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_FRACTION = 0.1  # 10% -> ~20GB -> 9.7M articles

RANDOM_SEED = 0.42


# ========= END CONFIG =========


def main():
    con = duckdb.connect()
    con.execute(f"SELECT setseed({RANDOM_SEED})")

    # Inspect the basic schema from one file
    print("Inspecting schema from one parquet file...")
    sample_file = next(DATA_DIR.glob("*.parquet"))
    print("Example file:", sample_file)

    schema_df = con.execute(f"""
        DESCRIBE SELECT * FROM read_parquet('{sample_file}')
    """).df()
    print(schema_df)

    # Count rows per 'year' (source column)
    print("\nCounting rows per year...")
    counts_df = con.execute(f"""
        SELECT
            CAST(source AS INTEGER) AS year,
            COUNT(*) AS n
        FROM read_parquet('{PARQUET_GLOB}')
        GROUP BY year
        ORDER BY year
    """).df()
    print(counts_df)

    years = counts_df["year"].tolist()

    # Sample per year into separate parquet files
    print("\nSampling data per year...")
    for year in years:
        out_file = OUTPUT_DIR / f"sample_{year}.parquet"
        print(f"  -> Year {year}: writing {out_file.name}")

        # filter by year, randomly sample, rename source to year
        con.execute(f"""
            COPY (
                SELECT
                    CAST(source AS INTEGER) AS year,
                    source_domain,
                    title,
                    text
                FROM read_parquet('{PARQUET_GLOB}')
                WHERE CAST(source AS INTEGER) = {year}
                  AND random() < {SAMPLE_FRACTION}
            )
            TO '{out_file}' (FORMAT PARQUET);
        """)

    # sanity check on the resulting subset
    print("\nSanity check on subset:")
    subset_glob = str(OUTPUT_DIR / "sample_*.parquet")
    subset_counts = con.execute(f"""
        SELECT
            year,
            COUNT(*) AS n
        FROM read_parquet('{subset_glob}')
        GROUP BY year
        ORDER BY year
    """).df()
    print(f"{subset_counts:_d}")

    con.close()

    print("\nDone. Now check disk usage of", OUTPUT_DIR)


if __name__ == "__main__":
    main()
