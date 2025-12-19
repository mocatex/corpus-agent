# CorpusAgent - A Hybrid NLP and LLM-Based System for Large Text Corpus Analysis

**Project thesis** (HS25)

**Authors:** Shpetim Veseli, Moritz Feuchter

CorpusAgent is an agent-based system for exploring and analyzing large news text corpora. It combines:

- **Hybrid retrieval** via **PostgreSQL** (structured access) and **OpenSearch** (full-text search)
- **NLP / analysis tools** (Python) orchestrated in a pipeline
- An **LLM-driven agent loop** that plans, retrieves, analyzes, and synthesizes answers
- A **Streamlit UI** for interactive querying plus a run/debug & visualization playground

---

## Repository structure (high level)

- `src/main.py` – Streamlit app entrypoint
- `src/pipeline.py` – main orchestration pipeline
- `corpus_postgres/` – Docker Compose setup + schema + import scripts
- `dataset/` – sample parquet files and Docker volumes (`pgdata/`, `osdata/`)
- `scripts/setup_dataset.sh` – one-shot setup that starts Docker, creates DB schema, imports parquet -> Postgres, and builds the OpenSearch index

---

## Prerequisites

- **Git**
- **Docker Desktop** (must be running)
- **Python 3.12+**

Recommended:
- **uv** (fast Python package manager). This repo also works with `python -m venv` + `pip` if you prefer.

---

## Installation (clone + Python deps)

1) **Clone the repository**

```bash
git clone git@github.com:mocatex/corpus-agent.git
cd corpus-agent
```

2) **Create a virtual environment and install dependencies**

### Option A (recommended): `uv`

```bash
uv sync
```

### Option B: `venv` + `pip`

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

---

## Dataset & backend services setup (Docker + Postgres + OpenSearch)

CorpusAgent expects a running **PostgreSQL** and **OpenSearch** backend. The repository includes a setup script that:

1. Starts the Docker services (`corpus_postgres/docker-compose.yml`)
2. Applies the database schema (`corpus_postgres/schema.sql`)
3. Imports a parquet subset into Postgres (`corpus_postgres/import_subset_to_postgres.py`)
4. Creates and populates an OpenSearch index (`corpus_postgres/setup_opensearch.py`)

### 1) Get the dataset (parquet)

This project is designed to work with the **CC-News sample** dataset:

- https://huggingface.co/datasets/mocatex/cc-news-sample

Download the parquet files and place them into `dataset/CC-News-sample/`.

### 2) Put parquet files into the dataset folder

- Place your `.parquet` files in `dataset/CC-News-sample/`.
- If you use a different folder, update the `SUBSET_DIR` variable in `corpus_postgres/import_subset_to_postgres.py`.

### 3) Run the setup script

From the repository root:

```bash
cd scripts
bash setup_dataset.sh
```

What you should see:
- Docker containers come up (Postgres + OpenSearch + Dashboards)
- DB schema is created inside the `corpus_postgres` container
- The subset is imported into Postgres
- An OpenSearch index named `article-corpus-opensearch` is created and populated

### Service endpoints (defaults)

- Postgres: `localhost:5432` (db: `corpus_db`, user: `corpus`, password: `corpus`)
- OpenSearch: `https://localhost:9200`
- OpenSearch Dashboards: `http://localhost:5601`

Note: OpenSearch uses a self-signed cert in this setup. The setup script uses `curl -k ...` for that reason.

---

## Configuration (LLM API key)

This project uses the `openai` Python SDK.

Create a `.env` file in the repository root (or export env vars in your shell). A typical `.env` looks like:

```dotenv
OPENAI_API_KEY=your_key_here
```

If you use a non-default base URL or model, check the code in `src/pipeline.py` / related modules and adapt accordingly.

---

## Run CorpusAgent (Streamlit UI)

Activate your environment if needed, then from the repository root:

```bash
streamlit run src/main.py
```

In the UI you can:
- Ask questions about the news corpus (2016–2021 sample)
- Inspect pipeline debug traces for the latest run
- Generate and run LLM-written visualization code that renders PNGs into `artifacts/`

---

## Troubleshooting

### Docker isn’t running / containers not found

- Ensure Docker Desktop is started.
- If you changed container names, note that `scripts/setup_dataset.sh` expects:
  - Postgres container: `corpus_postgres`
  - OpenSearch on `https://localhost:9200`

### OpenSearch index already exists

If you rerun setup and the index exists, either delete it manually in OpenSearch or adjust the script to use `PUT ...?ignore=400`.

### SSL errors when calling OpenSearch

This compose uses HTTPS with a self-signed cert. Use `-k` with `curl` or configure the Python OpenSearch client to disable cert verification for local development.

### Importing a larger dataset

The docker compose mounts `dataset/CC-News-sample/` into the Postgres container at `/data/subset`. If you swap in a larger dataset, ensure your disk space and Docker resource limits are sufficient.

---

## Citation

If you reference this work, please cite the thesis:

> Veseli, Shpetim and Feuchter, Moritz. *CorpusAgent - A Hybrid NLP and LLM-Based System for Large Text Corpus Analysis*. Project thesis, 2025.
