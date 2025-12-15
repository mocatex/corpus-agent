CREATE TABLE IF NOT EXISTS article_corpus (
    id            BIGSERIAL PRIMARY KEY,
    year          SMALLINT NOT NULL,
    source_domain TEXT,
    title         TEXT,
    body          TEXT
);

-- Stores per-pipeline-run article hits plus NLP fields (temporary analytics table)
CREATE TABLE IF NOT EXISTS pipeline_run_articles (
    run_id          UUID        NOT NULL,
    question        TEXT        NOT NULL,
    article_id      BIGINT      NOT NULL REFERENCES article_corpus(id),
    rank            INTEGER     NOT NULL,
    os_score        REAL        NOT NULL,
    sentiment_score REAL,
    relevance_score REAL,
    extra_metadata  JSONB,
    PRIMARY KEY (run_id, article_id)
);

-- Stores metadata about each pipeline run --
CREATE TABLE IF NOT EXISTS pipeline_runs (
  run_id UUID PRIMARY KEY,
  question TEXT NOT NULL,
  nlp_plan JSONB,
  mocked_tool_outputs JSONB,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
