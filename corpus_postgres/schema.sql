CREATE TABLE IF NOT EXISTS article_corpus (
    id            BIGSERIAL PRIMARY KEY,
    year          SMALLINT NOT NULL,
    source_domain TEXT,
    title         TEXT,
    body          TEXT
);