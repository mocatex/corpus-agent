"""
Database utility functions for connecting to PostgreSQL and executing queries.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

load_dotenv()

_param_pat = re.compile(r"%\((\w+)\)s")


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """
    Get a cached SQLAlchemy Engine for the PostgreSQL database connection.
    Connection parameters are read from environment variables with defaults.

    :return: SQLAlchemy Engine instance
    """
    # Defaults match docker-compose + tools_backend constants (maybe enter data in .env file)
    host = os.getenv("PG_HOST", "localhost")
    port = os.getenv("PG_PORT", "5432")
    db = os.getenv("PG_DBNAME", "corpus_db")
    user = os.getenv("PG_DBUSER", "corpus")
    pw = os.getenv("PG_DBPASS", "corpus")

    url = f"postgresql+psycopg2://{user}:{pw}@{host}:{port}/{db}"
    return create_engine(url, pool_pre_ping=True)


def query_df(sql: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
    """
    Execute a SQL query and return the results as a pandas DataFrame

    :param sql: the SQL query string
    :param params: optional dictionary of parameters for the query
    :return: pandas DataFrame with the query results
    """
    # accept psycopg2-style "%(name)s" and translate to SQLAlchemy ":name"
    sql = _param_pat.sub(r":\1", sql)

    with get_engine().connect() as conn:
        return pd.read_sql_query(text(sql), conn, params=params)


def exec_sql(sql: str, params: dict[str, Any] | None = None) -> None:
    """
    Execute a SQL command (INSERT, UPDATE, DELETE, etc.)

    :param sql: the SQL command string
    :param params: optional dictionary of parameters for the command
    """
    with get_engine().begin() as conn:
        conn.execute(text(sql), params or {})
