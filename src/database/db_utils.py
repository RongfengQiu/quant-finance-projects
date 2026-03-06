from __future__ import annotations

import sqlite3
from pathlib import Path
import pandas as pd


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS prices_cleaned (
    date TEXT PRIMARY KEY,
    close REAL NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    volume REAL
);

CREATE TABLE IF NOT EXISTS features (
    date TEXT PRIMARY KEY,
    close REAL NOT NULL,
    log_return REAL,
    rv_20d_ann REAL,
    rv_30d_ann REAL,
    rv_60d_ann REAL,
    ewma_vol_ann REAL,
    ma_20 REAL,
    ma_20_ratio REAL,
    ma_60 REAL,
    ma_60_ratio REAL,
    year INTEGER,
    month INTEGER
);

CREATE TABLE IF NOT EXISTS options_dataset (
    option_id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    split TEXT NOT NULL,
    S REAL NOT NULL,
    K REAL NOT NULL,
    T REAL NOT NULL,
    r REAL NOT NULL,
    sigma_used REAL NOT NULL,
    bs_price REAL NOT NULL,
    mc_price REAL NOT NULL,
    abs_error REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS ml_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    mae REAL NOT NULL,
    rmse REAL NOT NULL,
    created_at TEXT DEFAULT (datetime('now'))
);
"""


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_SQL)
    conn.commit()


def write_table(conn: sqlite3.Connection, name: str, df: pd.DataFrame) -> None:
    df = df.copy()
    if "date" in df.columns:
        df["date"] = df["date"].astype(str)
    df.to_sql(name, conn, if_exists="replace", index=False)


def append_table(conn: sqlite3.Connection, name: str, df: pd.DataFrame) -> None:
    df = df.copy()
    if "date" in df.columns:
        df["date"] = df["date"].astype(str)
    df.to_sql(name, conn, if_exists="append", index=False)


def run_sql_file(conn: sqlite3.Connection, sql_path: Path) -> None:
    if not sql_path.exists():
        return
    sql_text = sql_path.read_text(encoding="utf-8")
    conn.executescript(sql_text)
    conn.commit()


def query_to_df(conn: sqlite3.Connection, query: str) -> pd.DataFrame:
    return pd.read_sql_query(query, conn)
