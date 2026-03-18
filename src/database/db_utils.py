from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(db_path)


def write_table(conn: sqlite3.Connection, table_name: str, df: pd.DataFrame) -> None:
    out = df.copy()
    if "date" in out.columns:
        out["date"] = out["date"].astype(str)
    out.to_sql(table_name, conn, if_exists="replace", index=False)


def query_to_df(conn: sqlite3.Connection, query: str) -> pd.DataFrame:
    return pd.read_sql_query(query, conn)
