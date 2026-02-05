import os
from dotenv import load_dotenv
import pandas as pd
from sqlalchemy import create_engine, text

from . import config

load_dotenv()

def engine():
    url = os.getenv("DATABASE_URL")
    if not url:
        raise ValueError("DATABASE_URL missing. Put it in .env in repo root.")
    return create_engine(url, pool_pre_ping=True)

def create_tables(eng):
    # Creates tables if they don't exist (axes 1-8 only)
    cols = ",\n".join([f"{c} DOUBLE PRECISION" for c in config.DB_AXIS_COLS])
    sql_training = f"""
    CREATE TABLE IF NOT EXISTS {config.TRAINING_TABLE} (
        {config.DB_TIME} DOUBLE PRECISION,
        {cols},
        ingested_at TIMESTAMP DEFAULT NOW()
    );
    """
    sql_stream = f"""
    CREATE TABLE IF NOT EXISTS {config.STREAMING_TABLE} (
        {config.DB_TIME} DOUBLE PRECISION,
        {cols},
        ingested_at TIMESTAMP DEFAULT NOW()
    );
    """
    sql_events = f"""
    CREATE TABLE IF NOT EXISTS {config.EVENTS_TABLE} (
        id BIGSERIAL PRIMARY KEY,
        axis TEXT,
        event_type TEXT,
        start_time DOUBLE PRECISION,
        end_time DOUBLE PRECISION,
        duration_s DOUBLE PRECISION,
        max_residual DOUBLE PRECISION,
        threshold DOUBLE PRECISION,
        created_at TIMESTAMP DEFAULT NOW()
    );
    """
    with eng.begin() as conn:
        conn.execute(text(sql_training))
        conn.execute(text(sql_stream))
        conn.execute(text(sql_events))

def upload_df(df: pd.DataFrame, eng, table: str):
    df.to_sql(table, con=eng, if_exists="append", index=False, method="multi", chunksize=1000)

def fetch_training(eng) -> pd.DataFrame:
    cols = ", ".join([config.DB_TIME] + config.DB_AXIS_COLS)
    q = text(f"SELECT {cols} FROM {config.TRAINING_TABLE} ORDER BY {config.DB_TIME} ASC;")
    return pd.read_sql_query(q, eng)

def fetch_streaming(eng) -> pd.DataFrame:
    cols = ", ".join([config.DB_TIME] + config.DB_AXIS_COLS)
    q = text(f"SELECT {cols} FROM {config.STREAMING_TABLE} ORDER BY {config.DB_TIME} ASC;")
    return pd.read_sql_query(q, eng)

def clear_table(eng, table: str):
    with eng.begin() as conn:
        conn.execute(text(f"TRUNCATE TABLE {table};"))

def insert_events(eng, events_df: pd.DataFrame):
    events_df.to_sql(config.EVENTS_TABLE, con=eng, if_exists="append", index=False, method="multi", chunksize=500)
