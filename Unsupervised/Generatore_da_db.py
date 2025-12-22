import os
import psycopg2
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

def load_bitcoin_edges_from_db(limit=None):
    load_dotenv(dotenv_path=".env")

    print(os.getenv("DB_NAME"))

    conn = psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
    )

    query = f"""
      SELECT *
      FROM flows
    """

    if limit:
        query += f" LIMIT {limit}"

    df_edges = pd.read_sql(query, conn)
    print(df_edges)
    conn.close()

    # lista indirizzi
    addresses = pd.unique(
        df_edges[["from_address", "to_address"]].values.ravel()
    )

    return df_edges, addresses

def load_bitcoin_edges_from_db_without_warning():
  load_dotenv(dotenv_path=".env")
  print("Database:", os.getenv("DB_NAME"))

  engine = create_engine(
      f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
      f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
  )

  query = f"""
    SELECT *
    FROM flows
    """

  # Apri una connessione esplicita e usa sqlalchemy.text()
  with engine.connect() as conn:
    df = pd.read_sql(text(query), con=conn)  # <-- qui il fix

  # Lista indirizzi unici
  addresses = pd.unique(df[["from_address", "to_address"]].values.ravel())

  return df, addresses