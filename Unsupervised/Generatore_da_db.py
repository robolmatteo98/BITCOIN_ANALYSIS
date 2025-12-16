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
    SELECT
      addr_in.code  AS from_address,
      addr_out.code AS to_address,
      txo.amount    AS flow_amount,
      b.time        AS time,
      tx.txid       AS txid
    FROM bitcoin_transaction tx
    JOIN bitcoin_block b
      ON b.id = tx.fk_block_id

    JOIN bitcoin_tx_input txi
      ON txi.fk_transaction_id = tx.id
      AND txi.prev_transaction_id IS NOT NULL

    JOIN bitcoin_tx_output prev_out
      ON prev_out.fk_transaction_id = txi.prev_transaction_id
      AND prev_out.n = txi.prev_vout

    JOIN bitcoin_address addr_in
      ON addr_in.code = prev_out.fk_address_code

    JOIN bitcoin_tx_output txo
      ON txo.fk_transaction_id = tx.id

    JOIN bitcoin_address addr_out
      ON addr_out.code = txo.fk_address_code
    """

    if limit:
        query += f" LIMIT {limit}"

    df_edges = pd.read_sql(query, conn)
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
    SELECT
      addr_in.code  AS from_address,
      addr_out.code AS to_address,
      txo.amount    AS flow_amount,
      b.time        AS time,
      tx.txid       AS txid
    FROM bitcoin_transaction tx
    JOIN bitcoin_block b
      ON b.id = tx.fk_block_id

    JOIN bitcoin_tx_input txi
      ON txi.fk_transaction_id = tx.id
      AND txi.prev_transaction_id IS NOT NULL

    JOIN bitcoin_tx_output prev_out
      ON prev_out.fk_transaction_id = txi.prev_transaction_id
      AND prev_out.n = txi.prev_vout

    JOIN bitcoin_address addr_in
      ON addr_in.code = prev_out.fk_address_code

    JOIN bitcoin_tx_output txo
      ON txo.fk_transaction_id = tx.id

    JOIN bitcoin_address addr_out
      ON addr_out.code = txo.fk_address_code
    """

  # Apri una connessione esplicita e usa sqlalchemy.text()
  with engine.connect() as conn:
    df = pd.read_sql(text(query), con=conn)  # <-- qui il fix

  # Lista indirizzi unici
  addresses = pd.unique(df[["from_address", "to_address"]].values.ravel())

  return df, addresses
