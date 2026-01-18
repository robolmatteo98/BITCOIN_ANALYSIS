import os
import psycopg2
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

def load_bitcoin_edges_from_db_without_warning():
  load_dotenv(dotenv_path=".env")
  print("Database:", os.getenv("DB_NAME"))

  engine = create_engine(
      f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
      f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
  )

  query = f"""
    SELECT *
    FROM flows_view
    """

  # Apri una connessione esplicita e usa sqlalchemy.text()
  with engine.connect() as conn:
    df = pd.read_sql(text(query), con=conn)  # <-- qui il fix

  # Lista indirizzi unici
  addresses = pd.unique(df[["from_address", "to_address"]].values.ravel())

  return df, addresses