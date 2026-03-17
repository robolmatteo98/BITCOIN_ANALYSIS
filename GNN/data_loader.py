import pandas as pd
import psycopg2
import os
from dotenv import load_dotenv, find_dotenv

dotenv_path = find_dotenv()
load_dotenv(dotenv_path=".env")

def load_flows():
  DB_NAME = os.getenv("DB_NAME")
  DB_USER = os.getenv("DB_USER")
  DB_PASSWORD = os.getenv("DB_PASSWORD")
  DB_HOST = os.getenv("DB_HOST")
  DB_PORT = os.getenv("DB_PORT")

  conn = psycopg2.connect(
    database=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT
  )

  query = """
    SELECT 
        from_address, 
        to_address, 
        flow_amount,
        time
    FROM flows_view
    WHERE from_address IS NOT NULL
    AND to_address IS NOT NULL
  """

  df = pd.read_sql(query, conn)
  conn.close()

  # il tempo deve essere numerico (timestamp)
  df["time"] = pd.to_numeric(df["time"])

  return df