import pandas as pd
import networkx as nx
import psycopg2

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
    host=DB_HOST
  )

  query = """
    SELECT from_address, to_address, flow_amount
    FROM flows_view
    WHERE from_address IS NOT NULL
    AND to_address IS NOT NULL
  """

  return pd.read_sql(query, conn)