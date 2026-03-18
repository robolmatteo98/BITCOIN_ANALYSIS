import os

import pandas as pd
import psycopg2
from dotenv import find_dotenv, load_dotenv


dotenv_path = find_dotenv()
load_dotenv(dotenv_path=".env")


def _get_connection():
    DB_NAME = os.getenv("DB_NAME")
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_HOST = os.getenv("DB_HOST")
    DB_PORT = os.getenv("DB_PORT")

    return psycopg2.connect(
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
    )


def load_flows_with_address_features():
    conn = _get_connection()

    flows_query = """
        SELECT
            from_address,
            to_address,
            flow_amount,
            time
        FROM flows_view
        WHERE from_address IS NOT NULL
          AND to_address IS NOT NULL
    """

    address_features_query = """
        SELECT
            address,
            received_output_count,
            spent_output_count,
            unspent_output_count,
            total_received,
            total_sent,
            net_flow,
            estimated_balance,
            first_seen,
            last_seen,
            active_span,
            activity_count,
            active_days,
            region_id
        FROM address_feature_summary
    """

    df_flows = pd.read_sql(flows_query, conn)
    df_address_features = pd.read_sql(address_features_query, conn)
    conn.close()

    df_flows["time"] = pd.to_numeric(df_flows["time"])

    numeric_columns = [
        "received_output_count",
        "spent_output_count",
        "unspent_output_count",
        "total_received",
        "total_sent",
        "net_flow",
        "estimated_balance",
        "first_seen",
        "last_seen",
        "active_span",
        "activity_count",
        "active_days",
        "region_id",
    ]

    for column in numeric_columns:
        df_address_features[column] = pd.to_numeric(df_address_features[column], errors="coerce")

    return df_flows, df_address_features
