from pyspark.sql.types import LongType, StringType, StructField, StructType

# crea un dataframe tipato per scrivere su file .csv i risultati
# RDD dizionari --> DataFrame
def build_edges_df(spark, edges_rdd):
    schema = StructType([
      StructField("src", StringType(), False),
      StructField("dst", StringType(), False),
      StructField("value_sat", LongType(), False),
      StructField("timestamp", LongType(), True),
    ])

    return spark.createDataFrame(edges_rdd.map(normalize_edge_for_dataframe), schema=schema)

def normalize_edge_for_dataframe(edge):
    return {
        "src": edge["src"],
        "dst": edge["dst"],
        "value_sat": int(edge.get("value_sat") or 0),
        "timestamp": int(edge["timestamp"]) if edge.get("timestamp") is not None else None,
    }