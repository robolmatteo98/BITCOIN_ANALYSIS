import psycopg2
import datetime
from collections import Counter
from dotenv import load_dotenv
import os

# Carica variabili dal .env
load_dotenv(dotenv_path=".env")

DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")

print(f"Connecting to {DB_HOST}:{DB_PORT} as {DB_USER} to DB {DB_NAME}")

# Connessione al DB
conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
cursor = conn.cursor()

# Prendere tutte le transazioni di un indirizzo
address = "1H826bto12CBV9pnfvoPwS3hm4VBRHdre4"

cursor.execute(
  """
    SELECT b.time
    FROM bitcoin_tx_output o
    JOIN bitcoin_transaction t ON t.id = o.fk_transaction_id
    JOIN bitcoin_block b ON b.id = t.fk_block_id
    WHERE o.fk_address_code = %s
  """,
    (address,),
)

timestamps = [row[0] for row in cursor.fetchall()]

# Ore UTC delle transazioni
hours = [datetime.datetime.utcfromtimestamp(ts).hour for ts in timestamps]

hour_counts = Counter(hours)

# ordinato per ora
hour_activity = sorted(hour_counts.items())
for h, c in hour_activity:
    print(f"Hour {h}: {c} tx")

# Creiamo lista ripetuta di ore
active_hours = [h for h, c in hour_activity for _ in range(c)]

# Testiamo tutti i fusi orari (offset da -12 a +12)
best_offset = 0
best_activity_score = 0

# Consideriamo ore “attive” dalle 10 alle 22
active_range = list(range(10, 23))  # 10 inclusa, 22 inclusa

for offset in range(-12, 13):
    shifted_hours = [(h + offset) % 24 for h in active_hours]
    score = sum(1 for h in shifted_hours if h in active_range)
    if score > best_activity_score:
        best_activity_score = score
        best_offset = offset

print(f"\nProbabile fuso orario: UTC{best_offset:+}")


# Approssimiamo area geografica
def guess_region(utc_offset):
    if -12 <= utc_offset <= -8:
        return "America del Nord (West)"
    elif -7 <= utc_offset <= -3:
        return "America del Nord / Sud"
    elif -2 <= utc_offset <= 0:
        return "Europa Occidentale / Africa Occidentale"
    elif 1 <= utc_offset <= 3:
        return "Europa Centrale / Africa"
    elif 4 <= utc_offset <= 6:
        return "Asia Occidentale / India"
    elif 7 <= utc_offset <= 9:
        return "Asia Orientale"
    elif 10 <= utc_offset <= 12:
        return "Oceania / Asia Pacifico"
    else:
        return "Fuso non identificato"


region = guess_region(best_offset)
print(f"Area geografica approssimativa: {region}")
