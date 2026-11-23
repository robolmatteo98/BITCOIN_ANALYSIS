import argparse
import csv
import psycopg2
import datetime
from collections import Counter
from dotenv import load_dotenv
import os
from pathlib import Path

# Carica variabili dal .env
load_dotenv(dotenv_path=".env")

DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
ROOT = Path(__file__).resolve().parents[3]

print(f"Connecting to {DB_HOST}:{DB_PORT} as {DB_USER} to DB {DB_NAME}")

# Intervalli geografici usati per interpretare l'offset stimato.
REGIONS = [
    ("America del Nord (West)", -12, -8),
    ("America del Nord / Sud", -7, -3),
    ("Europa Occidentale / Africa Occidentale", -2, 0),
    ("Europa Centrale / Africa", 1, 3),
    ("Asia Occidentale / India", 4, 6),
    ("Asia Orientale", 7, 9),
    ("Oceania / Asia Pacifico", 10, 12),
]

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--output-dir",
    type=Path,
    default=ROOT / "07_reports" / "pre-processing" / "geolocation",
    help="cartella in cui salvare le stime",
)
args = parser.parse_args()

# Connessione al DB per leggere gli address e le loro transazioni.
conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
cursor = conn.cursor()

def guess_region_name(utc_offset):
    for r in REGIONS:
        name, start, end = r
        if start <= utc_offset <= end:
            return name
    return None


# Recupera tutti gli indirizzi
cursor.execute("SELECT from_address FROM count_tx")
addresses = [row[0] for row in cursor.fetchall()]
results = []

for address in addresses:
    # Prendiamo tutte le transazioni di quell'indirizzo
    cursor.execute("""
        SELECT b.time
        FROM tx_output o
        JOIN transaction t ON t.id = o.fk_transaction_id
        JOIN block b ON b.id = t.fk_block_id
        JOIN address a ON a.id = o.fk_address_id
        WHERE a.code = %s
    """, (address,))
    timestamps = [row[0] for row in cursor.fetchall()]

    n_tx = len(timestamps)

    # Ore UTC delle transazioni
    hours = [datetime.datetime.utcfromtimestamp(ts).hour for ts in timestamps]
    hour_counts = Counter(hours)
    hour_activity = sorted(hour_counts.items())

    # Lista ripetuta di ore
    active_hours = [h for h, c in hour_activity for _ in range(c)]

    # Testiamo tutti i fusi orari (offset da -12 a +12)
    best_offset = 0
    best_activity_score = 0
    active_range = list(range(10, 23))  # ore diurne 10-22

    for offset in range(-12, 13):
        shifted_hours = [(h + offset) % 24 for h in active_hours]
        score = sum(1 for h in shifted_hours if h in active_range)
        if score > best_activity_score:
            best_activity_score = score
            best_offset = offset

    # Stima area geografica
    region = guess_region_name(best_offset)

    results.append(
        {
            "address": address,
            "transaction_count": n_tx,
            "estimated_utc_offset": best_offset,
            "activity_score": best_activity_score,
            "region": region,
        }
    )

    print(f"Address: {address} | Tx: {n_tx} | UTC offset: {best_offset:+} | Region: {region}")

conn.close()

args.output_dir.mkdir(parents=True, exist_ok=True)
output_path = args.output_dir / "address_geolocation.csv"
with output_path.open("w", newline="", encoding="utf-8") as output_file:
    fieldnames = [
        "address", "transaction_count", "estimated_utc_offset",
        "activity_score", "region",
    ]
    writer = csv.DictWriter(output_file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print(f"Risultati salvati in: {output_path}")
