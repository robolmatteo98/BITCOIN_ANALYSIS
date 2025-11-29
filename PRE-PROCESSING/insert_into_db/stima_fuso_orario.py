import psycopg2
import datetime
from collections import Counter
import numpy as np

# connessione al DB
conn = psycopg2.connect(

)
cursor = conn.cursor()

# esempio: prendere tutte le transazioni di un indirizzo
address = "1DNpoCx2bpVuhb2HAnaiFVZrLFpbPpqAd2"

cursor.execute("""
    SELECT b.time
    FROM bitcoin_tx_output o
    JOIN bitcoin_transaction t ON t.id = o.fk_transaction_id
    JOIN bitcoin_block b ON b.id = t.fk_block_id
    WHERE o.fk_address_code = %s
""", (address,))

timestamps = [row[0] for row in cursor.fetchall()]

hours = [datetime.datetime.utcfromtimestamp(ts).hour for ts in timestamps]

hour_counts = Counter(hours)

# ordinato per ora
hour_activity = sorted(hour_counts.items())
for h, c in hour_activity:
  print(f"Hour {h}: {c} tx")


# Troviamo la fascia di ore con più transazioni (assumendo comportamento umano)
active_hours = [h for h, c in hour_activity for _ in range(c)]

# testiamo tutti i fusi orari (offset da -12 a +12)
best_offset = 0
best_activity_score = 0

for offset in range(-12, 13):
    shifted_hours = [(h + offset) % 24 for h in active_hours]
    # score: numero di transazioni nelle ore "diurne" 08-22
    score = sum(1 for h in shifted_hours if 8 <= h <= 22)
    if score > best_activity_score:
        best_activity_score = score
        best_offset = offset

print(f"Probabile fuso orario: UTC{best_offset:+}")
