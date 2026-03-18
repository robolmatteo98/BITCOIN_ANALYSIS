CREATE OR REPLACE VIEW address_balance_summary AS
WITH received AS (
    SELECT
        address,
        received_output_count,
        total_received
    FROM address_total_received
),
sent AS (
    SELECT
        address,
        spent_output_count,
        total_sent
    FROM address_total_sent
),
utxo AS (
    SELECT
        address,
        COUNT(*) AS unspent_output_count,
        SUM(amount) AS estimated_balance
    FROM address_utxo
    GROUP BY address
)
SELECT
    a.code AS address,
    COALESCE(received.received_output_count, 0) AS received_output_count,
    COALESCE(sent.spent_output_count, 0) AS spent_output_count,
    COALESCE(utxo.unspent_output_count, 0) AS unspent_output_count,
    COALESCE(received.total_received, 0) AS total_received,
    COALESCE(sent.total_sent, 0) AS total_sent,
    COALESCE(received.total_received, 0) - COALESCE(sent.total_sent, 0) AS net_flow,
    COALESCE(utxo.estimated_balance, 0) AS estimated_balance
FROM address a
LEFT JOIN received
  ON received.address = a.code
LEFT JOIN sent
  ON sent.address = a.code
LEFT JOIN utxo
  ON utxo.address = a.code;

-- Riassume tutto in una riga per indirizzo: ricevuto, speso, saldo stimato, numero di UTXO, net flow. Questa è già quasi una tabella di feature.