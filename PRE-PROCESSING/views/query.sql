CREATE VIEW input_values AS (
    SELECT
        inp.fk_transaction_id AS txid,
        prev_out.fk_address_code AS from_address,
        prev_out.amount AS input_amount
    FROM bitcoin_tx_input inp
    JOIN bitcoin_tx_output prev_out
         ON prev_out.fk_transaction_id = inp.prev_transaction_id
        AND prev_out.n = inp.prev_vout
);

CREATE VIEW total_input AS (
    SELECT 
        txid,
        SUM(input_amount) AS total_input_amount
    FROM input_values
    GROUP BY txid
);

CREATE VIEW outputs AS (
    SELECT
        out.fk_transaction_id AS txid,
        out.fk_address_code AS to_address,
        out.amount AS output_amount
    FROM bitcoin_tx_output out
);

CREATE VIEW flows AS (
    SELECT
        i.from_address,
        o.to_address,
        (i.input_amount / t.total_input_amount) * o.output_amount AS flow_amount
    FROM input_values i
    JOIN total_input t ON t.txid = i.txid
    JOIN outputs o ON o.txid = i.txid
);

-- calcola il portafoglio totale di ogni indirizzo
CREATE VIEW wallet AS (
    SELECT fk_address_code, SUM(amount) AS total_amount
    FROM bitcoin_tx_output
    GROUP BY fk_address_code
);

SELECT
    from_address,
    to_address,
    SUM(flow_amount) AS total_btc_sent,
    COUNT(*) AS total_tx_sent
FROM flows
WHERE from_address IS NOT NULL
  AND to_address IS NOT NULL
  AND from_address <> to_address
GROUP BY from_address, to_address
ORDER BY total_tx_sent DESC;