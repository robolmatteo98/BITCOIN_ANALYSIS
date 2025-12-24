CREATE VIEW flows_view AS
WITH input_values AS (
    SELECT
        inp.fk_transaction_id AS txid,
        prev_out.fk_address_code AS from_address,
        prev_out.amount AS input_amount
    FROM bitcoin_tx_input inp
    JOIN bitcoin_tx_output prev_out
      ON prev_out.fk_transaction_id = inp.prev_transaction_id
     AND prev_out.n = inp.prev_vout
),
total_input AS (
    SELECT
        txid,
        SUM(input_amount) AS total_input_amount
    FROM input_values
    GROUP BY txid
)
SELECT
    i.from_address,
    o.fk_address_code AS to_address,
    (i.input_amount / t.total_input_amount) * o.amount AS flow_amount,
    i.txid,
    b.time
FROM input_values i
JOIN total_input t       ON t.txid = i.txid
JOIN bitcoin_tx_output o ON o.fk_transaction_id = i.txid
JOIN bitcoin_transaction tra ON tra.id = i.txid
JOIN bitcoin_block b     ON b.id = tra.fk_block_id;