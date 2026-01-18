CREATE VIEW flows_view AS
WITH input_values AS (
    SELECT
        inp.fk_transaction_id AS txid,
        prev_out.fk_address_code AS from_address,
        prev_out.amount AS input_amount
    FROM tx_input inp
    JOIN tx_output prev_out
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
    b.time,
    tra.n_inputs,
    tra.n_outputs,
    tra.total_amount
FROM input_values i
JOIN total_input t       ON t.txid = i.txid
JOIN tx_output o ON o.fk_transaction_id = i.txid
JOIN transaction tra ON tra.id = i.txid
JOIN block b     ON b.id = tra.fk_block_id;

----
CREATE OR REPLACE VIEW flows_view AS
WITH input_values AS (
    SELECT
        inp.fk_transaction_id AS txid,
        prev_out.fk_address_code AS from_address,
        prev_out.amount AS input_amount
    FROM tx_input inp
    JOIN tx_output prev_out
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
    -- Se non ci sono input → coinbase → from_address = NULL
    i.from_address,
    o.fk_address_code AS to_address,

    CASE
        WHEN t.total_input_amount IS NULL THEN o.amount
        ELSE (i.input_amount / t.total_input_amount) * o.amount
    END AS flow_amount,

    tra.id AS txid,
    b.time,
    tra.n_inputs,
    tra.n_outputs,
    tra.total_amount
FROM transaction tra
JOIN block b ON b.id = tra.fk_block_id
JOIN tx_output o ON o.fk_transaction_id = tra.id
LEFT JOIN input_values i ON i.txid = tra.id
LEFT JOIN total_input t ON t.txid = tra.id;
