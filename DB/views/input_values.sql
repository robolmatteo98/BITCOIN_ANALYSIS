CREATE VIEW input_values AS (
    SELECT
        inp.fk_transaction_id AS txid,
        prev_out.fk_address_code AS from_address,
        prev_out.amount AS input_amount
    FROM tx_input inp
    JOIN tx_output prev_out
        ON prev_out.fk_transaction_id = inp.prev_transaction_id
        AND prev_out.n = inp.prev_vout
);