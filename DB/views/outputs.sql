CREATE VIEW outputs AS (
    SELECT
        out.fk_transaction_id AS txid,
        out.fk_address_code AS to_address,
        out.amount AS output_amount
    FROM tx_output out
);