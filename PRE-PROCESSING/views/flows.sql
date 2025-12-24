CREATE VIEW flows AS (
    SELECT
        i.from_address,
        o.to_address,
        (i.input_amount / t.total_input_amount) * o.output_amount AS flow_amount,
        i.txid,
        b.time
    FROM input_values i
    JOIN total_input t ON t.txid = i.txid
    JOIN outputs o ON o.txid = i.txid
    JOIN transaction tra ON i.txid = tra.id
    JOIN block b ON tra.fk_block_id = b.id
);