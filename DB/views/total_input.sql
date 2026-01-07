CREATE VIEW total_input AS (
    SELECT 
        txid,
        SUM(input_amount) AS total_input_amount
    FROM input_values
    GROUP BY txid
);