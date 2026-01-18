ALTER TABLE transaction
ADD COLUMN n_inputs INTEGER,
ADD COLUMN n_outputs INTEGER,
ADD COLUMN total_amount BIGINT;

UPDATE transaction t
SET
    n_inputs = i.cnt
FROM (
    SELECT
        prev.fk_transaction_id AS txid,
        COUNT(*) AS cnt,
        SUM(prev.amount) AS total
    FROM tx_input inp
    JOIN tx_output prev
      ON prev.fk_transaction_id = inp.prev_transaction_id
     AND prev.n = inp.prev_vout
    GROUP BY prev.fk_transaction_id
) i
WHERE t.id = i.txid;

UPDATE transaction t
SET
    n_outputs = o.cnt,
    total_amount = o.total
FROM (
    SELECT
        fk_transaction_id AS txid,
        COUNT(*) AS cnt,
        SUM(amount) AS total
    FROM tx_output
    GROUP BY fk_transaction_id
) o
WHERE t.id = o.txid;

-- default
ALTER TABLE transaction
ALTER COLUMN n_inputs SET DEFAULT 1;

ALTER TABLE transaction
ALTER COLUMN n_outputs SET DEFAULT 1;

-- set inputs and outputs NULL
UPDATE transaction
SET n_inputs = 1
WHERE n_inputs IS NULL;

UPDATE transaction
SET n_outputs = 1
WHERE n_outputs IS NULL;