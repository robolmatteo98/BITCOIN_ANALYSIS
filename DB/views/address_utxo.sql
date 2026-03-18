CREATE OR REPLACE VIEW address_utxo AS
SELECT
    out.fk_address_code AS address,
    out.fk_transaction_id AS txid,
    out.n AS vout,
    out.amount,
    t.fk_block_id,
    b.time
FROM tx_output out
JOIN transaction t
  ON t.id = out.fk_transaction_id
JOIN block b
  ON b.id = t.fk_block_id
LEFT JOIN tx_input inp
  ON inp.prev_transaction_id = out.fk_transaction_id
 AND inp.prev_vout = out.n
WHERE out.fk_address_code IS NOT NULL
  AND inp.id IS NULL;

-- Ti elenca gli output non ancora spesi. È la base corretta per stimare il saldo, perché in Bitcoin il saldo vero deriva dagli UTXO non spesi.