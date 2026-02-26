SELECT
  inp.fk_transaction_id AS txid,
  prev_out.fk_address_code AS address
FROM tx_input inp
JOIN tx_output prev_out
  ON prev_out.fk_transaction_id = inp.prev_transaction_id
 AND prev_out.n = inp.prev_vout;
