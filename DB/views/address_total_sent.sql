CREATE OR REPLACE VIEW address_total_sent AS
SELECT
    prev_out.fk_address_code AS address,
    COUNT(*) AS spent_output_count,
    SUM(prev_out.amount) AS total_sent
FROM tx_input inp
JOIN tx_output prev_out
  ON prev_out.fk_transaction_id = inp.prev_transaction_id
 AND prev_out.n = inp.prev_vout
WHERE prev_out.fk_address_code IS NOT NULL
GROUP BY prev_out.fk_address_code;

-- Ti dice quanto un indirizzo ha speso davvero, guardando gli output precedenti consumati come input. Serve per distinguere chi accumula da chi spende.