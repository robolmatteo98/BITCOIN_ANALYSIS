CREATE OR REPLACE VIEW address_activity_summary AS
WITH activity AS (
    SELECT
        out.fk_address_code AS address,
        b.time
    FROM tx_output out
    JOIN transaction t
      ON t.id = out.fk_transaction_id
    JOIN block b
      ON b.id = t.fk_block_id
    WHERE out.fk_address_code IS NOT NULL
)
SELECT
    address,
    MIN(time) AS first_seen,
    MAX(time) AS last_seen,
    MAX(time) - MIN(time) AS active_span,
    COUNT(*) AS activity_count,
    COUNT(DISTINCT to_timestamp(time)::date) AS active_days
FROM activity
GROUP BY address;

-- Ti dà feature temporali di base: prima attività, ultima attività, durata, giorni attivi, numero di eventi. Serve tantissimo per geolocalizzazione e analisi comportamentale.