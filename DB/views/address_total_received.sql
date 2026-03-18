CREATE OR REPLACE VIEW address_total_received AS
SELECT
    out.fk_address_code AS address,
    COUNT(*) AS received_output_count,
    SUM(out.amount) AS total_received
FROM tx_output out
WHERE out.fk_address_code IS NOT NULL
GROUP BY out.fk_address_code;

-- Ti dice quanto un indirizzo ha ricevuto in totale e quanti output ha ricevuto. Serve per misurare “peso economico” e attività in ingresso.