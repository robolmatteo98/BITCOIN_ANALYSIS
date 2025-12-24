-- calcola il portafoglio totale di ogni indirizzo
CREATE VIEW wallet AS (
    SELECT fk_address_code, SUM(amount) AS total_amount
    FROM tx_output
    GROUP BY fk_address_code
);