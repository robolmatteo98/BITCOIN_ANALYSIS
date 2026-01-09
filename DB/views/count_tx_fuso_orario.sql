CREATE VIEW count_tx AS (
  SELECT count(*) AS total_tx, from_address
  FROM flows_view
  GROUP BY from_address
  HAVING count(*) > 2
)