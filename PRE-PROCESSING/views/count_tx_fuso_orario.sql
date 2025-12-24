CREATE VIEW count_tx AS (
  select count(*) as total_tx, fk_address_code AS address 
  from tx_output 
  group by fk_address_code 
  having count(*) > 2
);
