CREATE TABLE IF NOT EXISTS bitcoin_block (
  id INTEGER PRIMARY KEY,
  hash TEXT UNIQUE,
  time INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS bitcoin_transaction (
  id SERIAL PRIMARY KEY,
  txid TEXT NOT NULL, -- unique
  fk_block_id INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS bitcoin_address (
  code TEXT PRIMARY KEY,
  region_id INTEGER
);

CREATE TABLE IF NOT EXISTS bitcoin_tx_input (
  id SERIAL PRIMARY KEY,
  fk_transaction_id INTEGER NOT NULL,
  prev_transaction_id INTEGER,
  prev_vout INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS bitcoin_tx_output (
  id SERIAL PRIMARY KEY,
  fk_transaction_id INTEGER NOT NULL,
  n INTEGER NOT NULL,
  amount DECIMAL NOT NULL,
  fk_address_code TEXT
);

CREATE TABLE region (
  id SERIAL PRIMARY KEY,
  name VARCHAR(50) NOT NULL,
  utc_start INT NOT NULL,
  utc_end INT NOT NULL
);

ALTER TABLE bitcoin_address ADD COLUMN region_id INT REFERENCES region(id);

ALTER TABLE bitcoin_transaction ADD CONSTRAINT FK_bitcoin_block FOREIGN KEY (fk_block_id) REFERENCES bitcoin_block (id);
ALTER TABLE bitcoin_tx_input ADD CONSTRAINT FK_bitcoin_transaction_tx_in FOREIGN KEY (fk_transaction_id) REFERENCES bitcoin_transaction (id);
ALTER TABLE bitcoin_tx_input ADD CONSTRAINT FK_prev_bitcoin_transaction_tx_in FOREIGN KEY (prev_transaction_id) REFERENCES bitcoin_transaction (id);
ALTER TABLE bitcoin_tx_output ADD CONSTRAINT FK_bitcoin_transaction_tx_out FOREIGN KEY (fk_transaction_id) REFERENCES bitcoin_transaction (id);

ALTER TABLE bitcoin_tx_output ADD CONSTRAINT FK_bitcoin_address_tx_out FOREIGN KEY (fk_address_code) REFERENCES bitcoin_address (code);
--ALTER TABLE bitcoin_address ADD CONSTRAINT FK_region FOREIGN KEY (fk_address_code) REFERENCES bitcoin_address (code);

INSERT INTO bitcoin_address (code)
SELECT DISTINCT fk_address_code
FROM bitcoin_tx_output
WHERE fk_address_code IS NOT NULL
ON CONFLICT (code) DO NOTHING;