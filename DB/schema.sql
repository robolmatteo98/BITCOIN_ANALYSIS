CREATE TABLE IF NOT EXISTS block (
  id INTEGER PRIMARY KEY,
  hash TEXT UNIQUE,
  time INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS transaction (
  id SERIAL PRIMARY KEY,
  txid TEXT NOT NULL, -- unique
  fk_block_id INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS address (
  code TEXT PRIMARY KEY,
  region_id INTEGER
);

CREATE TABLE IF NOT EXISTS tx_input (
  id SERIAL PRIMARY KEY,
  fk_transaction_id INTEGER NOT NULL,
  prev_transaction_id INTEGER,
  prev_vout INTEGER
);

CREATE TABLE IF NOT EXISTS tx_output (
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

ALTER TABLE transaction ADD CONSTRAINT FK_block FOREIGN KEY (fk_block_id) REFERENCES block (id);
ALTER TABLE tx_input ADD CONSTRAINT FK_transaction_tx_in FOREIGN KEY (fk_transaction_id) REFERENCES transaction (id);
ALTER TABLE tx_input ADD CONSTRAINT FK_prev_transaction_tx_in FOREIGN KEY (prev_transaction_id) REFERENCES transaction (id);
ALTER TABLE tx_output ADD CONSTRAINT FK_tx_out_transaction FOREIGN KEY (fk_transaction_id) REFERENCES transaction (id);

ALTER TABLE tx_output ADD CONSTRAINT FK_address_tx_out FOREIGN KEY (fk_address_code) REFERENCES address (code);