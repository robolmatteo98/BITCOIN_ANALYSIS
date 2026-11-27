CREATE TABLE IF NOT EXISTS block (
  id INTEGER PRIMARY KEY,
  hash TEXT UNIQUE,
  time INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS transaction (
  id SERIAL PRIMARY KEY,
  -- Normalmente un txid identifica una sola transazione. Esistono però due
  -- coinbase duplicate storiche, ammesse da BIP30 nei blocchi 91842 e 91880.
  txid TEXT NOT NULL,
  fk_block_id INTEGER NOT NULL,
  n_inputs INTEGER NOT NULL DEFAULT 1,
  n_outputs INTEGER NOT NULL DEFAULT 1,
  total_amount BIGINT NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS address (
  id BIGSERIAL PRIMARY KEY,
  code TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS script_type (
  id SMALLSERIAL PRIMARY KEY,
  code TEXT NOT NULL UNIQUE,
  description TEXT NOT NULL,
  is_addressable BOOLEAN NOT NULL DEFAULT TRUE
);

INSERT INTO script_type (code, description, is_addressable) VALUES
  ('P2PK', 'Pay to public key', TRUE),
  ('P2PKH', 'Pay to public key hash', TRUE),
  ('P2SH', 'Pay to script hash', TRUE),
  ('P2WPKH', 'Pay to witness public key hash', TRUE),
  ('P2WSH', 'Pay to witness script hash', TRUE),
  ('P2TR', 'Pay to taproot', TRUE),
  ('MULTISIG', 'Bare multisig', TRUE),
  ('OP_RETURN', 'Null data output', FALSE),
  ('UNKNOWN', 'Unrecognized script', FALSE)
ON CONFLICT (code) DO NOTHING;

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
  fk_address_id BIGINT,
  fk_script_type_id SMALLINT
);

ALTER TABLE transaction ADD CONSTRAINT FK_block FOREIGN KEY (fk_block_id) REFERENCES block (id);
ALTER TABLE tx_input ADD CONSTRAINT FK_transaction_tx_in FOREIGN KEY (fk_transaction_id) REFERENCES transaction (id);
ALTER TABLE tx_input ADD CONSTRAINT FK_prev_transaction_tx_in FOREIGN KEY (prev_transaction_id) REFERENCES transaction (id);
ALTER TABLE tx_output ADD CONSTRAINT FK_tx_out_transaction FOREIGN KEY (fk_transaction_id) REFERENCES transaction (id);

ALTER TABLE tx_output ADD CONSTRAINT FK_address_id_tx_out FOREIGN KEY (fk_address_id) REFERENCES address (id);
ALTER TABLE tx_output ADD CONSTRAINT FK_script_type_id_tx_out FOREIGN KEY (fk_script_type_id) REFERENCES script_type (id);
CREATE INDEX idx_transaction_txid ON transaction (txid);
CREATE INDEX idx_transaction_block ON transaction (fk_block_id);
CREATE INDEX idx_tx_input_transaction ON tx_input (fk_transaction_id);
CREATE INDEX idx_tx_input_prev_outpoint ON tx_input (prev_transaction_id, prev_vout);
CREATE INDEX idx_tx_output_tx_n ON tx_output (fk_transaction_id, n);
CREATE INDEX idx_tx_output_address_id ON tx_output (fk_address_id);
CREATE INDEX idx_tx_output_script_type ON tx_output (fk_script_type_id);
CREATE INDEX idx_block_time ON block (time);
CREATE INDEX idx_blockchain_statistics_dataset_time ON blockchain_statistics (dataset_name, computed_at DESC);
