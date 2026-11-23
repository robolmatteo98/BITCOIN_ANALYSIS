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

-- Snapshot delle statistiche calcolate sui blocchi JSON. Il report completo
-- resta disponibile in JSONB, mentre le colonne principali permettono query
-- rapide e confronti tra importazioni successive.
CREATE TABLE IF NOT EXISTS blockchain_statistics (
  id BIGSERIAL PRIMARY KEY,
  dataset_name TEXT NOT NULL,
  computed_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
  first_height BIGINT,
  last_height BIGINT,
  block_count BIGINT NOT NULL,
  transaction_count BIGINT NOT NULL,
  input_count BIGINT NOT NULL,
  output_count BIGINT NOT NULL,
  address_count BIGINT,
  total_output_satoshis NUMERIC,
  total_fee_satoshis NUMERIC,
  report JSONB NOT NULL
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
  fk_address_id BIGINT
);

ALTER TABLE transaction ADD CONSTRAINT FK_block FOREIGN KEY (fk_block_id) REFERENCES block (id);
ALTER TABLE tx_input ADD CONSTRAINT FK_transaction_tx_in FOREIGN KEY (fk_transaction_id) REFERENCES transaction (id);
ALTER TABLE tx_input ADD CONSTRAINT FK_prev_transaction_tx_in FOREIGN KEY (prev_transaction_id) REFERENCES transaction (id);
ALTER TABLE tx_output ADD CONSTRAINT FK_tx_out_transaction FOREIGN KEY (fk_transaction_id) REFERENCES transaction (id);

ALTER TABLE tx_output ADD CONSTRAINT FK_address_id_tx_out FOREIGN KEY (fk_address_id) REFERENCES address (id);
CREATE INDEX idx_transaction_txid ON transaction (txid);
CREATE INDEX idx_transaction_block ON transaction (fk_block_id);
CREATE INDEX idx_tx_input_transaction ON tx_input (fk_transaction_id);
CREATE INDEX idx_tx_input_prev_outpoint ON tx_input (prev_transaction_id, prev_vout);
CREATE INDEX idx_tx_output_tx_n ON tx_output (fk_transaction_id, n);
CREATE INDEX idx_tx_output_address_id ON tx_output (fk_address_id);
CREATE INDEX idx_block_time ON block (time);
CREATE INDEX idx_blockchain_statistics_dataset_time ON blockchain_statistics (dataset_name, computed_at DESC);