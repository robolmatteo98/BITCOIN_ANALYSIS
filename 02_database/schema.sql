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
  total_amount BIGINT NOT NULL DEFAULT 0,
  -- Score euristico CoinJoin calcolato in fase di preprocessing.
  coinjoin_score NUMERIC(6,5)
);

CREATE TABLE IF NOT EXISTS address_cluster (
  id BIGSERIAL PRIMARY KEY,
  cluster_key TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS address (
  id BIGSERIAL PRIMARY KEY,
  code TEXT NOT NULL UNIQUE,
  region_id INTEGER,
  cluster_id BIGINT REFERENCES address_cluster (id)
);

-- Tabella di appoggio con il risultato PageRank corrente per address.
-- I record vengono sostituiti a ogni esecuzione; non conserva lo storico dei
-- dataset o delle elaborazioni precedenti.
CREATE TABLE IF NOT EXISTS address_pagerank (
  id BIGSERIAL PRIMARY KEY,
  mode TEXT NOT NULL CHECK (mode IN ('basic', 'combined', 'paper')),
  iterations INTEGER NOT NULL CHECK (iterations > 0),
  damping NUMERIC(5,4) NOT NULL CHECK (damping >= 0 AND damping <= 1),
  graph_start_time BIGINT,
  graph_end_time BIGINT,
  node_count BIGINT NOT NULL CHECK (node_count >= 0),
  fk_address_id BIGINT NOT NULL REFERENCES address (id),
  rank_value DOUBLE PRECISION NOT NULL CHECK (rank_value >= 0),
  rank_position BIGINT,
  UNIQUE (mode, fk_address_id)
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

-- Etichette pubbliche sugli address. La tabella contiene una riga per ogni
-- evidenza/source assertion: lo stesso address può avere più fonti.
-- fk_address_id resta NULL quando l'address etichettato non è presente nel
-- sottoinsieme di blocchi importato localmente.
CREATE TABLE IF NOT EXISTS known_address_labels (
  id BIGSERIAL PRIMARY KEY,
  address_code TEXT NOT NULL,
  fk_address_id BIGINT REFERENCES address (id),
  entity_name TEXT NOT NULL,
  category TEXT NOT NULL,
  label TEXT NOT NULL,
  confidence NUMERIC(3,2) NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
  source_type TEXT NOT NULL,
  source_url TEXT NOT NULL,
  source_title TEXT NOT NULL,
  source_published_at DATE,
  source_accessed_at DATE NOT NULL DEFAULT CURRENT_DATE,
  notes TEXT,
  UNIQUE (address_code, entity_name, label, source_url)
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

CREATE TABLE region (
  id SERIAL PRIMARY KEY,
  name VARCHAR(50) NOT NULL,
  utc_start INT NOT NULL,
  utc_end INT NOT NULL
);

INSERT INTO region (name, utc_start, utc_end) VALUES
('America del Nord (West)', -12, -8),
('America del Nord / Sud', -7, -3),
('Europa Occidentale / Africa Occidentale', -2, 0),
('Europa Centrale / Africa', 1, 3),
('Asia Occidentale / India', 4, 6),
('Asia Orientale', 7, 9),
('Oceania / Asia Pacifico', 10, 12);

ALTER TABLE transaction ADD CONSTRAINT FK_block FOREIGN KEY (fk_block_id) REFERENCES block (id);
ALTER TABLE tx_input ADD CONSTRAINT FK_transaction_tx_in FOREIGN KEY (fk_transaction_id) REFERENCES transaction (id);
ALTER TABLE tx_input ADD CONSTRAINT FK_prev_transaction_tx_in FOREIGN KEY (prev_transaction_id) REFERENCES transaction (id);
ALTER TABLE tx_output ADD CONSTRAINT FK_tx_out_transaction FOREIGN KEY (fk_transaction_id) REFERENCES transaction (id);

ALTER TABLE tx_output ADD CONSTRAINT FK_address_id_tx_out FOREIGN KEY (fk_address_id) REFERENCES address (id);
ALTER TABLE address ADD CONSTRAINT FK_region_address FOREIGN KEY (region_id) REFERENCES region (id);

CREATE INDEX idx_transaction_txid ON transaction (txid);
CREATE INDEX idx_transaction_block ON transaction (fk_block_id);
CREATE INDEX idx_tx_input_transaction ON tx_input (fk_transaction_id);
CREATE INDEX idx_tx_input_prev_outpoint ON tx_input (prev_transaction_id, prev_vout);
CREATE INDEX idx_tx_output_tx_n ON tx_output (fk_transaction_id, n);
CREATE INDEX idx_tx_output_address_id ON tx_output (fk_address_id);
CREATE INDEX idx_address_cluster ON address (cluster_id);
CREATE INDEX idx_address_pagerank_mode ON address_pagerank (mode);
CREATE INDEX idx_address_pagerank_address ON address_pagerank (fk_address_id);
CREATE INDEX idx_address_pagerank_rank ON address_pagerank (mode, rank_value DESC);
CREATE INDEX idx_block_time ON block (time);
CREATE INDEX idx_blockchain_statistics_dataset_time
  ON blockchain_statistics (dataset_name, computed_at DESC);
CREATE INDEX idx_known_address_labels_code ON known_address_labels (address_code);
CREATE INDEX idx_known_address_labels_fk_address ON known_address_labels (fk_address_id);
CREATE INDEX idx_known_address_labels_category ON known_address_labels (category);
